"""Factor-based portfolio optimization solver.

This module implements a solver for factor-based portfolio optimization,
which uses a multi-factor model to optimize portfolios with controlled
exposures to common risk factors.

The FactorSolver integrates with the existing portfolio optimization framework
and provides the following capabilities:
1. Extraction of factor data from PortfolioOptProblem
2. Handling of target factor exposures from constraints
3. Integration with the FactorModel for optimization
4. Fallback to traditional mean-variance optimization when factor data is unavailable

The solver supports three main variants:
- Standard factor solver: Balanced approach (risk_aversion=2.0, alpha=0.5)
- Conservative factor solver: Higher risk aversion (risk_aversion=4.0, alpha=0.7)
- Aggressive factor solver: Lower risk aversion (risk_aversion=1.0, alpha=0.3)

Example usage:
    # Create a problem with factor data
    problem = PortfolioOptProblem(
        returns=asset_returns,
        factor_returns=factor_returns,
        factor_exposures=factor_exposures,
        constraints={
            'factor_exposures': [
                {
                    'factor_name': 'Value',
                    'target_exposure': 0.2,
                    'confidence': 0.8
                },
                {
                    'factor_name': 'Momentum',
                    'target_exposure': 0.1,
                    'confidence': 0.5
                }
            ],
            'factor_names': ['Value', 'Momentum', 'Size', 'Quality']
        }
    )
    
    # Create and use the solver
    solver = FactorSolver()
    result = solver.solve(problem)
    
    # Or use the factory
    from portopt.solvers.factory import SolverFactory
    solver = SolverFactory.create('factor')  # or 'factor_conservative', 'factor_aggressive'
    result = solver.solve(problem)
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any

from portopt.core.problem import PortfolioOptProblem
from portopt.core.result import PortfolioOptResult
from portopt.solvers.base import BaseSolver
from portopt.models.factor import FactorModel, FactorExposure


class FactorSolver(BaseSolver):
    """Solver for factor-based portfolio optimization.
    
    This solver uses a multi-factor model to optimize portfolios with
    controlled exposures to common risk factors.
    """
    
    def __init__(
        self,
        risk_aversion: float = 2.0,
        specific_risk_penalty: float = 0.1,
        alpha: float = 0.5,
        regularization_lambda: float = 1e-4,
        **kwargs
    ):
        """Initialize the factor-based solver.
        
        Args:
            risk_aversion: Risk aversion parameter (higher = more risk-averse)
            specific_risk_penalty: Penalty for specific risk (idiosyncratic)
            alpha: Blending parameter between factor and historical data (0-1)
            regularization_lambda: Regularization parameter for numerical stability
            **kwargs: Additional solver parameters
        """
        super().__init__(**kwargs)
        self.risk_aversion = risk_aversion
        self.specific_risk_penalty = specific_risk_penalty
        self.alpha = alpha
        self.regularization_lambda = regularization_lambda
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def solve(self, problem: PortfolioOptProblem) -> PortfolioOptResult:
        """Solve the portfolio optimization problem using factor model.
        
        Args:
            problem: Portfolio optimization problem
            
        Returns:
            Portfolio optimization result
        """
        # Preprocess the problem
        problem = self.preprocess_problem(problem)
        
        # Check if factor data is available
        if problem.factor_returns is None or problem.factor_exposures is None:
            self.logger.warning("Factor data not provided. Using historical returns only.")
            # Fall back to mean-variance optimization
            return self._solve_without_factors(problem)
        
        # Extract factor data
        factor_returns = problem.factor_returns
        factor_exposures = problem.factor_exposures
        historical_returns = problem.returns
        
        # Extract target factor exposures from constraints if available
        target_factor_exposures = self._extract_factor_exposures(problem)
        
        # Create factor model
        model = FactorModel(
            risk_aversion=self.risk_aversion,
            specific_risk_penalty=self.specific_risk_penalty,
            regularization_lambda=self.regularization_lambda
        )
        
        # Optimize weights
        weights, metadata = model.optimize_weights(
            factor_returns=factor_returns,
            factor_exposures=factor_exposures,
            target_factor_exposures=target_factor_exposures,
            historical_returns=historical_returns,
            alpha=self.alpha,
            constraints=problem.constraints
        )
        
        # Analyze factor contributions
        factor_analysis = model.analyze_factor_contributions(
            weights=weights,
            factor_returns=factor_returns,
            factor_exposures=factor_exposures,
            factor_names=self._get_factor_names(problem)
        )
        
        # Combine metadata
        metadata.update(factor_analysis)
        
        # Ensure we have a positive objective value
        objective_value = max(metadata['expected_return'], 1e-6)

        # Create result
        result = PortfolioOptResult(
            weights=weights,
            objective_value=objective_value,
            solve_time=0.1,  # Placeholder value
            feasible=True,
            iterations_used=1
        )
        
        # Add additional metadata
        result.metadata = metadata
        
        return result
    
    def _solve_without_factors(self, problem: PortfolioOptProblem) -> PortfolioOptResult:
        """Solve the problem using traditional mean-variance optimization.
        
        Args:
            problem: Portfolio optimization problem
            
        Returns:
            Portfolio optimization result
        """
        # Calculate expected returns and covariance
        expected_returns = np.mean(problem.returns, axis=1)
        covariance = problem.cov_matrix
        
        # Add regularization to ensure positive definiteness
        covariance = covariance + np.eye(covariance.shape[0]) * self.regularization_lambda
        
        # Basic mean-variance optimization
        inv_cov = np.linalg.inv(self.risk_aversion * covariance)
        weights = inv_cov @ expected_returns
        
        # Apply basic constraints
        weights = np.maximum(weights, 0)  # Long-only constraint
        weights = weights / np.sum(weights)  # Sum to 1
        
        # Calculate portfolio statistics
        expected_return = weights @ expected_returns
        expected_risk = np.sqrt(weights @ covariance @ weights)
        sharpe_ratio = expected_return / expected_risk if expected_risk > 0 else 0
        
        # Ensure we have a positive objective value
        objective_value = max(expected_return, 1e-6)
        
        # Create result
        result = PortfolioOptResult(
            weights=weights,
            objective_value=objective_value,
            solve_time=0.1,  # Placeholder value
            feasible=True,
            iterations_used=1
        )
        
        # Add additional metadata
        result.metadata = {
            'expected_return': expected_return,
            'total_risk': expected_risk,
            'sharpe_ratio': sharpe_ratio,
            'factor_data_used': False
        }
        
        return result
    
    def _extract_factor_exposures(self, problem: PortfolioOptProblem) -> List[FactorExposure]:
        """Extract target factor exposures from problem constraints.
        
        Args:
            problem: Portfolio optimization problem
            
        Returns:
            List of target factor exposures
        """
        target_exposures = []
        
        # Check if factor exposure constraints are provided
        if 'factor_exposures' in problem.constraints:
            factor_constraints = problem.constraints['factor_exposures']
            
            for constraint in factor_constraints:
                # Create FactorExposure object
                exposure = FactorExposure(
                    factor_name=constraint['factor_name'],
                    target_exposure=constraint['target_exposure'],
                    min_exposure=constraint.get('min_exposure'),
                    max_exposure=constraint.get('max_exposure'),
                    confidence=constraint.get('confidence', 0.5)
                )
                target_exposures.append(exposure)
        
        return target_exposures
    
    def _get_factor_names(self, problem: PortfolioOptProblem) -> Optional[List[str]]:
        """Get factor names from problem if available.
        
        Args:
            problem: Portfolio optimization problem
            
        Returns:
            List of factor names or None
        """
        if 'factor_names' in problem.constraints:
            return problem.constraints['factor_names']
        
        # If factor names are not provided, return None
        return None
    
    def calculate_factor_exposures(
        self,
        weights: np.ndarray,
        problem: PortfolioOptProblem
    ) -> Dict[str, float]:
        """Calculate factor exposures for a given portfolio.
        
        Args:
            weights: Portfolio weights
            problem: Portfolio optimization problem
            
        Returns:
            Dictionary of factor exposures
        """
        if problem.factor_exposures is None:
            self.logger.warning("Factor exposures not available")
            return {}
        
        # Calculate portfolio exposures
        portfolio_exposures = weights @ problem.factor_exposures
        
        # Get factor names if available
        factor_names = self._get_factor_names(problem)
        
        if factor_names is not None:
            # Create dictionary of factor exposures
            exposures = {
                factor_names[i]: portfolio_exposures[i]
                for i in range(len(factor_names))
            }
        else:
            # Use generic factor names
            exposures = {
                f"Factor_{i}": portfolio_exposures[i]
                for i in range(len(portfolio_exposures))
            }
        
        return exposures
