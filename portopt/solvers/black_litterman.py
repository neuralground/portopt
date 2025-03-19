"""Black-Litterman solver for portfolio optimization."""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

from portopt.core.problem import PortfolioOptProblem
from portopt.core.result import PortfolioOptResult
from portopt.solvers.base import BaseSolver
from portopt.models.black_litterman import BlackLittermanModel, InvestorView


class BlackLittermanSolver(BaseSolver):
    """Black-Litterman solver for portfolio optimization.
    
    This solver implements the Black-Litterman approach to asset allocation,
    which combines market equilibrium with investor views to produce more
    stable and intuitive portfolios.
    
    The solver works in several steps:
    1. Calculate market equilibrium returns using reverse optimization
    2. Process investor views (absolute and relative)
    3. Blend market equilibrium with views using Bayesian approach
    4. Construct optimal portfolio based on posterior estimates
    
    Attributes:
        risk_aversion: Risk aversion coefficient (lambda)
        tau: Uncertainty scaling parameter for prior
        use_market_caps: Whether to use market caps for equilibrium weights
        default_view_confidence: Default confidence level for views
        bl_model: The Black-Litterman model instance
    """
    
    def __init__(self, **kwargs):
        """Initialize the Black-Litterman solver.
        
        Args:
            risk_aversion: Risk aversion coefficient (default: 2.5)
            tau: Uncertainty scaling parameter for prior (default: 0.05)
            use_market_caps: Whether to use market caps for equilibrium weights (default: True)
            default_view_confidence: Default confidence level for views (default: 0.5)
        """
        super().__init__(**kwargs)
        
        self.risk_aversion = kwargs.get('risk_aversion', 2.5)
        self.tau = kwargs.get('tau', 0.05)
        self.use_market_caps = kwargs.get('use_market_caps', True)
        self.default_view_confidence = kwargs.get('default_view_confidence', 0.5)
        
        # Create Black-Litterman model
        self.bl_model = BlackLittermanModel(
            risk_aversion=self.risk_aversion,
            tau=self.tau,
            use_market_caps=self.use_market_caps,
            default_view_confidence=self.default_view_confidence
        )
        
    def _extract_views_from_problem(
        self, 
        problem: PortfolioOptProblem
    ) -> List[InvestorView]:
        """Extract investor views from the problem constraints.
        
        Args:
            problem: Portfolio optimization problem
            
        Returns:
            List of investor views
        """
        views = []
        
        # Extract views from problem constraints
        if 'views' in problem.constraints:
            raw_views = problem.constraints['views']
            
            for raw_view in raw_views:
                assets = raw_view.get('assets', [])
                weights = raw_view.get('weights', [1.0] * len(assets))
                value = raw_view.get('value', 0.0)
                confidence = raw_view.get('confidence', self.default_view_confidence)
                is_relative = raw_view.get('is_relative', False)
                
                view = InvestorView(
                    assets=assets,
                    weights=weights,
                    value=value,
                    confidence=confidence,
                    is_relative=is_relative
                )
                views.append(view)
                
        return views
    
    def _get_market_weights(self, problem: PortfolioOptProblem) -> np.ndarray:
        """Get market capitalization weights from the problem.
        
        Args:
            problem: Portfolio optimization problem
            
        Returns:
            Market capitalization weights
        """
        if not self.use_market_caps or problem.market_caps is None:
            # Use equal weights if market caps not available
            return np.ones(problem.n_assets) / problem.n_assets
            
        # Normalize market caps to get weights
        market_weights = problem.market_caps / np.sum(problem.market_caps)
        return market_weights
    
    def solve(self, problem: PortfolioOptProblem) -> PortfolioOptResult:
        """Solve the portfolio optimization problem using Black-Litterman approach.
        
        Args:
            problem: Portfolio optimization problem
            
        Returns:
            PortfolioOptResult containing the optimized weights
        """
        start_time = time.time()
        
        # Extract views from problem
        views = self._extract_views_from_problem(problem)
        
        # Get market weights
        market_weights = self._get_market_weights(problem)
        
        # Extract asset names if available
        asset_names = problem.constraints.get('asset_names')
        
        # Calculate optimal weights using Black-Litterman model
        weights = self.bl_model.optimize_portfolio(
            cov_matrix=problem.cov_matrix,
            views=views,
            market_weights=market_weights,
            asset_names=asset_names,
            risk_aversion=self.risk_aversion
        )
        
        # Apply minimum weight constraint if specified
        min_weight = problem.constraints.get('min_weight', 0.0)
        if min_weight > 0:
            weights[weights < min_weight] = 0.0
            if np.sum(weights) > 0:  # Ensure we don't divide by zero
                weights = weights / np.sum(weights)
        
        # Calculate objective value (portfolio variance)
        objective_value = weights.T @ problem.cov_matrix @ weights
        
        # Check if solution is feasible
        feasible = True
        
        # Check turnover constraint if specified
        if 'prev_weights' in problem.constraints and 'turnover_limit' in problem.constraints:
            prev_weights = problem.constraints['prev_weights']
            turnover_limit = problem.constraints['turnover_limit']
            actual_turnover = np.sum(np.abs(weights - prev_weights))
            
            if actual_turnover > turnover_limit:
                feasible = False
        
        # Create and return result
        solve_time = time.time() - start_time
        result = PortfolioOptResult(
            weights=weights,
            objective_value=objective_value,
            solve_time=solve_time,
            feasible=feasible
        )
        
        return result
    
    def calculate_weight_uncertainty(
        self,
        problem: PortfolioOptProblem,
        n_samples: int = 1000
    ) -> Dict[str, np.ndarray]:
        """Calculate uncertainty in optimal weights.
        
        Args:
            problem: Portfolio optimization problem
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary containing mean and standard deviation of weights
        """
        # Extract views from problem
        views = self._extract_views_from_problem(problem)
        
        # Get market weights
        market_weights = self._get_market_weights(problem)
        
        # Extract asset names if available
        asset_names = problem.constraints.get('asset_names')
        
        # Calculate weight uncertainty
        uncertainty = self.bl_model.calculate_weight_uncertainty(
            cov_matrix=problem.cov_matrix,
            views=views,
            market_weights=market_weights,
            asset_names=asset_names,
            risk_aversion=self.risk_aversion,
            n_samples=n_samples
        )
        
        return uncertainty
