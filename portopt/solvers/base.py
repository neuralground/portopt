"""Base solver interface for portfolio optimization."""

import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Set
from portopt.core.problem import PortfolioOptProblem
from portopt.core.result import PortfolioOptResult

class BaseSolver(ABC):
    """Abstract base class for portfolio optimization solvers.
    
    This class defines the common interface and utility functions for all portfolio
    optimization solvers. It provides methods for:
    
    1. Problem validation and preprocessing
    2. Common constraint handling
    3. Solution post-processing and validation
    4. Performance metrics calculation
    
    Derived solver classes should implement the solve() method and can leverage
    the utility functions provided here for common tasks.
    """
    
    def __init__(self, **kwargs):
        """Initialize the solver with common parameters.
        
        Args:
            **kwargs: Solver-specific parameters
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.warnings: Dict[str, Any] = {}
        
    @abstractmethod
    def solve(self, problem: PortfolioOptProblem) -> PortfolioOptResult:
        """Solve the given portfolio optimization problem.
        
        Args:
            problem: The portfolio optimization problem to solve
            
        Returns:
            A PortfolioOptResult containing the optimal weights and metadata
        """
        pass
    
    def preprocess_problem(self, problem: PortfolioOptProblem) -> PortfolioOptProblem:
        """Preprocess the problem before solving.
        
        This method performs common preprocessing steps such as:
        - Validating problem inputs
        - Handling missing data
        - Normalizing constraints
        
        Args:
            problem: The original problem instance
            
        Returns:
            Preprocessed problem instance
        """
        # Validate the problem
        problem.validate()
        
        # Create a copy of constraints with defaults for common constraints
        constraints = problem.constraints.copy()
        
        # Set default constraints if not provided
        if 'min_weight' not in constraints:
            constraints['min_weight'] = 0.0
        
        if 'max_weight' not in constraints:
            constraints['max_weight'] = 1.0
            
        if 'sum_to_one' not in constraints:
            constraints['sum_to_one'] = True
            
        # Create a new problem with processed constraints
        return PortfolioOptProblem(
            returns=problem.returns,
            constraints=constraints,
            target_return=problem.target_return,
            volumes=problem.volumes,
            spreads=problem.spreads,
            market_caps=problem.market_caps,
            factor_returns=problem.factor_returns,
            factor_exposures=problem.factor_exposures,
            classifications=problem.classifications,
            asset_classes=problem.asset_classes,
            currencies=problem.currencies,
            credit_profiles=problem.credit_profiles
        )
    
    def create_standard_constraints(self, problem: PortfolioOptProblem) -> List[Dict]:
        """Create standard optimization constraints in a solver-agnostic format.
        
        Args:
            problem: The portfolio optimization problem
            
        Returns:
            List of constraint dictionaries in a standard format
        """
        constraints = []
        
        # Sum to one constraint (if required)
        if problem.constraints.get('sum_to_one', True):
            constraints.append({
                'type': 'eq',
                'description': 'sum_to_one',
                'fun': lambda x: np.sum(x) - 1.0
            })
        
        # Minimum number of stocks constraint
        min_stocks = problem.constraints.get('min_stocks_held')
        if min_stocks is not None:
            constraints.append({
                'type': 'ineq',
                'description': 'min_stocks_held',
                'fun': lambda x: np.sum(x > problem.constraints.get('min_weight', 0.0)) - min_stocks
            })
        
        # Maximum number of stocks constraint
        max_stocks = problem.constraints.get('max_stocks_held')
        if max_stocks is not None:
            constraints.append({
                'type': 'ineq',
                'description': 'max_stocks_held',
                'fun': lambda x: max_stocks - np.sum(x > problem.constraints.get('min_weight', 0.0))
            })
            
        # Sector constraints
        if problem.classifications is not None:
            sector_limits = problem.constraints.get('sector_limits', {})
            if sector_limits:
                sectors = set(c.sector for c in problem.classifications)
                for sector in sectors:
                    if sector in sector_limits:
                        sector_mask = np.array([c.sector == sector for c in problem.classifications])
                        constraints.append({
                            'type': 'ineq',
                            'description': f'sector_limit_{sector}',
                            'fun': lambda x, mask=sector_mask, limit=sector_limits[sector]: 
                                   limit - np.sum(x * mask)
                        })
        
        return constraints
    
    def create_bounds(self, problem: PortfolioOptProblem) -> List[Tuple[float, float]]:
        """Create bounds for the optimization variables.
        
        Args:
            problem: The portfolio optimization problem
            
        Returns:
            List of (lower_bound, upper_bound) tuples for each asset
        """
        min_weight = problem.constraints.get('min_weight', 0.0)
        max_weight = problem.constraints.get('max_weight', 1.0)
        
        # Create uniform bounds for all assets
        bounds = [(min_weight, max_weight) for _ in range(problem.n_assets)]
        
        # Apply asset-specific bounds if provided
        asset_min_weights = problem.constraints.get('asset_min_weights', {})
        asset_max_weights = problem.constraints.get('asset_max_weights', {})
        
        for i in range(problem.n_assets):
            if i in asset_min_weights:
                bounds[i] = (asset_min_weights[i], bounds[i][1])
            if i in asset_max_weights:
                bounds[i] = (bounds[i][0], asset_max_weights[i])
                
        return bounds
    
    def calculate_objective(self, weights: np.ndarray, problem: PortfolioOptProblem) -> float:
        """Calculate the objective function value for a given weight vector.
        
        The default implementation calculates the portfolio variance.
        
        Args:
            weights: Portfolio weights
            problem: Portfolio optimization problem
            
        Returns:
            Objective function value
        """
        return weights.T @ problem.cov_matrix @ weights
    
    def process_weights(self, weights: np.ndarray, min_weight: float = 0.0) -> np.ndarray:
        """Process weights to ensure they satisfy basic constraints.
        
        This method:
        1. Clips weights to be non-negative
        2. Sets weights below threshold to zero
        3. Normalizes weights to sum to 1
        
        Args:
            weights: Raw weights from the optimizer
            min_weight: Minimum weight threshold
            
        Returns:
            Processed weights
        """
        # Ensure non-negative weights
        weights = np.maximum(weights, 0.0)
        
        # Set small weights to zero
        weights[weights < min_weight] = 0.0
        
        # Normalize to sum to 1
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
            
        return weights
    
    def check_constraints(self, weights: np.ndarray, problem: PortfolioOptProblem) -> Dict[str, bool]:
        """Check if weights satisfy all constraints.
        
        Args:
            weights: Portfolio weights
            problem: Portfolio optimization problem
            
        Returns:
            Dictionary of constraint names and whether they are satisfied
        """
        results = {}
        
        # Check sum to one
        results['sum_to_one'] = np.isclose(np.sum(weights), 1.0)
        
        # Check min/max weight constraints
        min_weight = problem.constraints.get('min_weight', 0.0)
        max_weight = problem.constraints.get('max_weight', 1.0)
        
        # Only check min weight for non-zero positions
        non_zero = weights > 0
        results['min_weight'] = np.all(weights[non_zero] >= min_weight)
        results['max_weight'] = np.all(weights <= max_weight)
        
        # Check number of stocks constraints
        min_stocks = problem.constraints.get('min_stocks_held')
        if min_stocks is not None:
            results['min_stocks_held'] = np.sum(weights > 0) >= min_stocks
            
        max_stocks = problem.constraints.get('max_stocks_held')
        if max_stocks is not None:
            results['max_stocks_held'] = np.sum(weights > 0) <= max_stocks
        
        # Check cardinality constraint
        cardinality = problem.constraints.get('cardinality')
        if cardinality is not None:
            # Count assets with weight above min_weight
            non_zero_weights = np.sum(weights > min_weight)
            results['cardinality'] = non_zero_weights <= cardinality
        
        # Check turnover constraint
        prev_weights = problem.constraints.get('prev_weights')
        turnover_limit = problem.constraints.get('turnover_limit')
        
        if prev_weights is not None and turnover_limit is not None:
            turnover = np.sum(np.abs(weights - prev_weights))
            results['turnover'] = turnover <= turnover_limit
        
        return results
