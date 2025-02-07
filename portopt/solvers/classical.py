"""Classical optimization solver using sequential relaxation."""
import numpy as np
import time
import logging
import warnings
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Tuple
from scipy.optimize import minimize, OptimizeResult
from .base import BaseSolver
from ..core.problem import PortfolioOptProblem
from ..core.result import PortfolioOptResult

@contextmanager
def suppress_slsqp_warnings():
    """Context manager to suppress SLSQP bounds warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                              message='Values in x were outside bounds during a minimize step, clipping to bounds')
        yield

class ClassicalSolver(BaseSolver):
    """Classical portfolio optimization solver using sequential relaxation."""
    
    def __init__(self, **kwargs):
        """Initialize solver with configurable parameters."""
        self.max_iterations = kwargs.get('max_iterations', 5)
        self.initial_penalty = kwargs.get('initial_penalty', 100.0)
        self.penalty_multiplier = kwargs.get('penalty_multiplier', 10.0)
        self.perturbation_size = kwargs.get('perturbation_size', 0.01)
        self.logger = logging.getLogger(__name__)
        self.warnings: Dict[str, Any] = {}

    def _run_single_optimization(self, 
                               problem: PortfolioOptProblem,
                               target_weights: np.ndarray,
                               init_weights: np.ndarray,
                               prev_weights: np.ndarray,
                               turnover_limit: float,
                               bounds: List[Tuple[float, float]],
                               base_constraints: List[Dict]) -> OptimizeResult:
        """Run a single optimization trial."""
        def objective(x):
            return np.sum((x - target_weights) ** 2)
        
        constraints = base_constraints.copy()
        constraints.append({
            'type': 'ineq',
            'fun': lambda x: turnover_limit - np.sum(np.abs(x - prev_weights))
        })
        
        with suppress_slsqp_warnings():
            result = minimize(
                objective,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
        
        return result

    def _find_minimum_variance(self, problem: PortfolioOptProblem, 
                             bounds: List[Tuple[float, float]], 
                             base_constraints: List[Dict],
                             prev_weights: np.ndarray) -> Optional[OptimizeResult]:
        """Find the minimum variance portfolio without turnover constraint."""
        with suppress_slsqp_warnings():
            result = minimize(
                lambda x: x.T @ problem.cov_matrix @ x,
                prev_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=base_constraints,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
        
        if not result.success:
            self.warnings['min_var'] = "Failed to find minimum variance portfolio"
            return None
            
        return result

    def _create_base_constraints(self, problem: PortfolioOptProblem) -> List[Dict]:
        """Create base constraints for the optimization problem.
        
        Args:
            problem: The portfolio optimization problem instance
            
        Returns:
            List of constraint dictionaries for scipy.optimize.minimize
        """
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Sum to 1
        ]

        # Sector constraints if sector map is provided
        if 'sector_map' in problem.constraints:
            sector_map = problem.constraints['sector_map']
            max_sector_weight = problem.constraints['max_sector_weight']

            for sector in np.unique(sector_map):
                sector_mask = sector_map == sector
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, mask=sector_mask: max_sector_weight - np.sum(x[mask])
                })

        return constraints

    def _perturb_weights(self, weights: np.ndarray, alpha: float = 0.0) -> np.ndarray:
        """Generate perturbed weights for new optimization trial.
        
        Args:
            weights: Current weights to perturb
            alpha: Relaxation factor
            
        Returns:
            Perturbed weights that sum to 1 and respect bounds
        """
        noise = np.random.normal(0, self.perturbation_size * (1 - alpha), 
                               size=len(weights))
        perturbed = weights + noise
        perturbed = np.clip(perturbed, 0, 1)  # Ensure non-negative and <= 1
        return perturbed / np.sum(perturbed)  # Normalize to sum to 1

    def _process_weights(self, weights: np.ndarray, min_weight: float) -> np.ndarray:
        """Process optimization weights to ensure they meet minimum weight constraint.
        
        Args:
            weights: Raw optimization weights
            min_weight: Minimum allowed weight
            
        Returns:
            Processed weights that sum to 1 and meet minimum weight constraint
        """
        weights = weights.copy()
        weights[weights < min_weight] = 0.0
        return weights / np.sum(weights)

    def _calculate_objective(self, weights: np.ndarray, 
                           problem: PortfolioOptProblem) -> float:
        """Calculate the objective value (portfolio variance) for given weights.
        
        Args:
            weights: Portfolio weights
            problem: Portfolio optimization problem
            
        Returns:
            Portfolio variance
        """
        return weights.T @ problem.cov_matrix @ weights

    def _create_fallback_result(self, weights: np.ndarray,
                              problem: PortfolioOptProblem,
                              start_time: float) -> PortfolioOptResult:
        """Create a fallback result when optimization fails.
        
        Args:
            weights: Fallback weights to use
            problem: Portfolio optimization problem
            start_time: Start time of optimization
            
        Returns:
            PortfolioOptResult with fallback solution
        """
        return PortfolioOptResult(
            weights=weights,
            objective_value=self._calculate_objective(weights, problem),
            solve_time=time.time() - start_time,
            feasible=False
        )

    def solve(self, problem: PortfolioOptProblem) -> PortfolioOptResult:
        """Solve the portfolio optimization problem.
        
        Args:
            problem: Portfolio optimization problem instance
            
        Returns:
            PortfolioOptResult containing the solution
        """
        start_time = time.time()
        self.warnings.clear()
        
        # Get problem parameters
        prev_weights = problem.constraints.get('prev_weights')
        turnover_limit = problem.constraints.get('turnover_limit')
        min_weight = problem.constraints.get('min_weight', 0.0)
        max_weight = problem.constraints.get('max_weight', 1.0)

        # Setup optimization constraints
        bounds = [(0, max_weight) for _ in range(problem.n_assets)]
        base_constraints = self._create_base_constraints(problem)

        # Stage 1: Find minimum variance portfolio
        min_var_result = self._find_minimum_variance(
            problem, bounds, base_constraints, prev_weights
        )
        if min_var_result is None:
            return self._create_fallback_result(prev_weights, problem, start_time)

        min_var_weights = min_var_result.x
        best_result = None
        best_turnover = float('inf')
        best_weights = None
        current_weights = prev_weights.copy()

        # Stage 2: Sequential relaxation
        for alpha in np.linspace(0, 1, self.max_iterations):
            self.logger.debug(f"Relaxation step: alpha = {alpha:.3f}")
            target_weights = alpha * min_var_weights + (1 - alpha) * prev_weights

            # Try multiple initial points
            for trial in range(3):
                init_weights = self._perturb_weights(current_weights, alpha)
                result = self._run_single_optimization(
                    problem, target_weights, init_weights, prev_weights,
                    turnover_limit, bounds, base_constraints
                )

                if result.success:
                    weights = self._process_weights(result.x, min_weight)
                    turnover = np.sum(np.abs(weights - prev_weights))
                    self.logger.debug(f"Trial {trial + 1}: turnover = {turnover:.4f}")

                    if turnover < best_turnover:
                        best_turnover = turnover
                        best_result = result
                        best_weights = weights.copy()
                        current_weights = weights.copy()

                        if turnover <= turnover_limit * 1.001:
                            self.logger.debug("Found feasible solution")
                            return PortfolioOptResult(
                                weights=weights,
                                objective_value=self._calculate_objective(weights, problem),
                                solve_time=time.time() - start_time,
                                feasible=True
                            )

        # Return best solution found, even if not fully feasible
        return self._finalize_result(
            best_weights if best_weights is not None else prev_weights,
            problem, prev_weights, turnover_limit, min_weight, start_time
        )

    def _finalize_result(self, weights: np.ndarray,
                        problem: PortfolioOptProblem,
                        prev_weights: np.ndarray,
                        turnover_limit: float,
                        min_weight: float,
                        start_time: float) -> PortfolioOptResult:
        """Create final optimization result with proper feasibility checks.
        
        Args:
            weights: Portfolio weights
            problem: Portfolio optimization problem
            prev_weights: Previous portfolio weights
            turnover_limit: Maximum allowed turnover
            min_weight: Minimum allowed weight
            start_time: Start time of optimization
            
        Returns:
            PortfolioOptResult with final solution
        """
        weights = self._process_weights(weights, min_weight)
        actual_turnover = np.sum(np.abs(weights - prev_weights))
        feasible = actual_turnover <= turnover_limit * 1.001
        
        if not feasible:
            self.warnings['turnover'] = {
                'actual': actual_turnover,
                'limit': turnover_limit
            }
        
        return PortfolioOptResult(
            weights=weights,
            objective_value=self._calculate_objective(weights, problem),
            solve_time=time.time() - start_time,
            feasible=feasible
        )

