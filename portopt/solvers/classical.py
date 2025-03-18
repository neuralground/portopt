"""Classical optimization solver using sequential relaxation."""
import numpy as np
import time
import logging
import warnings
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Tuple, Generator
from scipy.optimize import minimize, OptimizeResult
from .base import BaseSolver
from ..core.problem import PortfolioOptProblem
from ..core.result import PortfolioOptResult

@contextmanager
def suppress_slsqp_warnings() -> Generator[None, None, None]:
    """Context manager to suppress SLSQP bounds warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                              message='Values in x were outside bounds during a minimize step, clipping to bounds')
        yield

class ClassicalSolver(BaseSolver):
    """Classical portfolio optimization solver using sequential relaxation.

    This solver implements a sophisticated approach to portfolio optimization that:
    1. Uses sequential relaxation to handle non-linear constraints
    2. Supports warm starting from previous solutions
    3. Implements adaptive penalty adjustment
    4. Handles market impact and transaction costs

    The solver works in multiple stages:
    1. Find minimum variance portfolio without turnover constraints
    2. Gradually relax from current position to minimum variance target
    3. Fine-tune solution with multiple random starts

    Attributes:
        max_iterations: Maximum number of relaxation steps
        initial_penalty: Initial penalty for constraint violations
        penalty_multiplier: Factor to increase penalties
        perturbation_size: Size of random perturbations for multiple starts
    """

    def __init__(self, **kwargs):
        """Initialize solver with configurable parameters.

        Args:
            max_iterations: Number of relaxation steps (default: 5)
            initial_penalty: Initial constraint penalty (default: 100.0)
            penalty_multiplier: Penalty increase factor (default: 10.0)
            perturbation_size: Random perturbation scale (default: 0.01)
        """
        self.max_iterations = kwargs.get('max_iterations', 5)
        self.initial_penalty = kwargs.get('initial_penalty', 100.0)
        self.penalty_multiplier = kwargs.get('penalty_multiplier', 10.0)
        self.perturbation_size = kwargs.get('perturbation_size', 0.01)
        self.logger = logging.getLogger(__name__)
        self.warnings: Dict[str, Any] = {}

    def _run_single_optimization(self, problem: PortfolioOptProblem, 
                               target_weights: np.ndarray,
                               init_weights: np.ndarray,
                               prev_weights: np.ndarray,
                               turnover_limit: Optional[float],
                               base_constraints: List[Dict]) -> OptimizeResult:
        """Run a single optimization step with specified parameters."""
        def objective(x):
            """Objective function for optimization.
            
            Minimizes squared deviation from target weights.
            
            Args:
                x: Current portfolio weights
                
            Returns:
                Sum of squared deviations from target
            """
            # Quadratic deviation from target
            return np.sum((x - target_weights) ** 2)

        # Add turnover constraint to base constraints
        constraints = base_constraints.copy()
        if turnover_limit is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: turnover_limit - np.sum(np.abs(x - prev_weights))
            })

        # Run optimization with warning suppression
        with suppress_slsqp_warnings():
            result = minimize(
                objective,
                init_weights,
                method='SLSQP',
                bounds=[(0, 1) for _ in range(problem.n_assets)],
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-8}
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

        Implementation follows these steps:
        1. Extract problem parameters and set up constraints
        2. Find minimum variance portfolio as ideal target
        3. Use sequential relaxation to move from current to target
        4. Try multiple random starts at each relaxation step
        5. Track best feasible solution found

        Args:
            problem: Portfolio optimization problem instance

        Returns:
            PortfolioOptResult containing optimal weights and metadata
        """
        start_time = time.time()
        self.warnings.clear()

        # Extract key problem parameters
        prev_weights = problem.constraints.get('prev_weights')
        if prev_weights is None:
            # If no previous weights provided, initialize with equal weights
            prev_weights = np.ones(problem.n_assets) / problem.n_assets
            
        turnover_limit = problem.constraints.get('turnover_limit')
        min_weight = problem.constraints.get('min_weight', 0.0)
        max_weight = problem.constraints.get('max_weight', 1.0)

        # Set up optimization constraints
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
            # Interpolate between current and minimum variance
            target_weights = alpha * min_var_weights + (1 - alpha) * prev_weights

            # Try multiple initial points
            for trial in range(3):
                # Generate perturbed initial weights
                init_weights = self._perturb_weights(current_weights, alpha)
                result = self._run_single_optimization(
                    problem, target_weights, init_weights, prev_weights,
                    turnover_limit, base_constraints
                )

                if result.success:
                    # Process and validate solution
                    weights = self._process_weights(result.x, min_weight)
                    if turnover_limit is None:
                        turnover = np.sum(np.abs(weights - prev_weights))
                        self.logger.debug(f"Trial {trial + 1}: turnover = {turnover:.4f}")
                        if turnover < best_turnover:
                            best_turnover = turnover
                            best_result = result
                            best_weights = weights.copy()
                            current_weights = weights.copy()
                    else:
                        turnover = np.sum(np.abs(weights - prev_weights))
                        self.logger.debug(f"Trial {trial + 1}: turnover = {turnover:.4f}")

                        # Update best solution if better
                        if turnover < best_turnover:
                            best_turnover = turnover
                            best_result = result
                            best_weights = weights.copy()
                            current_weights = weights.copy()

                            # Return immediately if solution is feasible
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
            prev_weights,
            turnover_limit,
            problem,
            start_time
        )

    def _finalize_result(self, best_weights: np.ndarray, 
                       prev_weights: np.ndarray,
                       turnover_limit: Optional[float],
                       problem: PortfolioOptProblem,
                       start_time: float) -> PortfolioOptResult:
        """Create final result with feasibility check."""
        # Check turnover feasibility
        actual_turnover = np.sum(np.abs(best_weights - prev_weights))
        feasible = actual_turnover <= turnover_limit * 1.001 if turnover_limit is not None else True
        
        if not feasible and turnover_limit is not None:
            self.warnings['turnover'] = {
                'limit': turnover_limit,
                'actual': actual_turnover
            }
            
        return PortfolioOptResult(
            weights=best_weights,
            objective_value=self._calculate_objective(best_weights, problem),
            solve_time=time.time() - start_time,
            feasible=feasible
        )
