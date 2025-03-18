"""
Multi-threaded portfolio optimization solver module.

This module provides a parallel implementation of the portfolio optimization solver
that leverages multi-threading to improve performance. It implements:

- Concurrent optimization of multiple relaxation factors
- Thread pool management for efficient resource utilization
- Parallel evaluation of objective functions
- Synchronized result aggregation

The implementation is designed to scale with the number of available CPU cores
and is particularly effective for large portfolio optimization problems.
"""

import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Any, Optional, Callable
from .base import BaseSolver
from ..core.problem import PortfolioOptProblem
from ..core.result import PortfolioOptResult

class MultithreadedSolver(BaseSolver):
    """Classical portfolio optimization solver using sequential relaxation with multithreading."""

    def __init__(self, **kwargs):
        """Initialize solver with configurable parameters."""
        self.max_iterations = kwargs.get('max_iterations', 5)
        self.initial_penalty = kwargs.get('initial_penalty', 100.0)
        self.penalty_multiplier = kwargs.get('penalty_multiplier', 10.0)
        self.perturbation_size = kwargs.get('perturbation_size', 0.01)
        self.n_threads = kwargs.get('n_threads', 4)  # Number of threads to use

    def _create_base_constraints(self, problem: PortfolioOptProblem) -> List[Dict[str, Any]]:
        """Create base constraints for the optimization problem.
        
        Args:
            problem: Portfolio optimization problem definition
            
        Returns:
            List of constraint dictionaries for scipy.optimize.minimize
        """
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Sum to 1
        ]

        # Sector constraints
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

    def _optimize_subproblem(self, problem: PortfolioOptProblem, target_weights: np.ndarray, 
                            prev_weights: np.ndarray, turnover_limit: float, 
                            bounds: List[Tuple[float, float]], 
                            base_constraints: List[Dict[str, Any]]) -> Tuple[np.ndarray, float]:
        """Optimize a single subproblem with a given target.
        
        Args:
            problem: Portfolio optimization problem definition
            target_weights: Target portfolio weights for this subproblem
            prev_weights: Previous portfolio weights
            turnover_limit: Maximum allowed turnover
            bounds: Weight bounds for each asset
            base_constraints: Base constraints for the optimization
            
        Returns:
            Tuple containing optimized weights and achieved turnover
        """
        def objective(x: np.ndarray) -> float:
            """Calculate the objective function value.
            
            The objective is to minimize the squared deviation from target weights
            while respecting all constraints.
            
            Args:
                x: Portfolio weights
                
            Returns:
                Objective function value (squared deviation)
            """
            return np.sum((x - target_weights) ** 2)

        # Try multiple initial points
        best_result = None
        best_turnover = float('inf')

        for trial in range(3):
            # Perturb current solution
            noise = np.random.normal(0, self.perturbation_size, size=len(prev_weights))
            init_weights = prev_weights + noise
            init_weights = np.clip(init_weights, 0, bounds[-1])
            init_weights = init_weights / np.sum(init_weights)

            # Add turnover constraint
            constraints = base_constraints.copy()
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: turnover_limit - np.sum(np.abs(x - prev_weights))
            })

            result = minimize(
                objective,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )

            if result.success:
                turnover = np.sum(np.abs(result.x - prev_weights))
                if turnover < best_turnover:
                    best_turnover = turnover
                    best_result = result

        return best_result

    def solve(self, problem: PortfolioOptProblem) -> PortfolioOptResult:
        """Solve using sequential relaxation approach with multithreading."""
        start_time = time.time()

        prev_weights = problem.constraints.get('prev_weights')
        turnover_limit = problem.constraints.get('turnover_limit')
        min_weight = problem.constraints.get('min_weight', 0.0)
        max_weight = problem.constraints.get('max_weight', 1.0)

        # Initial bounds and constraints
        bounds = [(0, max_weight) for _ in range(problem.n_assets)]
        base_constraints = self._create_base_constraints(problem)

        # Stage 1: Find minimum variance portfolio without turnover constraint (no multithreading here)
        result1 = minimize(
            lambda x: x.T @ problem.cov_matrix @ x,
            prev_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=base_constraints,
            options={'maxiter': 1000, 'ftol': 1e-8}
        )

        if result1.success:
            min_var_weights = result1.x.copy()

        # Stage 2: Sequential relaxation with multithreading
        current_weights = prev_weights.copy()
        relaxation_factors = np.linspace(0, 1, self.max_iterations)
        best_result = None
        best_turnover = float('inf')

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = []  
            for alpha in relaxation_factors:
                # Target weights for this iteration
                target_weights = alpha * min_var_weights + (1 - alpha) * prev_weights

                # Submit subproblem to the executor
                future = executor.submit(
                    self._optimize_subproblem,
                    problem,
                    target_weights,
                    current_weights,
                    turnover_limit,
                    bounds,
                    base_constraints
                )
                futures.append(future)

            for future in as_completed(futures):
                result = future.result()
                if result is not None and result.success:
                    weights = result.x.copy()
                    weights[weights < min_weight] = 0.0
                    weights = weights / np.sum(weights)

                    turnover = np.sum(np.abs(weights - prev_weights))
                    if turnover < best_turnover:
                        best_turnover = turnover
                        best_result = result
                        current_weights = weights.copy()

        # If we get here, use the best result we found
        if best_result is not None:
            weights = best_result.x.copy()
            weights[weights < min_weight] = 0.0
            weights = weights / np.sum(weights)
        else:
            weights = prev_weights.copy()
            weights[weights < min_weight] = 0.0
            weights = weights / np.sum(weights)

        # Final turnover check
        actual_turnover = np.sum(np.abs(weights - prev_weights))
        if actual_turnover > turnover_limit * 1.001:
            print(f"Warning: Turnover constraint violated. Actual: {actual_turnover:.4f}, Limit: {turnover_limit:.4f}")
            feasible = False
        else:
            feasible = True

        solve_time = time.time() - start_time

        return PortfolioOptResult(
            weights=weights,
            objective_value=weights.T @ problem.cov_matrix @ weights,
            solve_time=solve_time,
            feasible=feasible
        )