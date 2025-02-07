"""Classical optimization solver."""
import numpy as np
import time
from scipy.optimize import minimize
from .base import BaseSolver
from ..core.problem import PortfolioOptProblem
from ..core.result import PortfolioOptResult

class ClassicalSolver(BaseSolver):
    """Classical portfolio optimization solver using two-stage approach."""

    def _create_base_constraints(self, problem, x0=None):
        """Create base constraints excluding turnover."""
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

        # Minimum holdings constraint
        if 'min_stocks_held' in problem.constraints:
            min_stocks = problem.constraints['min_stocks_held']
            min_weight = problem.constraints['min_weight']
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: np.sum(x > min_weight) - min_stocks
            })

        return constraints

    def _create_turnover_constraint(self, prev_weights, turnover_limit):
        """Create turnover constraint."""
        return {
            'type': 'ineq',
            'fun': lambda x: turnover_limit - np.sum(np.abs(x - prev_weights))
        }

    def solve(self, problem: PortfolioOptProblem) -> PortfolioOptResult:
        start_time = time.time()

        prev_weights = problem.constraints.get('prev_weights')
        turnover_limit = problem.constraints.get('turnover_limit')
        min_weight = problem.constraints.get('min_weight', 0.0)
        max_weight = problem.constraints.get('max_weight', 1.0)

        # Set up bounds
        bounds = [(0, max_weight) for _ in range(problem.n_assets)]

        # Stage 1: Minimize tracking error to get close to previous portfolio
        if prev_weights is not None:
            base_constraints = self._create_base_constraints(problem)

            def tracking_error_objective(x):
                return np.sum((x - prev_weights) ** 2)

            result1 = minimize(
                tracking_error_objective,
                prev_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=base_constraints,
                options={'maxiter': 500, 'ftol': 1e-8}
            )

            if not result1.success:
                print("Warning: Stage 1 optimization failed")

            initial_weights = result1.x
        else:
            initial_weights = np.ones(problem.n_assets) / problem.n_assets

        # Stage 2: Minimize variance while respecting turnover
        base_constraints = self._create_base_constraints(problem)
        if prev_weights is not None:
            base_constraints.append(
                self._create_turnover_constraint(prev_weights, turnover_limit)
            )

        def variance_objective(x):
            return x.T @ problem.cov_matrix @ x

        result2 = minimize(
            variance_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=base_constraints,
            options={'maxiter': 1000, 'ftol': 1e-8}
        )

        # Clean up the solution
        weights = result2.x.copy()
        weights[weights < min_weight] = 0.0
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)

        solve_time = time.time() - start_time

        return PortfolioOptResult(
            weights=weights,
            objective_value=variance_objective(weights),
            solve_time=solve_time,
            feasible=result2.success
        )

