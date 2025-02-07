"""Classical optimization solver using sequential relaxation."""
import numpy as np
import time
from scipy.optimize import minimize
from .base import BaseSolver
from ..core.problem import PortfolioOptProblem
from ..core.result import PortfolioOptResult

class ClassicalSolver(BaseSolver):
    """Classical portfolio optimization solver using sequential relaxation."""
    
    def __init__(self, **kwargs):
        """Initialize solver with configurable parameters."""
        self.max_iterations = kwargs.get('max_iterations', 5)
        self.initial_penalty = kwargs.get('initial_penalty', 100.0)
        self.penalty_multiplier = kwargs.get('penalty_multiplier', 10.0)
        self.perturbation_size = kwargs.get('perturbation_size', 0.01)

    def _create_base_constraints(self, problem):
        """Create base constraints."""
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

    def solve(self, problem: PortfolioOptProblem) -> PortfolioOptResult:
        """Solve using sequential relaxation approach."""
        start_time = time.time()
        
        prev_weights = problem.constraints.get('prev_weights')
        turnover_limit = problem.constraints.get('turnover_limit')
        min_weight = problem.constraints.get('min_weight', 0.0)
        max_weight = problem.constraints.get('max_weight', 1.0)

        # Initial bounds and constraints
        bounds = [(0, max_weight) for _ in range(problem.n_assets)]
        base_constraints = self._create_base_constraints(problem)

        best_result = None
        best_turnover = float('inf')
        best_weights = None

        # Stage 1: Find minimum variance portfolio without turnover constraint
        print("\nStage 1: Finding minimum variance portfolio...")
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
            min_var_turnover = np.sum(np.abs(min_var_weights - prev_weights))
            print(f"Minimum variance turnover: {min_var_turnover:.4f}")

        # Stage 2: Sequential relaxation
        print("\nStage 2: Sequential relaxation...")
        current_weights = prev_weights.copy()
        relaxation_factors = np.linspace(0, 1, self.max_iterations)

        for i, alpha in enumerate(relaxation_factors):
            print(f"\nIteration {i+1}/{len(relaxation_factors)} (alpha={alpha:.3f})")
            
            # Target weights for this iteration
            target_weights = alpha * min_var_weights + (1 - alpha) * prev_weights

            # Create the tracking error objective
            def objective(x):
                return np.sum((x - target_weights) ** 2)

            # Try multiple initial points
            for trial in range(3):
                # Perturb current solution
                noise = np.random.normal(0, self.perturbation_size * (1 - alpha), 
                                      size=len(current_weights))
                init_weights = current_weights + noise
                init_weights = np.clip(init_weights, 0, max_weight)
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
                    weights = result.x.copy()
                    weights[weights < min_weight] = 0.0
                    weights = weights / np.sum(weights)
                    
                    turnover = np.sum(np.abs(weights - prev_weights))
                    print(f"  Trial {trial + 1}: turnover = {turnover:.4f}")

                    if turnover < best_turnover:
                        best_turnover = turnover
                        best_result = result
                        best_weights = weights.copy()
                        current_weights = weights.copy()

                        if turnover <= turnover_limit * 1.001:
                            print(f"\nFound feasible solution!")
                            solve_time = time.time() - start_time
                            return PortfolioOptResult(
                                weights=weights,
                                objective_value=weights.T @ problem.cov_matrix @ weights,
                                solve_time=solve_time,
                                feasible=True
                            )

        # If we get here, use the best result we found
        if best_weights is not None:
            weights = best_weights
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

