"""Constraint adapter module for converting between different solver formats."""

import numpy as np
from typing import Dict, List, Tuple, Callable, Any, Optional, Union
from portopt.core.problem import PortfolioOptProblem

class ConstraintAdapter:
    """Adapter for converting constraints between different solver formats.
    
    This class provides methods to convert portfolio constraints from the standard
    format used in PortfolioOptProblem to formats required by different solvers:
    
    - SciPy format (for classical solvers)
    - CVXPY format (for convex optimization solvers)
    - Penalty functions (for heuristic solvers)
    - Hamiltonian terms (for quantum solvers)
    
    It also provides utility functions for constraint validation and preprocessing.
    """
    
    @staticmethod
    def to_scipy_constraints(problem: PortfolioOptProblem) -> List[Dict[str, Any]]:
        """Convert constraints to SciPy format for classical solvers.
        
        Args:
            problem: Portfolio optimization problem
            
        Returns:
            List of constraint dictionaries in SciPy format
        """
        constraints = []
        
        # Sum to one constraint
        if problem.constraints.get('sum_to_one', True):
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sum(x) - 1.0
            })
        
        # Target return constraint
        if problem.target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: x @ problem.exp_returns - problem.target_return
            })
        
        # Turnover constraint
        prev_weights = problem.constraints.get('prev_weights')
        turnover_limit = problem.constraints.get('turnover_limit')
        
        if prev_weights is not None and turnover_limit is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: turnover_limit - np.sum(np.abs(x - prev_weights))
            })
        
        # Sector constraints
        if problem.classifications is not None:
            sector_limits = problem.constraints.get('sector_limits', {})
            for sector, limit in sector_limits.items():
                sector_mask = np.array([c.sector == sector for c in problem.classifications])
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, mask=sector_mask, lim=limit: lim - np.sum(x * mask)
                })
        
        # Asset class constraints
        if problem.asset_classes is not None:
            asset_class_limits = problem.constraints.get('asset_class_limits', {})
            for asset_class, limit in asset_class_limits.items():
                class_mask = np.array([c.primary == asset_class for c in problem.asset_classes])
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, mask=class_mask, lim=limit: lim - np.sum(x * mask)
                })
        
        # Factor exposure constraints
        if problem.factor_exposures is not None:
            factor_min = problem.constraints.get('factor_min_exposures', {})
            factor_max = problem.constraints.get('factor_max_exposures', {})
            
            for factor_idx, min_exp in factor_min.items():
                factor_exposure = problem.factor_exposures[:, factor_idx]
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, exp=factor_exposure, min_val=min_exp: x @ exp - min_val
                })
                
            for factor_idx, max_exp in factor_max.items():
                factor_exposure = problem.factor_exposures[:, factor_idx]
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, exp=factor_exposure, max_val=max_exp: max_val - x @ exp
                })
        
        return constraints
    
    @staticmethod
    def to_penalty_functions(problem: PortfolioOptProblem) -> List[Tuple[Callable, float]]:
        """Convert constraints to penalty functions for heuristic solvers.
        
        Args:
            problem: Portfolio optimization problem
            
        Returns:
            List of (penalty_function, weight) tuples
        """
        penalties = []
        
        # Sum to one constraint
        if problem.constraints.get('sum_to_one', True):
            penalties.append((
                lambda x: (np.sum(x) - 1.0) ** 2,
                100.0  # High weight for this critical constraint
            ))
        
        # Min/max weight constraints
        min_weight = problem.constraints.get('min_weight', 0.0)
        max_weight = problem.constraints.get('max_weight', 1.0)
        
        penalties.append((
            lambda x: np.sum(np.maximum(0, min_weight - x) ** 2),
            10.0
        ))
        
        penalties.append((
            lambda x: np.sum(np.maximum(0, x - max_weight) ** 2),
            10.0
        ))
        
        # Target return constraint
        if problem.target_return is not None:
            penalties.append((
                lambda x: (x @ problem.exp_returns - problem.target_return) ** 2,
                50.0
            ))
        
        # Turnover constraint
        prev_weights = problem.constraints.get('prev_weights')
        turnover_limit = problem.constraints.get('turnover_limit')
        
        if prev_weights is not None and turnover_limit is not None:
            penalties.append((
                lambda x: np.maximum(0, np.sum(np.abs(x - prev_weights)) - turnover_limit) ** 2,
                20.0
            ))
        
        # Sector constraints
        if problem.classifications is not None:
            sector_limits = problem.constraints.get('sector_limits', {})
            for sector, limit in sector_limits.items():
                sector_mask = np.array([c.sector == sector for c in problem.classifications])
                penalties.append((
                    lambda x, mask=sector_mask, lim=limit: np.maximum(0, np.sum(x * mask) - lim) ** 2,
                    5.0
                ))
        
        return penalties
    
    @staticmethod
    def create_bounds(problem: PortfolioOptProblem) -> List[Tuple[float, float]]:
        """Create bounds for the optimization variables.
        
        Args:
            problem: Portfolio optimization problem
            
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
    
    @staticmethod
    def to_hamiltonian_terms(problem: PortfolioOptProblem) -> List[Tuple[str, float]]:
        """Convert portfolio optimization problem to Hamiltonian terms for quantum solvers.
        
        This method converts the portfolio optimization problem into a list of
        Hamiltonian terms that can be used by quantum solvers like QAOA and VQE.
        Each term is a tuple of (operator_string, coefficient).
        
        Args:
            problem: Portfolio optimization problem
            
        Returns:
            List of Hamiltonian terms as tuples (operator_string, coefficient)
        """
        n_assets = problem.n_assets
        returns = problem.returns
        cov_matrix = problem.cov_matrix
        risk_aversion = 1.0  # Default risk aversion parameter
        
        hamiltonian_terms = []
        
        # Convert return maximization objective to energy minimization
        # (negate returns since we want to maximize returns but quantum algorithms minimize energy)
        for i in range(n_assets):
            # Linear terms for returns (Z_i terms)
            expected_return = np.mean(returns[:, i])
            hamiltonian_terms.append((f"Z{i}", -expected_return))
        
        # Risk minimization terms (quadratic terms)
        for i in range(n_assets):
            for j in range(i, n_assets):
                if i == j:
                    # Diagonal terms (Z_i terms)
                    hamiltonian_terms.append((f"Z{i}", risk_aversion * cov_matrix[i, i]))
                else:
                    # Off-diagonal terms (Z_i Z_j terms)
                    hamiltonian_terms.append((f"Z{i} Z{j}", risk_aversion * cov_matrix[i, j]))
        
        # Add constraint penalty terms
        # Sum-to-one constraint as a penalty term
        sum_penalty = 10.0  # High penalty for violating sum-to-one constraint
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                hamiltonian_terms.append((f"Z{i} Z{j}", sum_penalty))
        
        # Add additional constraints based on problem.constraints if available
        if problem.constraints:
            # Min weight constraint
            min_weight = problem.constraints.get('min_weight', 0.0)
            if min_weight > 0:
                min_weight_penalty = 5.0
                for i in range(n_assets):
                    hamiltonian_terms.append((f"Z{i}", -min_weight_penalty * min_weight))
            
            # Max weight constraint
            max_weight = problem.constraints.get('max_weight', 1.0)
            if max_weight < 1.0:
                max_weight_penalty = 5.0
                for i in range(n_assets):
                    hamiltonian_terms.append((f"Z{i}", max_weight_penalty * max_weight))
        
        return hamiltonian_terms
    
    @staticmethod
    def to_qubo(problem: PortfolioOptProblem) -> Dict[Tuple[int, int], float]:
        """Convert portfolio optimization problem to QUBO format for quantum solvers.
        
        This method converts the portfolio optimization problem into a QUBO (Quadratic
        Unconstrained Binary Optimization) dictionary that can be used by quantum solvers.
        The QUBO format represents the problem as a dictionary where keys are tuples of
        variable indices and values are coefficients.
        
        Args:
            problem: Portfolio optimization problem
            
        Returns:
            QUBO dictionary with keys as tuples of variable indices and values as coefficients
        """
        n_assets = problem.n_assets
        returns = problem.returns
        cov_matrix = problem.cov_matrix
        risk_aversion = 1.0  # Default risk aversion parameter
        
        qubo = {}
        
        # Convert return maximization objective to energy minimization
        # (negate returns since we want to maximize returns but quantum algorithms minimize energy)
        for i in range(n_assets):
            # Linear terms for returns
            expected_return = np.mean(returns[:, i])
            qubo[(i, i)] = qubo.get((i, i), 0.0) - expected_return
        
        # Risk minimization terms (quadratic terms)
        for i in range(n_assets):
            for j in range(n_assets):
                if i == j:
                    # Diagonal terms
                    qubo[(i, i)] = qubo.get((i, i), 0.0) + risk_aversion * cov_matrix[i, i]
                else:
                    # Off-diagonal terms
                    key = (min(i, j), max(i, j))  # Ensure i <= j for consistency
                    qubo[key] = qubo.get(key, 0.0) + risk_aversion * cov_matrix[i, j]
        
        # Add constraint penalty terms
        # Sum-to-one constraint as a penalty term
        sum_penalty = 10.0  # High penalty for violating sum-to-one constraint
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                key = (i, j)
                qubo[key] = qubo.get(key, 0.0) + sum_penalty
        
        # Add additional constraints based on problem.constraints if available
        if problem.constraints:
            # Min weight constraint
            min_weight = problem.constraints.get('min_weight', 0.0)
            if min_weight > 0:
                min_weight_penalty = 5.0
                for i in range(n_assets):
                    qubo[(i, i)] = qubo.get((i, i), 0.0) - min_weight_penalty * min_weight
            
            # Max weight constraint
            max_weight = problem.constraints.get('max_weight', 1.0)
            if max_weight < 1.0:
                max_weight_penalty = 5.0
                for i in range(n_assets):
                    qubo[(i, i)] = qubo.get((i, i), 0.0) + max_weight_penalty * max_weight
        
        return qubo
    
    @staticmethod
    def to_quadratic_program(problem: PortfolioOptProblem):
        """Convert portfolio optimization problem to QuadraticProgram for quantum solvers.
        
        This method converts the portfolio optimization problem into a QuadraticProgram
        that can be used by quantum solvers like QAOA and VQE.
        
        Args:
            problem: Portfolio optimization problem
            
        Returns:
            QuadraticProgram representation of the problem
        """
        # Import here to avoid circular imports
        from qiskit_optimization.problems import QuadraticProgram
        
        n_assets = problem.n_assets
        
        # Create a quadratic program
        qp = QuadraticProgram(name="Portfolio Optimization")
        
        # Add binary variables for each asset
        for i in range(n_assets):
            qp.binary_var(name=f"x{i}")
        
        # Set up the objective function (maximize expected return)
        expected_returns = problem.exp_returns
        linear = {f"x{i}": expected_returns[i] for i in range(n_assets)}
        
        # Get covariance matrix
        covariance = problem.cov_matrix
        
        # Add risk term using the covariance matrix
        quadratic = {}
        for i in range(n_assets):
            for j in range(n_assets):
                if covariance[i, j] != 0:
                    quadratic[(f"x{i}", f"x{j}")] = -0.5 * covariance[i, j]
        
        # Set the objective to maximize return and minimize risk
        qp.maximize(linear=linear, quadratic=quadratic)
        
        # Add constraints
        constraints = problem.constraints
        
        # Sum to one constraint
        qp.linear_constraint(
            linear={f"x{i}": 1 for i in range(n_assets)},
            sense="==",
            rhs=1,
            name="sum_to_one"
        )
        
        # Min weight constraint
        min_weight = constraints.get('min_weight', 0.0)
        if min_weight > 0:
            for i in range(n_assets):
                qp.linear_constraint(
                    linear={f"x{i}": 1},
                    sense=">=",
                    rhs=min_weight,
                    name=f"min_weight_{i}"
                )
        
        # Max weight constraint
        max_weight = constraints.get('max_weight', 1.0)
        if max_weight < 1.0:
            for i in range(n_assets):
                qp.linear_constraint(
                    linear={f"x{i}": 1},
                    sense="<=",
                    rhs=max_weight,
                    name=f"max_weight_{i}"
                )
        
        return qp
    
    @staticmethod
    def validate_constraints(weights: np.ndarray, problem: PortfolioOptProblem) -> Dict[str, bool]:
        """Validate if weights satisfy all constraints.
        
        Args:
            weights: Portfolio weights
            problem: Portfolio optimization problem
            
        Returns:
            Dictionary of constraint names and whether they are satisfied
        """
        results = {}
        
        # Check sum to one
        if problem.constraints.get('sum_to_one', True):
            results['sum_to_one'] = np.isclose(np.sum(weights), 1.0)
        
        # Check min/max weight constraints
        min_weight = problem.constraints.get('min_weight', 0.0)
        max_weight = problem.constraints.get('max_weight', 1.0)
        
        # Only check min weight for non-zero positions
        non_zero = weights > 0
        results['min_weight'] = np.all(weights[non_zero] >= min_weight)
        results['max_weight'] = np.all(weights <= max_weight)
        
        # Check target return constraint
        if problem.target_return is not None:
            expected_return = weights @ problem.exp_returns
            results['target_return'] = np.isclose(expected_return, problem.target_return, rtol=1e-3)
        
        # Check turnover constraint
        prev_weights = problem.constraints.get('prev_weights')
        turnover_limit = problem.constraints.get('turnover_limit')
        
        if prev_weights is not None and turnover_limit is not None:
            turnover = np.sum(np.abs(weights - prev_weights))
            results['turnover'] = turnover <= turnover_limit
        
        # Check sector constraints
        if problem.classifications is not None:
            sector_limits = problem.constraints.get('sector_limits', {})
            for sector, limit in sector_limits.items():
                sector_mask = np.array([c.sector == sector for c in problem.classifications])
                sector_exposure = np.sum(weights * sector_mask)
                results[f'sector_{sector}'] = sector_exposure <= limit
        
        # Check asset class constraints
        if problem.asset_classes is not None:
            asset_class_limits = problem.constraints.get('asset_class_limits', {})
            for asset_class, limit in asset_class_limits.items():
                class_mask = np.array([c.primary == asset_class for c in problem.asset_classes])
                class_exposure = np.sum(weights * class_mask)
                results[f'asset_class_{asset_class}'] = class_exposure <= limit
        
        # Check factor exposure constraints
        if problem.factor_exposures is not None:
            factor_min = problem.constraints.get('factor_min_exposures', {})
            factor_max = problem.constraints.get('factor_max_exposures', {})
            
            for factor_idx, min_exp in factor_min.items():
                factor_exposure = problem.factor_exposures[:, factor_idx]
                exposure = weights @ factor_exposure
                results[f'factor_min_{factor_idx}'] = exposure >= min_exp
                
            for factor_idx, max_exp in factor_max.items():
                factor_exposure = problem.factor_exposures[:, factor_idx]
                exposure = weights @ factor_exposure
                results[f'factor_max_{factor_idx}'] = exposure <= max_exp
        
        return results
