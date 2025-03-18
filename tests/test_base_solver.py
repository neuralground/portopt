"""Tests for the BaseSolver and related components."""

import pytest
import numpy as np
from typing import Dict, Any

from portopt.core.problem import PortfolioOptProblem
from portopt.core.result import PortfolioOptResult
from portopt.solvers.base import BaseSolver
from portopt.solvers.constraint_adapter import ConstraintAdapter
from portopt.solvers.factory import SolverFactory
from portopt.solvers.classical import ClassicalSolver

# Simple concrete solver implementation for testing the BaseSolver functionality
class SimpleSolver(BaseSolver):
    """Simple solver implementation for testing BaseSolver functionality."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.equal_weights = kwargs.get('equal_weights', False)
    
    def solve(self, problem: PortfolioOptProblem) -> PortfolioOptResult:
        """Solve using equal weights or minimum variance."""
        processed_problem = self.preprocess_problem(problem)
        
        if self.equal_weights:
            # Equal weight solution
            weights = np.ones(processed_problem.n_assets) / processed_problem.n_assets
        else:
            # Simple minimum variance solution (not optimal, just for testing)
            weights = 1.0 / np.diag(processed_problem.cov_matrix)
            weights = weights / np.sum(weights)
        
        # Process weights to ensure they satisfy constraints
        min_weight = processed_problem.constraints.get('min_weight', 0.0)
        weights = self.process_weights(weights, min_weight)
        
        # Check constraints
        constraint_results = self.check_constraints(weights, processed_problem)
        
        # Determine if solution is feasible based on constraint results
        feasible = all(constraint_results.values())
        
        # Create and return result
        return PortfolioOptResult(
            weights=weights,
            objective_value=self.calculate_objective(weights, processed_problem),
            solve_time=0.1,  # Mock solve time
            feasible=feasible,
            iterations_used=1
        )


class TestBaseSolver:
    """Tests for BaseSolver functionality."""
    
    @pytest.fixture
    def sample_problem(self) -> PortfolioOptProblem:
        """Create a sample portfolio optimization problem."""
        n_assets = 10
        n_periods = 100
        
        # Generate random returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, (n_assets, n_periods))
        
        # Create problem with basic constraints
        return PortfolioOptProblem(
            returns=returns,
            constraints={
                'min_weight': 0.01,
                'max_weight': 0.3,
                'sum_to_one': True
            }
        )
    
    def test_preprocess_problem(self, sample_problem):
        """Test problem preprocessing."""
        solver = SimpleSolver()
        processed = solver.preprocess_problem(sample_problem)
        
        # Check that constraints are preserved
        assert processed.constraints['min_weight'] == 0.01
        assert processed.constraints['max_weight'] == 0.3
        assert processed.constraints['sum_to_one'] is True
        
        # Test with missing constraints
        minimal_problem = PortfolioOptProblem(
            returns=sample_problem.returns,
            constraints={}
        )
        
        processed = solver.preprocess_problem(minimal_problem)
        
        # Check that default constraints are added
        assert processed.constraints['min_weight'] == 0.0
        assert processed.constraints['max_weight'] == 1.0
        assert processed.constraints['sum_to_one'] is True
    
    def test_create_bounds(self, sample_problem):
        """Test bounds creation."""
        solver = SimpleSolver()
        bounds = solver.create_bounds(sample_problem)
        
        # Check that all bounds match the constraints
        for lower, upper in bounds:
            assert lower == 0.01
            assert upper == 0.3
        
        # Test with asset-specific bounds
        problem_with_specific_bounds = PortfolioOptProblem(
            returns=sample_problem.returns,
            constraints={
                'min_weight': 0.01,
                'max_weight': 0.3,
                'asset_min_weights': {0: 0.05, 2: 0.1},
                'asset_max_weights': {1: 0.2, 3: 0.15}
            }
        )
        
        bounds = solver.create_bounds(problem_with_specific_bounds)
        
        # Check specific asset bounds
        assert bounds[0] == (0.05, 0.3)  # Asset 0 has specific min
        assert bounds[1] == (0.01, 0.2)  # Asset 1 has specific max
        assert bounds[2] == (0.1, 0.3)   # Asset 2 has specific min
        assert bounds[3] == (0.01, 0.15) # Asset 3 has specific max
        
        # Check other assets have default bounds
        for i in range(4, len(bounds)):
            assert bounds[i] == (0.01, 0.3)
    
    def test_process_weights(self, sample_problem):
        """Test weight processing."""
        solver = SimpleSolver()
        
        # Test with weights below threshold
        weights = np.array([0.005, 0.3, 0.005, 0.2, 0.49])
        processed = solver.process_weights(weights, min_weight=0.01)
        
        # Check that small weights are set to zero
        assert processed[0] == 0.0
        assert processed[2] == 0.0
        
        # Check that weights sum to 1
        assert np.isclose(np.sum(processed), 1.0)
        
        # Test with negative weights
        weights = np.array([-0.1, 0.3, 0.4, 0.5])
        processed = solver.process_weights(weights, min_weight=0.01)
        
        # Check that negative weights are set to zero
        assert processed[0] == 0.0
        
        # Check that weights sum to 1
        assert np.isclose(np.sum(processed), 1.0)
    
    def test_check_constraints(self, sample_problem):
        """Test constraint checking."""
        solver = SimpleSolver()
        
        # Valid weights
        valid_weights = np.array([0.1, 0.2, 0.2, 0.2, 0.3])
        results = solver.check_constraints(valid_weights, sample_problem)
        
        # All constraints should be satisfied
        assert results['sum_to_one'] == True
        assert results['min_weight'] == True
        assert results['max_weight'] == True
        
        # Invalid weights (sum > 1)
        invalid_weights = np.array([0.3, 0.3, 0.3, 0.3, 0.3])
        results = solver.check_constraints(invalid_weights, sample_problem)
        
        # Sum to one should be violated
        assert results['sum_to_one'] == False
        
        # Invalid weights (below min)
        invalid_weights = np.array([0.005, 0.2, 0.2, 0.2, 0.395])
        results = solver.check_constraints(invalid_weights, sample_problem)
        
        # Min weight should be violated
        assert results['min_weight'] == False
        
        # Invalid weights (above max)
        invalid_weights = np.array([0.1, 0.4, 0.1, 0.1, 0.3])
        results = solver.check_constraints(invalid_weights, sample_problem)
        
        # Max weight should be violated
        assert results['max_weight'] == False
    
    def test_simple_solver_equal_weights(self, sample_problem):
        """Test SimpleSolver with equal weights."""
        solver = SimpleSolver(equal_weights=True)
        result = solver.solve(sample_problem)
        
        # Check that weights are equal
        expected_weight = 1.0 / sample_problem.n_assets
        assert np.allclose(result.weights, expected_weight)
        
        # Check that solution is feasible
        assert result.feasible == True
    
    def test_simple_solver_min_variance(self, sample_problem):
        """Test SimpleSolver with minimum variance."""
        solver = SimpleSolver(equal_weights=False)
        result = solver.solve(sample_problem)
        
        # Check that weights sum to 1
        assert np.isclose(np.sum(result.weights), 1.0)
        
        # Check that solution is feasible
        assert result.feasible == True


class TestConstraintAdapter:
    """Tests for ConstraintAdapter functionality."""
    
    @pytest.fixture
    def sample_problem(self) -> PortfolioOptProblem:
        """Create a sample portfolio optimization problem."""
        n_assets = 5
        n_periods = 50
        
        # Generate random returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, (n_assets, n_periods))
        
        # Create problem with various constraints
        return PortfolioOptProblem(
            returns=returns,
            constraints={
                'min_weight': 0.01,
                'max_weight': 0.4,
                'sum_to_one': True,
                'turnover_limit': 0.2,
                'prev_weights': np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            }
        )
    
    def test_to_scipy_constraints(self, sample_problem):
        """Test conversion to SciPy constraints."""
        constraints = ConstraintAdapter.to_scipy_constraints(sample_problem)
        
        # Check that we have the expected constraints
        assert len(constraints) == 2  # sum_to_one and turnover
        
        # Test sum to one constraint
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        assert np.isclose(constraints[0]['fun'](weights), 0.0)
        
        # Test turnover constraint
        weights = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        turnover = np.sum(np.abs(weights - sample_problem.constraints['prev_weights']))
        assert np.isclose(turnover, 0.2)
        assert np.isclose(constraints[1]['fun'](weights), 0.0)  # At the limit
        
        # Test violated turnover constraint
        weights = np.array([0.4, 0.2, 0.2, 0.1, 0.1])
        assert constraints[1]['fun'](weights) < 0  # Constraint violated
    
    def test_to_penalty_functions(self, sample_problem):
        """Test conversion to penalty functions."""
        penalties = ConstraintAdapter.to_penalty_functions(sample_problem)
        
        # Check that we have the expected penalties
        assert len(penalties) == 4  # sum_to_one, min_weight, max_weight, turnover
        
        # Test sum to one penalty
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        assert np.isclose(penalties[0][0](weights), 0.0)  # No penalty
        
        weights = np.array([0.3, 0.3, 0.3, 0.3, 0.3])
        assert penalties[0][0](weights) > 0  # Penalty applied
        
        # Test min weight penalty
        weights = np.array([0.005, 0.2, 0.2, 0.2, 0.395])
        assert penalties[1][0](weights) > 0  # Penalty applied
        
        # Test max weight penalty
        weights = np.array([0.1, 0.5, 0.1, 0.1, 0.2])
        assert penalties[2][0](weights) > 0  # Penalty applied
        
        # Test turnover penalty
        weights = np.array([0.4, 0.2, 0.2, 0.1, 0.1])
        assert penalties[3][0](weights) > 0  # Penalty applied
    
    def test_create_bounds(self, sample_problem):
        """Test bounds creation."""
        bounds = ConstraintAdapter.create_bounds(sample_problem)
        
        # Check that all bounds match the constraints
        for lower, upper in bounds:
            assert lower == 0.01
            assert upper == 0.4
    
    def test_validate_constraints(self, sample_problem):
        """Test constraint validation."""
        # Valid weights
        valid_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        results = ConstraintAdapter.validate_constraints(valid_weights, sample_problem)
        
        # All constraints should be satisfied
        assert results['sum_to_one'] == True
        assert results['min_weight'] == True
        assert results['max_weight'] == True
        assert results['turnover'] == True
        
        # Invalid weights (violate turnover)
        invalid_weights = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
        results = ConstraintAdapter.validate_constraints(invalid_weights, sample_problem)
        
        # Turnover should be violated
        assert results['turnover'] == False


class TestSolverFactory:
    """Tests for SolverFactory functionality."""
    
    def test_create_classical_solver(self):
        """Test creation of classical solver."""
        factory = SolverFactory()
        solver = factory.create_solver('classical')
        
        # Check that we got the right type of solver
        assert isinstance(solver, ClassicalSolver)
        
        # Check that default parameters were applied
        assert solver.max_iterations == 5
        assert solver.initial_penalty == 100.0
        
        # Test with custom parameters
        solver = factory.create_solver('classical', max_iterations=10, initial_penalty=200.0)
        assert solver.max_iterations == 10
        assert solver.initial_penalty == 200.0
    
    def test_register_solver(self):
        """Test registration of a new solver."""
        factory = SolverFactory()
        
        # Register our test solver
        factory.register_solver('simple', SimpleSolver, {'equal_weights': True})
        
        # Create the solver
        solver = factory.create_solver('simple')
        
        # Check that we got the right type of solver
        assert isinstance(solver, SimpleSolver)
        
        # Check that default parameters were applied
        assert solver.equal_weights == True
        
        # Test with custom parameters
        solver = factory.create_solver('simple', equal_weights=False)
        assert solver.equal_weights == False
    
    def test_get_available_solvers(self):
        """Test getting available solvers."""
        factory = SolverFactory()
        solvers = factory.get_available_solvers()
        
        # Check that classical solver is available
        assert 'classical' in solvers
        
        # Register a new solver and check again
        factory.register_solver('simple', SimpleSolver)
        solvers = factory.get_available_solvers()
        
        # Classical should still be there
        assert 'classical' in solvers
    
    def test_get_solver_parameters(self):
        """Test getting solver parameters."""
        factory = SolverFactory()
        params = factory.get_solver_parameters('classical')
        
        # Check that we got the expected parameters
        assert 'max_iterations' in params
        assert 'initial_penalty' in params
        assert 'penalty_multiplier' in params
        assert 'perturbation_size' in params
        
        # Check parameter values
        assert params['max_iterations'] == 5
        assert params['initial_penalty'] == 100.0
        
        # Test with unknown solver
        with pytest.raises(ValueError):
            factory.get_solver_parameters('unknown_solver')
