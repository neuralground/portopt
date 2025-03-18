"""Tests for the Advanced Genetic Algorithm solver."""

import pytest
import numpy as np
from typing import Dict

from portopt.core.problem import PortfolioOptProblem
from portopt.solvers.advanced_genetic import AdvancedGeneticSolver


@pytest.fixture
def sample_problem():
    """Create a sample portfolio optimization problem for testing."""
    # Create random returns data
    np.random.seed(42)  # For reproducibility
    n_assets = 10
    n_periods = 50
    returns = np.random.normal(0.01, 0.05, (n_assets, n_periods))
    
    # Create constraints
    constraints = {
        'min_weight': 0.0,
        'max_weight': 1.0,
        'sum_to_one': True,
        'cardinality': 5  # Limit to 5 assets
    }
    
    return PortfolioOptProblem(returns=returns, constraints=constraints)


@pytest.fixture
def small_problem():
    """Create a smaller problem for faster tests."""
    # Create random returns data
    np.random.seed(42)  # For reproducibility
    n_assets = 5
    n_periods = 20
    returns = np.random.normal(0.01, 0.05, (n_assets, n_periods))
    
    # Create constraints
    constraints = {
        'min_weight': 0.0,
        'max_weight': 1.0,
        'sum_to_one': True
    }
    
    return PortfolioOptProblem(returns=returns, constraints=constraints)


class TestAdvancedGeneticSolver:
    """Test suite for the AdvancedGeneticSolver class."""
    
    def test_initialization(self):
        """Test that the solver initializes correctly with default and custom parameters."""
        # Default initialization
        solver = AdvancedGeneticSolver()
        assert solver.population_size == 200
        assert solver.generations == 100
        assert solver.mutation_rate == 0.1
        assert solver.crossover_rate == 0.8
        
        # Custom initialization
        custom_solver = AdvancedGeneticSolver(
            population_size=50,
            generations=20,
            mutation_rate=0.2,
            crossover_rate=0.7,
            num_islands=2
        )
        assert custom_solver.population_size == 50
        assert custom_solver.generations == 20
        assert custom_solver.mutation_rate == 0.2
        assert custom_solver.crossover_rate == 0.7
        assert custom_solver.num_islands == 2
    
    def test_solve_small_problem(self, small_problem):
        """Test solving a small portfolio optimization problem."""
        # Create solver with reduced parameters for faster testing
        solver = AdvancedGeneticSolver(
            population_size=20,
            generations=5,
            num_islands=2,
            early_stopping=False
        )
        
        # Solve the problem
        result = solver.solve(small_problem)
        
        # Check that the result has the expected properties
        assert result is not None
        assert hasattr(result, 'weights')
        assert hasattr(result, 'objective_value')
        assert hasattr(result, 'solve_time')
        assert hasattr(result, 'feasible')
        
        # Check weights
        weights = result.weights
        assert len(weights) == small_problem.n_assets
        assert np.isclose(np.sum(weights), 1.0, atol=1e-6)
        assert np.all(weights >= 0)
        
        # Check that the objective value is reasonable
        # Note: For portfolio optimization, the objective value is the portfolio variance,
        # which is always positive. In the fitness function, we negate it for minimization.
        assert result.objective_value > 0  # Portfolio variance is always positive
    
    def test_cardinality_constraint(self, sample_problem):
        """Test that the solver respects cardinality constraints."""
        # Create solver with reduced parameters for faster testing
        solver = AdvancedGeneticSolver(
            population_size=20,
            generations=5,
            num_islands=2,
            early_stopping=False
        )
        
        # Solve the problem
        result = solver.solve(sample_problem)
        
        # Check that the cardinality constraint is satisfied
        weights = result.weights
        non_zero_weights = np.sum(weights > 1e-6)
        assert non_zero_weights <= sample_problem.constraints['cardinality']
    
    def test_multi_objective_optimization(self, small_problem):
        """Test multi-objective optimization mode."""
        # Create solver with multi-objective enabled
        solver = AdvancedGeneticSolver(
            population_size=20,
            generations=5,
            num_islands=2,
            multi_objective=True,
            risk_weight=0.3,
            return_weight=0.7,
            early_stopping=False
        )
        
        # Solve the problem
        result = solver.solve(small_problem)
        
        # Check that the result is valid
        assert result is not None
        assert hasattr(result, 'weights')
        assert np.isclose(np.sum(result.weights), 1.0, atol=1e-6)
        
        # Compare with single-objective solver
        single_obj_solver = AdvancedGeneticSolver(
            population_size=20,
            generations=5,
            num_islands=2,
            multi_objective=False,
            early_stopping=False
        )
        single_obj_result = single_obj_solver.solve(small_problem)
        
        # Results should be different due to different objective functions
        # but both should be valid portfolios
        assert np.isclose(np.sum(single_obj_result.weights), 1.0, atol=1e-6)
    
    def test_adaptive_rates(self, small_problem):
        """Test that adaptive rates mode works."""
        # Create solver with adaptive rates enabled
        solver = AdvancedGeneticSolver(
            population_size=20,
            generations=5,
            adaptive_rates=True,
            early_stopping=False
        )
        
        # Solve the problem
        result = solver.solve(small_problem)
        
        # Check that the result is valid
        assert result is not None
        assert hasattr(result, 'weights')
        assert np.isclose(np.sum(result.weights), 1.0, atol=1e-6)
    
    def test_island_model(self, small_problem):
        """Test that island model works with different numbers of islands."""
        # Create solver with multiple islands
        solver = AdvancedGeneticSolver(
            population_size=20,
            generations=5,
            num_islands=3,
            migration_interval=2,
            early_stopping=False
        )
        
        # Solve the problem
        result = solver.solve(small_problem)
        
        # Check that the result is valid
        assert result is not None
        assert hasattr(result, 'weights')
        assert np.isclose(np.sum(result.weights), 1.0, atol=1e-6)
    
    def test_early_stopping(self, small_problem):
        """Test that early stopping works."""
        # Create solver with early stopping enabled
        solver = AdvancedGeneticSolver(
            population_size=20,
            generations=20,  # Set high, but should stop earlier
            early_stopping=True,
            early_stopping_generations=2
        )
        
        # Solve the problem
        result = solver.solve(small_problem)
        
        # Check that the result is valid and iterations_used is less than generations
        assert result is not None
        assert hasattr(result, 'iterations_used')
        assert result.iterations_used <= solver.generations
    
    def test_constraint_satisfaction(self, sample_problem):
        """Test that the solver produces solutions that satisfy constraints."""
        # Create solver
        solver = AdvancedGeneticSolver(
            population_size=30,
            generations=10,
            early_stopping=False
        )
        
        # Solve the problem
        result = solver.solve(sample_problem)
        
        # Check constraints
        weights = result.weights
        
        # Sum to one
        assert np.isclose(np.sum(weights), 1.0, atol=1e-6)
        
        # Non-negative weights
        assert np.all(weights >= 0)
        
        # Max weight constraint
        max_weight = sample_problem.constraints.get('max_weight', 1.0)
        assert np.all(weights <= max_weight + 1e-6)
        
        # Cardinality constraint
        if 'cardinality' in sample_problem.constraints:
            non_zero_count = np.sum(weights > 1e-6)
            assert non_zero_count <= sample_problem.constraints['cardinality']
    
    def test_process_weights(self, sample_problem):
        """Test the process_weights method."""
        solver = AdvancedGeneticSolver()
        
        # Create test weights
        raw_weights = np.array([0.1, 0.2, 0.3, 0.05, 0.02, 0.15, 0.08, 0.03, 0.04, 0.03])
        
        # Process with min_weight
        min_weight = 0.05
        processed = solver.process_weights(raw_weights, min_weight, sample_problem)
        
        # Check that weights below min_weight are zero
        assert np.all(processed[raw_weights < min_weight] == 0)
        
        # Check sum to one
        assert np.isclose(np.sum(processed), 1.0, atol=1e-6)
        
        # Check cardinality
        non_zero_count = np.sum(processed > 0)
        assert non_zero_count <= sample_problem.constraints['cardinality']
