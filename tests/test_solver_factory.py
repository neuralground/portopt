"""Tests for the SolverFactory class."""

import pytest
import numpy as np
from typing import Dict, Any

from portopt.core.problem import PortfolioOptProblem
from portopt.solvers.base import BaseSolver
from portopt.solvers.factory import SolverFactory
from portopt.solvers.classical import ClassicalSolver


# Simple mock solver for testing factory registration
class MockSolver(BaseSolver):
    """Mock solver for testing factory registration."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mock_param = kwargs.get('mock_param', 'default')
        self.iterations = kwargs.get('iterations', 10)
    
    def solve(self, problem: PortfolioOptProblem):
        """Mock solve method."""
        # Just return a simple result with equal weights
        weights = np.ones(problem.n_assets) / problem.n_assets
        return self.create_result(weights, problem, 0.1, True, 1)


class TestSolverFactory:
    """Tests for SolverFactory functionality."""
    
    def test_default_solvers(self):
        """Test that default solvers are available."""
        factory = SolverFactory()
        solvers = factory.get_available_solvers()
        
        # Check that classical solver is available by default
        assert 'classical' in solvers
    
    def test_create_classical_solver(self):
        """Test creation of classical solver with default and custom parameters."""
        factory = SolverFactory()
        
        # Create with default parameters
        solver = factory.create_solver('classical')
        assert isinstance(solver, ClassicalSolver)
        
        # Default parameters should be applied
        assert solver.max_iterations == 5
        assert solver.initial_penalty == 100.0
        
        # Create with custom parameters
        solver = factory.create_solver('classical', max_iterations=10, initial_penalty=200.0)
        assert isinstance(solver, ClassicalSolver)
        assert solver.max_iterations == 10
        assert solver.initial_penalty == 200.0
    
    def test_register_new_solver(self):
        """Test registration of a new solver type."""
        factory = SolverFactory()
        
        # Register a new solver type
        factory.register_solver('mock', MockSolver, {'mock_param': 'test', 'iterations': 5})
        
        # Check that it's now available
        solvers = factory.get_available_solvers()
        assert 'mock' in solvers
        
        # Create the solver with default parameters
        solver = factory.create_solver('mock')
        assert isinstance(solver, MockSolver)
        assert solver.mock_param == 'test'
        assert solver.iterations == 5
        
        # Create with custom parameters
        solver = factory.create_solver('mock', mock_param='custom', iterations=20)
        assert solver.mock_param == 'custom'
        assert solver.iterations == 20
    
    def test_get_solver_parameters(self):
        """Test getting solver parameters."""
        factory = SolverFactory()
        
        # Get parameters for classical solver
        params = factory.get_solver_parameters('classical')
        assert 'max_iterations' in params
        assert 'initial_penalty' in params
        
        # Register a new solver and get its parameters
        factory.register_solver('mock', MockSolver, {'mock_param': 'test', 'iterations': 5})
        params = factory.get_solver_parameters('mock')
        assert 'mock_param' in params
        assert params['mock_param'] == 'test'
        assert params['iterations'] == 5
    
    def test_invalid_solver_type(self):
        """Test error handling for invalid solver types."""
        factory = SolverFactory()
        
        # Try to create a non-existent solver
        with pytest.raises(ValueError):
            factory.create_solver('non_existent')
        
        # Try to get parameters for a non-existent solver
        with pytest.raises(ValueError):
            factory.get_solver_parameters('non_existent')
    
    def test_override_existing_solver(self):
        """Test overriding an existing solver registration."""
        factory = SolverFactory()
        
        # Register a mock solver with the same name as an existing one
        factory.register_solver('classical', MockSolver, {'mock_param': 'override'})
        
        # Create the solver and check it's now the mock version
        solver = factory.create_solver('classical')
        assert isinstance(solver, MockSolver)
        assert solver.mock_param == 'override'
