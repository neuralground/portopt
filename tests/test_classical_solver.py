"""Tests for classical solver."""

import numpy as np
from portopt.solvers.classical import ClassicalSolver
from portopt.core.problem import PortfolioOptProblem

def test_classical_solver_basic():
    # Create a simple test problem
    returns = np.random.randn(3, 100)
    constraints = {'sum_to_one': True}
    problem = PortfolioOptProblem(returns=returns, constraints=constraints)
    
    # Solve it
    solver = ClassicalSolver()
    result = solver.solve(problem)
    
    # Basic checks
    assert result.weights.shape == (3,)
    assert np.isclose(np.sum(result.weights), 1)  # Sum to 1
    assert np.all(result.weights >= 0)  # No short selling
    assert result.feasible

