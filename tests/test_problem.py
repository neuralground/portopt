"""Tests for problem definition."""

import pytest
import numpy as np
from portopt.core.problem import PortfolioOptProblem

def test_problem_initialization():
    returns = np.random.randn(3, 100)
    constraints = {'sum_to_one': True}
    
    problem = PortfolioOptProblem(returns=returns, constraints=constraints)
    
    assert problem.n_assets == 3
    assert problem.n_periods == 100
    assert problem.exp_returns.shape == (3,)
    assert problem.cov_matrix.shape == (3, 3)

def test_problem_validation():
    """Test various invalid input cases."""

    # Test case 1: 1D array (invalid shape)
    with pytest.raises(ValueError, match="must be a 2D array"):
        PortfolioOptProblem(
            returns=np.random.randn(10),  # 1D array
            constraints={}
        )

    # Test case 2: NaN values
    returns = np.random.randn(3, 100)
    returns[0, 0] = np.nan
    with pytest.raises(ValueError, match="contain NaN values"):
        PortfolioOptProblem(
            returns=returns,
            constraints={}
        )

