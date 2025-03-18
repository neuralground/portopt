"""Tests for problem definition."""

import pytest
import numpy as np
from portopt.core.problem import PortfolioOptProblem
from portopt.constraints.constraint_types import IndustryClassification

def test_problem_initialization():
    returns = np.random.randn(3, 100)
    constraints = {'sum_to_one': True}
    
    problem = PortfolioOptProblem(returns=returns, constraints=constraints)
    
    assert problem.n_assets == 3
    assert problem.n_periods == 100
    assert problem.exp_returns.shape == (3,)
    assert problem.cov_matrix.shape == (3, 3)

def test_problem_with_enhanced_data():
    """Test problem initialization with enhanced data types."""
    n_assets = 10
    n_periods = 100
    returns = np.random.randn(n_assets, n_periods)
    volumes = np.random.rand(n_assets, n_periods) * 1000
    factor_returns = np.random.randn(3, n_periods)
    factor_exposures = np.random.randn(n_assets, 3)

    classifications = [
        IndustryClassification("Tech", "Software", "Apps", "Mobile")
        for _ in range(n_assets)
    ]

    problem = PortfolioOptProblem(
        returns=returns,
        constraints={},
        volumes=volumes,
        factor_returns=factor_returns,
        factor_exposures=factor_exposures,
        classifications=classifications
    )

    assert problem.n_assets == n_assets
    assert problem.n_periods == n_periods
    assert problem.volumes is not None
    assert len(problem.classifications) == n_assets

def test_problem_validation():
    """Test validation of enhanced data."""
    n_assets = 10
    n_periods = 100
    returns = np.random.randn(n_assets, n_periods)

    # Test mismatched volumes shape
    with pytest.raises(ValueError, match="Volumes array must match returns shape"):
        PortfolioOptProblem(
            returns=returns,
            constraints={},
            volumes=np.random.rand(n_assets, n_periods + 1)
        )

    # Test factor model consistency
    with pytest.raises(ValueError, match="Factor exposures required"):
        PortfolioOptProblem(
            returns=returns,
            constraints={},
            factor_returns=np.random.randn(3, n_periods)
        )
