"""Test data generation module."""

import numpy as np
from typing import List, Tuple, Optional
from portopt.core.problem import PortfolioOptProblem

class TestDataGenerator:
    """Generates test data for portfolio optimization."""
    
    @staticmethod
    def generate_realistic_problem(
        n_assets: int,
        n_periods: int,
        volatility_range: Tuple[float, float] = (0.1, 0.3),
        correlation_range: Tuple[float, float] = (0.1, 0.7),
        seed: Optional[int] = None
    ) -> PortfolioOptProblem:
        """Generate a realistic portfolio optimization problem."""
        if seed is not None:
            np.random.seed(seed)
            
        # Generate random volatilities
        vols = np.random.uniform(*volatility_range, size=n_assets)
        
        # Generate random correlation matrix
        corr = np.random.uniform(*correlation_range, size=(n_assets, n_assets))
        corr = (corr + corr.T) / 2  # Make symmetric
        np.fill_diagonal(corr, 1)  # Diagonal should be 1
        
        # Ensure positive semi-definite
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, 0)
        corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # Convert correlation to covariance
        cov = np.diag(vols) @ corr @ np.diag(vols)
        
        # Generate returns
        returns = np.random.multivariate_normal(
            mean=np.random.randn(n_assets) * 0.1,
            cov=cov,
            size=n_periods
        ).T
        
        constraints = {
            'sum_to_one': True,
            'no_short': True
        }
        
        return PortfolioOptProblem(returns=returns, constraints=constraints)

