"""Problem definition module."""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

@dataclass
class PortfolioOptProblem:
    """Represents a portfolio optimization problem instance."""

    returns: np.ndarray  # Historical returns matrix (n_assets x n_periods)
    constraints: Dict    # Dictionary of constraints
    target_return: Optional[float] = None

    def validate(self) -> bool:
        """Validates the problem instance."""
        if not isinstance(self.returns, np.ndarray):
            raise ValueError("Returns must be a numpy array")
        if len(self.returns.shape) != 2:
            raise ValueError("Returns must be a 2D array")
        if np.isnan(self.returns).any():
            raise ValueError("Returns contain NaN values")
        return True

    def __post_init__(self):
        """Initialize derived attributes after validation."""
        self.validate()  # Validate before accessing shape
        self.n_assets = self.returns.shape[0]
        self.n_periods = self.returns.shape[1]
        self.exp_returns = np.mean(self.returns, axis=1)
        self.cov_matrix = np.cov(self.returns)

