"""Problem definition module."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from portopt.constraints.constraint_types import (
    IndustryClassification, AssetClass, CurrencyExposure, CreditProfile
)

@dataclass
class PortfolioOptProblem:
    """Represents a portfolio optimization problem instance."""

    returns: np.ndarray  # Historical returns matrix (n_assets x n_periods)
    constraints: Dict    # Dictionary of constraints
    target_return: Optional[float] = None
    
    # Market impact data
    volumes: Optional[np.ndarray] = None      # Trading volumes
    spreads: Optional[np.ndarray] = None      # Bid-ask spreads
    market_caps: Optional[np.ndarray] = None  # Market capitalizations
    
    # Factor model data
    factor_returns: Optional[np.ndarray] = None    # Factor returns
    factor_exposures: Optional[np.ndarray] = None  # Factor loadings
    
    # Classification data
    classifications: Optional[List[IndustryClassification]] = None
    asset_classes: Optional[List[AssetClass]] = None
    currencies: Optional[List[CurrencyExposure]] = None
    credit_profiles: Optional[List[CreditProfile]] = None

    def validate(self) -> bool:
        """Validates the problem instance."""
        if not isinstance(self.returns, np.ndarray):
            raise ValueError("Returns must be a numpy array")
        if len(self.returns.shape) != 2:
            raise ValueError("Returns must be a 2D array")
        if np.isnan(self.returns).any():
            raise ValueError("Returns contain NaN values")
            
        # Validate shapes of optional arrays
        n_assets, n_periods = self.returns.shape
        
        if self.volumes is not None and self.volumes.shape != (n_assets, n_periods):
            raise ValueError("Volumes array must match returns shape")
        if self.spreads is not None and self.spreads.shape != (n_assets, n_periods):
            raise ValueError("Spreads array must match returns shape")
            
        if self.factor_returns is not None:
            if self.factor_exposures is None:
                raise ValueError("Factor exposures required if factor returns provided")
            if self.factor_returns.shape[1] != n_periods:
                raise ValueError("Factor returns must have same number of periods as returns")
            if self.factor_exposures.shape[0] != n_assets:
                raise ValueError("Factor exposures must have same number of assets as returns")
            if self.factor_exposures.shape[1] != self.factor_returns.shape[0]:
                raise ValueError("Factor exposures and returns dimensions must match")
                
        # Validate classification data
        if self.classifications is not None and len(self.classifications) != n_assets:
            raise ValueError("Number of classifications must match number of assets")
        if self.asset_classes is not None and len(self.asset_classes) != n_assets:
            raise ValueError("Number of asset classes must match number of assets")
        if self.currencies is not None and len(self.currencies) != n_assets:
            raise ValueError("Number of currency exposures must match number of assets")
        if self.credit_profiles is not None and len(self.credit_profiles) != n_assets:
            raise ValueError("Number of credit profiles must match number of assets")
            
        return True

    def __post_init__(self):
        """Initialize derived attributes after validation."""
        self.validate()
        self.n_assets = self.returns.shape[0]
        self.n_periods = self.returns.shape[1]
        
        # Calculate returns statistics
        self.exp_returns = np.mean(self.returns, axis=1)
        
        # Calculate covariance either from factor model or empirically
        if self.factor_returns is not None and self.factor_exposures is not None:
            factor_cov = np.cov(self.factor_returns)
            specific_var = np.var(
                self.returns - self.factor_exposures @ self.factor_returns, 
                axis=1
            )
            self.cov_matrix = (
                self.factor_exposures @ factor_cov @ self.factor_exposures.T + 
                np.diag(specific_var)
            )
        else:
            self.cov_matrix = np.cov(self.returns)
            
        # Pre-calculate market impact metrics if data available
        if self.volumes is not None and self.spreads is not None:
            self.avg_daily_volumes = np.mean(self.volumes, axis=1)
            self.avg_spreads = np.mean(self.spreads, axis=1)
