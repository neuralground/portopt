"""Black-Litterman model implementation for portfolio optimization.

The Black-Litterman model is a sophisticated approach to asset allocation that
combines market equilibrium returns with investor views to produce more stable
and intuitive portfolios compared to traditional mean-variance optimization.

Key features of this implementation:
1. Market equilibrium return calculation using reverse optimization
2. Flexible specification of investor views (absolute and relative)
3. Bayesian blending of market equilibrium with investor views
4. Optimal portfolio construction based on the posterior estimates
5. Confidence parameters for both prior and view uncertainties
6. Support for sparse and partial views on subsets of assets

References:
    [1] Black, F., & Litterman, R. (1992). Global portfolio optimization.
        Financial Analysts Journal, 48(5), 28-43.
    [2] He, G., & Litterman, R. (1999). The intuition behind Black-Litterman
        model portfolios. Goldman Sachs Investment Management Research.
    [3] Idzorek, T. M. (2005). A step-by-step guide to the Black-Litterman model.
        Forecasting Expected Returns in the Financial Markets, 17.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class InvestorView:
    """Represents an investor view for the Black-Litterman model.
    
    Attributes:
        assets: List of asset indices or names the view applies to
        weights: Weights for the assets in the view (for relative views)
        value: Expected return value for the view
        confidence: Confidence level in the view (0-1, higher means more confident)
        is_relative: Whether this is a relative view (comparing assets) or absolute
    """
    assets: List[Union[int, str]]
    weights: List[float]
    value: float
    confidence: float = 0.5
    is_relative: bool = False
    
    def __post_init__(self):
        """Validate the view parameters."""
        if len(self.assets) != len(self.weights):
            raise ValueError("Number of assets must match number of weights")
        
        if self.is_relative and abs(sum(self.weights)) > 1e-10:
            logger.warning("Weights in relative view don't sum to zero. Normalizing.")
            # For relative views, weights should sum to zero
            mean_weight = sum(self.weights) / len(self.weights)
            self.weights = [w - mean_weight for w in self.weights]


class BlackLittermanModel:
    """Black-Litterman model for portfolio optimization.
    
    This class implements the Black-Litterman approach to asset allocation,
    which combines market equilibrium with investor views to produce more
    stable and intuitive portfolios.
    
    Attributes:
        risk_aversion: Risk aversion coefficient (lambda)
        tau: Uncertainty scaling parameter for prior
        use_market_caps: Whether to use market caps for equilibrium weights
        default_view_confidence: Default confidence level for views
    """
    
    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        use_market_caps: bool = True,
        default_view_confidence: float = 0.5
    ):
        """Initialize the Black-Litterman model.
        
        Args:
            risk_aversion: Risk aversion coefficient (default: 2.5)
            tau: Uncertainty scaling parameter for prior (default: 0.05)
            use_market_caps: Whether to use market caps for equilibrium weights (default: True)
            default_view_confidence: Default confidence level for views (default: 0.5)
        """
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.use_market_caps = use_market_caps
        self.default_view_confidence = default_view_confidence
        
    def calculate_equilibrium_returns(
        self,
        cov_matrix: np.ndarray,
        market_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculate implied equilibrium returns using reverse optimization.
        
        Args:
            cov_matrix: Covariance matrix of asset returns
            market_weights: Market capitalization weights (if None, equal weights are used)
            
        Returns:
            Vector of implied equilibrium returns
        """
        n_assets = cov_matrix.shape[0]
        
        # If market weights not provided, use equal weights
        if market_weights is None:
            market_weights = np.ones(n_assets) / n_assets
            
        # Reverse optimization: π = λΣw
        # where π is the implied excess returns, λ is risk aversion,
        # Σ is the covariance matrix, and w is the market weights
        implied_returns = self.risk_aversion * cov_matrix @ market_weights
        
        return implied_returns
    
    def _create_view_matrices(
        self,
        views: List[InvestorView],
        n_assets: int,
        asset_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create the view matrices (P, Q, Omega) from investor views.
        
        Args:
            views: List of investor views
            n_assets: Number of assets in the portfolio
            asset_names: Optional list of asset names for string-based view specification
            
        Returns:
            Tuple containing:
            - P: View picking matrix
            - Q: View expected returns vector
            - Omega: View uncertainty matrix
        """
        n_views = len(views)
        
        # Initialize matrices
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        Omega = np.zeros((n_views, n_views))
        
        # Process each view
        for i, view in enumerate(views):
            Q[i] = view.value
            
            # Map asset names to indices if needed
            asset_indices = []
            if asset_names is not None and isinstance(view.assets[0], str):
                asset_indices = [asset_names.index(asset) for asset in view.assets]
            else:
                asset_indices = view.assets
                
            # Fill in the picking matrix row
            for j, (asset_idx, weight) in enumerate(zip(asset_indices, view.weights)):
                P[i, asset_idx] = weight
                
            # Set confidence in the view
            confidence = view.confidence if view.confidence is not None else self.default_view_confidence
            
            # Ensure confidence is not too close to 1.0 to avoid singular matrix
            confidence = min(confidence, 0.9999)
            
            # Omega is diagonal with each element representing view uncertainty
            # Lower confidence means higher uncertainty
            # Add a small constant to ensure Omega is never singular
            Omega[i, i] = max((1.0 / confidence - 1.0) * 0.1, 1e-6)
            
        return P, Q, Omega
    
    def blend_returns(
        self,
        cov_matrix: np.ndarray,
        views: List[InvestorView],
        market_weights: Optional[np.ndarray] = None,
        asset_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Blend market equilibrium returns with investor views.
        
        Args:
            cov_matrix: Covariance matrix of asset returns
            views: List of investor views
            market_weights: Market capitalization weights (if None, equal weights are used)
            asset_names: Optional list of asset names for string-based view specification
            
        Returns:
            Tuple containing:
            - Posterior expected returns
            - Posterior covariance matrix
        """
        n_assets = cov_matrix.shape[0]
        
        # Calculate implied equilibrium returns
        pi = self.calculate_equilibrium_returns(cov_matrix, market_weights)
        
        # If no views provided, return the prior
        if not views:
            return pi, cov_matrix
        
        # Create view matrices
        P, Q, Omega = self._create_view_matrices(views, n_assets, asset_names)
        
        # Calculate posterior expected returns
        # Formula: E[R] = [(τΣ)^-1 + P'Ω^-1P]^-1 * [(τΣ)^-1*π + P'Ω^-1*Q]
        
        # Prior precision matrix
        prior_precision = np.linalg.inv(self.tau * cov_matrix)
        
        # View precision matrix (inverse of view uncertainty)
        try:
            view_precision = np.linalg.inv(Omega)
        except np.linalg.LinAlgError:
            # If Omega is singular, add a small regularization term
            logger.warning("Omega matrix is singular. Adding regularization.")
            Omega_reg = Omega + np.eye(Omega.shape[0]) * 1e-6
            view_precision = np.linalg.inv(Omega_reg)
        
        # Combined precision matrix
        combined_precision = prior_precision + P.T @ view_precision @ P
        
        # Combined information
        combined_info = prior_precision @ pi + P.T @ view_precision @ Q
        
        # Posterior expected returns
        try:
            posterior_returns = np.linalg.inv(combined_precision) @ combined_info
        except np.linalg.LinAlgError:
            # If combined_precision is singular, add a small regularization term
            logger.warning("Combined precision matrix is singular. Adding regularization.")
            combined_precision_reg = combined_precision + np.eye(combined_precision.shape[0]) * 1e-6
            posterior_returns = np.linalg.inv(combined_precision_reg) @ combined_info
        
        # Posterior covariance (optional, can be used for uncertainty analysis)
        posterior_covariance = np.linalg.inv(combined_precision)
        
        return posterior_returns, posterior_covariance
    
    def optimize_portfolio(
        self,
        cov_matrix: np.ndarray,
        views: List[InvestorView],
        market_weights: Optional[np.ndarray] = None,
        asset_names: Optional[List[str]] = None,
        risk_aversion: Optional[float] = None
    ) -> np.ndarray:
        """Generate optimal portfolio weights using the Black-Litterman model.
        
        Args:
            cov_matrix: Covariance matrix of asset returns
            views: List of investor views
            market_weights: Market capitalization weights (if None, equal weights are used)
            asset_names: Optional list of asset names for string-based view specification
            risk_aversion: Risk aversion parameter (if None, use instance value)
            
        Returns:
            Optimal portfolio weights
        """
        # Use instance risk aversion if not provided
        if risk_aversion is None:
            risk_aversion = self.risk_aversion
            
        # Calculate posterior expected returns and covariance
        posterior_returns, posterior_covariance = self.blend_returns(
            cov_matrix, views, market_weights, asset_names
        )
        
        # Calculate optimal portfolio weights
        # Formula: w* = (λΣ)^-1 * E[R]
        optimal_weights = np.linalg.inv(risk_aversion * cov_matrix) @ posterior_returns
        
        # Normalize weights to sum to 1
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        # Ensure no negative weights (optional, can be removed if short selling is allowed)
        optimal_weights = np.maximum(optimal_weights, 0)
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        return optimal_weights
    
    def calculate_weight_uncertainty(
        self,
        cov_matrix: np.ndarray,
        views: List[InvestorView],
        market_weights: Optional[np.ndarray] = None,
        asset_names: Optional[List[str]] = None,
        risk_aversion: Optional[float] = None,
        n_samples: int = 1000
    ) -> Dict[str, np.ndarray]:
        """Calculate uncertainty in optimal weights using Monte Carlo simulation.
        
        Args:
            cov_matrix: Covariance matrix of asset returns
            views: List of investor views
            market_weights: Market capitalization weights
            asset_names: Optional list of asset names
            risk_aversion: Risk aversion parameter
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary containing mean and standard deviation of weights
        """
        # Use instance risk aversion if not provided
        if risk_aversion is None:
            risk_aversion = self.risk_aversion
            
        # Calculate posterior expected returns and covariance
        posterior_returns, posterior_covariance = self.blend_returns(
            cov_matrix, views, market_weights, asset_names
        )
        
        # Add small regularization to posterior covariance to ensure stability
        posterior_covariance = posterior_covariance + np.eye(posterior_covariance.shape[0]) * 1e-6
        
        # Generate samples from the posterior distribution
        np.random.seed(42)  # For reproducibility
        samples = np.random.multivariate_normal(
            mean=posterior_returns,
            cov=posterior_covariance,
            size=n_samples
        )
        
        # Calculate optimal weights for each sample
        all_weights = []
        for sample in samples:
            try:
                # Calculate weights
                weights = np.linalg.inv(risk_aversion * cov_matrix) @ sample
                
                # Normalize and ensure non-negative
                weights = np.maximum(weights, 0)
                
                # Check for zero sum to avoid division by zero
                weights_sum = np.sum(weights)
                if weights_sum > 1e-10:
                    weights = weights / weights_sum
                else:
                    # If all weights are zero or very small, use equal weights
                    weights = np.ones_like(weights) / len(weights)
                
                all_weights.append(weights)
            except np.linalg.LinAlgError:
                # If we encounter a linear algebra error, use equal weights
                weights = np.ones(len(posterior_returns)) / len(posterior_returns)
                all_weights.append(weights)
            
        # Convert to array
        all_weights = np.array(all_weights)
        
        # Filter out any rows with NaN values
        valid_rows = ~np.isnan(all_weights).any(axis=1)
        if np.any(valid_rows):
            all_weights = all_weights[valid_rows]
        else:
            # If all rows have NaN, return equal weights with zero std
            equal_weights = np.ones(len(posterior_returns)) / len(posterior_returns)
            return {
                'mean_weights': equal_weights,
                'std_weights': np.zeros_like(equal_weights)
            }
        
        # Calculate statistics
        mean_weights = np.mean(all_weights, axis=0)
        std_weights = np.std(all_weights, axis=0)
        
        return {
            'mean_weights': mean_weights,
            'std_weights': std_weights
        }
