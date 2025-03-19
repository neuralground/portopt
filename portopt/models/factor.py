"""Factor-based portfolio optimization model.

This module implements a multi-factor model for portfolio optimization,
allowing for exposures to common risk factors to be controlled and optimized.

The factor model provides several key capabilities:
1. Estimation of expected returns based on factor exposures and returns
2. Estimation of covariance matrices using factor structure
3. Optimization of portfolio weights with target factor exposures
4. Analysis of factor contributions to portfolio risk and return

The implementation supports:
- Blending between factor-based and historical estimates
- Targeting specific factor exposures with confidence levels
- Regularization to ensure numerical stability
- Decomposition of risk into factor and specific components

Example usage:
    # Create a factor model
    model = FactorModel(risk_aversion=2.0)
    
    # Optimize weights with target factor exposures
    weights, metadata = model.optimize_weights(
        factor_returns=factor_returns,
        factor_exposures=factor_exposures,
        target_factor_exposures=[
            FactorExposure(factor_name="Value", target_exposure=0.2, confidence=0.8),
            FactorExposure(factor_name="Momentum", target_exposure=0.1, confidence=0.5)
        ],
        historical_returns=historical_returns
    )
    
    # Analyze factor contributions
    factor_analysis = model.analyze_factor_contributions(
        weights=weights,
        factor_returns=factor_returns,
        factor_exposures=factor_exposures,
        factor_names=["Value", "Momentum", "Size", "Quality"]
    )
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class FactorExposure:
    """Represents a target exposure to a specific factor."""
    factor_name: str
    target_exposure: float
    min_exposure: Optional[float] = None
    max_exposure: Optional[float] = None
    confidence: float = 0.5  # Confidence in the target exposure (0-1)


class FactorModel:
    """Implements a multi-factor model for portfolio optimization.
    
    This model allows for:
    1. Estimating expected returns based on factor exposures
    2. Decomposing risk into factor and specific components
    3. Optimizing portfolios with controlled factor exposures
    4. Analyzing factor contributions to portfolio risk and return
    """
    
    def __init__(
        self,
        risk_aversion: float = 2.0,
        specific_risk_penalty: float = 0.1,
        regularization_lambda: float = 1e-4
    ):
        """Initialize the factor model.
        
        Args:
            risk_aversion: Risk aversion parameter (higher = more risk-averse)
            specific_risk_penalty: Penalty for specific risk (idiosyncratic)
            regularization_lambda: Regularization parameter for numerical stability
        """
        self.risk_aversion = risk_aversion
        self.specific_risk_penalty = specific_risk_penalty
        self.regularization_lambda = regularization_lambda
    
    def estimate_returns(
        self,
        factor_returns: np.ndarray,
        factor_exposures: np.ndarray,
        historical_returns: Optional[np.ndarray] = None,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Estimate expected returns using factor model.
        
        Args:
            factor_returns: Historical factor returns (n_factors x n_periods)
            factor_exposures: Asset exposures to factors (n_assets x n_factors)
            historical_returns: Historical asset returns (n_assets x n_periods)
            alpha: Blending parameter between factor and historical returns (0-1)
                  0 = use only historical returns, 1 = use only factor-based returns
        
        Returns:
            Expected returns vector (n_assets)
        """
        # Calculate factor expected returns (average of historical factor returns)
        factor_expected_returns = np.mean(factor_returns, axis=1)
        
        # Calculate expected returns based on factor model
        factor_based_returns = factor_exposures @ factor_expected_returns
        
        # If historical returns are provided, blend with factor-based returns
        if historical_returns is not None and alpha < 1.0:
            historical_expected_returns = np.mean(historical_returns, axis=1)
            return alpha * factor_based_returns + (1 - alpha) * historical_expected_returns
        
        return factor_based_returns
    
    def estimate_covariance(
        self,
        factor_returns: np.ndarray,
        factor_exposures: np.ndarray,
        historical_returns: Optional[np.ndarray] = None,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Estimate covariance matrix using factor model.
        
        Args:
            factor_returns: Historical factor returns (n_factors x n_periods)
            factor_exposures: Asset exposures to factors (n_assets x n_factors)
            historical_returns: Historical asset returns (n_assets x n_periods)
            alpha: Blending parameter between factor and historical covariance (0-1)
                  0 = use only historical covariance, 1 = use only factor-based covariance
        
        Returns:
            Covariance matrix (n_assets x n_assets)
        """
        # Calculate factor covariance matrix
        factor_cov = np.cov(factor_returns)
        
        # Add regularization to ensure positive definiteness
        factor_cov = factor_cov + np.eye(factor_cov.shape[0]) * self.regularization_lambda
        
        # Calculate specific variance (idiosyncratic risk)
        if historical_returns is not None:
            # Estimate specific variance as residual variance
            factor_returns_expanded = factor_exposures @ factor_returns
            residual_returns = historical_returns - factor_returns_expanded
            specific_var = np.var(residual_returns, axis=1)
        else:
            # If no historical returns, use a small default value
            specific_var = np.ones(factor_exposures.shape[0]) * 0.01
        
        # Calculate factor-based covariance matrix
        factor_based_cov = (
            factor_exposures @ factor_cov @ factor_exposures.T +
            np.diag(specific_var)
        )
        
        # If historical returns are provided, blend with factor-based covariance
        if historical_returns is not None and alpha < 1.0:
            historical_cov = np.cov(historical_returns)
            # Add regularization to ensure positive definiteness
            historical_cov = historical_cov + np.eye(historical_cov.shape[0]) * self.regularization_lambda
            return alpha * factor_based_cov + (1 - alpha) * historical_cov
        
        return factor_based_cov
    
    def optimize_weights(
        self,
        factor_returns: np.ndarray,
        factor_exposures: np.ndarray,
        target_factor_exposures: Optional[List[FactorExposure]] = None,
        historical_returns: Optional[np.ndarray] = None,
        alpha: float = 0.5,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Optimize portfolio weights using factor model.
        
        Args:
            factor_returns: Factor returns matrix (n_factors x n_periods)
            factor_exposures: Asset exposures to factors (n_assets x n_factors)
            target_factor_exposures: List of target factor exposures
            historical_returns: Historical asset returns (n_assets x n_periods)
            alpha: Weight between factor and historical estimates (0 = only historical, 1 = only factor)
            constraints: Additional constraints
            
        Returns:
            Tuple of (weights, metadata)
        """
        # Get dimensions
        n_assets = factor_exposures.shape[0]
        n_factors = factor_returns.shape[0]
        
        # Ensure factor exposures and returns have compatible dimensions
        assert factor_exposures.shape[1] == n_factors, \
            f"Factor exposures shape {factor_exposures.shape} incompatible with factor returns shape {factor_returns.shape}"
        
        # Calculate expected returns and covariance
        expected_returns = self.estimate_returns(
            factor_returns, factor_exposures, historical_returns, alpha
        )
        covariance = self.estimate_covariance(
            factor_returns, factor_exposures, historical_returns, alpha
        )
        
        # Basic mean-variance optimization
        inv_cov = np.linalg.inv(self.risk_aversion * covariance)
        weights = inv_cov @ expected_returns
        
        # Apply basic constraints
        weights = np.maximum(weights, 0)  # Long-only constraint
        weights = weights / np.sum(weights)  # Sum to 1
        
        # Adjust for target factor exposures if provided
        if target_factor_exposures:
            weights = self._adjust_for_factor_exposures(
                weights, factor_exposures, target_factor_exposures, covariance
            )
            
            # Ensure we still have a positive expected return after adjustment
            # If not, blend with the original weights to maintain some return
            portfolio_return = weights @ expected_returns
            if portfolio_return <= 0:
                # Blend with original weights to ensure positive return
                original_weights = inv_cov @ expected_returns
                original_weights = np.maximum(original_weights, 0)
                original_weights = original_weights / np.sum(original_weights)
                
                # Find minimum blend that gives positive return
                blend_factor = 0.5
                blended_weights = (1 - blend_factor) * weights + blend_factor * original_weights
                blended_weights = blended_weights / np.sum(blended_weights)
                
                # Ensure positive return
                while (blended_weights @ expected_returns <= 0) and (blend_factor < 0.95):
                    blend_factor += 0.1
                    blended_weights = (1 - blend_factor) * weights + blend_factor * original_weights
                    blended_weights = blended_weights / np.sum(blended_weights)
                
                weights = blended_weights
        
        # Calculate portfolio statistics
        portfolio_return = weights @ expected_returns
        
        # Calculate factor and specific risk
        factor_cov = np.cov(factor_returns)
        factor_risk = np.sqrt(weights @ factor_exposures @ factor_cov @ factor_exposures.T @ weights)
        
        # Calculate specific risk
        specific_var = np.diag(covariance) - np.diag(factor_exposures @ factor_cov @ factor_exposures.T)
        specific_var = np.maximum(specific_var, 0)  # Ensure non-negative
        specific_risk = np.sqrt(weights @ (specific_var * np.eye(n_assets)) @ weights)
        
        # Calculate total risk
        total_risk = np.sqrt(factor_risk**2 + specific_risk**2)
        
        # Calculate Sharpe ratio
        sharpe_ratio = portfolio_return / total_risk if total_risk > 0 else 0
        
        # Calculate portfolio factor exposures
        portfolio_exposures = weights @ factor_exposures
        
        # Return weights and metadata
        metadata = {
            'expected_return': portfolio_return,
            'total_risk': total_risk,
            'factor_risk': factor_risk,
            'specific_risk': specific_risk,
            'sharpe_ratio': sharpe_ratio,
            'portfolio_exposures': portfolio_exposures
        }
        
        return weights, metadata
    
    def _adjust_for_factor_exposures(
        self,
        initial_weights: np.ndarray,
        factor_exposures: np.ndarray,
        target_exposures: List[FactorExposure],
        covariance: np.ndarray
    ) -> np.ndarray:
        """Adjust weights to meet target factor exposures.
        
        Args:
            initial_weights: Initial portfolio weights
            factor_exposures: Asset exposures to factors
            target_exposures: List of target factor exposures
            covariance: Covariance matrix
            
        Returns:
            Adjusted weights
        """
        weights = initial_weights.copy()
        n_assets = len(weights)
        
        # Get factor names and indices
        factor_names = [te.factor_name for te in target_exposures]
        
        # Create a mapping of factor names to column indices
        # For simplicity, we'll assume factors are indexed 0...n-1
        factor_indices = list(range(len(factor_names)))
        
        # Extract the relevant factor exposures
        relevant_exposures = factor_exposures[:, factor_indices]
        
        # Calculate current factor exposures
        current_exposures = weights @ relevant_exposures
        
        # Calculate target exposures and confidence weights
        target_values = np.array([te.target_exposure for te in target_exposures])
        confidence_weights = np.array([te.confidence for te in target_exposures])
        
        # Calculate exposure differences
        exposure_diff = target_values - current_exposures
        
        # Weight differences by confidence
        weighted_diff = exposure_diff * confidence_weights
        
        # If all differences are small, return the current weights
        if np.all(np.abs(weighted_diff) < 1e-4):
            return weights
        
        # Iterative adjustment to gradually move toward target exposures
        max_iterations = 10
        learning_rate = 0.5
        
        for iteration in range(max_iterations):
            # Calculate current exposures
            current_exposures = weights @ relevant_exposures
            
            # Calculate exposure differences
            exposure_diff = target_values - current_exposures
            
            # Weight differences by confidence
            weighted_diff = exposure_diff * confidence_weights
            
            # If all differences are small, break
            if np.all(np.abs(weighted_diff) < 1e-4):
                break
            
            # Calculate the adjustment to weights
            adjustment = np.zeros(n_assets)
            for i, diff in enumerate(weighted_diff):
                if abs(diff) > 1e-4:  # Only adjust if difference is significant
                    # Gradient direction for this factor
                    factor_gradient = relevant_exposures[:, i]
                    
                    # Scale the adjustment by learning rate and difference
                    adjustment += learning_rate * diff * factor_gradient
            
            # Apply the adjustment
            weights += adjustment
            
            # Re-normalize and apply constraints
            weights = np.maximum(weights, 0)  # Long-only constraint
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)  # Sum to 1
            else:
                # If all weights are zero, revert to equal weights
                weights = np.ones(n_assets) / n_assets
        
        return weights
    
    def analyze_factor_contributions(
        self,
        weights: np.ndarray,
        factor_returns: np.ndarray,
        factor_exposures: np.ndarray,
        factor_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze factor contributions to portfolio risk and return.
        
        Args:
            weights: Portfolio weights
            factor_returns: Historical factor returns
            factor_exposures: Asset exposures to factors
            factor_names: Names of the factors
            
        Returns:
            Dictionary with factor contribution analysis
        """
        # Calculate portfolio factor exposures
        portfolio_exposures = weights @ factor_exposures
        
        # Calculate factor expected returns
        factor_expected_returns = np.mean(factor_returns, axis=1)
        
        # Calculate factor covariance
        factor_cov = np.cov(factor_returns)
        
        # Calculate factor contribution to return
        return_contribution = portfolio_exposures * factor_expected_returns
        
        # Calculate factor contribution to risk (marginal contribution)
        risk_contribution = portfolio_exposures @ factor_cov
        
        # Calculate total portfolio risk from factors
        portfolio_factor_risk = np.sqrt(portfolio_exposures @ factor_cov @ portfolio_exposures)
        
        # Calculate percentage contribution to risk
        if portfolio_factor_risk > 0:
            pct_risk_contribution = (risk_contribution * portfolio_exposures) / portfolio_factor_risk
        else:
            pct_risk_contribution = np.zeros_like(portfolio_exposures)
        
        # Create result dictionary
        result = {
            'portfolio_exposures': portfolio_exposures,
            'return_contribution': return_contribution,
            'risk_contribution': risk_contribution,
            'pct_risk_contribution': pct_risk_contribution,
            'total_factor_risk': portfolio_factor_risk,
            'factor_contribution': portfolio_exposures  # Add this for backward compatibility
        }
        
        # Add factor names if provided
        if factor_names is not None:
            result['factor_names'] = factor_names
            result['exposure_by_factor'] = {
                factor_names[i]: portfolio_exposures[i]
                for i in range(len(factor_names))
            }
            result['return_contribution_by_factor'] = {
                factor_names[i]: return_contribution[i]
                for i in range(len(factor_names))
            }
            result['risk_contribution_by_factor'] = {
                factor_names[i]: pct_risk_contribution[i]
                for i in range(len(factor_names))
            }
        
        return result
