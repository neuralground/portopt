import numpy as np
from typing import Dict, Optional, Tuple
from scipy import stats

class EnhancedRiskMetrics:
    """Advanced risk metrics calculator for portfolio optimization.
    
    This class provides comprehensive risk analysis capabilities including:
    - Standard risk measures (volatility, VaR, CVaR)
    - Factor model risk decomposition
    - Liquidity-adjusted risk metrics
    - Tracking error and active risk
    - Stress testing and scenario analysis
    
    The implementation supports multiple risk measurement approaches:
    1. Empirical calculation using historical returns
    2. Factor model decomposition when factor data available
    3. Liquidity-adjusted measures when market data available
    4. Relative risk metrics against benchmark
    
    Typical usage:
        >>> metrics = EnhancedRiskMetrics(returns, factor_returns, factor_exposures)
        >>> risk_analysis = metrics.calculate_all_metrics(
        ...     weights=portfolio_weights,
        ...     benchmark_weights=benchmark_weights
        ... )
    """
    
    def __init__(self, returns: np.ndarray, 
                 factor_returns: Optional[np.ndarray] = None,
                 factor_exposures: Optional[np.ndarray] = None):
        """Initialize risk metrics calculator.
        
        Args:
            returns: Asset returns matrix (n_assets x n_periods)
            factor_returns: Optional factor returns (n_factors x n_periods)
            factor_exposures: Optional factor loadings (n_assets x n_factors)
            
        The class supports two operating modes:
        1. Empirical mode: Using only historical returns
        2. Factor model mode: Using returns and factor data for decomposition
        """
        self.returns = returns
        self.factor_returns = factor_returns
        self.factor_exposures = factor_exposures
        
    def calculate_var_cvar(self, weights: np.ndarray, 
                          confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional Value at Risk.
        
        Uses historical simulation approach to estimate tail risk:
        1. Compute portfolio historical returns
        2. Sort returns to find VaR threshold
        3. Calculate CVaR as average of tail events
        
        Args:
            weights: Portfolio weights
            confidence_level: Statistical confidence (e.g., 0.95 for 95% VaR)
            
        Returns:
            Tuple containing:
            - Value at Risk (VaR) at specified confidence level
            - Conditional Value at Risk (CVaR) measuring tail risk
        """
        # Calculate portfolio historical returns
        portfolio_returns = self.returns.T @ weights
        sorted_returns = np.sort(portfolio_returns)
        
        # Find VaR threshold index
        var_idx = int((1 - confidence_level) * len(sorted_returns))
        
        # Calculate VaR and CVaR
        var = -sorted_returns[var_idx]
        cvar = -np.mean(sorted_returns[:var_idx])
        
        return var, cvar
        
    def calculate_tracking_error(self, weights: np.ndarray, 
                               benchmark_weights: np.ndarray) -> float:
        """Calculate ex-post tracking error against benchmark.
        
        Tracking error calculation process:
        1. Compute portfolio and benchmark returns
        2. Calculate return differences
        3. Compute annualized standard deviation
        
        Args:
            weights: Portfolio weights
            benchmark_weights: Benchmark portfolio weights
            
        Returns:
            Annualized tracking error (std dev of active returns)
        """
        # Calculate return series
        portfolio_returns = self.returns.T @ weights
        benchmark_returns = self.returns.T @ benchmark_weights
        
        # Calculate and annualize tracking error
        tracking_diff = portfolio_returns - benchmark_returns
        return np.std(tracking_diff) * np.sqrt(252)  # Annualize assuming daily data
        
    def calculate_factor_exposures(self, weights: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio factor exposures and risk decomposition.
        
        Factor analysis process:
        1. Calculate portfolio factor exposures
        2. Compute factor covariances
        3. Decompose risk into factor and specific components
        4. Calculate contribution percentages
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Dictionary containing:
            - factor_exposures: Exposure to each risk factor
            - risk_contribution: Risk contribution from each factor
            - total_factor_risk: Total systematic risk
            - specific_risk: Residual (idiosyncratic) risk
        """
        if self.factor_returns is None or self.factor_exposures is None:
            return {}
            
        # Calculate factor exposures
        portfolio_exposures = self.factor_exposures.T @ weights
        factor_covar = np.cov(self.factor_returns)
        factor_vol = np.sqrt(np.diag(factor_covar))
        
        # Calculate risk contributions
        factor_contrib = portfolio_exposures * (factor_vol @ portfolio_exposures)
        total_risk = np.sum(factor_contrib)
        
        # Calculate specific (residual) risk
        predicted_returns = self.factor_returns.T @ portfolio_exposures
        residual_returns = (self.returns.T @ weights) - predicted_returns
        specific_risk = np.std(residual_returns) * np.sqrt(252)
        
        return {
            'exposures': portfolio_exposures,
            'risk_contribution': factor_contrib / total_risk if total_risk > 0 else factor_contrib,
            'total_factor_risk': total_risk,
            'specific_risk': specific_risk
        }
        
    def calculate_liquidity_score(self, weights: np.ndarray,
                                volumes: np.ndarray,
                                spreads: np.ndarray) -> Dict[str, float]:
        """Calculate composite liquidity risk score.
        
        Incorporates multiple liquidity dimensions:
        1. Trading volume participation
        2. Bid-ask spread impact
        3. Price impact sensitivity
        4. Market depth measures
        
        Args:
            weights: Portfolio weights
            volumes: Trading volumes (n_assets x n_periods)
            spreads: Bid-ask spreads (n_assets x n_periods)
            
        Returns:
            Dictionary containing:
            - adv_participation: Volume participation metrics
            - illiquidity: Amihud illiquidity measure
            - spread_score: Spread-based liquidity score
            - composite_score: Combined liquidity risk measure
        """
        # Calculate volume participation
        adv_participation = weights / np.mean(volumes, axis=1)
        
        # Calculate Amihud illiquidity ratio
        daily_returns = np.diff(self.returns, axis=1)
        daily_dollar_volume = volumes[:, 1:] * np.abs(self.returns[:, 1:])
        illiquidity = np.mean(np.abs(daily_returns) / daily_dollar_volume, axis=1)
        
        # Calculate composite score (lower is more liquid)
        liquidity_score = (
            0.4 * stats.zscore(adv_participation) +
            0.4 * stats.zscore(illiquidity) +
            0.2 * stats.zscore(spreads.mean(axis=1))
        )
        
        return {
            'adv_participation': float(np.max(adv_participation)),
            'illiquidity': float(np.mean(illiquidity)),
            'spread_score': float(np.mean(spreads)),
            'composite_score': float(np.mean(liquidity_score))
        }
        
    def calculate_all_metrics(self, weights: np.ndarray,
                            benchmark_weights: Optional[np.ndarray] = None,
                            volumes: Optional[np.ndarray] = None,
                            spreads: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate comprehensive set of risk metrics.
        
        This method combines all available risk measures:
        1. Standard risk metrics (VaR, CVaR, volatility)
        2. Factor model analysis if factor data available
        3. Liquidity risk if market data available
        4. Relative risk if benchmark provided
        
        Args:
            weights: Portfolio weights
            benchmark_weights: Optional benchmark weights
            volumes: Optional trading volumes
            spreads: Optional bid-ask spreads
            
        Returns:
            Dictionary containing all calculated risk metrics
        """
        # Calculate basic risk metrics
        var, cvar = self.calculate_var_cvar(weights)
        portfolio_returns = self.returns.T @ weights
        
        metrics = {
            'var_95': var,
            'cvar_95': cvar,
            'volatility': float(np.std(portfolio_returns) * np.sqrt(252)),
            'skewness': float(stats.skew(portfolio_returns)),
            'kurtosis': float(stats.kurtosis(portfolio_returns))
        }
        
        # Add benchmark-relative metrics if available
        if benchmark_weights is not None:
            metrics['tracking_error'] = self.calculate_tracking_error(
                weights, benchmark_weights
            )
            metrics['active_risk'] = metrics['tracking_error']
            
        # Add factor analysis if available
        factor_metrics = self.calculate_factor_exposures(weights)
        if factor_metrics:
            metrics.update({
                'factor_exposures': factor_metrics['exposures'].tolist(),
                'factor_contribution': factor_metrics['risk_contribution'].tolist(),
                'systematic_risk': factor_metrics['total_factor_risk'],
                'specific_risk': factor_metrics['specific_risk']
            })
            
        # Add liquidity metrics if market data available
        if volumes is not None and spreads is not None:
            liquidity_metrics = self.calculate_liquidity_score(weights, volumes, spreads)
            metrics.update(liquidity_metrics)
            
        return metrics
    
 