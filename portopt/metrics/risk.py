import numpy as np
from typing import Dict, Optional, Tuple
from scipy import stats

class EnhancedRiskMetrics:
    """Enhanced risk metrics calculator with VaR, CVaR, and factor analysis."""
    
    def __init__(self, returns: np.ndarray, factor_returns: Optional[np.ndarray] = None,
                 factor_exposures: Optional[np.ndarray] = None):
        self.returns = returns
        self.factor_returns = factor_returns
        self.factor_exposures = factor_exposures
        
    def calculate_var_cvar(self, weights: np.ndarray, 
                          confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional Value at Risk."""
        portfolio_returns = self.returns.T @ weights
        sorted_returns = np.sort(portfolio_returns)
        var_idx = int((1 - confidence_level) * len(sorted_returns))
        
        var = -sorted_returns[var_idx]
        cvar = -np.mean(sorted_returns[:var_idx])
        
        return var, cvar
        
    def calculate_tracking_error(self, weights: np.ndarray, 
                               benchmark_weights: np.ndarray) -> float:
        """Calculate ex-post tracking error."""
        portfolio_returns = self.returns.T @ weights
        benchmark_returns = self.returns.T @ benchmark_weights
        tracking_diff = portfolio_returns - benchmark_returns
        return np.std(tracking_diff) * np.sqrt(252)  # Annualized
        
    def calculate_factor_exposures(self, weights: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio factor exposures and risk decomposition."""
        if self.factor_returns is None or self.factor_exposures is None:
            return {}
            
        portfolio_exposures = self.factor_exposures.T @ weights
        factor_covar = np.cov(self.factor_returns)
        factor_vol = np.sqrt(np.diag(factor_covar))
        
        # Calculate factor contributions to risk
        factor_contrib = portfolio_exposures * (factor_vol @ portfolio_exposures)
        total_risk = np.sum(factor_contrib)
        
        return {
            'exposures': portfolio_exposures,
            'contribution': factor_contrib / total_risk if total_risk > 0 else factor_contrib,
            'total_factor_risk': total_risk
        }
        
    def calculate_liquidity_score(self, weights: np.ndarray, 
                                volumes: np.ndarray, 
                                spreads: np.ndarray) -> Dict[str, float]:
        """Calculate composite liquidity score."""
        # ADV participation
        adv_participation = weights / np.mean(volumes, axis=1)
        
        # Amihud illiquidity ratio
        daily_returns = np.diff(self.returns, axis=1)
        daily_dollar_volume = volumes[:, 1:] * np.abs(self.returns[:, 1:])
        illiquidity = np.mean(np.abs(daily_returns) / daily_dollar_volume, axis=1)
        
        # Composite score (lower is more liquid)
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
        """Calculate all risk and performance metrics."""
        var, cvar = self.calculate_var_cvar(weights)
        metrics = {
            'var_95': var,
            'cvar_95': cvar
        }
        
        if benchmark_weights is not None:
            metrics['tracking_error'] = self.calculate_tracking_error(
                weights, benchmark_weights
            )
            
        factor_metrics = self.calculate_factor_exposures(weights)
        if factor_metrics:
            metrics.update({
                'factor_exposures': factor_metrics['exposures'].tolist(),
                'factor_contribution': factor_metrics['contribution'].tolist(),
                'total_factor_risk': factor_metrics['total_factor_risk']
            })
            
        if volumes is not None and spreads is not None:
            liquidity_metrics = self.calculate_liquidity_score(weights, volumes, spreads)
            metrics.update(liquidity_metrics)
            
        return metrics

