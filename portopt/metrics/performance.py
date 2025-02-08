"""Performance metrics calculation module."""

import numpy as np
from typing import Dict, Optional, List
from scipy import stats

class PerformanceMetrics:
    """Calculates portfolio performance metrics."""
    
    def __init__(self, returns: np.ndarray, risk_free_rate: float = 0.02):
        """
        Initialize performance metrics calculator.
        
        Args:
            returns: Asset returns matrix (n_assets x n_periods)
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.returns = returns
        self.rf_daily = (1 + risk_free_rate) ** (1/252) - 1
        self.n_assets, self.n_periods = returns.shape
        
    def calculate_metrics(self, weights: np.ndarray,
                         benchmark_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            weights: Portfolio weights
            benchmark_weights: Optional benchmark portfolio weights
            
        Returns:
            Dictionary of performance metrics
        """
        # Calculate portfolio returns
        portfolio_returns = self.returns.T @ weights
        
        # Basic metrics
        metrics = {
            'total_return': self._calculate_total_return(portfolio_returns),
            'annualized_return': self._calculate_annualized_return(portfolio_returns),
            'volatility': self._calculate_volatility(portfolio_returns),
            'sharpe_ratio': self._calculate_sharpe_ratio(portfolio_returns),
            'sortino_ratio': self._calculate_sortino_ratio(portfolio_returns),
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'skewness': float(stats.skew(portfolio_returns)),
            'kurtosis': float(stats.kurtosis(portfolio_returns))
        }
        
        # Add benchmark-relative metrics if benchmark provided
        if benchmark_weights is not None:
            benchmark_returns = self.returns.T @ benchmark_weights
            metrics.update(self._calculate_relative_metrics(
                portfolio_returns, benchmark_returns
            ))
            
        return metrics
    
    def _calculate_total_return(self, returns: np.ndarray) -> float:
        """Calculate total return."""
        return float(np.prod(1 + returns) - 1)
    
    def _calculate_annualized_return(self, returns: np.ndarray) -> float:
        """Calculate annualized return."""
        total_return = self._calculate_total_return(returns)
        years = self.n_periods / 252
        return float((1 + total_return) ** (1/years) - 1)
    
    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility."""
        return float(np.std(returns) * np.sqrt(252))
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - self.rf_daily
        if np.std(excess_returns) == 0:
            return 0.0
        return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - self.rf_daily
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        return float(np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252))
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative / running_max - 1
        return float(np.min(drawdowns))
    
    def _calculate_relative_metrics(self, portfolio_returns: np.ndarray,
                                  benchmark_returns: np.ndarray) -> Dict[str, float]:
        """Calculate benchmark-relative metrics."""
        # Tracking error
        tracking_diff = portfolio_returns - benchmark_returns
        tracking_error = np.std(tracking_diff) * np.sqrt(252)
        
        # Information ratio
        if tracking_error == 0:
            information_ratio = 0.0
        else:
            information_ratio = np.mean(tracking_diff) / np.std(tracking_diff) * np.sqrt(252)
        
        # Beta
        portfolio_excess = portfolio_returns - self.rf_daily
        benchmark_excess = benchmark_returns - self.rf_daily
        beta = np.cov(portfolio_excess, benchmark_excess)[0,1] / np.var(benchmark_excess)
        
        # Alpha (annualized)
        alpha = (np.mean(portfolio_excess) - beta * np.mean(benchmark_excess)) * 252
        
        return {
            'tracking_error': float(tracking_error),
            'information_ratio': float(information_ratio),
            'beta': float(beta),
            'alpha': float(alpha),
            'active_return': float(np.mean(tracking_diff) * 252),
            'r_squared': float(np.corrcoef(portfolio_returns, benchmark_returns)[0,1] ** 2)
        }
        
    def calculate_rolling_metrics(self, weights: np.ndarray,
                                window: int = 63) -> Dict[str, List[float]]:
        """
        Calculate rolling performance metrics.
        
        Args:
            weights: Portfolio weights
            window: Rolling window size in days (default 63 = quarterly)
            
        Returns:
            Dictionary of rolling metrics
        """
        portfolio_returns = self.returns.T @ weights
        rolling_metrics = {}
        
        for i in range(window, len(portfolio_returns)):
            window_returns = portfolio_returns[i-window:i]
            
            metrics = {
                'rolling_return': self._calculate_total_return(window_returns),
                'rolling_vol': self._calculate_volatility(window_returns),
                'rolling_sharpe': self._calculate_sharpe_ratio(window_returns)
            }
            
            for key, value in metrics.items():
                if key not in rolling_metrics:
                    rolling_metrics[key] = []
                rolling_metrics[key].append(value)
                
        return rolling_metrics

