"""Performance metrics calculation module."""

import numpy as np
from typing import Dict, Optional, List
from scipy import stats

class PerformanceMetrics:
    """Advanced performance metrics calculator for portfolio analysis.
    
    This class provides comprehensive performance analysis including:
    - Return metrics (total return, annualized return)
    - Risk-adjusted measures (Sharpe ratio, Sortino ratio)
    - Drawdown analysis
    - Rolling metrics calculation
    - Benchmark-relative performance
    
    The implementation supports:
    1. Single portfolio analysis
    2. Benchmark comparison
    3. Rolling window analysis
    4. Return distribution analysis
    5. Risk-adjusted performance measures
    
    Typical usage:
        >>> metrics = PerformanceMetrics(returns, risk_free_rate=0.02)
        >>> perf_analysis = metrics.calculate_metrics(
        ...     weights=portfolio_weights,
        ...     benchmark_weights=benchmark_weights
        ... )
    """
    
    def __init__(self, returns: np.ndarray, risk_free_rate: float = 0.02):
        """Initialize performance metrics calculator.
        
        Args:
            returns: Asset returns matrix (n_assets x n_periods)
            risk_free_rate: Annual risk-free rate (default 2%)
            
        Note:
            Risk-free rate is converted to daily rate assuming 252 trading days
        """
        self.returns = returns
        self.rf_daily = (1 + risk_free_rate) ** (1/252) - 1  # Convert to daily
        self.n_assets, self.n_periods = returns.shape
        
    def calculate_metrics(self, weights: np.ndarray,
                         benchmark_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate comprehensive performance metrics.
        
        Calculates a full suite of performance measures:
        1. Basic return metrics
        2. Risk-adjusted performance
        3. Distribution statistics
        4. Benchmark-relative metrics if benchmark provided
        
        Args:
            weights: Portfolio weights
            benchmark_weights: Optional benchmark portfolio weights
            
        Returns:
            Dictionary containing performance metrics including:
            - total_return: Total portfolio return
            - annualized_return: Annualized return
            - volatility: Annualized volatility
            - sharpe_ratio: Risk-adjusted return
            - sortino_ratio: Downside risk-adjusted return
            - max_drawdown: Maximum portfolio drawdown
            - skewness: Return distribution skewness
            - kurtosis: Return distribution kurtosis
        """
        # Calculate portfolio returns
        portfolio_returns = self.returns.T @ weights
        
        # Calculate basic metrics
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
        """Calculate total compound return.
        
        Args:
            returns: Array of period returns
            
        Returns:
            Total compound return over the period
        """
        return float(np.prod(1 + returns) - 1)
    
    def _calculate_annualized_return(self, returns: np.ndarray) -> float:
        """Calculate annualized return.
        
        Args:
            returns: Array of period returns
            
        Returns:
            Annualized return assuming 252 trading days per year
        """
        total_return = self._calculate_total_return(returns)
        years = self.n_periods / 252
        return float((1 + total_return) ** (1/years) - 1)
    
    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility.
        
        Args:
            returns: Array of period returns
            
        Returns:
            Annualized return volatility
        """
        return float(np.std(returns) * np.sqrt(252))
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio.
        
        Computes the annualized Sharpe ratio:
        excess_return / volatility
        
        Args:
            returns: Array of period returns
            
        Returns:
            Annualized Sharpe ratio
        """
        excess_returns = returns - self.rf_daily
        if np.std(excess_returns) == 0:
            return 0.0
        return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio.
        
        Computes ratio using downside deviation instead of
        total volatility to focus on harmful volatility.
        
        Args:
            returns: Array of period returns
            
        Returns:
            Annualized Sortino ratio
        """
        excess_returns = returns - self.rf_daily
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        return float(np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252))
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown.
        
        Process:
        1. Calculate cumulative returns
        2. Track running maximum
        3. Calculate drawdowns from peaks
        4. Find maximum drawdown
        
        Args:
            returns: Array of period returns
            
        Returns:
            Maximum portfolio drawdown
        """
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative / running_max - 1
        return float(np.min(drawdowns))
    
    def _calculate_relative_metrics(self, portfolio_returns: np.ndarray,
                                  benchmark_returns: np.ndarray) -> Dict[str, float]:
        """Calculate benchmark-relative performance metrics.
        
        Calculates metrics including:
        1. Tracking error
        2. Information ratio
        3. Beta
        4. Alpha
        5. R-squared
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            
        Returns:
            Dictionary of benchmark-relative metrics
        """
        # Calculate tracking error
        tracking_diff = portfolio_returns - benchmark_returns
        tracking_error = np.std(tracking_diff) * np.sqrt(252)
        
        # Calculate information ratio
        if tracking_error == 0:
            information_ratio = 0.0
        else:
            information_ratio = np.mean(tracking_diff) / np.std(tracking_diff) * np.sqrt(252)
        
        # Calculate beta and alpha
        portfolio_excess = portfolio_returns - self.rf_daily
        benchmark_excess = benchmark_returns - self.rf_daily
        beta = np.cov(portfolio_excess, benchmark_excess)[0,1] / np.var(benchmark_excess)
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
        """Calculate rolling performance metrics.
        
        Computes rolling window metrics including:
        1. Returns
        2. Volatility
        3. Sharpe ratio
        
        Args:
            weights: Portfolio weights
            window: Rolling window size in days (default 63 = quarterly)
            
        Returns:
            Dictionary containing lists of rolling metrics
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
