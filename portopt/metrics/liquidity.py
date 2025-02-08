"""Liquidity metrics calculation module."""

import numpy as np
from typing import Dict, Optional, List
from scipy import stats

class LiquidityMetrics:
    """Advanced liquidity analysis for portfolio optimization.
    
    This class provides comprehensive liquidity analysis including:
    - Trading volume analysis
    - Bid-ask spread impact
    - Market impact estimation
    - Liquidation analysis
    - Multi-day trading scheduling
    
    The implementation considers multiple liquidity dimensions:
    1. Volume-based measures (ADV participation)
    2. Cost-based measures (spread costs)
    3. Price impact sensitivity
    4. Time to liquidation
    5. Trading capacity constraints
    
    Typical usage:
        >>> metrics = LiquidityMetrics(volumes, prices, spreads, returns)
        >>> liquidity_analysis = metrics.calculate_metrics(
        ...     weights=portfolio_weights,
        ...     trade_horizon=5
        ... )
    """
    
    def __init__(self, volumes: np.ndarray, prices: np.ndarray, 
                 spreads: np.ndarray, returns: np.ndarray):
        """Initialize liquidity metrics calculator.
        
        Args:
            volumes: Trading volumes (n_assets x n_periods)
            prices: Asset prices (n_assets x n_periods)
            spreads: Bid-ask spreads (n_assets x n_periods)
            returns: Asset returns (n_assets x n_periods)
            
        All input arrays should have matching dimensions where applicable
        """
        self.volumes = volumes
        self.prices = prices
        self.spreads = spreads
        self.returns = returns
        self.n_assets, self.n_periods = volumes.shape
        
    def calculate_metrics(self, weights: np.ndarray,
                         trade_horizon: int = 5) -> Dict[str, float]:
        """Calculate comprehensive liquidity metrics.
        
        Computes multiple liquidity measures:
        1. Basic liquidity metrics (volume, spread)
        2. Advanced metrics (price impact, stability)
        3. Trading capacity analysis
        
        Args:
            weights: Portfolio weights
            trade_horizon: Trading horizon in days for liquidation analysis
            
        Returns:
            Dictionary containing all liquidity metrics
        """
        metrics = {}
        
        # Calculate basic liquidity metrics
        metrics.update(self._calculate_basic_metrics(weights))
        
        # Calculate advanced liquidity metrics
        metrics.update(self._calculate_advanced_metrics(weights))
        
        # Calculate liquidation metrics
        metrics.update(self._calculate_liquidation_metrics(weights, trade_horizon))
        
        return metrics
    
    def _calculate_basic_metrics(self, weights: np.ndarray) -> Dict[str, float]:
        """Calculate basic liquidity metrics.
        
        Computes fundamental liquidity measures:
        1. ADV participation rates
        2. Spread costs
        3. Basic capacity measures
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Dictionary of basic liquidity metrics
        """
        # Calculate average daily volume participation
        avg_volumes = np.mean(self.volumes, axis=1)
        participation_rates = weights / avg_volumes
        
        # Calculate spread costs
        avg_spreads = np.mean(self.spreads, axis=1)
        spread_costs = weights * avg_spreads
        
        return {
            'avg_participation': float(np.mean(participation_rates)),
            'max_participation': float(np.max(participation_rates)),
            'avg_spread_cost': float(np.mean(spread_costs)),
            'max_spread_cost': float(np.max(spread_costs))
        }
    
    def _calculate_advanced_metrics(self, weights: np.ndarray) -> Dict[str, float]:
        """Calculate advanced liquidity metrics.
        
        Computes sophisticated liquidity measures:
        1. Amihud illiquidity ratio
        2. Relative spread score
        3. Volume stability measures
        4. Price impact sensitivity
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Dictionary of advanced liquidity metrics
        """
        # Calculate Amihud illiquidity ratio
        daily_dollar_volume = self.volumes * self.prices
        amihud_ratio = np.abs(self.returns) / daily_dollar_volume
        portfolio_amihud = np.sum(weights.reshape(-1, 1) * amihud_ratio, axis=0)
        
        # Calculate relative spread score
        spread_score = np.sum(weights * np.mean(self.spreads, axis=1))
        
        # Calculate volume stability
        volume_volatility = np.std(self.volumes, axis=1) / np.mean(self.volumes, axis=1)
        portfolio_vol_stability = np.sum(weights * volume_volatility)
        
        return {
            'amihud_ratio': float(np.mean(portfolio_amihud)),
            'spread_score': float(spread_score),
            'volume_stability': float(portfolio_vol_stability)
        }
    
    def _calculate_liquidation_metrics(self, weights: np.ndarray,
                                    trade_horizon: int) -> Dict[str, float]:
        """Calculate liquidation-based metrics.
        
        Analyzes portfolio liquidation characteristics:
        1. Days to liquidate positions
        2. Liquidation cost estimates
        3. Trading capacity analysis
        4. Impact cost projections
        
        Args:
            weights: Portfolio weights
            trade_horizon: Trading horizon in days
            
        Returns:
            Dictionary of liquidation-based metrics
        """
        # Calculate daily liquidation capacity
        avg_daily_volume = np.mean(self.volumes, axis=1)
        position_sizes = weights * np.mean(self.prices, axis=1)
        
        # Maximum participation of 20% of daily volume
        max_daily_trade = 0.20 * avg_daily_volume * np.mean(self.prices, axis=1)
        
        # Calculate days to liquidate
        days_to_liquidate = position_sizes / max_daily_trade
        
        # Calculate liquidation cost estimate using square root model
        price_impact = 0.1 * np.sqrt(position_sizes / (avg_daily_volume * trade_horizon))
        spread_impact = np.mean(self.spreads, axis=1) / 2
        total_cost = price_impact + spread_impact
        
        return {
            'wtd_avg_liquidation_days': float(np.sum(weights * days_to_liquidate)),
            'max_liquidation_days': float(np.max(days_to_liquidate)),
            'pct_liquid_1day': float(np.sum(weights[days_to_liquidate <= 1])),
            'estimated_liquidation_cost': float(np.sum(weights * total_cost))
        }
    
    def calculate_liquidity_buckets(self, weights: np.ndarray) -> Dict[str, float]:
        """Categorize portfolio into liquidity buckets.
        
        Segments portfolio by liquidity levels:
        1. Highly liquid (daily volume > $100M)
        2. Moderately liquid ($10M-$100M)
        3. Less liquid ($1M-$10M)
        4. Illiquid (< $1M)
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Dictionary with allocation to liquidity buckets
        """
        # Calculate average daily volume in dollars
        avg_daily_volume = np.mean(self.volumes * self.prices, axis=1)
        
        # Define bucket thresholds (in millions)
        thresholds = [1, 5, 10, 50, 100]
        buckets = {}
        
        for i, threshold in enumerate(thresholds):
            if i == 0:
                mask = avg_daily_volume < threshold * 1e6
                bucket_name = f"<{threshold}M"
            else:
                mask = (avg_daily_volume >= thresholds[i-1] * 1e6) & \
                       (avg_daily_volume < threshold * 1e6)
                bucket_name = f"{thresholds[i-1]}M-{threshold}M"
            
            buckets[bucket_name] = float(np.sum(weights[mask]))
            
        # Add final bucket
        mask = avg_daily_volume >= thresholds[-1] * 1e6
        buckets[f">{thresholds[-1]}M"] = float(np.sum(weights[mask]))
        
        return buckets
    
    def calculate_time_to_liquidate(self, weights: np.ndarray,
                                  participation_rate: float = 0.20,
                                  max_price_impact: float = 0.02) -> Dict[str, np.ndarray]:
        """Calculate optimal liquidation timeline.
        
        Creates liquidation schedule considering:
        1. Volume participation constraints
        2. Price impact limits
        3. Market liquidity profile
        4. Trading costs
        
        Args:
            weights: Portfolio weights
            participation_rate: Maximum daily participation rate
            max_price_impact: Maximum acceptable price impact
            
        Returns:
            Dictionary containing:
            - schedule: Daily liquidation amounts
            - remaining: Remaining position after schedule
            - total_days: Days needed for liquidation
        """
        position_values = weights * np.mean(self.prices, axis=1)
        avg_daily_volume = np.mean(self.volumes * self.prices, axis=1)
        
        # Initialize liquidation schedule
        max_days = 20  # Maximum days to consider
        liquidation_schedule = np.zeros((self.n_assets, max_days))
        remaining_position = position_values.copy()
        
        for day in range(max_days):
            # Calculate today's maximum trading capacity
            max_trade = np.minimum(
                participation_rate * avg_daily_volume,
                remaining_position
            )
            
            # Calculate price impact
            price_impact = 0.1 * np.sqrt(max_trade / avg_daily_volume)
            
            # Reduce trade size if price impact exceeds threshold
            max_trade = np.where(
                price_impact > max_price_impact,
                max_price_impact * avg_daily_volume / 0.1,
                max_trade
            )
            
            # Record today's trades
            liquidation_schedule[:, day] = max_trade
            remaining_position -= max_trade
            
            # Stop if fully liquidated
            if np.sum(remaining_position) < 1e-6:
                break
                
        return {
            'schedule': liquidation_schedule,
            'remaining': remaining_position,
            'total_days': day + 1
        }
