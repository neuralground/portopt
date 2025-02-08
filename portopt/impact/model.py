import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class MarketImpactParams:
    """Parameters for market impact model."""
    permanent_impact: float = 0.1    # Permanent impact coefficient
    temporary_impact: float = 0.2    # Temporary impact coefficient
    decay_rate: float = 0.85        # Impact decay rate
    participation_limit: float = 0.3 # Maximum participation rate
    volatility_factor: float = 0.1  # Volatility scaling factor
    spread_factor: float = 1.0      # Spread cost scaling
    
class MarketImpactModel:
    """Enhanced market impact model with sophisticated cost estimation."""
    
    def __init__(self, volumes: np.ndarray, spreads: np.ndarray, 
                 volatility: np.ndarray, params: Optional[MarketImpactParams] = None):
        """
        Initialize market impact model.
        
        Args:
            volumes: Historical trading volumes (n_assets x n_periods)
            spreads: Historical bid-ask spreads (n_assets x n_periods)
            volatility: Asset volatilities (n_assets)
            params: Market impact model parameters
        """
        self.volumes = volumes
        self.spreads = spreads
        self.volatility = volatility
        self.params = params or MarketImpactParams()
        
    def estimate_spread_costs(self, weights: np.ndarray, 
                            prev_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Estimate trading costs from bid-ask spreads.
        
        Args:
            weights: Target portfolio weights
            prev_weights: Previous portfolio weights (optional)
            
        Returns:
            Dictionary of spread cost metrics
        """
        if prev_weights is None:
            trading_value = np.abs(weights)
        else:
            trading_value = np.abs(weights - prev_weights)
            
        # Half-spread cost for each trade
        half_spreads = self.spreads.mean(axis=1) / 2
        spread_costs = trading_value * half_spreads * self.params.spread_factor
        
        # Adjust for size-based spread widening
        adv = np.mean(self.volumes, axis=1)
        size_adjustment = 1 + np.maximum(0, trading_value / adv - 0.1) * 2
        adjusted_costs = spread_costs * size_adjustment
        
        return {
            'total_spread_cost': float(np.sum(adjusted_costs)),
            'avg_spread_cost': float(np.mean(adjusted_costs)),
            'max_spread_cost': float(np.max(adjusted_costs)),
            'size_adjustments': size_adjustment.tolist()
        }
        
    def estimate_market_impact(self, weights: np.ndarray,
                             prev_weights: Optional[np.ndarray] = None,
                             trade_duration: int = 1) -> Dict[str, float]:
        """
        Estimate market impact costs using square-root model with decay.
        
        Args:
            weights: Target portfolio weights
            prev_weights: Previous portfolio weights (optional)
            trade_duration: Number of days to execute trades
            
        Returns:
            Dictionary of market impact metrics
        """
        if prev_weights is None:
            trades = np.abs(weights)
        else:
            trades = np.abs(weights - prev_weights)
            
        # Average daily volume
        adv = np.mean(self.volumes, axis=1)
        
        # Participation rates
        participation = trades / (adv * trade_duration)
        
        # Check participation limit violations
        violations = participation > self.params.participation_limit
        if np.any(violations):
            participation[violations] = self.params.participation_limit
            trades[violations] = self.params.participation_limit * adv[violations] * trade_duration
        
        # Permanent impact using square root model
        permanent_impact = (
            self.params.permanent_impact * 
            self.volatility * 
            np.sqrt(participation) * 
            trades
        )
        
        # Temporary impact with decay
        temporary_impact = np.zeros_like(trades)
        remaining_trades = trades.copy()
        
        for day in range(trade_duration):
            daily_trade = remaining_trades / (trade_duration - day)
            daily_participation = daily_trade / adv
            
            daily_impact = (
                self.params.temporary_impact * 
                self.volatility * 
                np.sqrt(daily_participation) * 
                daily_trade
            )
            
            # Apply decay to previous day's impact
            if day > 0:
                daily_impact += temporary_impact * self.params.decay_rate
                
            temporary_impact = daily_impact
            remaining_trades -= daily_trade
            
        total_impact = permanent_impact + temporary_impact
        
        return {
            'total_impact': float(np.sum(total_impact)),
            'permanent_impact': float(np.sum(permanent_impact)),
            'temporary_impact': float(np.sum(temporary_impact)),
            'max_participation': float(np.max(participation)),
            'avg_participation': float(np.mean(participation)),
            'participation_violations': int(np.sum(violations))
        }
        
    def calculate_liquidation_horizon(self, weights: np.ndarray,
                                   max_price_impact: float = 0.02) -> Dict[str, float]:
        """
        Calculate the optimal liquidation horizon to minimize impact.
        
        Args:
            weights: Portfolio weights to liquidate
            max_price_impact: Maximum acceptable price impact
            
        Returns:
            Dictionary with liquidation analysis
        """
        adv = np.mean(self.volumes, axis=1)
        
        # Calculate minimum days needed based on participation limit
        min_days = np.ceil(np.abs(weights) / (adv * self.params.participation_limit))
        
        # Calculate days needed based on price impact threshold
        impact_days = np.ceil(
            (weights * self.volatility * self.params.temporary_impact) ** 2 /
            (max_price_impact * adv)
        )
        
        # Take the maximum of participation and impact constraints
        optimal_days = np.maximum(min_days, impact_days)
        
        return {
            'max_days': float(np.max(optimal_days)),
            'avg_days': float(np.mean(optimal_days)),
            'total_days': float(np.sum(optimal_days)),
            'impact_constrained': float(np.sum(impact_days > min_days))
        }
        
    def estimate_total_costs(self, weights: np.ndarray,
                           prev_weights: Optional[np.ndarray] = None,
                           trade_duration: Optional[int] = None) -> Dict[str, float]:
        """
        Estimate total trading costs including spread and impact.
        
        Args:
            weights: Target portfolio weights
            prev_weights: Previous portfolio weights (optional)
            trade_duration: Number of days to execute trades (optional)
            
        Returns:
            Dictionary with complete cost analysis
        """
        # If trade_duration not provided, calculate optimal horizon
        if trade_duration is None:
            liquidation = self.calculate_liquidation_horizon(weights)
            trade_duration = int(np.ceil(liquidation['max_days']))
            
        # Get spread costs
        spread_costs = self.estimate_spread_costs(weights, prev_weights)
        
        # Get impact costs
        impact_costs = self.estimate_market_impact(
            weights, prev_weights, trade_duration
        )
        
        # Combine all metrics
        total_costs = {
            'total_cost': spread_costs['total_spread_cost'] + impact_costs['total_impact'],
            'spread_costs': spread_costs,
            'impact_costs': impact_costs,
            'trade_duration': trade_duration,
            'cost_breakdown': {
                'spread_pct': spread_costs['total_spread_cost'] / 
                    (spread_costs['total_spread_cost'] + impact_costs['total_impact']),
                'impact_pct': impact_costs['total_impact'] / 
                    (spread_costs['total_spread_cost'] + impact_costs['total_impact'])
            }
        }
        
        return total_costs

