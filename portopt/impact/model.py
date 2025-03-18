"""
Market impact and transaction cost modeling module.

This module provides classes and functions for estimating the market impact and
transaction costs associated with portfolio trades. It implements sophisticated
models that account for:

- Permanent and temporary price impacts
- Bid-ask spread costs
- Volume-dependent effects
- Price volatility scaling
- Impact decay over time
- Participation rate limits

The implementation is based on academic research and industry best practices
for transaction cost analysis and optimal trade execution.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class MarketImpactParams:
    """Parameters for market impact and transaction cost modeling.
    
    This class encapsulates all parameters needed for market impact estimation:
    - Core impact parameters
    - Volatility scaling factors
    - Spread-related parameters
    - Capacity constraints
    - Time decay parameters
    - Market condition adjustments
    
    The parameters are based on academic research and empirical studies of 
    market impact. Default values are set based on typical market behavior
    but can be adjusted for different market conditions or asset classes.
    
    Typical usage:
        >>> params = MarketImpactParams()  # Use defaults
        >>> params = MarketImpactParams.high_urgency()  # Preset for urgent trading
        >>> params = MarketImpactParams(permanent_impact=0.15)  # Custom parameters
    
    Attributes:
        permanent_impact (float): Coefficient for permanent price impact (default: 0.1)
            - Measures lasting effect of trades on market prices
            - Typically ranges from 0.05 to 0.2
            - Higher values for less liquid markets
        
        temporary_impact (float): Coefficient for temporary price impact (default: 0.2)
            - Measures transient price effects during trading
            - Usually larger than permanent impact
            - Decays according to decay_rate
        
        decay_rate (float): Rate of temporary impact decay (default: 0.85)
            - Controls how quickly temporary impact dissipates
            - 1.0 means no decay, 0.0 means immediate decay
            - Typically between 0.7 and 0.9
        
        participation_limit (float): Maximum market participation rate (default: 0.3)
            - Upper limit on trading as fraction of volume
            - Helps prevent excessive market impact
            - Usually between 0.1 and 0.4
        
        volatility_factor (float): Volatility scaling factor (default: 0.1)
            - Adjusts impact based on asset volatility
            - Higher values mean stronger volatility effect
        
        min_volatility (float): Minimum volatility floor (default: 0.001)
            - Prevents near-zero volatility issues
            - Used in volatility scaling calculations
        
        spread_factor (float): Spread cost scaling factor (default: 1.0)
            - Multiplier for bid-ask spread costs
            - Adjust based on market conditions
        
        spread_power (float): Spread cost nonlinearity (default: 0.5)
            - Controls how spread cost scales with size
            - 0.5 corresponds to square root model
        
        min_spread (float): Minimum spread floor (default: 0.0001)
            - Prevents numerical issues with tiny spreads
            - Sets lower bound for spread calculations
        
        max_pct_adv (float): Maximum percentage of ADV (default: 0.3)
            - Upper limit on daily volume participation
            - Helps ensure realistic trading capacity
        
        max_pct_spread (float): Maximum spread widening (default: 0.25)
            - Limits impact on bid-ask spreads
            - Represents maximum spread increase
        
        time_decay_factor (float): Time decay exponent (default: 0.5)
            - Controls impact decay over time
            - Used in multi-day trading schedules
        
        min_time_window (int): Minimum trading window in days (default: 1)
            - Shortest allowed trading period
            - Used in trading schedule optimization
        
        max_time_window (int): Maximum trading window in days (default: 20)
            - Longest allowed trading period
            - Limits trading schedule length
        
        volatility_adjustment (bool): Whether to adjust for volatility (default: True)
            - Enables volatility-based impact scaling
            - Can be disabled for simpler models
        
        spread_adjustment (bool): Whether to adjust for spreads (default: True)
            - Enables spread-based cost adjustments
            - Can be disabled for simpler models
        
        volume_adjustment (bool): Whether to adjust for volume (default: True)
            - Enables volume-based impact scaling
            - Can be disabled for simpler models
    """
    
    # Core impact parameters
    permanent_impact: float = 0.1    
    temporary_impact: float = 0.2    
    decay_rate: float = 0.85        
    participation_limit: float = 0.3 
    
    # Volatility scaling
    volatility_factor: float = 0.1   
    min_volatility: float = 0.001    
    
    # Spread-related parameters
    spread_factor: float = 1.0       
    spread_power: float = 0.5        
    min_spread: float = 0.0001       
    
    # Capacity constraints
    max_pct_adv: float = 0.3         
    max_pct_spread: float = 0.25     
    
    # Time decay parameters
    time_decay_factor: float = 0.5   
    min_time_window: int = 1         
    max_time_window: int = 20        
    
    # Market condition adjustments
    volatility_adjustment: bool = True
    spread_adjustment: bool = True    
    volume_adjustment: bool = True    
    
    def __post_init__(self):
        """Validate parameters after initialization.
        
        Raises:
            ValueError: If parameters are outside valid ranges
        """
        if not 0 <= self.permanent_impact <= 1:
            raise ValueError("Permanent impact must be between 0 and 1")
        if not 0 <= self.temporary_impact <= 1:
            raise ValueError("Temporary impact must be between 0 and 1")
        if not 0 <= self.decay_rate <= 1:
            raise ValueError("Decay rate must be between 0 and 1")
        if not 0 < self.participation_limit <= 1:
            raise ValueError("Participation limit must be between 0 and 1")
            
    @classmethod
    def high_urgency(cls) -> "MarketImpactParams":
        """Create parameters optimized for high-urgency trading.
        
        Returns parameters configured for scenarios requiring rapid execution:
        - Higher impact coefficients
        - Faster decay
        - Higher participation limits
        - Shorter trading windows
        
        Returns:
            MarketImpactParams with settings for urgent trading
        """
        return cls(
            permanent_impact=0.15,     # Higher permanent impact
            temporary_impact=0.3,      # Higher temporary impact
            decay_rate=0.9,           # Faster decay
            participation_limit=0.4,   # Higher participation
            time_decay_factor=0.3,    # Faster time decay
            max_time_window=5         # Shorter trading window
        )
    
    @classmethod
    def low_urgency(cls) -> "MarketImpactParams":
        """Create parameters optimized for low-urgency trading.
        
        Returns parameters configured for patient execution:
        - Lower impact coefficients
        - Slower decay
        - Lower participation limits
        - Longer trading windows
        
        Returns:
            MarketImpactParams with settings for patient trading
        """
        return cls(
            permanent_impact=0.08,     # Lower permanent impact
            temporary_impact=0.15,     # Lower temporary impact
            decay_rate=0.8,           # Slower decay
            participation_limit=0.2,   # Lower participation
            time_decay_factor=0.7,    # Slower time decay
            max_time_window=30        # Longer trading window
        )
    
    def adjust_for_market_cap(self, market_cap: float) -> "MarketImpactParams":
        """Adjust parameters based on market capitalization.
        
        Scales impact parameters based on asset size:
        - Large cap: Lower impact, higher participation
        - Mid cap: Standard parameters
        - Small cap: Higher impact, lower participation
        
        Args:
            market_cap: Market capitalization in standard currency units
            
        Returns:
            New MarketImpactParams adjusted for market cap
        """
        if market_cap > 10e9:  # Large cap
            factor = 0.8
        elif market_cap > 2e9:  # Mid cap
            factor = 1.0
        else:  # Small cap
            factor = 1.2
            
        return MarketImpactParams(
            permanent_impact=self.permanent_impact * factor,
            temporary_impact=self.temporary_impact * factor,
            decay_rate=self.decay_rate,
            participation_limit=self.participation_limit * (1/factor),
            volatility_factor=self.volatility_factor * factor,
            spread_factor=self.spread_factor * factor
        )
    
    def adjust_for_volatility(self, volatility: float) -> "MarketImpactParams":
        """Adjust parameters based on market volatility.
        
        Scales impact parameters based on asset volatility:
        - Higher volatility: Higher impact, lower participation
        - Lower volatility: Lower impact, higher participation
        
        Args:
            volatility: Asset or market volatility (annualized)
            
        Returns:
            New MarketImpactParams adjusted for volatility
        """
        if not self.volatility_adjustment:
            return self
            
        vol_ratio = volatility / 0.2  # Compare to baseline 20% volatility
        
        return MarketImpactParams(
            permanent_impact=self.permanent_impact * vol_ratio**0.5,
            temporary_impact=self.temporary_impact * vol_ratio**0.5,
            decay_rate=self.decay_rate,
            participation_limit=self.participation_limit * (1/vol_ratio**0.5),
            volatility_factor=self.volatility_factor,
            spread_factor=self.spread_factor * vol_ratio**0.3
        )
    
    def to_dict(self) -> dict:
        """Convert parameters to dictionary format.
        
        Useful for:
        - Serialization
        - Configuration storage
        - Parameter comparison
        
        Returns:
            Dictionary containing all parameter values
        """
        return {
            'permanent_impact': self.permanent_impact,
            'temporary_impact': self.temporary_impact,
            'decay_rate': self.decay_rate,
            'participation_limit': self.participation_limit,
            'volatility_factor': self.volatility_factor,
            'spread_factor': self.spread_factor,
            'time_decay_factor': self.time_decay_factor,
            'max_time_window': self.max_time_window
        }
    
class MarketImpactModel:
    """Models market impact and transaction costs for portfolio optimization.
    
    This class implements a sophisticated market impact model that accounts for:
    - Permanent market impact (lasting price changes)
    - Temporary market impact (transient effects)
    - Bid-ask spread costs
    - Volume-dependent effects
    - Price volatility scaling
    - Impact decay over time
    - Participation rate limits
    
    The model is based on academic research and industry practice, using:
    - Square-root price impact model for market impact
    - Linear model for spread costs
    - Exponential decay for temporary impact
    - Volume-weighted price adjustments
    
    Typical usage:
        >>> model = MarketImpactModel(volumes, spreads, volatility)
        >>> costs = model.estimate_total_costs(
        ...     weights=target_weights,
        ...     prev_weights=current_weights
        ... )
    """
    
    def __init__(self, volumes: np.ndarray, spreads: np.ndarray, 
                 volatility: np.ndarray, params: Optional[MarketImpactParams] = None):
        """Initialize market impact model.
        
        Args:
            volumes: Historical trading volumes (n_assets x n_periods)
            spreads: Historical bid-ask spreads (n_assets x n_periods)
            volatility: Asset volatilities (n_assets)
            params: Optional market impact parameters. If not provided, uses defaults
        """
        self.volumes = volumes
        self.spreads = spreads
        self.volatility = volatility
        self.params = params or MarketImpactParams()
        
    def estimate_spread_costs(self, weights: np.ndarray, 
                            prev_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Estimate trading costs from bid-ask spreads.
        
        Calculation process:
        1. Determine trading values from weight changes
        2. Apply half-spread costs
        3. Adjust for size-based spread widening
        4. Account for market impact on spreads
        
        Args:
            weights: Target portfolio weights
            prev_weights: Previous portfolio weights (optional)
            
        Returns:
            Dictionary containing:
            - total_spread_cost: Total cost from spreads
            - avg_spread_cost: Average cost per trade
            - max_spread_cost: Maximum cost for any trade
            - size_adjustments: Size-based spread adjustments
        """
        # Calculate trading values
        if prev_weights is None:
            trading_value = np.abs(weights)
        else:
            trading_value = np.abs(weights - prev_weights)
            
        # Calculate base spread costs
        half_spreads = self.spreads.mean(axis=1) / 2
        spread_costs = trading_value * half_spreads * self.params.spread_factor
        
        # Adjust for trade size impact on spreads
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
        """Estimate market impact costs using square-root model with decay.
        
        Implementation features:
        1. Square-root price impact model
        2. Separate permanent and temporary effects
        3. Time decay of impact
        4. Volume-based participation limits
        
        Args:
            weights: Target portfolio weights
            prev_weights: Previous portfolio weights (optional)
            trade_duration: Number of days to execute trades
            
        Returns:
            Dictionary containing:
            - total_impact: Total price impact cost
            - permanent_impact: Lasting price effects
            - temporary_impact: Transient price effects
            - max_participation: Maximum daily participation rate
            - avg_participation: Average daily participation rate
            - participation_violations: Count of participation limit violations
        """
        # Calculate required trades
        if prev_weights is None:
            trades = np.abs(weights)
        else:
            trades = np.abs(weights - prev_weights)
            
        # Calculate average daily volume and participation
        adv = np.mean(self.volumes, axis=1)
        participation = trades / (adv * trade_duration)
        
        # Check and adjust for participation limits
        violations = participation > self.params.participation_limit
        if np.any(violations):
            participation[violations] = self.params.participation_limit
            trades[violations] = self.params.participation_limit * adv[violations] * trade_duration
        
        # Calculate permanent impact using square root model
        permanent_impact = (
            self.params.permanent_impact * 
            self.volatility * 
            np.sqrt(participation) * 
            trades
        )
        
        # Calculate temporary impact with decay
        temporary_impact = np.zeros_like(trades)
        remaining_trades = trades.copy()
        
        for day in range(trade_duration):
            # Calculate daily trading amount
            daily_trade = remaining_trades / (trade_duration - day)
            daily_participation = daily_trade / adv
            
            # Calculate daily impact
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
        """Calculate the optimal liquidation horizon to minimize impact.
        
        This method:
        1. Calculates minimum days based on participation limits
        2. Calculates days needed based on price impact threshold
        3. Determines optimal trading horizon considering both constraints
        
        Args:
            weights: Portfolio weights to liquidate
            max_price_impact: Maximum acceptable price impact (default 2%)
            
        Returns:
            Dictionary containing:
            - max_days: Maximum days needed for any position
            - avg_days: Average days needed across positions
            - total_days: Sum of days needed for all positions
            - impact_constrained: Number of positions constrained by impact
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
        """Estimate total trading costs including spread and impact.
        
        This method:
        1. Determines optimal trading horizon if not specified
        2. Calculates spread costs
        3. Estimates market impact
        4. Combines all cost components
        
        Args:
            weights: Target portfolio weights
            prev_weights: Previous portfolio weights (optional)
            trade_duration: Number of days to execute trades (optional)
            
        Returns:
            Dictionary containing:
            - total_cost: Combined trading costs
            - spread_costs: Detailed spread cost breakdown
            - impact_costs: Detailed impact cost breakdown
            - trade_duration: Trading horizon used
            - cost_breakdown: Relative contribution of each component
        """
        # Calculate optimal trade duration if not provided
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