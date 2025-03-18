"""Market impact model parameters."""

from dataclasses import dataclass
from typing import Optional

@dataclass
class MarketImpactParams:
    """Parameters for market impact model."""
    # Core impact parameters
    permanent_impact: float = 0.1    # Permanent impact coefficient
    temporary_impact: float = 0.2    # Temporary impact coefficient
    decay_rate: float = 0.85        # Impact decay rate
    participation_limit: float = 0.3 # Maximum participation rate
    
    # Volatility scaling
    volatility_factor: float = 0.1   # Volatility scaling factor
    min_volatility: float = 0.001    # Minimum volatility floor
    
    # Spread-related parameters
    spread_factor: float = 1.0       # Base spread cost scaling
    spread_power: float = 0.5        # Spread cost nonlinearity
    min_spread: float = 0.0001       # Minimum spread floor
    
    # Capacity constraints
    max_pct_adv: float = 0.3         # Maximum % of ADV per day
    max_pct_spread: float = 0.25     # Maximum % of spread widening
    
    # Time decay parameters
    time_decay_factor: float = 0.5   # Time decay exponent
    min_time_window: int = 1         # Minimum trading window (days)
    max_time_window: int = 20        # Maximum trading window (days)
    
    # Market condition adjustments
    volatility_adjustment: bool = True  # Whether to adjust for volatility
    spread_adjustment: bool = True      # Whether to adjust for spread changes
    volume_adjustment: bool = True      # Whether to adjust for volume changes
    
    def __post_init__(self):
        """Validate parameters after initialization."""
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
        """Create parameters for high-urgency trading."""
        return cls(
            permanent_impact=0.15,
            temporary_impact=0.3,
            decay_rate=0.9,
            participation_limit=0.4,
            time_decay_factor=0.3,
            max_time_window=5
        )
    
    @classmethod
    def low_urgency(cls) -> "MarketImpactParams":
        """Create parameters for low-urgency trading."""
        return cls(
            permanent_impact=0.08,
            temporary_impact=0.15,
            decay_rate=0.8,
            participation_limit=0.2,
            time_decay_factor=0.7,
            max_time_window=30
        )
    
    def adjust_for_market_cap(self, market_cap: float) -> 'MarketImpactParams':
        """Adjust parameters based on market capitalization."""
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
    
    def adjust_for_volatility(self, volatility: float) -> 'MarketImpactParams':
        """Adjust parameters based on market volatility."""
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
        """Convert parameters to dictionary."""
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
