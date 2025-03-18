"""
Portfolio constraint definitions and validation module.

This module defines the data structures and classes for representing and validating
portfolio constraints. It includes:
- Industry classification structures
- Asset class definitions
- Currency exposure tracking
- Credit profile representation
- Comprehensive portfolio constraint validation

These components enable the specification and enforcement of complex constraints
in portfolio optimization problems, including sector limits, asset class diversification,
currency exposure management, and credit quality requirements.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Set
import numpy as np

@dataclass
class IndustryClassification:
    """Multi-level industry classification (e.g., GICS)."""
    sector: str          # Level 1 (e.g., Information Technology)
    industry_group: str  # Level 2 (e.g., Software & Services)
    industry: str        # Level 3 (e.g., Software)
    sub_industry: str    # Level 4 (e.g., Application Software)

@dataclass
class AssetClass:
    """Asset class classification with sub-types."""
    primary: str        # e.g., Equity, Fixed Income, Alternative
    sub_type: str      # e.g., Large Cap, Government Bond, Real Estate
    region: str        # e.g., North America, Europe, Asia
    style: Optional[str] = None  # e.g., Value, Growth, Blend

@dataclass
class CurrencyExposure:
    """Currency exposure information."""
    direct: str        # Direct currency of the asset
    effective: Dict[str, float]  # Dictionary of currency: exposure pairs for multinational companies

@dataclass
class CreditProfile:
    """Credit rating and related information."""
    rating: str        # Standard rating (e.g., AAA, BBB+)
    agency: str        # Rating agency (e.g., S&P, Moody's)
    outlook: str       # Rating outlook
    watch: bool       # Whether on credit watch
    
class PortfolioConstraints:
    """Enhanced portfolio constraints handler."""
    
    def __init__(self):
        # Basic constraints
        self.min_weight: float = 0.0
        self.max_weight: float = 1.0
        self.min_stocks: int = 0
        self.max_stocks: Optional[int] = None
        
        # Industry constraints
        self.sector_limits: Dict[str, float] = {}
        self.industry_group_limits: Dict[str, float] = {}
        self.industry_limits: Dict[str, float] = {}
        self.sub_industry_limits: Dict[str, float] = {}
        
        # Asset class constraints
        self.asset_class_limits: Dict[str, float] = {}
        self.asset_subtype_limits: Dict[str, float] = {}
        self.region_limits: Dict[str, float] = {}
        self.style_limits: Dict[str, float] = {}
        
        # Currency constraints
        self.currency_limits: Dict[str, float] = {}
        self.base_currency: str = 'USD'
        self.max_unhedged_exposure: float = 1.0
        
        # Credit constraints
        self.min_rating: str = 'C'
        self.rating_limits: Dict[str, float] = {}
        self.max_watch_list_exposure: float = 1.0
        
        # Risk constraints
        self.max_volatility: Optional[float] = None
        self.max_var: Optional[float] = None
        self.max_cvar: Optional[float] = None
        self.tracking_error_limit: Optional[float] = None
        
    def check_industry_constraints(self, 
                                 weights: np.ndarray,
                                 classifications: List[IndustryClassification]) -> Dict[str, bool]:
        """Check all industry-related constraints."""
        results = {}
        
        # Check sector constraints
        sector_exposures = {}
        for sector in set(c.sector for c in classifications):
            mask = [c.sector == sector for c in classifications]
            exposure = np.sum(weights[mask])
            limit = self.sector_limits.get(sector, 1.0)
            results[f'sector_{sector}'] = exposure <= limit
            sector_exposures[sector] = exposure
            
        # Similar checks for industry_group, industry, and sub_industry...
        return results, sector_exposures
    
    def check_asset_class_constraints(self,
                                    weights: np.ndarray,
                                    asset_classes: List[AssetClass]) -> Dict[str, bool]:
        """Check all asset class related constraints."""
        results = {}
        
        # Check primary asset class constraints
        for asset_type in set(a.primary for a in asset_classes):
            mask = [a.primary == asset_type for a in asset_classes]
            exposure = np.sum(weights[mask])
            limit = self.asset_class_limits.get(asset_type, 1.0)
            results[f'asset_class_{asset_type}'] = exposure <= limit
            
        # Similar checks for sub_type, region, and style...
        return results
    
    def check_currency_constraints(self,
                                 weights: np.ndarray,
                                 currencies: List[CurrencyExposure]) -> Dict[str, bool]:
        """Check all currency-related constraints."""
        results = {}
        
        # Calculate direct currency exposures
        direct_exposures = {}
        for curr in set(c.direct for c in currencies):
            mask = [c.direct == curr for c in currencies]
            exposure = np.sum(weights[mask])
            direct_exposures[curr] = exposure
            
        # Calculate effective currency exposures including multinationals
        effective_exposures = direct_exposures.copy()
        for i, curr in enumerate(currencies):
            for eff_curr, exposure in curr.effective.items():
                effective_exposures[eff_curr] = (
                    effective_exposures.get(eff_curr, 0) + 
                    weights[i] * exposure
                )
                
        # Check against limits
        for curr, exposure in effective_exposures.items():
            limit = self.currency_limits.get(curr, 1.0)
            results[f'currency_{curr}'] = exposure <= limit
            
        # Check unhedged exposure
        non_base = sum(exp for curr, exp in effective_exposures.items() 
                      if curr != self.base_currency)
        results['unhedged_exposure'] = non_base <= self.max_unhedged_exposure
        
        return results, effective_exposures
    
    def check_credit_constraints(self,
                               weights: np.ndarray,
                               credit_profiles: List[CreditProfile]) -> Dict[str, bool]:
        """Check all credit rating related constraints."""
        results = {}
        
        # Check rating-based constraints
        rating_exposures = {}
        for rating in set(c.rating for c in credit_profiles):
            mask = [c.rating == rating for c in credit_profiles]
            exposure = np.sum(weights[mask])
            rating_exposures[rating] = exposure
            limit = self.rating_limits.get(rating, 1.0)
            results[f'rating_{rating}'] = exposure <= limit
            
        # Check watch list exposure
        watch_mask = [c.watch for c in credit_profiles]
        watch_exposure = np.sum(weights[watch_mask])
        results['watch_list'] = watch_exposure <= self.max_watch_list_exposure
        
        return results, rating_exposures

    def check_all_constraints(self,
                            weights: np.ndarray,
                            problem_data: Dict) -> Dict[str, bool]:
        """Check all constraints and return detailed results."""
        results = {}
        exposures = {}
        
        # Basic constraints
        results['sum_to_one'] = np.isclose(np.sum(weights), 1.0)
        results['min_weight'] = np.all(weights[weights > 0] >= self.min_weight)
        results['max_weight'] = np.all(weights <= self.max_weight)
        active_positions = np.sum(weights > 0)
        results['min_stocks'] = active_positions >= self.min_stocks
        if self.max_stocks:
            results['max_stocks'] = active_positions <= self.max_stocks
            
        # Industry constraints
        if 'classifications' in problem_data:
            industry_results, sector_exp = self.check_industry_constraints(
                weights, problem_data['classifications']
            )
            results.update(industry_results)
            exposures['sectors'] = sector_exp
            
        # Asset class constraints
        if 'asset_classes' in problem_data:
            asset_results = self.check_asset_class_constraints(
                weights, problem_data['asset_classes']
            )
            results.update(asset_results)
            
        # Currency constraints
        if 'currencies' in problem_data:
            currency_results, curr_exp = self.check_currency_constraints(
                weights, problem_data['currencies']
            )
            results.update(currency_results)
            exposures['currencies'] = curr_exp
            
        # Credit constraints
        if 'credit_profiles' in problem_data:
            credit_results, rating_exp = self.check_credit_constraints(
                weights, problem_data['credit_profiles']
            )
            results.update(credit_results)
            exposures['ratings'] = rating_exp
            
        return results, exposures
