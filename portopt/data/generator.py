"""Enhanced test data generation module."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from scipy import stats
from portopt.core.problem import PortfolioOptProblem
from portopt.constraints.constraint_types import (
    IndustryClassification, AssetClass, CurrencyExposure, CreditProfile
)

@dataclass
class MarketData:
    """Container for generated market data."""
    returns: np.ndarray              # Asset returns (n_assets x n_periods)
    volumes: np.ndarray             # Trading volumes (n_assets x n_periods)
    spreads: np.ndarray            # Bid-ask spreads (n_assets x n_periods)
    factor_returns: np.ndarray     # Factor returns (n_factors x n_periods)
    factor_exposures: np.ndarray   # Factor exposures (n_assets x n_factors)
    classifications: List[IndustryClassification]  # Industry classifications
    asset_classes: List[AssetClass]               # Asset class assignments
    currencies: List[CurrencyExposure]            # Currency exposures
    credit_profiles: List[CreditProfile]          # Credit ratings
    market_caps: np.ndarray        # Market capitalizations

class EnhancedTestDataGenerator:
    """Generates realistic test data for portfolio optimization."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize the generator with optional seed."""
        self.rng = np.random.RandomState(seed)
        self.logger = logging.getLogger(__name__)

        # Define common factors
        self.factors = [
            'Market', 'Size', 'Value', 'Momentum', 'Quality',
            'LowVol', 'Yield', 'Growth', 'Liquidity'
        ]

        # Define sectors and industry hierarchy
        self.industry_hierarchy = {
            'Information Technology': {
                'Software & Services': {
                    'Software': ['Application Software', 'Systems Software'],
                    'IT Services': ['Data Processing', 'IT Consulting']
                },
                'Hardware': {
                    'Components': ['Semiconductors', 'Electronics'],
                    'Equipment': ['PCs', 'Servers', 'Storage']
                }
            },
            'Financials': {
                'Banks': {
                    'Commercial Banks': ['Regional Banks', 'Global Banks'],
                    'Investment Banks': ['Brokers', 'Asset Managers']
                },
                'Insurance': {
                    'Life Insurance': ['Life', 'Health'],
                    'P&C Insurance': ['Commercial', 'Personal']
                }
            },
            'Healthcare': {
                'Equipment': {
                    'Medical Devices': ['Imaging', 'Surgical', 'Monitoring'],
                    'Supplies': ['Consumables', 'PPE']
                },
                'Services': {
                    'Providers': ['Hospitals', 'Clinics'],
                    'Insurance': ['Managed Care', 'Benefits']
                }
            }
        }

        # Define asset class hierarchy
        self.asset_hierarchy = {
            'Equity': {
                'sub_types': ['Large Cap', 'Mid Cap', 'Small Cap'],
                'regions': ['North America', 'Europe', 'Asia Pacific'],
                'styles': ['Value', 'Growth', 'Blend']
            },
            'Fixed Income': {
                'sub_types': ['Government', 'Corporate', 'Municipal'],
                'regions': ['Domestic', 'International', 'Emerging'],
                'styles': ['Investment Grade', 'High Yield']
            },
            'Alternative': {
                'sub_types': ['Real Estate', 'Private Equity', 'Commodities'],
                'regions': ['Global', 'Domestic', 'International'],
                'styles': ['Core', 'Opportunistic']
            }
        }

        # Define currency zones
        self.currency_zones = {
            'USD': ['USD', 'CAD', 'MXN'],
            'EUR': ['EUR', 'GBP', 'CHF'],
            'APAC': ['JPY', 'KRW', 'CNY', 'AUD']
        }

        # Define credit ratings
        self.ratings = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-',
                       'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-',
                       'B+', 'B', 'B-', 'CCC+', 'CCC', 'CCC-', 'CC', 'C']
        self.rating_agencies = ['S&P', 'Moody\'s', 'Fitch']
        self.rating_outlooks = ['Stable', 'Positive', 'Negative']

    def generate_market_data(self, n_assets: int, n_periods: int) -> MarketData:
        """Generate complete set of realistic market data."""
        # Generate factor-based returns
        factor_returns = self.generate_factor_returns(n_periods)
        factor_exposures = self.generate_factor_exposures(n_assets)

        # Generate market caps with realistic distribution
        log_market_caps = stats.norm.rvs(loc=8.0, scale=1.5, size=n_assets,
                                       random_state=self.rng)
        market_caps = np.exp(log_market_caps)  # In millions

        # Generate returns
        specific_vol = 0.20 / np.sqrt(252)  # 20% annualized specific vol
        specific_returns = self.rng.normal(0, specific_vol,
                                         size=(n_assets, n_periods))
        returns = factor_exposures @ factor_returns + specific_returns

        # Generate volumes and spreads
        volumes = self.generate_volumes(n_assets, n_periods, market_caps)
        spreads = self.generate_spreads(n_assets, n_periods, market_caps)

        # Generate classifications and profiles
        classifications = self.generate_classifications(n_assets)
        asset_classes = self.generate_asset_classes(n_assets)
        currencies = self.generate_currency_exposures(n_assets)
        credit_profiles = self.generate_credit_profiles(n_assets)

        return MarketData(
            returns=returns,
            volumes=volumes,
            spreads=spreads,
            factor_returns=factor_returns,
            factor_exposures=factor_exposures,
            classifications=classifications,
            asset_classes=asset_classes,
            currencies=currencies,
            credit_profiles=credit_profiles,
            market_caps=market_caps
        )

    def generate_factor_returns(self, n_periods: int) -> np.ndarray:
        """Generate realistic factor returns with appropriate correlations."""
        n_factors = len(self.factors)

        # Define factor volatilities (annualized)
        factor_vols = {
            'Market': 0.15,    # Market factor has highest vol
            'Size': 0.10,
            'Value': 0.08,
            'Momentum': 0.12,
            'Quality': 0.06,
            'LowVol': 0.05,
            'Yield': 0.07,
            'Growth': 0.09,
            'Liquidity': 0.04
        }

        # Create correlation matrix
        corr = np.eye(n_factors)
        factor_corr = {
            ('Market', 'Size'): 0.3,
            ('Market', 'Value'): -0.2,
            ('Market', 'Momentum'): -0.1,
            ('Market', 'Quality'): -0.3,
            ('Market', 'LowVol'): -0.8,
            ('Size', 'Value'): 0.2,
            ('Quality', 'LowVol'): 0.4,
            ('Growth', 'Quality'): 0.3,
            ('Yield', 'Value'): 0.4
        }

        for i, f1 in enumerate(self.factors):
            for j, f2 in enumerate(self.factors):
                if i != j:
                    corr[i, j] = corr[j, i] = factor_corr.get(
                        (f1, f2), factor_corr.get((f2, f1), 0.0)
                    )

        # Ensure matrix is positive definite
        corr = self._nearest_psd(corr)

        # Convert volatilities to array
        vols = np.array([factor_vols[f] for f in self.factors])

        # Create covariance matrix
        cov = np.diag(vols) @ corr @ np.diag(vols)

        # Generate daily factor returns
        daily_vol = vols / np.sqrt(252)
        returns = self.rng.multivariate_normal(
            mean=daily_vol * 0.1,  # Small positive risk premia
            cov=cov / 252,
            size=n_periods
        ).T

        return returns

    def generate_volumes(self, n_assets: int, n_periods: int,
                        market_caps: np.ndarray) -> np.ndarray:
        """Generate realistic trading volumes."""
        base_volumes = market_caps * 0.004  # 0.4% daily turnover
        volumes = np.zeros((n_assets, n_periods))

        for i in range(n_assets):
            daily_changes = self.rng.lognormal(mean=0.0, sigma=0.5, size=n_periods)
            volumes[i] = base_volumes[i] * daily_changes

        # Add day-of-week effect
        dow_factors = np.array([1.1, 1.0, 0.9, 0.95, 1.05])
        for i in range(n_periods):
            volumes[:, i] *= dow_factors[i % 5]

        return volumes

    def generate_spreads(self, n_assets: int, n_periods: int,
                        market_caps: np.ndarray) -> np.ndarray:
        """Generate realistic bid-ask spreads."""
        base_spreads = 5 / np.sqrt(market_caps)  # Basis points
        base_spreads = np.clip(base_spreads, 0.5, 50.0)
        spreads = np.zeros((n_assets, n_periods))

        for i in range(n_assets):
            daily_changes = self.rng.lognormal(mean=0.0, sigma=0.2, size=n_periods)
            spreads[i] = base_spreads[i] * daily_changes

        return spreads

    def generate_factor_exposures(self, n_assets: int) -> np.ndarray:
        """Generate realistic factor exposures for assets."""
        n_factors = len(self.factors)
        exposures = np.zeros((n_assets, n_factors))

        # Generate market betas with realistic distribution
        market_betas = stats.norm.rvs(loc=1.0, scale=0.3, size=n_assets,
                                    random_state=self.rng)
        exposures[:, 0] = market_betas

        # Generate other factor exposures
        for i in range(1, n_factors):
            exposures[:, i] = stats.norm.rvs(loc=0.0, scale=0.15, size=n_assets,
                                           random_state=self.rng)

        return exposures

    def generate_classifications(self, n_assets: int) -> List[IndustryClassification]:
        """Generate realistic industry classifications."""
        classifications = []

        for _ in range(n_assets):
            sector = self.rng.choice(list(self.industry_hierarchy.keys()))
            industry_groups = self.industry_hierarchy[sector]
            industry_group = self.rng.choice(list(industry_groups.keys()))
            industries = industry_groups[industry_group]
            industry = self.rng.choice(list(industries.keys()))
            sub_industry = self.rng.choice(industries[industry])

            classifications.append(IndustryClassification(
                sector=sector,
                industry_group=industry_group,
                industry=industry,
                sub_industry=sub_industry
            ))

        return classifications

    def generate_asset_classes(self, n_assets: int) -> List[AssetClass]:
        """Generate realistic asset class assignments."""
        asset_classes = []

        for _ in range(n_assets):
            primary = self.rng.choice(list(self.asset_hierarchy.keys()))
            sub_type = self.rng.choice(self.asset_hierarchy[primary]['sub_types'])
            region = self.rng.choice(self.asset_hierarchy[primary]['regions'])
            style = self.rng.choice(self.asset_hierarchy[primary]['styles'])

            asset_classes.append(AssetClass(
                primary=primary,
                sub_type=sub_type,
                region=region,
                style=style
            ))

        return asset_classes

    def generate_currency_exposures(self, n_assets: int) -> List[CurrencyExposure]:
        """Generate realistic currency exposures."""
        exposures = []
        all_currencies = [curr for zone in self.currency_zones.values()
                         for curr in zone]

        for _ in range(n_assets):
            zone = self.rng.choice(list(self.currency_zones.keys()))
            direct = self.rng.choice(self.currency_zones[zone])

            if self.rng.random() < 0.3:  # 30% chance of multinational
                effective = {}
                n_currencies = self.rng.randint(2, 4)
                weights = self.rng.dirichlet(np.ones(n_currencies))
                currencies = self.rng.choice(all_currencies, size=n_currencies,
                                          replace=False)

                for curr, weight in zip(currencies, weights):
                    effective[curr] = weight
            else:
                effective = {direct: 1.0}

            exposures.append(CurrencyExposure(
                direct=direct,
                effective=effective
            ))

        return exposures

    def generate_credit_profiles(self, n_assets: int) -> List[CreditProfile]:
        """Generate realistic credit profiles."""
        profiles = []

        # Generate skewed rating distribution
        rating_weights = stats.beta(a=6, b=4).rvs(len(self.ratings))
        rating_weights = rating_weights / np.sum(rating_weights)

        for _ in range(n_assets):
            rating = self.rng.choice(self.ratings, p=rating_weights)
            agency = self.rng.choice(self.rating_agencies)
            outlook = self.rng.choice(self.rating_outlooks)
            watch = self.rng.random() < 0.05  # 5% chance of being on watch

            profiles.append(CreditProfile(
                rating=rating,
                agency=agency,
                outlook=outlook,
                watch=watch
            ))

        return profiles

    def _nearest_psd(self, A: np.ndarray) -> np.ndarray:
        """Find the nearest positive-definite matrix to input matrix A."""
        B = (A + A.T) / 2
        eigvals, eigvecs = np.linalg.eigh(B)
        eigvals = np.maximum(eigvals, 0)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def create_stress_scenario(self, base_data: MarketData, 
                             scenario_type: str) -> MarketData:
        """Create stress scenario by modifying base market data.
        
        Args:
            base_data: Base market data
            scenario_type: Type of stress scenario ('market_crash', 'liquidity_crisis',
                         'sector_rotation', 'currency_crisis', 'credit_event')
        """
        stress_data = MarketData(
            returns=base_data.returns.copy(),
            volumes=base_data.volumes.copy(),
            spreads=base_data.spreads.copy(),
            factor_returns=base_data.factor_returns.copy(),
            factor_exposures=base_data.factor_exposures.copy(),
            classifications=base_data.classifications.copy(),
            asset_classes=base_data.asset_classes.copy(),
            currencies=base_data.currencies.copy(),
            credit_profiles=base_data.credit_profiles.copy(),
            market_caps=base_data.market_caps.copy()
        )
        
        if scenario_type == "market_crash":
            # Increase volatility and correlations
            stress_data.returns *= 2.0
            stress_data.volumes *= 3.0
            stress_data.spreads *= 5.0
            # Increase factor correlations
            stress_data.factor_returns *= 1.5
            
        elif scenario_type == "liquidity_crisis":
            # Reduce volumes, increase spreads
            stress_data.volumes *= 0.2
            stress_data.spreads *= 10.0
            # Affect small caps more severely
            for i, ac in enumerate(stress_data.asset_classes):
                if ac.sub_type == 'Small Cap':
                    stress_data.volumes[i] *= 0.5
                    stress_data.spreads[i] *= 2.0
            
        elif scenario_type == "sector_rotation":
            # Create sector-specific shocks
            sector_shocks = self.rng.normal(0, 0.02, 
                                          len(set(c.sector for c in stress_data.classifications)))
            sector_map = {s: i for i, s in 
                         enumerate(set(c.sector for c in stress_data.classifications))}
            
            for i, classification in enumerate(stress_data.classifications):
                shock = sector_shocks[sector_map[classification.sector]]
                stress_data.returns[i] += shock
                
        elif scenario_type == "currency_crisis":
            # Simulate currency crisis in a specific region
            crisis_region = self.rng.choice(list(self.currency_zones.keys()))
            affected_currencies = set(self.currency_zones[crisis_region])
            
            # Apply shocks to assets with exposure to affected currencies
            for i, curr_exp in enumerate(stress_data.currencies):
                exposure = sum(weight for curr, weight in curr_exp.effective.items()
                             if curr in affected_currencies)
                if exposure > 0:
                    stress_data.returns[i] *= (1 - exposure * 0.3)  # Up to 30% loss
                    stress_data.volumes[i] *= (1 - exposure * 0.5)  # Up to 50% volume reduction
                    stress_data.spreads[i] *= (1 + exposure * 4)    # Up to 5x spread increase
                    
        elif scenario_type == "credit_event":
            # Simulate credit market stress
            for i, profile in enumerate(stress_data.credit_profiles):
                if profile.rating in ['B+', 'B', 'B-', 'CCC+', 'CCC', 'CCC-', 'CC', 'C']:
                    # High yield stress
                    stress_data.returns[i] *= 0.7  # 30% loss
                    stress_data.volumes[i] *= 0.3  # 70% volume reduction
                    stress_data.spreads[i] *= 8.0  # 8x spread increase
                elif profile.rating.startswith('BB'):
                    # Crossover stress
                    stress_data.returns[i] *= 0.85
                    stress_data.volumes[i] *= 0.5
                    stress_data.spreads[i] *= 4.0
                    
        return stress_data

    def generate_realistic_problem(self,
                                    n_assets: int,
                                    n_periods: int,
                                    include_market_impact: bool = True,
                                    include_factors: bool = True,
                                    include_classifications: bool = True,
                                    seed: Optional[int] = None) -> PortfolioOptProblem:
        """Generate a realistic portfolio optimization problem.
        
        Args:
            n_assets: Number of assets
            n_periods: Number of time periods
            include_market_impact: Whether to include volume and spread data
            include_factors: Whether to include factor model data
            include_classifications: Whether to include classification data
            seed: Random seed
        """
        # Generate full market data
        market_data = self.generate_market_data(n_assets, n_periods)
        
        # Create problem with appropriate data based on flags
        problem_data = {
            'returns': market_data.returns,
            'constraints': {
                'sum_to_one': True,
                'no_short': True
            }
        }
        
        if include_market_impact:
            problem_data.update({
                'volumes': market_data.volumes,
                'spreads': market_data.spreads,
                'market_caps': market_data.market_caps
            })
            
        if include_factors:
            problem_data.update({
                'factor_returns': market_data.factor_returns,
                'factor_exposures': market_data.factor_exposures
            })
            
        if include_classifications:
            problem_data.update({
                'classifications': market_data.classifications,
                'asset_classes': market_data.asset_classes,
                'currencies': market_data.currencies,
                'credit_profiles': market_data.credit_profiles
            })
        
        return PortfolioOptProblem(**problem_data)

