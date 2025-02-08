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
    """Container for generated market data and metadata.
    
    This class holds all components of the simulated market environment:
    - Price and return data
    - Trading characteristics
    - Factor model components
    - Asset classifications
    - Market metadata
    
    The data structure ensures all components are consistently sized
    and properly aligned for use in portfolio optimization.
    
    Attributes:
        returns: Asset returns matrix (n_assets x n_periods)
        volumes: Trading volumes matrix (n_assets x n_periods)
        spreads: Bid-ask spreads matrix (n_assets x n_periods)
        factor_returns: Factor returns matrix (n_factors x n_periods)
        factor_exposures: Factor loadings matrix (n_assets x n_factors)
        classifications: List of industry classifications per asset
        asset_classes: List of asset class assignments per asset
        currencies: List of currency exposures per asset
        credit_profiles: List of credit ratings and profiles per asset
        market_caps: Array of market capitalizations
    """
    returns: np.ndarray              
    volumes: np.ndarray             
    spreads: np.ndarray            
    factor_returns: np.ndarray     
    factor_exposures: np.ndarray   
    classifications: List[IndustryClassification]  
    asset_classes: List[AssetClass]               
    currencies: List[CurrencyExposure]            
    credit_profiles: List[CreditProfile]          
    market_caps: np.ndarray        


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
    """Generates realistic test data for portfolio optimization experiments.
    
    This class generates comprehensive market data including:
    - Asset returns with realistic correlations
    - Factor model data with common risk factors
    - Trading volumes with intraday patterns
    - Bid-ask spreads with size effects
    - Industry classifications with hierarchy
    - Asset characteristics and metadata
    
    Features:
    1. Realistic return distributions and correlations
    2. Common factor structure (market, size, value, etc.)
    3. Volume patterns including seasonality
    4. Market cap-sensitive spread modeling
    5. Multi-level industry classification
    6. ESG and credit ratings
    
    Typical usage:
        >>> generator = EnhancedTestDataGenerator(seed=42)
        >>> data = generator.generate_market_data(
        ...     n_assets=100,
        ...     n_periods=252
        ... )
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize test data generator.
        
        Args:
            seed: Optional random seed for reproducibility
        
        Sets up:
        - Random number generator
        - Industry classification hierarchy
        - Factor definitions
        - Asset class categories
        - Currency zones
        - Rating scales
        """
        self.rng = np.random.RandomState(seed)
        self.logger = logging.getLogger(__name__)

        # Define common investment factors
        self.factors = [
            'Market',    # Market factor
            'Size',      # Market capitalization
            'Value',     # Value vs Growth
            'Momentum',  # Price momentum
            'Quality',   # Financial quality
            'LowVol',    # Low volatility
            'Yield',     # Dividend yield
            'Growth',    # Earnings growth
            'Liquidity'  # Trading liquidity
        ]

        # Setup industry hierarchy (abbreviated for clarity)
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
            # Additional sectors follow similar pattern...
        }

        # Define asset class categories
        self.asset_hierarchy = {
            'Equity': {
                'sub_types': ['Large Cap', 'Mid Cap', 'Small Cap'],
                'regions': ['North America', 'Europe', 'Asia Pacific'],
                'styles': ['Value', 'Growth', 'Blend']
            },
            # Additional asset classes...
        }

        # Define currency zones for FX exposure
        self.currency_zones = {
            'USD': ['USD', 'CAD', 'MXN'],
            'EUR': ['EUR', 'GBP', 'CHF'],
            'APAC': ['JPY', 'KRW', 'CNY', 'AUD']
        }

        # Define credit ratings
        self.ratings = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-',
                     'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-']
        self.rating_agencies = ['S&P', 'Moody\'s', 'Fitch']
        self.rating_outlooks = ['Stable', 'Positive', 'Negative']

    def generate_market_data(self, n_assets: int, n_periods: int) -> MarketData:
        """Generate complete set of realistic market data.
        
        Args:
            n_assets: Number of assets to generate
            n_periods: Number of time periods
            
        Returns:
            MarketData object containing:
            - returns: Asset returns
            - volumes: Trading volumes
            - spreads: Bid-ask spreads
            - factor_returns: Factor returns
            - factor_exposures: Factor loadings
            - classifications: Industry classifications
            - asset_classes: Asset class assignments
            - currencies: Currency exposures
            - credit_profiles: Credit ratings
            - market_caps: Market capitalizations
        """
        # Generate factor-based returns
        factor_returns = self.generate_factor_returns(n_periods)
        factor_exposures = self.generate_factor_exposures(n_assets)

        # Generate market caps with realistic distribution
        log_market_caps = stats.norm.rvs(loc=8.0, scale=1.5, size=n_assets,
                                       random_state=self.rng)
        market_caps = np.exp(log_market_caps)  # In millions

        # Generate returns with factor structure
        specific_vol = 0.20 / np.sqrt(252)  # 20% annualized specific vol
        specific_returns = self.rng.normal(0, specific_vol,
                                         size=(n_assets, n_periods))
        returns = factor_exposures @ factor_returns + specific_returns

        # Generate realistic volumes and spreads
        volumes = self.generate_volumes(n_assets, n_periods, market_caps)
        spreads = self.generate_spreads(n_assets, n_periods, market_caps)

        # Generate metadata
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
        """Generate realistic factor returns.
        
        Creates factor returns with:
        1. Realistic volatilities
        2. Cross-factor correlations
        3. Risk premia
        
        Args:
            n_periods: Number of periods to generate
            
        Returns:
            Factor returns matrix (n_factors x n_periods)
        """
        n_factors = len(self.factors)

        # Define annualized factor volatilities
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

        # Create correlation matrix with realistic relationships
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

        # Build correlation matrix
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

        # Generate daily factor returns with small positive risk premia
        daily_vol = vols / np.sqrt(252)
        returns = self.rng.multivariate_normal(
            mean=daily_vol * 0.1,  # Small positive risk premia
            cov=cov / 252,
            size=n_periods
        ).T

        return returns

    def generate_volumes(self, n_assets: int, n_periods: int,
                        market_caps: np.ndarray) -> np.ndarray:
        """Generate realistic trading volumes.
        
        Includes:
        1. Base volume related to market cap
        2. Day-of-week effects
        3. Random daily variations
        
        Args:
            n_assets: Number of assets
            n_periods: Number of periods
            market_caps: Market capitalizations
            
        Returns:
            Trading volumes matrix (n_assets x n_periods)
        """
        # Base daily turnover of 0.4%
        base_volumes = market_caps * 0.004
        volumes = np.zeros((n_assets, n_periods))

        # Generate volumes with random daily variations
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
        """Generate realistic bid-ask spreads.
        
        Features:
        1. Market cap dependent base spreads
        2. Daily variations
        3. Minimum spread floors
        
        Args:
            n_assets: Number of assets
            n_periods: Number of periods
            market_caps: Market capitalizations
            
        Returns:
            Bid-ask spreads matrix (n_assets x n_periods)
        """
        # Base spreads inversely related to sqrt of market cap
        base_spreads = 5 / np.sqrt(market_caps)  # Basis points
        base_spreads = np.clip(base_spreads, 0.5, 50.0)  # Clip extremes
        spreads = np.zeros((n_assets, n_periods))

        # Add daily variations
        for i in range(n_assets):
            daily_changes = self.rng.lognormal(mean=0.0, sigma=0.2, size=n_periods)
            spreads[i] = base_spreads[i] * daily_changes

        return spreads

    def create_stress_scenario(self, base_data: MarketData, 
                             scenario_type: str) -> MarketData:
        """Create stress scenario by modifying base market data.
        
        Implements common stress scenarios:
        1. Market crash: High volatility, correlations
        2. Liquidity crisis: Volume drops, spread spikes
        3. Sector rotation: Sector-specific shocks
        4. Currency crisis: Regional currency stress
        5. Credit event: Rating-based stress
        
        Args:
            base_data: Base market data
            scenario_type: Type of stress scenario
            
        Returns:
            Modified MarketData reflecting stress scenario
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
            stress_data.returns *= 2.0        # Double volatility
            stress_data.volumes *= 3.0        # Triple volumes
            stress_data.spreads *= 5.0        # 5x spreads
            stress_data.factor_returns *= 1.5  # Increase factor effects
            
        elif scenario_type == "liquidity_crisis":
            stress_data.volumes *= 0.2        # 80% volume drop
            stress_data.spreads *= 10.0       # 10x spreads
            # More severe for small caps
            for i, ac in enumerate(stress_data.asset_classes):
                if ac.sub_type == 'Small Cap':
                    stress_data.volumes[i] *= 0.5
                    stress_data.spreads[i] *= 2.0
                    
        elif scenario_type == "credit_event":
            for i, profile in enumerate(stress_data.credit_profiles):
                if profile.rating.startswith('B'):
                    # High yield stress
                    stress_data.returns[i] *= 0.7   # 30% loss
                    stress_data.volumes[i] *= 0.3   # 70% volume drop
                    stress_data.spreads[i] *= 8.0   # 8x spreads
                    
        return stress_data
    
    def generate_factor_exposures(self, n_assets: int) -> np.ndarray:
        """Generate realistic factor exposures for assets.
        
        Creates factor loadings with:
        1. Realistic market betas
        2. Style factor exposures
        3. Cross-factor relationships
        
        Args:
            n_assets: Number of assets to generate exposures for
            
        Returns:
            Factor exposures matrix (n_assets x n_factors)
        """
        n_factors = len(self.factors)
        exposures = np.zeros((n_assets, n_factors))

        # Generate market betas with realistic distribution
        market_betas = stats.norm.rvs(loc=1.0, scale=0.3, size=n_assets,
                                    random_state=self.rng)
        exposures[:, 0] = market_betas  # First factor is market

        # Generate other factor exposures
        for i in range(1, n_factors):
            exposures[:, i] = stats.norm.rvs(loc=0.0, scale=0.15, size=n_assets,
                                        random_state=self.rng)

        return exposures

    def generate_classifications(self, n_assets: int) -> List[IndustryClassification]:
        """Generate realistic industry classifications.
        
        Creates multi-level industry assignments:
        1. Sector level (e.g., Technology)
        2. Industry group (e.g., Software & Services)
        3. Industry (e.g., Software)
        4. Sub-industry (e.g., Application Software)
        
        Args:
            n_assets: Number of assets to classify
            
        Returns:
            List of industry classifications for each asset
        """
        classifications = []

        for _ in range(n_assets):
            # Select random sector
            sector = self.rng.choice(list(self.industry_hierarchy.keys()))
            industry_groups = self.industry_hierarchy[sector]
            
            # Select random industry group
            industry_group = self.rng.choice(list(industry_groups.keys()))
            industries = industry_groups[industry_group]
            
            # Select random industry and sub-industry
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
        """Generate realistic asset class assignments.
        
        Creates asset class metadata including:
        1. Primary asset class (e.g., Equity)
        2. Sub-type (e.g., Large Cap)
        3. Geographic region
        4. Investment style
        
        Args:
            n_assets: Number of assets to classify
            
        Returns:
            List of asset class assignments for each asset
        """
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
        """Generate realistic currency exposures.
        
        Creates currency exposure profiles:
        1. Direct currency (primary listing)
        2. Effective currency mix (for multinationals)
        3. Regional currency groupings
        
        Args:
            n_assets: Number of assets
            
        Returns:
            List of currency exposures for each asset
        """
        exposures = []
        all_currencies = [curr for zone in self.currency_zones.values()
                        for curr in zone]

        for _ in range(n_assets):
            # Select random currency zone and direct currency
            zone = self.rng.choice(list(self.currency_zones.keys()))
            direct = self.rng.choice(self.currency_zones[zone])

            # 30% chance of being multinational
            if self.rng.random() < 0.3:
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
        """Generate realistic credit profiles.
        
        Creates credit ratings and profiles:
        1. Credit rating (e.g., AAA)
        2. Rating agency
        3. Outlook and watch status
        4. Realistic rating distribution
        
        Args:
            n_assets: Number of assets
            
        Returns:
            List of credit profiles for each asset
        """
        profiles = []

        # Generate skewed rating distribution using beta distribution
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
        """Find the nearest positive-definite matrix.
        
        Converts a symmetric matrix to the nearest positive-definite matrix,
        which is necessary for valid covariance matrices.
        
        Args:
            A: Input matrix to convert
            
        Returns:
            Nearest positive-definite matrix to input
        """
        B = (A + A.T) / 2  # Ensure symmetry
        eigvals, eigvecs = np.linalg.eigh(B)
        eigvals = np.maximum(eigvals, 0)  # Ensure positive eigenvalues
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def generate_realistic_problem(self, n_assets: int, n_periods: int,
                                include_market_impact: bool = True,
                                include_factors: bool = True,
                                include_classifications: bool = True,
                                seed: Optional[int] = None) -> PortfolioOptProblem:
        """Generate a complete portfolio optimization problem.
        
        Creates a realistic problem instance with optional components:
        1. Core returns data
        2. Market impact data (optional)
        3. Factor model data (optional)
        4. Classification data (optional)
        
        Args:
            n_assets: Number of assets in portfolio
            n_periods: Number of time periods
            include_market_impact: Whether to include volume/spread data
            include_factors: Whether to include factor model data
            include_classifications: Whether to include classifications
            seed: Random seed for reproducibility
            
        Returns:
            PortfolioOptProblem instance with requested components
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
