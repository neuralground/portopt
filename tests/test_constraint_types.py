"""Tests for constraint types and validation."""

import pytest
import numpy as np

from portopt.constraints.constraint_types import (
    IndustryClassification, 
    AssetClass, 
    CurrencyExposure, 
    CreditProfile,
    PortfolioConstraints
)


class TestConstraintTypes:
    """Tests for the constraint data structures and validation."""
    
    def test_industry_classification(self):
        """Test IndustryClassification data structure."""
        classification = IndustryClassification(
            sector="Information Technology",
            industry_group="Software & Services",
            industry="Software",
            sub_industry="Application Software"
        )
        
        assert classification.sector == "Information Technology"
        assert classification.industry_group == "Software & Services"
        assert classification.industry == "Software"
        assert classification.sub_industry == "Application Software"
    
    def test_asset_class(self):
        """Test AssetClass data structure."""
        asset_class = AssetClass(
            primary="Equity",
            sub_type="Large Cap",
            region="North America",
            style="Value"
        )
        
        assert asset_class.primary == "Equity"
        assert asset_class.sub_type == "Large Cap"
        assert asset_class.region == "North America"
        assert asset_class.style == "Value"
        
        # Test with optional field omitted
        asset_class_no_style = AssetClass(
            primary="Fixed Income",
            sub_type="Government",
            region="Europe"
        )
        
        assert asset_class_no_style.primary == "Fixed Income"
        assert asset_class_no_style.sub_type == "Government"
        assert asset_class_no_style.region == "Europe"
        assert asset_class_no_style.style is None
    
    def test_currency_exposure(self):
        """Test CurrencyExposure data structure."""
        currency_exposure = CurrencyExposure(
            direct="USD",
            effective={"USD": 0.7, "EUR": 0.2, "JPY": 0.1}
        )
        
        assert currency_exposure.direct == "USD"
        assert currency_exposure.effective["USD"] == 0.7
        assert currency_exposure.effective["EUR"] == 0.2
        assert currency_exposure.effective["JPY"] == 0.1
        assert sum(currency_exposure.effective.values()) == 1.0
    
    def test_credit_profile(self):
        """Test CreditProfile data structure."""
        credit_profile = CreditProfile(
            rating="BBB+",
            agency="S&P",
            outlook="Stable",
            watch=False
        )
        
        assert credit_profile.rating == "BBB+"
        assert credit_profile.agency == "S&P"
        assert credit_profile.outlook == "Stable"
        assert credit_profile.watch is False
        
        # Test with watch=True
        credit_profile_watch = CreditProfile(
            rating="A-",
            agency="Moody's",
            outlook="Negative",
            watch=True
        )
        
        assert credit_profile_watch.rating == "A-"
        assert credit_profile_watch.agency == "Moody's"
        assert credit_profile_watch.outlook == "Negative"
        assert credit_profile_watch.watch is True


class TestPortfolioConstraints:
    """Tests for the PortfolioConstraints class."""
    
    @pytest.fixture
    def sample_constraints(self):
        """Create a sample PortfolioConstraints object."""
        constraints = PortfolioConstraints()
        
        # Set basic constraints
        constraints.min_weight = 0.01
        constraints.max_weight = 0.3
        constraints.min_stocks = 5
        constraints.max_stocks = 20
        
        # Set industry constraints
        constraints.sector_limits = {
            "Information Technology": 0.4,
            "Financials": 0.3,
            "Healthcare": 0.25
        }
        
        # Set asset class constraints
        constraints.asset_class_limits = {
            "Equity": 0.7,
            "Fixed Income": 0.3
        }
        
        # Set currency constraints
        constraints.currency_limits = {
            "USD": 0.6,
            "EUR": 0.3,
            "JPY": 0.1
        }
        constraints.base_currency = "USD"
        
        # Set credit constraints
        constraints.min_rating = "BBB-"
        constraints.rating_limits = {
            "AAA": 0.2,
            "AA": 0.3,
            "A": 0.3,
            "BBB": 0.2
        }
        
        return constraints
    
    @pytest.fixture
    def sample_classifications(self):
        """Create sample industry classifications."""
        return [
            IndustryClassification(
                sector="Information Technology",
                industry_group="Software & Services",
                industry="Software",
                sub_industry="Application Software"
            ),
            IndustryClassification(
                sector="Information Technology",
                industry_group="Hardware",
                industry="Computers",
                sub_industry="Servers"
            ),
            IndustryClassification(
                sector="Financials",
                industry_group="Banks",
                industry="Commercial Banks",
                sub_industry="Regional Banks"
            ),
            IndustryClassification(
                sector="Healthcare",
                industry_group="Pharmaceuticals",
                industry="Biotech",
                sub_industry="Research"
            ),
            IndustryClassification(
                sector="Consumer Discretionary",
                industry_group="Retail",
                industry="Specialty Retail",
                sub_industry="Apparel"
            )
        ]
    
    @pytest.fixture
    def sample_asset_classes(self):
        """Create sample asset classes."""
        return [
            AssetClass(primary="Equity", sub_type="Large Cap", region="North America", style="Value"),
            AssetClass(primary="Equity", sub_type="Mid Cap", region="Europe", style="Growth"),
            AssetClass(primary="Fixed Income", sub_type="Government", region="North America"),
            AssetClass(primary="Fixed Income", sub_type="Corporate", region="Europe"),
            AssetClass(primary="Alternative", sub_type="Real Estate", region="Asia")
        ]
    
    def test_check_industry_constraints(self, sample_constraints, sample_classifications):
        """Test checking industry-related constraints."""
        # Create weights that violate sector constraints
        weights = np.array([0.3, 0.3, 0.2, 0.1, 0.1])  # IT sector = 0.6 > limit of 0.4
        
        results, _ = sample_constraints.check_industry_constraints(weights, sample_classifications)
        
        # IT sector constraint should be violated
        assert results["sector_Information Technology"] == False
        
        # Financials and Healthcare should be within limits
        assert results["sector_Financials"] == True
        assert results["sector_Healthcare"] == True
        
        # Fix weights to comply with constraints
        weights = np.array([0.2, 0.2, 0.3, 0.2, 0.1])
        
        results, _ = sample_constraints.check_industry_constraints(weights, sample_classifications)
        
        # All sector constraints should now be satisfied
        assert results["sector_Information Technology"] == True
        assert results["sector_Financials"] == True
        assert results["sector_Healthcare"] == True
    
    def test_check_asset_class_constraints(self, sample_constraints, sample_asset_classes):
        """Test checking asset class constraints."""
        # Create weights that violate asset class constraints
        weights = np.array([0.4, 0.4, 0.1, 0.05, 0.05])  # Equity = 0.8 > limit of 0.7
        
        results = sample_constraints.check_asset_class_constraints(weights, sample_asset_classes)
        
        # Equity constraint should be violated
        assert results["asset_class_Equity"] == False
        
        # Fixed Income should be within limits
        assert results["asset_class_Fixed Income"] == True
        
        # Fix weights to comply with constraints
        weights = np.array([0.35, 0.35, 0.15, 0.15, 0.0])
        
        results = sample_constraints.check_asset_class_constraints(weights, sample_asset_classes)
        
        # All asset class constraints should now be satisfied
        assert results["asset_class_Equity"] == True
        assert results["asset_class_Fixed Income"] == True
