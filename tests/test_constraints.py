import pytest
import numpy as np
from portopt.constraints.constraint_types import (
    IndustryClassification, AssetClass, CurrencyExposure, CreditProfile
)

def test_industry_classification():
    classification = IndustryClassification(
        sector="Technology",
        industry_group="Software",
        industry="Application Software",
        sub_industry="Enterprise Software"
    )
    assert classification.sector == "Technology"
    assert classification.industry == "Application Software"

def test_asset_class():
    asset_class = AssetClass(
        primary="Equity",
        sub_type="Large Cap",
        region="North America",
        style="Value"
    )
    assert asset_class.primary == "Equity"
    assert asset_class.style == "Value"

def test_currency_exposure():
    exposure = CurrencyExposure(
        direct="USD",
        effective={"USD": 0.7, "EUR": 0.3}
    )
    assert exposure.direct == "USD"
    assert sum(exposure.effective.values()) == pytest.approx(1.0)

def test_credit_profile():
    profile = CreditProfile(
        rating="AAA",
        agency="S&P",
        outlook="Stable",
        watch=False
    )
    assert profile.rating == "AAA"
    assert not profile.watch

