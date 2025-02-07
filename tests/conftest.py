import pytest

def pytest_addoption(parser):
    """Add custom command line options for portfolio optimization tests."""
    # Problem scale parameters
    parser.addoption("--n-assets", type=int, default=100,
                    help="Number of assets in portfolio")
    parser.addoption("--n-periods", type=int, default=252,
                    help="Number of time periods for historical data")
    
    # Portfolio constraints
    parser.addoption("--min-weight", type=float, default=0.01,
                    help="Minimum weight for any asset")
    parser.addoption("--max-weight", type=float, default=0.20,
                    help="Maximum weight for any asset")
    parser.addoption("--max-sector-weight", type=float, default=0.30,
                    help="Maximum weight for any sector")
    parser.addoption("--min-stocks-held", type=int, default=20,
                    help="Minimum number of stocks with non-zero weights")
    parser.addoption("--turnover-limit", type=float, default=0.20,
                    help="Maximum turnover from previous portfolio")

@pytest.fixture
def opt_params(request):
    """Fixture to provide optimization parameters from command line."""
    return {
        'n_assets': request.config.getoption('--n-assets'),
        'n_periods': request.config.getoption('--n-periods'),
        'min_weight': request.config.getoption('--min-weight'),
        'max_weight': request.config.getoption('--max-weight'),
        'max_sector_weight': request.config.getoption('--max-sector-weight'),
        'min_stocks_held': request.config.getoption('--min-stocks-held'),
        'turnover_limit': request.config.getoption('--turnover-limit')
    }

