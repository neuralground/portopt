import pytest
from portopt.config.manager import ConfigManager

def pytest_addoption(parser):
    """Add custom command line options for portfolio optimization tests."""
    parser.addoption(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )

@pytest.fixture
def config_manager(request):
    """Fixture to provide configuration manager."""
    config_path = request.config.getoption('--config')
    return ConfigManager(config_path)

@pytest.fixture
def opt_params(config_manager):
    """Fixture to provide optimization parameters from config."""
    params = {}
    
    # Combine test parameters and portfolio constraints
    params.update(config_manager.get_test_params())
    params.update(config_manager.get_portfolio_constraints())
    
    # Add solver parameters
    params.update(config_manager.get_solver_params('classical_solver'))
    
    return params

