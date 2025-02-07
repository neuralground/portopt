import os
import configparser
from typing import Dict, Any

class ConfigManager:
    """Manages configuration settings for portfolio optimization."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config = configparser.ConfigParser()
        self.converted_values = {
            'test_parameters': {},
            'portfolio_constraints': {},
            'classical_solver': {},
            'multithreaded_solver': {},  # Added this initialization
            'performance_metrics': {}
        }
        
        # Load default configuration
        default_path = os.path.join(
            os.path.dirname(__file__),
            'default.ini'
        )
        self.config.read(default_path)
        
        # Override with user config if provided
        if config_path and os.path.exists(config_path):
            self.config.read(config_path)
        
        # Convert types for numerical values
        self._convert_types()

    def _convert_types(self):
        """Convert string values to appropriate types."""
        # Test parameters
        for key in ['n_assets', 'n_periods', 'n_simulations']:
            if self.config.has_option('test_parameters', key):
                self.converted_values['test_parameters'][key] = self.config.getint('test_parameters', key)
        
        # Portfolio constraints
        float_keys = ['min_weight', 'max_weight', 'max_sector_weight', 'turnover_limit']
        for key in float_keys:
            if self.config.has_option('portfolio_constraints', key):
                self.converted_values['portfolio_constraints'][key] = self.config.getfloat('portfolio_constraints', key)
        
        if self.config.has_option('portfolio_constraints', 'min_stocks_held'):
            self.converted_values['portfolio_constraints']['min_stocks_held'] = self.config.getint(
                'portfolio_constraints', 'min_stocks_held'
            )
        
        # Classical solver parameters
        solver_float_keys = ['initial_penalty', 'penalty_multiplier', 'perturbation_size']
        
        if self.config.has_option('classical_solver', 'max_iterations'):
            self.converted_values['classical_solver']['max_iterations'] = self.config.getint(
                'classical_solver', 'max_iterations'
            )
        
        for key in solver_float_keys:
            if self.config.has_option('classical_solver', key):
                self.converted_values['classical_solver'][key] = self.config.getfloat('classical_solver', key)
        
        # Multithreaded solver parameters (only if section exists)
        if 'multithreaded_solver' in self.config:
            if self.config.has_option('multithreaded_solver', 'max_iterations'):
                self.converted_values['multithreaded_solver']['max_iterations'] = self.config.getint(
                    'multithreaded_solver', 'max_iterations'
                )
            
            for key in solver_float_keys:
                if self.config.has_option('multithreaded_solver', key):
                    self.converted_values['multithreaded_solver'][key] = self.config.getfloat(
                        'multithreaded_solver', key
                    )
            
            if self.config.has_option('multithreaded_solver', 'n_threads'):
                self.converted_values['multithreaded_solver']['n_threads'] = self.config.getint(
                    'multithreaded_solver', 'n_threads'
                )
        
        # Performance metrics
        if 'performance_metrics' in self.config:
            for key in self.config['performance_metrics']:
                self.converted_values['performance_metrics'][key] = self.config.getboolean('performance_metrics', key)
    
    def get_test_params(self) -> Dict[str, Any]:
        """Get test parameters."""
        return self.converted_values['test_parameters'].copy()
    
    def get_portfolio_constraints(self) -> Dict[str, Any]:
        """Get portfolio constraints."""
        return self.converted_values['portfolio_constraints'].copy()
    
    def get_solver_params(self, solver_name: str) -> Dict[str, Any]:
        """Get solver-specific parameters."""
        if solver_name not in self.converted_values:
            raise ValueError(f"No configuration found for solver: {solver_name}")
        return self.converted_values[solver_name].copy()
    
    def get_performance_metrics(self) -> Dict[str, bool]:
        """Get performance metrics settings."""
        return self.converted_values['performance_metrics'].copy()

