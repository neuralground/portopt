import pytest
import numpy as np
from portopt.solvers.classical import ClassicalSolver
from portopt.benchmark.runner import BenchmarkRunner

def test_solver_scaling(config_manager):
    """Test solver performance scaling with problem size."""
    # Get solver parameters from config
    solver_params = config_manager.get_solver_params('classical_solver')
    
    # Create benchmark runner
    runner = BenchmarkRunner(output_dir="benchmark_results")
    
    # Run size scaling benchmark with smaller ranges for testing
    results = runner.run_size_scaling_benchmark(
        solver_class=ClassicalSolver,
        solver_params=solver_params,
        n_assets_range=[20, 50],  # Smaller range for testing
        n_periods_range=[252],
        n_trials=2  # Fewer trials for testing
    )
    
    # Basic checks on results
    assert len(results) > 0, "No benchmark results generated"
    assert results['solve_time'].mean() > 0, "Invalid solve times"
    assert 'feasible' in results.columns, "Missing feasibility information"
    
    # Check scaling behavior
    small_problems = results[results['n_assets'] == 20]['solve_time'].mean()
    large_problems = results[results['n_assets'] == 50]['solve_time'].mean()
    assert large_problems > small_problems, "No apparent size scaling"

def test_constraint_sensitivity(config_manager):
    """Test solver sensitivity to constraint parameters."""
    # Get solver parameters from config
    solver_params = config_manager.get_solver_params('classical_solver')
    
    # Base constraints
    base_constraints = {
        'n_assets': 30,  # Smaller for testing
        'n_periods': 252,
        'min_weight': 0.005,
        'max_weight': 0.15,
        'max_sector_weight': 0.25,
        'min_stocks_held': 10,
        'turnover_limit': 0.15
    }
    
    # Parameters to vary (limited range for testing)
    param_ranges = {
        'turnover_limit': [0.10, 0.20],
        'max_weight': [0.10, 0.20]
    }
    
    # Create benchmark runner
    runner = BenchmarkRunner(output_dir="benchmark_results")
    
    # Run sensitivity analysis
    results = runner.run_constraint_sensitivity(
        solver_class=ClassicalSolver,
        solver_params=solver_params,
        base_constraints=base_constraints,
        param_ranges=param_ranges,
        n_trials=2
    )
    
    # Basic checks on results
    assert len(results) > 0, "No sensitivity results generated"
    assert all(col in results.columns for col in ['parameter', 'value', 'solve_time'])
    
    # Check impact of constraints
    for param in param_ranges:
        param_results = results[results['parameter'] == param]
        assert len(param_results) > 0, f"No results for {param}"

if __name__ == '__main__':
    # For manual running with different configurations
    from portopt.config.manager import ConfigManager
    
    config_manager = ConfigManager()
    test_solver_scaling(config_manager)
    test_constraint_sensitivity(config_manager)

