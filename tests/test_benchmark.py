import pytest
import numpy as np
from portopt.solvers.classical import ClassicalSolver
from portopt.benchmark.runner import BenchmarkRunner
from portopt.impact import MarketImpactParams

def test_solver_scaling(config_manager):
    """Test solver performance scaling with problem size."""
    # Get solver parameters from config
    solver_params = config_manager.get_solver_params('classical_solver')

    # Create benchmark runner
    runner = BenchmarkRunner(output_dir="benchmark_results")

    # Define problem size ranges for testing
    n_assets_range = [20, 50, 100]  # Small, medium, large portfolios
    n_periods_range = [252, 504]    # 1 year and 2 years of daily data

    # Optional: Define stress scenarios
    stress_scenarios = ['market_crash', 'liquidity_crisis']

    # Run size scaling benchmark
    results = runner.run_size_scaling_benchmark(
        solver_classes=[ClassicalSolver],
        solver_params=solver_params,
        n_assets_range=n_assets_range,
        n_periods_range=n_periods_range,
        n_trials=2,
        stress_scenarios=stress_scenarios,
        log_level="INFO"
    )

    # Basic checks on results
    assert len(results) > 0, "No benchmark results generated"
    assert 'solve_time' in results.columns, "Missing solve time data"
    assert 'feasible' in results.columns, "Missing feasibility information"
    assert 'total_cost' in results.columns, "Missing cost analysis"
    assert 'var_95' in results.columns, "Missing risk metrics"

    # Check scaling behavior
    solver_results = results[results['solver_class'] == ClassicalSolver.__name__]
    small_problems = solver_results[solver_results['n_assets'] == 20]['solve_time'].mean()
    large_problems = solver_results[solver_results['n_assets'] == 100]['solve_time'].mean()
    assert large_problems > small_problems, f"No apparent size scaling for {ClassicalSolver.__name__}"

    # Check stress scenario impact
    if stress_scenarios:
        base_results = results[results['scenario'] == 'base']
        stress_results = results[results['scenario'] == 'market_crash']
        assert stress_results['total_cost'].mean() > base_results['total_cost'].mean(), \
            "Market stress not reflected in costs"

def test_detailed_metrics(config_manager):
    """Test detailed metrics calculation and analysis."""
    # Get solver parameters
    solver_params = config_manager.get_solver_params('classical_solver')

    # Create benchmark runner
    runner = BenchmarkRunner(output_dir="benchmark_results")

    # Run benchmark with focus on single size but detailed metrics
    results = runner.run_size_scaling_benchmark(
        solver_classes=[ClassicalSolver],
        solver_params=solver_params,
        n_assets_range=[50],  # Single size for detailed analysis
        n_periods_range=[252],
        n_trials=3,
        stress_scenarios=None  # No stress scenarios for this test
    )

    # Check presence of all metrics
    required_metrics = [
        'var_95', 'cvar_95', 'tracking_error',
        'total_cost', 'spread_costs', 'impact_costs',
        'active_positions', 'concentration'
    ]

    for metric in required_metrics:
        assert metric in results.columns, f"Missing metric: {metric}"

    # Verify reasonable ranges for metrics
    assert all(results['var_95'] > 0), "Invalid VaR values"
    assert all(results['total_cost'] > 0), "Invalid cost values"
    assert all(results['active_positions'] > 0), "Invalid position counts"

if __name__ == '__main__':
    # For manual running with different configurations
    from portopt.config.manager import ConfigManager

    config_manager = ConfigManager()
    test_solver_scaling(config_manager)
    test_detailed_metrics(config_manager)

