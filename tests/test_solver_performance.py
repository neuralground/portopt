import numpy as np
from portopt.solvers.classical import ClassicalSolver
from portopt.benchmark.runner import BenchmarkRunner
from portopt.impact import MarketImpactParams
from datetime import datetime

def test_solver_comprehensive(opt_params, config_manager):
    """Comprehensive solver testing with all metrics."""
    # Get performance metrics configuration
    metrics_config = config_manager.get_performance_metrics()
    
    # Create benchmark runner
    runner = BenchmarkRunner(output_dir="benchmark_results")

    # Run comprehensive benchmark
    results = runner.run_size_scaling_benchmark(
        solver_classes=[ClassicalSolver],
        solver_params={
            'max_iterations': opt_params['max_iterations'],
            'initial_penalty': opt_params['initial_penalty'],
            'penalty_multiplier': opt_params['penalty_multiplier'],
            'perturbation_size': opt_params['perturbation_size']
        },
        n_assets_range=[opt_params['n_assets']],
        n_periods_range=[opt_params['n_periods']],
        n_trials=opt_params['n_simulations'],
        stress_scenarios=['market_crash', 'liquidity_crisis'] if metrics_config.get('track_stress_scenarios') else None
    )

    # Basic assertions
    assert len(results) > 0, "No results generated"
    assert all(results['feasible']), "Not all solutions are feasible"
    
    # Check tracked metrics based on configuration
    if metrics_config.get('track_var_cvar'):
        assert 'var_95' in results.columns
        assert 'cvar_95' in results.columns
        
    if metrics_config.get('track_market_impact'):
        assert 'total_cost' in results.columns
        assert 'spread_costs' in results.columns
        assert 'impact_costs' in results.columns
        
    if metrics_config.get('track_factor_exposures'):
        assert 'factor_exposures' in results.columns

def test_stress_scenarios(opt_params, config_manager):
    """Test solver performance under different stress scenarios."""
    # Create benchmark runner
    runner = BenchmarkRunner(output_dir="benchmark_results")
    
    # Define stress scenarios
    stress_scenarios = ['market_crash', 'liquidity_crisis', 'sector_rotation']
    
    # Run stress tests
    results = runner.run_size_scaling_benchmark(
        solver_classes=[ClassicalSolver],
        solver_params={
            'max_iterations': opt_params['max_iterations'],
            'initial_penalty': opt_params['initial_penalty'],
            'penalty_multiplier': opt_params['penalty_multiplier'],
            'perturbation_size': opt_params['perturbation_size']
        },
        n_assets_range=[opt_params['n_assets']],
        n_periods_range=[opt_params['n_periods']],
        n_trials=2,  # Reduced trials for stress testing
        stress_scenarios=stress_scenarios
    )
    
    # Analyze results for each scenario
    for scenario in stress_scenarios:
        scenario_results = results[results['scenario'] == scenario]
        base_results = results[results['scenario'] == 'base']
        
        # Compare metrics
        assert len(scenario_results) > 0, f"No results for {scenario}"
        assert scenario_results['total_cost'].mean() > base_results['total_cost'].mean(), \
            f"Unexpected cost behavior in {scenario}"
        assert scenario_results['solve_time'].mean() >= base_results['solve_time'].mean() * 0.5, \
            f"Unexpected performance in {scenario}"

if __name__ == '__main__':
    from portopt.config.manager import ConfigManager
    
    # For manual testing
    config_manager = ConfigManager()
    opt_params = {
        'n_assets': 50,
        'n_periods': 252,
        'n_simulations': 3,
        'max_iterations': 20,
        'initial_penalty': 100.0,
        'penalty_multiplier': 2.0,
        'perturbation_size': 0.01
    }
    
    test_solver_comprehensive(opt_params, config_manager)
    test_stress_scenarios(opt_params, config_manager)

