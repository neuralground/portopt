import numpy as np
from portopt.solvers.classical import ClassicalSolver
from portopt.data.generator import EnhancedTestDataGenerator
from tests.utils import TestDataHandler, TestMetricsCalculator, TestResult
from datetime import datetime

def test_solver_with_constraints(opt_params, config_manager):
    """Test solver with realistic portfolio constraints."""
    # Get performance metrics configuration
    metrics = config_manager.get_performance_metrics()
    
    # Initialize handlers
    test_handler = TestDataHandler()
    metrics_calculator = TestMetricsCalculator()
    
    # Generate test data using enhanced generator
    generator = EnhancedTestDataGenerator()
    market_data = generator.generate_market_data(
        n_assets=opt_params['n_assets'],
        n_periods=opt_params['n_periods']
    )
    
    # Create problem instance
    problem = PortfolioOptProblem(
        returns=market_data.returns,
        volumes=market_data.volumes,
        spreads=market_data.spreads,
        factor_returns=market_data.factor_returns,
        factor_exposures=market_data.factor_exposures,
        market_caps=market_data.market_caps,
        currencies=market_data.currencies,
        constraints={}  # Will be updated below
    )
    
    # Add constraints
    problem.constraints.update({
        'min_weight': opt_params['min_weight'],
        'max_weight': opt_params['max_weight'],
        'max_sector_weight': opt_params['max_sector_weight'],
        'min_stocks_held': opt_params['min_stocks_held'],
        'sector_map': market_data.sector_map,
        'max_participation': 0.3  # Maximum market participation rate
    })
    
    # Generate previous portfolio for turnover constraint
    prev_weights = np.random.dirichlet(np.ones(opt_params['n_assets']))
    problem.constraints['prev_weights'] = prev_weights
    problem.constraints['turnover_limit'] = opt_params['turnover_limit']
    
    # Create solver
    solver = ClassicalSolver(
        max_iterations=opt_params['max_iterations'],
        initial_penalty=opt_params['initial_penalty'],
        penalty_multiplier=opt_params['penalty_multiplier'],
        perturbation_size=opt_params['perturbation_size']
    )
    
    # Time and solve
    start_time = datetime.now()
    result = solver.solve(problem)
    end_time = datetime.now()
    
    # Check constraints
    constraints_satisfied = test_handler.check_constraints(
        result.weights,
        opt_params,
        market_data
    )
    
    # Calculate metrics
    portfolio_metrics = metrics_calculator.calculate_portfolio_metrics(
        result.weights,
        problem
    )
    
    # Create test result
    test_result = TestResult(
        test_name="solver_with_constraints",
        start_time=start_time,
        end_time=end_time,
        parameters=opt_params,
        metrics={
            'solve_time': result.solve_time,
            'objective_value': result.objective_value,
            **portfolio_metrics
        },
        constraints_satisfied=constraints_satisfied,
        additional_info={
            'n_active_positions': np.sum(result.weights > 0),
            'feasible': result.feasible,
            'factor_exposures': (result.weights @ market_data.factor_exposures).tolist() 
                if metrics.get('track_factor_exposures', False) else None,
            'sector_weights': test_handler.calculate_sector_weights(
                result.weights, 
                market_data.sector_map
            ).tolist() if metrics.get('track_sector_weights', False) else None,
            'market_impact_metrics': {
                'max_participation': float(np.max(result.weights / market_data.volumes.mean(axis=1))),
                'avg_spread_cost': float(np.mean(result.weights * market_data.spreads.mean(axis=1)))
            } if metrics.get('track_market_impact', False) else None
        }
    )
    
    # Print results
    print_test_report(test_result)
    
    # Assertions
    assert all(constraints_satisfied.values()), "Not all constraints satisfied"
    assert result.feasible, "Solution not marked as feasible"
    
    return test_result

def test_stress_scenarios(opt_params, config_manager):
    """Test solver under different market stress scenarios."""
    generator = EnhancedTestDataGenerator()
    base_market_data = generator.generate_market_data(
        n_assets=opt_params['n_assets'],
        n_periods=opt_params['n_periods']
    )
    
    scenarios = ["market_crash", "liquidity_crisis", "sector_rotation"]
    results = []
    
    for scenario in scenarios:
        # Generate stress scenario
        stress_data = generator.create_stress_scenario(
            base_market_data,
            scenario
        )
        
        # Create and solve problem under stress
        problem = PortfolioOptProblem(
            returns=stress_data.returns,
            volumes=stress_data.volumes,
            spreads=stress_data.spreads,
            factor_returns=stress_data.factor_returns,
            factor_exposures=stress_data.factor_exposures,
            market_caps=stress_data.market_caps,
            currencies=stress_data.currencies,
            constraints=opt_params
        )
        
        solver = ClassicalSolver(**opt_params)
        result = solver.solve(problem)
        
        results.append({
            'scenario': scenario,
            'feasible': result.feasible,
            'solve_time': result.solve_time,
            'objective_value': result.objective_value
        })
    
    return results

