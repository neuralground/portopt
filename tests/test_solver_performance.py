import numpy as np
from portopt.solvers.classical import ClassicalSolver
from portopt.data.generator import TestDataGenerator
from tests.utils import (
    TestDataHandler, 
    TestMetricsCalculator, 
    TestResult,
    print_test_report
)
from datetime import datetime

def test_solver_with_constraints(opt_params, config_manager):
    """Test solver with realistic portfolio constraints."""
    # Get performance metrics configuration
    metrics = config_manager.get_performance_metrics()
    
    # Initialize handlers
    test_handler = TestDataHandler()
    metrics_calculator = TestMetricsCalculator()
    
    # Generate test problem with sectors
    generator = TestDataGenerator()
    problem = generator.generate_realistic_problem(
        n_assets=opt_params['n_assets'],
        n_periods=opt_params['n_periods']
    )
    
    # Add sector information
    sector_map = test_handler.generate_sector_map(opt_params['n_assets'])
    problem.constraints.update({
        'min_weight': opt_params['min_weight'],
        'max_weight': opt_params['max_weight'],
        'max_sector_weight': opt_params['max_sector_weight'],
        'min_stocks_held': opt_params['min_stocks_held'],
        'sector_map': sector_map
    })
    
    # Generate previous portfolio for turnover constraint
    prev_weights = np.random.dirichlet(np.ones(opt_params['n_assets']))
    problem.constraints['prev_weights'] = prev_weights
    problem.constraints['turnover_limit'] = opt_params['turnover_limit']
    
    # Create solver with optimization parameters
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
        sector_map,
        prev_weights
    )
    
    # Calculate metrics
    portfolio_metrics = metrics_calculator.calculate_portfolio_metrics(
        result.weights,
        problem.returns,
        problem.cov_matrix
    )
    
    # Create test result object
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
            'sector_weights': test_handler.calculate_sector_weights(
                result.weights, 
                sector_map
            ).tolist() if metrics.get('track_sector_weights', False) else None
        }
    )
    
    # Print results
    print_test_report(test_result)
    
    # Assert all constraints are satisfied
    assert all(constraints_satisfied.values()), "Not all constraints satisfied"
    assert result.feasible, "Solution not marked as feasible"
    
    return test_result  # Return result for potential aggregation

