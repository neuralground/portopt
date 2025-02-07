import numpy as np
import pytest
from portopt.data.generator import TestDataGenerator
from portopt.solvers.classical import ClassicalSolver

def _generate_sector_map(n_assets, n_sectors=11):  # 11 GICS sectors
    """Generate random sector assignments for assets."""
    return np.random.randint(0, n_sectors, size=n_assets)

def _calculate_sector_weights(weights, sector_map):
    """Calculate total weight per sector."""
    n_sectors = len(np.unique(sector_map))
    sector_weights = np.zeros(n_sectors)
    for i in range(n_sectors):
        sector_weights[i] = np.sum(weights[sector_map == i])
    return sector_weights

def _check_constraints(weights, params, sector_map=None, prev_weights=None):
    """Check if all portfolio constraints are satisfied."""
    constraints_satisfied = {
        'sum_to_one': np.isclose(np.sum(weights), 1.0, rtol=1e-5),
        'min_weight': np.all(weights[weights > 0] >= params['min_weight']),
        'max_weight': np.all(weights <= params['max_weight']),
        'min_stocks_held': np.sum(weights > 0) >= params['min_stocks_held']
    }
    
    if sector_map is not None:
        sector_weights = _calculate_sector_weights(weights, sector_map)
        constraints_satisfied['sector_limits'] = np.all(sector_weights <= params['max_sector_weight'])
    
    if prev_weights is not None:
        turnover = np.sum(np.abs(weights - prev_weights))
        constraints_satisfied['turnover'] = turnover <= params['turnover_limit']
    
    return constraints_satisfied

def test_solver_with_constraints(opt_params):
    """Test solver with realistic portfolio constraints."""
    # Generate test problem with sectors
    generator = TestDataGenerator()
    problem = generator.generate_realistic_problem(
        n_assets=opt_params['n_assets'],
        n_periods=opt_params['n_periods']
    )
    
    # Add sector information
    sector_map = _generate_sector_map(opt_params['n_assets'])
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
    
    # Solve and check results
    solver = ClassicalSolver()
    result = solver.solve(problem)
    
    # Check constraints
    constraints_satisfied = _check_constraints(
        result.weights,
        opt_params,
        sector_map,
        prev_weights
    )
    
    # Print detailed results
    print("\nOptimization Results:")
    print(f"Solve Time: {result.solve_time:.4f} seconds")
    print(f"Objective Value: {result.objective_value:.6f}")
    print(f"Number of Assets: {opt_params['n_assets']}")
    print(f"Active Positions: {np.sum(result.weights > 0)}")
    print("\nConstraint Satisfaction:")
    for constraint, satisfied in constraints_satisfied.items():
        print(f"{constraint}: {'✓' if satisfied else '✗'}")
    
    # Portfolio characteristics
    portfolio_return = np.dot(result.weights, problem.exp_returns)
    portfolio_vol = np.sqrt(result.weights.T @ problem.cov_matrix @ result.weights)
    
    print("\nPortfolio Characteristics:")
    print(f"Expected Return: {portfolio_return:.4%}")
    print(f"Volatility: {portfolio_vol:.4%}")
    print(f"Sharpe Ratio: {portfolio_return/portfolio_vol:.4f}")
    
    # Assert all constraints are satisfied
    assert all(constraints_satisfied.values()), "Not all constraints satisfied"
    assert result.feasible, "Solution not marked as feasible"

