# Quick Start Guide

This guide will help you get up and running with the Portfolio Optimization Testbed quickly.

## Basic Usage

### Step 1: Import Required Modules

```python
from portopt import TestDataGenerator, ClassicalSolver
from portopt.utils.logging import setup_logging, OptimizationLogger
from portopt.constraints.basic import FullInvestmentConstraint, PositionLimitConstraint
from portopt.core.objective import MinimumVarianceObjective
```

### Step 2: Set Up Logging

```python
# Set up logging
setup_logging(level="INFO")
logger = OptimizationLogger("quick_start")
```

### Step 3: Generate Test Data

```python
# Generate test data
generator = TestDataGenerator(seed=42)
problem = generator.generate_realistic_problem(
    n_assets=50,
    n_periods=252
)

logger.info(f"Generated problem with {problem.n_assets} assets and {problem.n_periods} periods")
```

### Step 4: Create and Run Solver

```python
# Create solver
solver = ClassicalSolver(
    max_iterations=20,
    initial_penalty=100.0
)

# Define constraints
constraints = [
    FullInvestmentConstraint(),
    PositionLimitConstraint(lower=0.0, upper=0.05)
]

# Define objective
objective = MinimumVarianceObjective()

# Solve the problem
result = solver.solve(
    problem=problem,
    constraints=constraints,
    objective=objective
)

logger.info(f"Optimization completed in {result.iterations} iterations")
logger.info(f"Objective value: {result.objective_value:.6f}")
```

### Step 5: Analyze Results

```python
# Get optimal weights
weights = result.weights
top_positions = sorted(zip(range(len(weights)), weights), key=lambda x: x[1], reverse=True)[:10]

print("Top 10 positions:")
for asset_idx, weight in top_positions:
    print(f"Asset {asset_idx}: {weight:.4f}")

# Calculate risk metrics
from portopt.metrics.risk import calculate_portfolio_volatility, calculate_var, calculate_cvar

volatility = calculate_portfolio_volatility(weights, problem.get_covariance_matrix())
var_95 = calculate_var(weights, problem.returns, confidence_level=0.95)
cvar_95 = calculate_cvar(weights, problem.returns, confidence_level=0.95)

print(f"Portfolio volatility: {volatility:.4f}")
print(f"VaR (95%): {var_95:.4f}")
print(f"CVaR (95%): {cvar_95:.4f}")
```

## Complete Example

Here's a complete example that you can run:

```python
import numpy as np
from portopt import TestDataGenerator, ClassicalSolver
from portopt.utils.logging import setup_logging, OptimizationLogger
from portopt.constraints.basic import FullInvestmentConstraint, PositionLimitConstraint
from portopt.core.objective import MinimumVarianceObjective
from portopt.metrics.risk import calculate_portfolio_volatility, calculate_var, calculate_cvar
from portopt.visualization.dashboard import create_dashboard

# Set up logging
setup_logging(level="INFO")
logger = OptimizationLogger("complete_example")

# Generate test data
logger.info("Generating test data...")
generator = TestDataGenerator(seed=42)
problem = generator.generate_realistic_problem(
    n_assets=50,
    n_periods=252
)

# Create solver
logger.info("Creating solver...")
solver = ClassicalSolver(
    max_iterations=50,
    initial_penalty=100.0,
    tolerance=1e-6,
    use_warm_start=True
)

# Define constraints
constraints = [
    FullInvestmentConstraint(),
    PositionLimitConstraint(lower=0.0, upper=0.05)
]

# Define objective
objective = MinimumVarianceObjective()

# Solve the problem
logger.info("Solving optimization problem...")
result = solver.solve(
    problem=problem,
    constraints=constraints,
    objective=objective
)

# Log results
logger.info(f"Optimization completed in {result.iterations} iterations")
logger.info(f"Objective value: {result.objective_value:.6f}")

# Analyze results
weights = result.weights
top_positions = sorted(zip(range(len(weights)), weights), key=lambda x: x[1], reverse=True)[:10]

print("\nTop 10 positions:")
for asset_idx, weight in top_positions:
    print(f"Asset {asset_idx}: {weight:.4f}")

# Calculate risk metrics
volatility = calculate_portfolio_volatility(weights, problem.get_covariance_matrix())
var_95 = calculate_var(weights, problem.returns, confidence_level=0.95)
cvar_95 = calculate_cvar(weights, problem.returns, confidence_level=0.95)

print(f"\nRisk Metrics:")
print(f"Portfolio volatility: {volatility:.4f}")
print(f"VaR (95%): {var_95:.4f}")
print(f"CVaR (95%): {cvar_95:.4f}")

# Create and run dashboard
logger.info("Creating dashboard...")
dashboard = create_dashboard(
    optimization_result=result,
    problem=problem,
    port=8050
)

logger.info("Dashboard created. Run dashboard.run_server() to start the dashboard.")
```

## Next Steps

Now that you've run your first optimization, you can:

1. **Explore Different Constraints**
   - Try sector constraints
   - Add factor exposure constraints
   - Implement turnover limits

2. **Try Different Objectives**
   - Maximum Sharpe ratio
   - Mean-variance optimization
   - Risk parity

3. **Analyze Market Impact**
   - Include market impact in optimization
   - Analyze trading costs
   - Optimize execution

4. **Run Benchmarks**
   - Compare different solvers
   - Analyze scaling behavior
   - Stress test with larger problems

## Common Patterns

### Loading Market Data

```python
import pandas as pd
from portopt.core.problem import PortfolioOptProblem

# Load returns data
returns_df = pd.read_csv("market_data.csv", index_col=0)
returns = returns_df.values

# Create problem instance
problem = PortfolioOptProblem(returns=returns)
```

### Adding Constraints

```python
from portopt.constraints.sector import SectorConstraint
from portopt.constraints.factor import FactorExposureConstraint

# Create sector constraint
sector_constraint = SectorConstraint(
    sector_ids=industry_classes,
    lower_bounds=np.full(n_industries, 0.05),
    upper_bounds=np.full(n_industries, 0.25)
)

# Create factor exposure constraint
factor_constraint = FactorExposureConstraint(
    factor_exposures=factor_exposures,
    lower_bounds=np.array([-0.2, -0.2, -0.2, -0.2, -0.2]),
    upper_bounds=np.array([0.2, 0.2, 0.2, 0.2, 0.2])
)

# Add to constraints list
constraints = [
    FullInvestmentConstraint(),
    PositionLimitConstraint(lower=0.0, upper=0.05),
    sector_constraint,
    factor_constraint
]
```

### Visualizing Results

```python
from portopt.visualization.plots import (
    plot_weights_distribution,
    plot_efficient_frontier,
    plot_risk_decomposition
)

# Plot weights distribution
plot_weights_distribution(weights, figsize=(10, 6))

# Plot efficient frontier
plot_efficient_frontier(
    problem,
    n_portfolios=50,
    highlight_min_variance=True,
    highlight_max_sharpe=True,
    figsize=(10, 6)
)

# Plot risk decomposition
plot_risk_decomposition(
    weights,
    problem.get_covariance_matrix(),
    problem.factor_exposures,
    problem.factor_returns,
    figsize=(10, 6)
)
```

## Troubleshooting

### Common Issues

1. **Solver Not Converging**
   - Increase `max_iterations`
   - Adjust `initial_penalty` and `penalty_growth_factor`
   - Check constraint feasibility
   - Try a different solver backend

2. **Memory Issues with Large Problems**
   - Use factor models to reduce dimensionality
   - Try the `MultiThreadedSolver` with fewer threads
   - Reduce the problem size for testing
   - Use sparse matrix representations

3. **Numerical Stability Issues**
   - Normalize input data
   - Use shrinkage for covariance estimation
   - Increase solver tolerance
   - Try a more robust solver backend

## Getting Help

If you encounter issues:

1. Check the [API Reference](../reference/api-reference.md) for detailed documentation
2. Look at the [Examples](../examples/minimum-variance-portfolio.md) for similar use cases
3. Consult the [Troubleshooting Guide](../developer-guides/troubleshooting.md)
4. Ask for help in the [GitHub Issues](https://github.com/example/portopt/issues)
