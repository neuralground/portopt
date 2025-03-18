# Portfolio Optimization Workflow Guide

This guide outlines the typical workflow for performing portfolio optimization using the Portfolio Optimization Testbed. It provides a step-by-step approach to help you understand the process and make the most of the library's capabilities.

## Workflow Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  1. Data        │────►│  2. Problem     │────►│  3. Constraints │
│  Preparation    │     │  Definition     │     │  & Objectives   │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         │
┌─────────────────┐     ┌─────────────────┐     ┌────────▼────────┐
│                 │     │                 │     │                 │
│  6. Analysis    │◄────│  5. Results     │◄────│  4. Solver      │
│  & Visualization│     │  Processing     │     │  Execution      │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Detailed Workflow Steps

### 1. Data Preparation

The first step in any portfolio optimization process is to prepare your data. This typically involves:

- Collecting historical price data for your assets
- Calculating returns from price data
- Handling missing values and outliers
- Optionally preparing additional data like volumes, market impact models, etc.

```python
import numpy as np
import pandas as pd
from portopt.utils.data import load_data, clean_data, calculate_returns

# Load historical price data
prices = load_data('path/to/price_data.csv')

# Clean the data (handle missing values, outliers)
prices = clean_data(prices)

# Calculate returns
returns = calculate_returns(prices, method='log')

# Optional: Load or calculate additional data
volumes = load_data('path/to/volume_data.csv')
```

Alternatively, you can use the test data generator for experimentation:

```python
from portopt.utils.data import TestDataGenerator

# Create a data generator
generator = TestDataGenerator(seed=42)

# Generate synthetic data
returns, volumes = generator.generate_returns_and_volumes(
    n_assets=10,
    n_periods=252,  # One year of daily data
    volatility_range=(0.1, 0.3),
    correlation_range=(0.1, 0.7)
)
```

### 2. Problem Definition

Once your data is prepared, you need to define the portfolio optimization problem:

```python
from portopt.core.problem import PortfolioOptProblem

# Create a problem instance
problem = PortfolioOptProblem(
    returns=returns,
    volumes=volumes,
    initial_weights=None,  # Optional starting point
    risk_free_rate=0.02,   # Optional risk-free rate
    transaction_costs=0.001  # Optional transaction costs
)

# Validate the problem
problem.validate()

# Examine problem properties
print(f"Number of assets: {problem.n_assets}")
print(f"Number of periods: {problem.n_periods}")
print(f"Covariance matrix:\n{problem.get_covariance()}")
```

### 3. Constraints and Objectives

Define the constraints and objectives for your optimization problem:

```python
from portopt.constraints.basic import (
    FullInvestmentConstraint,
    LongOnlyConstraint,
    MaxWeightConstraint,
    MinWeightConstraint,
    SectorConstraint
)
from portopt.core.objective import (
    MinimumVarianceObjective,
    MaximumSharpeRatioObjective,
    MeanVarianceObjective
)

# Define constraints
constraints = [
    FullInvestmentConstraint(),  # Sum of weights = 1
    LongOnlyConstraint(),        # No short selling (weights >= 0)
    MaxWeightConstraint(max_weight=0.2),  # No asset can exceed 20% of portfolio
    # Add sector constraints if needed
    SectorConstraint(
        sector_mapping={0: 'Tech', 1: 'Tech', 2: 'Finance', 3: 'Finance', 4: 'Energy'},
        min_weights={'Tech': 0.1, 'Finance': 0.1, 'Energy': 0.05},
        max_weights={'Tech': 0.5, 'Finance': 0.4, 'Energy': 0.3}
    )
]

# Define objective function
objective = MinimumVarianceObjective()  # Minimize portfolio variance

# Alternatively, use maximum Sharpe ratio objective
# objective = MaximumSharpeRatioObjective(risk_free_rate=0.02)

# Or use mean-variance objective with risk aversion parameter
# objective = MeanVarianceObjective(risk_aversion=2.0)
```

### 4. Solver Execution

Execute the solver to find the optimal portfolio:

```python
from portopt.solvers.classical import ClassicalSolver
from portopt.solvers.risk_parity import RiskParitySolver

# Create a solver
solver = ClassicalSolver(
    max_iterations=100,
    tolerance=1e-8,
    verbose=True
)

# Solve the problem
result = solver.solve(
    problem=problem,
    constraints=constraints,
    objective=objective,
    initial_weights=None  # Optional starting point
)

# Check if optimization was successful
if result.success:
    print("Optimization successful!")
    print(f"Optimal weights: {result.weights}")
    print(f"Objective value: {result.objective_value}")
    print(f"Number of iterations: {result.n_iterations}")
else:
    print(f"Optimization failed: {result.message}")
```

For specific types of portfolios, you might use specialized solvers:

```python
# For risk parity portfolios
risk_parity_solver = RiskParitySolver(max_iterations=100, tolerance=1e-8)
risk_parity_result = risk_parity_solver.solve(problem)
```

### 5. Results Processing

Process the optimization results to extract useful information:

```python
from portopt.metrics.performance import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_maximum_drawdown
)
from portopt.metrics.risk import (
    calculate_portfolio_volatility,
    calculate_value_at_risk,
    calculate_conditional_value_at_risk,
    calculate_risk_contribution
)

# Extract optimal weights
weights = result.weights

# Calculate expected return and risk
expected_return = np.dot(problem.get_expected_returns(), weights)
volatility = calculate_portfolio_volatility(weights, problem.get_covariance())

# Calculate performance metrics
sharpe_ratio = calculate_sharpe_ratio(
    returns=np.dot(problem.returns, weights),
    risk_free_rate=problem.risk_free_rate
)

sortino_ratio = calculate_sortino_ratio(
    returns=np.dot(problem.returns, weights),
    risk_free_rate=problem.risk_free_rate
)

# Calculate risk metrics
var_95 = calculate_value_at_risk(
    returns=np.dot(problem.returns, weights),
    confidence_level=0.95
)

cvar_95 = calculate_conditional_value_at_risk(
    returns=np.dot(problem.returns, weights),
    confidence_level=0.95
)

# Calculate risk contributions
risk_contrib = calculate_risk_contribution(weights, problem.get_covariance())
risk_contrib_pct = risk_contrib / sum(risk_contrib)

print(f"Expected annual return: {expected_return * 252:.4f}")
print(f"Expected annual volatility: {volatility * np.sqrt(252):.4f}")
print(f"Sharpe ratio: {sharpe_ratio:.4f}")
print(f"Sortino ratio: {sortino_ratio:.4f}")
print(f"95% VaR: {var_95:.4f}")
print(f"95% CVaR: {cvar_95:.4f}")
print("Risk contribution percentages:")
for i, rc in enumerate(risk_contrib_pct):
    print(f"Asset {i}: {rc:.4f}")
```

### 6. Analysis and Visualization

Analyze and visualize the results to gain insights:

```python
import matplotlib.pyplot as plt
import seaborn as sns
from portopt.utils.visualization import (
    plot_efficient_frontier,
    plot_weights,
    plot_risk_contribution,
    plot_performance
)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Plot portfolio weights
plot_weights(weights, title="Optimal Portfolio Weights")

# Plot risk contribution
plot_risk_contribution(weights, problem.get_covariance(), title="Risk Contribution")

# Generate and plot the efficient frontier
plot_efficient_frontier(
    problem=problem,
    constraints=constraints,
    n_points=20,
    highlight_portfolio=weights,
    highlight_label="Optimal Portfolio",
    risk_free_rate=problem.risk_free_rate
)

# Plot portfolio performance over time
historical_returns = np.dot(problem.returns, weights)
plot_performance(
    returns=historical_returns,
    benchmark_returns=problem.returns[:, 0],  # Using first asset as benchmark
    title="Portfolio Performance",
    benchmark_label="Benchmark"
)

plt.tight_layout()
plt.show()
```

## Advanced Workflow Considerations

### Backtesting

To evaluate how your portfolio would have performed historically:

```python
from portopt.utils.backtest import backtest_portfolio

# Define a rebalancing strategy
def rebalance_strategy(historical_data, current_index):
    # Use data up to current_index to make decisions
    historical_returns = historical_data[:current_index]
    
    # Create problem
    problem = PortfolioOptProblem(returns=historical_returns)
    
    # Solve
    solver = ClassicalSolver()
    result = solver.solve(
        problem=problem,
        constraints=[FullInvestmentConstraint(), LongOnlyConstraint()],
        objective=MinimumVarianceObjective()
    )
    
    return result.weights

# Run backtest
backtest_results = backtest_portfolio(
    returns=returns,
    rebalance_strategy=rebalance_strategy,
    rebalance_frequency=21,  # Monthly (assuming 21 trading days)
    transaction_costs=0.001
)

# Analyze backtest results
print(f"Cumulative return: {backtest_results['cumulative_return']:.4f}")
print(f"Annualized return: {backtest_results['annualized_return']:.4f}")
print(f"Annualized volatility: {backtest_results['annualized_volatility']:.4f}")
print(f"Sharpe ratio: {backtest_results['sharpe_ratio']:.4f}")
print(f"Maximum drawdown: {backtest_results['maximum_drawdown']:.4f}")
```

### Sensitivity Analysis

Perform sensitivity analysis to understand how your portfolio responds to changes in inputs:

```python
from portopt.utils.sensitivity import perform_sensitivity_analysis

# Define parameter ranges
parameter_ranges = {
    'risk_aversion': [1.0, 2.0, 3.0, 4.0, 5.0],
    'max_weight': [0.1, 0.2, 0.3, 0.4, 0.5]
}

# Perform sensitivity analysis
sensitivity_results = perform_sensitivity_analysis(
    problem=problem,
    base_constraints=constraints,
    base_objective=objective,
    parameter_ranges=parameter_ranges,
    solver=solver
)

# Plot sensitivity results
plt.figure(figsize=(12, 8))
for param, results in sensitivity_results.items():
    plt.subplot(1, len(parameter_ranges), list(parameter_ranges.keys()).index(param) + 1)
    plt.plot(list(parameter_ranges[param]), [r['sharpe_ratio'] for r in results])
    plt.xlabel(param)
    plt.ylabel('Sharpe Ratio')
    plt.title(f'Sensitivity to {param}')

plt.tight_layout()
plt.show()
```

### Multi-Period Optimization

For more advanced use cases, consider multi-period optimization:

```python
from portopt.core.multi_period import MultiPeriodOptProblem
from portopt.solvers.multi_period import MultiPeriodSolver

# Create a multi-period problem
multi_period_problem = MultiPeriodOptProblem(
    returns=returns,
    volumes=volumes,
    n_periods=4,  # Number of future periods to optimize for
    transaction_costs=0.001
)

# Create a multi-period solver
multi_period_solver = MultiPeriodSolver(max_iterations=100, tolerance=1e-8)

# Solve the multi-period problem
multi_period_result = multi_period_solver.solve(
    problem=multi_period_problem,
    constraints=constraints,
    objective=objective
)

# Extract the optimal weights for each period
for period, weights in enumerate(multi_period_result.weights_by_period):
    print(f"Period {period} optimal weights: {weights}")
```

## Common Workflow Patterns

### Pattern 1: Minimum Variance Portfolio

```python
# Create problem
problem = PortfolioOptProblem(returns=returns)

# Define constraints
constraints = [FullInvestmentConstraint(), LongOnlyConstraint()]

# Define objective
objective = MinimumVarianceObjective()

# Solve
solver = ClassicalSolver()
result = solver.solve(problem, constraints, objective)
```

### Pattern 2: Maximum Sharpe Ratio Portfolio

```python
# Create problem
problem = PortfolioOptProblem(returns=returns, risk_free_rate=0.02)

# Define constraints
constraints = [FullInvestmentConstraint(), LongOnlyConstraint()]

# Define objective
objective = MaximumSharpeRatioObjective(risk_free_rate=0.02)

# Solve
solver = ClassicalSolver()
result = solver.solve(problem, constraints, objective)
```

### Pattern 3: Risk Parity Portfolio

```python
# Create problem
problem = PortfolioOptProblem(returns=returns)

# Use specialized risk parity solver
solver = RiskParitySolver()
result = solver.solve(problem)
```

### Pattern 4: Target Return Portfolio

```python
# Create problem
problem = PortfolioOptProblem(returns=returns)

# Define constraints including target return
constraints = [
    FullInvestmentConstraint(),
    LongOnlyConstraint(),
    TargetReturnConstraint(target_return=0.1)  # 10% target return
]

# Define objective to minimize variance
objective = MinimumVarianceObjective()

# Solve
solver = ClassicalSolver()
result = solver.solve(problem, constraints, objective)
```

## Workflow Tips and Best Practices

1. **Start Simple**: Begin with simple constraints and objectives, then gradually add complexity.

2. **Validate Inputs**: Always validate your input data before optimization to catch issues early.

3. **Check Feasibility**: If optimization fails, check if your constraints create a feasible solution space.

4. **Use Warm Starting**: For related problems, use previous solutions as starting points to speed up convergence.

5. **Regularize**: Consider adding regularization to prevent extreme allocations:

   ```python
   from portopt.constraints.advanced import L2RegularizationConstraint
   
   constraints.append(L2RegularizationConstraint(lambda_reg=0.1))
   ```

6. **Sensitivity Testing**: Always test how sensitive your optimal portfolio is to changes in inputs.

7. **Out-of-Sample Testing**: Validate your strategy on out-of-sample data to avoid overfitting.

8. **Benchmark Comparison**: Always compare your optimized portfolio against relevant benchmarks.

9. **Transaction Costs**: Include transaction costs in your optimization for more realistic results.

10. **Rebalancing Frequency**: Consider the trade-off between frequent rebalancing (tracking the optimal portfolio) and transaction costs.

## Conclusion

This workflow guide provides a structured approach to portfolio optimization using the Portfolio Optimization Testbed. By following these steps and best practices, you can develop effective portfolio strategies tailored to your specific investment objectives.

For more detailed information on specific components, refer to:
- [API Reference](../reference/api-reference.md)
- [Examples](../examples/README.md)
- [Glossary](../reference/glossary.md)
