# Minimum Variance Portfolio Example

This example demonstrates how to use the Portfolio Optimization Testbed to construct a minimum variance portfolio.

## Overview

A minimum variance portfolio is a portfolio that has the lowest possible risk (measured by variance) for a given set of assets. This is a common optimization objective in portfolio management, especially for risk-averse investors.

## Prerequisites

Before running this example, make sure you have:

1. Installed the Portfolio Optimization Testbed (see [Installation Guide](../getting-started/installation.md))
2. Basic understanding of portfolio optimization concepts (see [Portfolio Tutorial](../concepts/portfolio-tutorial.md))

## Step 1: Import Required Modules

First, let's import the necessary modules:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from portopt import TestDataGenerator
from portopt.core.problem import PortfolioOptProblem
from portopt.core.objective import MinimumVarianceObjective
from portopt.constraints.basic import FullInvestmentConstraint, LongOnlyConstraint
from portopt.solvers.classical import ClassicalSolver
from portopt.metrics.risk import calculate_portfolio_volatility
from portopt.metrics.performance import calculate_portfolio_return
from portopt.utils.logging import setup_logging
```

## Step 2: Set Up Logging

Set up logging to monitor the optimization process:

```python
setup_logging(level="INFO")
```

## Step 3: Generate Test Data

For this example, we'll use synthetic data. In a real-world scenario, you would use actual market data.

```python
# Create a test data generator with a fixed seed for reproducibility
generator = TestDataGenerator(seed=42)

# Generate a problem with 20 assets and 252 trading days (approximately 1 year)
problem = generator.generate_realistic_problem(
    n_assets=20,
    n_periods=252,
    volatility_range=(0.1, 0.4),  # Annual volatility between 10% and 40%
    correlation_range=(0.1, 0.7)  # Correlations between 0.1 and 0.7
)

# Get the returns data
returns = problem.returns

# Print basic statistics
print(f"Number of assets: {problem.n_assets}")
print(f"Number of time periods: {problem.n_periods}")
print(f"Mean returns: {np.mean(returns, axis=0)}")
print(f"Asset volatilities: {np.std(returns, axis=0) * np.sqrt(252)}")
```

## Step 4: Define Constraints

For a minimum variance portfolio, we'll use two basic constraints:

1. Full investment: The sum of weights must equal 1
2. Long-only: All weights must be non-negative

```python
constraints = [
    FullInvestmentConstraint(),  # Sum of weights = 1
    LongOnlyConstraint()         # All weights >= 0
]
```

## Step 5: Define the Objective

We want to minimize the portfolio variance:

```python
objective = MinimumVarianceObjective()
```

## Step 6: Create and Configure the Solver

```python
solver = ClassicalSolver(
    max_iterations=100,    # Maximum number of iterations
    tolerance=1e-6,        # Convergence tolerance
    initial_penalty=10.0,  # Initial penalty parameter
    penalty_growth_factor=2.0  # How quickly the penalty increases
)
```

## Step 7: Solve the Optimization Problem

```python
# Solve the problem
result = solver.solve(
    problem=problem,
    constraints=constraints,
    objective=objective
)

# Print the result summary
print(f"Converged: {result.converged}")
print(f"Number of iterations: {result.iterations}")
print(f"Objective value: {result.objective_value:.6f}")
```

## Step 8: Analyze the Results

Let's analyze the optimized portfolio:

```python
# Get the optimized weights
weights = result.weights

# Print the weights
print("\nPortfolio Weights:")
for i, weight in enumerate(weights):
    print(f"Asset {i+1}: {weight:.4f}")

# Calculate portfolio metrics
cov_matrix = problem.get_covariance_matrix()
portfolio_volatility = calculate_portfolio_volatility(weights, cov_matrix)
portfolio_annual_volatility = portfolio_volatility * np.sqrt(252)
portfolio_return = calculate_portfolio_return(weights, np.mean(returns, axis=0))
portfolio_annual_return = portfolio_return * 252

print(f"\nPortfolio Metrics:")
print(f"Annual Return: {portfolio_annual_return:.4f}")
print(f"Annual Volatility: {portfolio_annual_volatility:.4f}")
print(f"Sharpe Ratio (assuming 0% risk-free rate): {portfolio_annual_return / portfolio_annual_volatility:.4f}")
```

## Step 9: Visualize the Portfolio

Let's create some visualizations to better understand the portfolio:

```python
# Create a pie chart of the portfolio weights
plt.figure(figsize=(10, 6))
plt.pie(weights, labels=[f"Asset {i+1}" for i in range(len(weights))], 
        autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Minimum Variance Portfolio Allocation')
plt.tight_layout()
plt.savefig('minimum_variance_allocation.png')
plt.show()

# Create a bar chart of asset volatilities vs. weights
plt.figure(figsize=(12, 6))
asset_vols = np.std(returns, axis=0) * np.sqrt(252)
sorted_indices = np.argsort(asset_vols)
sorted_vols = asset_vols[sorted_indices]
sorted_weights = weights[sorted_indices]

x = np.arange(len(sorted_vols))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, sorted_vols, width, label='Annual Volatility')
ax.bar(x + width/2, sorted_weights, width, label='Portfolio Weight')

ax.set_xlabel('Assets (sorted by volatility)')
ax.set_ylabel('Value')
ax.set_title('Asset Volatilities vs. Portfolio Weights')
ax.set_xticks(x)
ax.set_xticklabels([f"Asset {i+1}" for i in sorted_indices])
ax.legend()

plt.tight_layout()
plt.savefig('volatility_vs_weights.png')
plt.show()
```

## Step 10: Compare with Equal-Weight Portfolio

Let's compare our minimum variance portfolio with a simple equal-weight portfolio:

```python
# Create equal-weight portfolio
equal_weights = np.ones(problem.n_assets) / problem.n_assets

# Calculate metrics for equal-weight portfolio
equal_portfolio_volatility = calculate_portfolio_volatility(equal_weights, cov_matrix)
equal_portfolio_annual_volatility = equal_portfolio_volatility * np.sqrt(252)
equal_portfolio_return = calculate_portfolio_return(equal_weights, np.mean(returns, axis=0))
equal_portfolio_annual_return = equal_portfolio_return * 252

print("\nComparison with Equal-Weight Portfolio:")
print(f"Minimum Variance Portfolio Annual Volatility: {portfolio_annual_volatility:.4f}")
print(f"Equal-Weight Portfolio Annual Volatility: {equal_portfolio_annual_volatility:.4f}")
print(f"Volatility Reduction: {(1 - portfolio_annual_volatility / equal_portfolio_annual_volatility) * 100:.2f}%")

print(f"\nMinimum Variance Portfolio Annual Return: {portfolio_annual_return:.4f}")
print(f"Equal-Weight Portfolio Annual Return: {equal_portfolio_annual_return:.4f}")

print(f"\nMinimum Variance Portfolio Sharpe Ratio: {portfolio_annual_return / portfolio_annual_volatility:.4f}")
print(f"Equal-Weight Portfolio Sharpe Ratio: {equal_portfolio_annual_return / equal_portfolio_annual_volatility:.4f}")
```

## Step 11: Efficient Frontier Analysis

Let's generate the efficient frontier to see where our minimum variance portfolio lies:

```python
from portopt.core.objective import MeanVarianceObjective

# Generate a range of target returns
min_return = np.min(np.mean(returns, axis=0))
max_return = np.max(np.mean(returns, axis=0))
target_returns = np.linspace(min_return, max_return, 50)

# Store the results
efficient_frontier = []

# For each target return, find the minimum variance portfolio
for target_return in target_returns:
    # Create a mean-variance objective with the target return
    mv_objective = MeanVarianceObjective(
        target_return=target_return,
        return_weight=1.0,
        risk_weight=1.0
    )
    
    # Solve the problem
    mv_result = solver.solve(
        problem=problem,
        constraints=constraints,
        objective=mv_objective
    )
    
    if mv_result.converged:
        # Calculate portfolio metrics
        mv_weights = mv_result.weights
        mv_volatility = calculate_portfolio_volatility(mv_weights, cov_matrix)
        mv_return = calculate_portfolio_return(mv_weights, np.mean(returns, axis=0))
        
        efficient_frontier.append({
            'target_return': target_return,
            'return': mv_return,
            'volatility': mv_volatility,
            'weights': mv_weights
        })

# Convert to DataFrame for easier analysis
ef_df = pd.DataFrame(efficient_frontier)

# Plot the efficient frontier
plt.figure(figsize=(10, 6))
plt.plot(ef_df['volatility'], ef_df['return'], 'b-', label='Efficient Frontier')
plt.scatter(portfolio_volatility, portfolio_return, color='r', marker='*', s=200, label='Minimum Variance Portfolio')
plt.scatter(equal_portfolio_volatility, equal_portfolio_return, color='g', marker='o', s=100, label='Equal-Weight Portfolio')

# Add labels and title
plt.xlabel('Portfolio Volatility')
plt.ylabel('Portfolio Return')
plt.title('Efficient Frontier')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('efficient_frontier.png')
plt.show()
```

## Complete Example Code

Here's the complete code for this example:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from portopt import TestDataGenerator
from portopt.core.problem import PortfolioOptProblem
from portopt.core.objective import MinimumVarianceObjective, MeanVarianceObjective
from portopt.constraints.basic import FullInvestmentConstraint, LongOnlyConstraint
from portopt.solvers.classical import ClassicalSolver
from portopt.metrics.risk import calculate_portfolio_volatility
from portopt.metrics.performance import calculate_portfolio_return
from portopt.utils.logging import setup_logging

# Set up logging
setup_logging(level="INFO")

# Create a test data generator with a fixed seed for reproducibility
generator = TestDataGenerator(seed=42)

# Generate a problem with 20 assets and 252 trading days (approximately 1 year)
problem = generator.generate_realistic_problem(
    n_assets=20,
    n_periods=252,
    volatility_range=(0.1, 0.4),  # Annual volatility between 10% and 40%
    correlation_range=(0.1, 0.7)  # Correlations between 0.1 and 0.7
)

# Get the returns data
returns = problem.returns

# Print basic statistics
print(f"Number of assets: {problem.n_assets}")
print(f"Number of time periods: {problem.n_periods}")
print(f"Mean returns: {np.mean(returns, axis=0)}")
print(f"Asset volatilities: {np.std(returns, axis=0) * np.sqrt(252)}")

# Define constraints
constraints = [
    FullInvestmentConstraint(),  # Sum of weights = 1
    LongOnlyConstraint()         # All weights >= 0
]

# Define the objective
objective = MinimumVarianceObjective()

# Create and configure the solver
solver = ClassicalSolver(
    max_iterations=100,    # Maximum number of iterations
    tolerance=1e-6,        # Convergence tolerance
    initial_penalty=10.0,  # Initial penalty parameter
    penalty_growth_factor=2.0  # How quickly the penalty increases
)

# Solve the problem
result = solver.solve(
    problem=problem,
    constraints=constraints,
    objective=objective
)

# Print the result summary
print(f"Converged: {result.converged}")
print(f"Number of iterations: {result.iterations}")
print(f"Objective value: {result.objective_value:.6f}")

# Get the optimized weights
weights = result.weights

# Print the weights
print("\nPortfolio Weights:")
for i, weight in enumerate(weights):
    print(f"Asset {i+1}: {weight:.4f}")

# Calculate portfolio metrics
cov_matrix = problem.get_covariance_matrix()
portfolio_volatility = calculate_portfolio_volatility(weights, cov_matrix)
portfolio_annual_volatility = portfolio_volatility * np.sqrt(252)
portfolio_return = calculate_portfolio_return(weights, np.mean(returns, axis=0))
portfolio_annual_return = portfolio_return * 252

print(f"\nPortfolio Metrics:")
print(f"Annual Return: {portfolio_annual_return:.4f}")
print(f"Annual Volatility: {portfolio_annual_volatility:.4f}")
print(f"Sharpe Ratio (assuming 0% risk-free rate): {portfolio_annual_return / portfolio_annual_volatility:.4f}")

# Create a pie chart of the portfolio weights
plt.figure(figsize=(10, 6))
plt.pie(weights, labels=[f"Asset {i+1}" for i in range(len(weights))], 
        autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Minimum Variance Portfolio Allocation')
plt.tight_layout()
plt.savefig('minimum_variance_allocation.png')
plt.show()

# Create a bar chart of asset volatilities vs. weights
plt.figure(figsize=(12, 6))
asset_vols = np.std(returns, axis=0) * np.sqrt(252)
sorted_indices = np.argsort(asset_vols)
sorted_vols = asset_vols[sorted_indices]
sorted_weights = weights[sorted_indices]

x = np.arange(len(sorted_vols))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, sorted_vols, width, label='Annual Volatility')
ax.bar(x + width/2, sorted_weights, width, label='Portfolio Weight')

ax.set_xlabel('Assets (sorted by volatility)')
ax.set_ylabel('Value')
ax.set_title('Asset Volatilities vs. Portfolio Weights')
ax.set_xticks(x)
ax.set_xticklabels([f"Asset {i+1}" for i in sorted_indices])
ax.legend()

plt.tight_layout()
plt.savefig('volatility_vs_weights.png')
plt.show()

# Create equal-weight portfolio
equal_weights = np.ones(problem.n_assets) / problem.n_assets

# Calculate metrics for equal-weight portfolio
equal_portfolio_volatility = calculate_portfolio_volatility(equal_weights, cov_matrix)
equal_portfolio_annual_volatility = equal_portfolio_volatility * np.sqrt(252)
equal_portfolio_return = calculate_portfolio_return(equal_weights, np.mean(returns, axis=0))
equal_portfolio_annual_return = equal_portfolio_return * 252

print("\nComparison with Equal-Weight Portfolio:")
print(f"Minimum Variance Portfolio Annual Volatility: {portfolio_annual_volatility:.4f}")
print(f"Equal-Weight Portfolio Annual Volatility: {equal_portfolio_annual_volatility:.4f}")
print(f"Volatility Reduction: {(1 - portfolio_annual_volatility / equal_portfolio_annual_volatility) * 100:.2f}%")

print(f"\nMinimum Variance Portfolio Annual Return: {portfolio_annual_return:.4f}")
print(f"Equal-Weight Portfolio Annual Return: {equal_portfolio_annual_return:.4f}")

print(f"\nMinimum Variance Portfolio Sharpe Ratio: {portfolio_annual_return / portfolio_annual_volatility:.4f}")
print(f"Equal-Weight Portfolio Sharpe Ratio: {equal_portfolio_annual_return / equal_portfolio_annual_volatility:.4f}")

# Generate a range of target returns
min_return = np.min(np.mean(returns, axis=0))
max_return = np.max(np.mean(returns, axis=0))
target_returns = np.linspace(min_return, max_return, 50)

# Store the results
efficient_frontier = []

# For each target return, find the minimum variance portfolio
for target_return in target_returns:
    # Create a mean-variance objective with the target return
    mv_objective = MeanVarianceObjective(
        target_return=target_return,
        return_weight=1.0,
        risk_weight=1.0
    )
    
    # Solve the problem
    mv_result = solver.solve(
        problem=problem,
        constraints=constraints,
        objective=mv_objective
    )
    
    if mv_result.converged:
        # Calculate portfolio metrics
        mv_weights = mv_result.weights
        mv_volatility = calculate_portfolio_volatility(mv_weights, cov_matrix)
        mv_return = calculate_portfolio_return(mv_weights, np.mean(returns, axis=0))
        
        efficient_frontier.append({
            'target_return': target_return,
            'return': mv_return,
            'volatility': mv_volatility,
            'weights': mv_weights
        })

# Convert to DataFrame for easier analysis
ef_df = pd.DataFrame(efficient_frontier)

# Plot the efficient frontier
plt.figure(figsize=(10, 6))
plt.plot(ef_df['volatility'], ef_df['return'], 'b-', label='Efficient Frontier')
plt.scatter(portfolio_volatility, portfolio_return, color='r', marker='*', s=200, label='Minimum Variance Portfolio')
plt.scatter(equal_portfolio_volatility, equal_portfolio_return, color='g', marker='o', s=100, label='Equal-Weight Portfolio')

# Add labels and title
plt.xlabel('Portfolio Volatility')
plt.ylabel('Portfolio Return')
plt.title('Efficient Frontier')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('efficient_frontier.png')
plt.show()
```

## Expected Output

When you run this example, you should see:

1. Information about the problem size and asset characteristics
2. The optimization results, including convergence status and iterations
3. The portfolio weights for each asset
4. Portfolio metrics such as return, volatility, and Sharpe ratio
5. Comparison with an equal-weight portfolio
6. Visualizations of the portfolio allocation, asset characteristics, and efficient frontier

## Conclusion

This example demonstrates how to use the Portfolio Optimization Testbed to construct a minimum variance portfolio. The key steps are:

1. Define the problem with historical returns data
2. Set up constraints (full investment and long-only)
3. Define the objective (minimum variance)
4. Solve the optimization problem
5. Analyze and visualize the results

The minimum variance portfolio typically has lower risk than an equal-weight portfolio, but may also have lower expected return. The efficient frontier analysis shows the trade-off between risk and return, allowing investors to choose a portfolio that matches their risk preferences.

## Related Examples

- [Maximum Sharpe Ratio Portfolio](./maximum-sharpe-ratio.md)
- [Risk Parity Portfolio](./risk-parity.md)
- [Portfolio with Sector Constraints](./sector-constraints.md)
- [Portfolio with Transaction Costs](./transaction-costs.md)

## Related Resources

- [Quick Start Guide](../getting-started/quick-start.md)
- [API Reference](../reference/api-reference.md)
- [Portfolio Optimization Concepts](../concepts/portfolio-tutorial.md)
