# Maximum Sharpe Ratio Portfolio Example

This example demonstrates how to construct a portfolio that maximizes the Sharpe ratio using the Portfolio Optimization Testbed.

## Introduction

The Sharpe ratio is a measure of risk-adjusted return, calculated as the ratio of excess return to volatility. A portfolio with a higher Sharpe ratio provides more return for the same amount of risk. The maximum Sharpe ratio portfolio is the portfolio on the efficient frontier with the highest risk-adjusted return.

## Mathematical Formulation

The maximum Sharpe ratio portfolio can be formulated as:

$$
\begin{align}
\max_w \quad & \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma w}} \\
\text{subject to} \quad & \sum_{i=1}^{n} w_i = 1 \\
& w_i \geq 0, \quad i = 1, 2, \ldots, n
\end{align}
$$

Where:
- $w$ is the vector of portfolio weights
- $\mu$ is the vector of expected returns
- $r_f$ is the risk-free rate
- $\Sigma$ is the covariance matrix of asset returns
- $n$ is the number of assets

## Implementation

### Step 1: Import Required Modules

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from portopt.core.problem import PortfolioOptProblem
from portopt.core.objective import MaximumSharpeRatioObjective
from portopt.constraints.basic import FullInvestmentConstraint, LongOnlyConstraint
from portopt.solvers.classical import ClassicalSolver
from portopt.metrics.performance import calculate_portfolio_return, calculate_sharpe_ratio
from portopt.metrics.risk import calculate_portfolio_volatility
from portopt.utils.plotting import plot_efficient_frontier
```

### Step 2: Create or Load Data

For this example, we'll generate synthetic data using the `TestDataGenerator`:

```python
from portopt.utils.data import TestDataGenerator

# Create a data generator with a fixed seed for reproducibility
generator = TestDataGenerator(seed=42)

# Generate a realistic problem with 20 assets and 252 periods (1 year of daily returns)
problem = generator.generate_realistic_problem(
    n_assets=20,
    n_periods=252,
    n_factors=5,
    n_industries=8
)

# Extract the returns data
returns = problem.returns
```

Alternatively, you can load your own data:

```python
# Load returns data from a CSV file
returns_df = pd.read_csv('asset_returns.csv', index_col=0, parse_dates=True)
returns = returns_df.values

# Create a problem instance
problem = PortfolioOptProblem(returns=returns)
```

### Step 3: Define Constraints

For a basic maximum Sharpe ratio portfolio, we'll use two constraints:
1. Full investment constraint (weights sum to 1)
2. Long-only constraint (no short selling)

```python
constraints = [
    FullInvestmentConstraint(),
    LongOnlyConstraint()
]
```

### Step 4: Define the Objective

We'll use the `MaximumSharpeRatioObjective` with a risk-free rate of 0.02 (2%):

```python
risk_free_rate = 0.02
objective = MaximumSharpeRatioObjective(risk_free_rate=risk_free_rate)
```

### Step 5: Solve the Optimization Problem

```python
# Create a solver
solver = ClassicalSolver(max_iterations=100, tolerance=1e-6)

# Solve the problem
result = solver.solve(problem, constraints=constraints, objective=objective)

# Extract the optimal weights
weights = result.weights
```

### Step 6: Analyze the Results

```python
# Calculate expected return
expected_returns = problem.get_expected_returns()
portfolio_return = calculate_portfolio_return(weights, expected_returns)

# Calculate portfolio volatility
covariance_matrix = problem.get_covariance_matrix()
portfolio_volatility = calculate_portfolio_volatility(weights, covariance_matrix)

# Calculate Sharpe ratio
sharpe_ratio = calculate_sharpe_ratio(
    portfolio_return, 
    portfolio_volatility, 
    risk_free_rate
)

print(f"Portfolio Expected Return: {portfolio_return:.4f}")
print(f"Portfolio Volatility: {portfolio_volatility:.4f}")
print(f"Portfolio Sharpe Ratio: {sharpe_ratio:.4f}")
```

### Step 7: Visualize the Results

#### Plot Asset Weights

```python
plt.figure(figsize=(12, 6))
plt.bar(range(len(weights)), weights)
plt.xlabel('Asset')
plt.ylabel('Weight')
plt.title('Maximum Sharpe Ratio Portfolio Weights')
plt.xticks(range(len(weights)), [f'Asset {i+1}' for i in range(len(weights))])
plt.grid(True, alpha=0.3)
plt.show()
```

#### Plot Efficient Frontier with Maximum Sharpe Ratio Portfolio

```python
# Generate points on the efficient frontier
n_points = 50
min_return = np.min(expected_returns)
max_return = np.max(expected_returns)
target_returns = np.linspace(min_return, max_return, n_points)

# Calculate efficient frontier
frontier_returns = []
frontier_volatilities = []

for target_return in target_returns:
    # Define mean-variance objective with target return
    mv_objective = MeanVarianceObjective(target_return=target_return)
    
    # Solve the problem
    mv_result = solver.solve(problem, constraints=constraints, objective=mv_objective)
    
    # Calculate portfolio return and volatility
    mv_weights = mv_result.weights
    mv_return = calculate_portfolio_return(mv_weights, expected_returns)
    mv_volatility = calculate_portfolio_volatility(mv_weights, covariance_matrix)
    
    frontier_returns.append(mv_return)
    frontier_volatilities.append(mv_volatility)

# Plot efficient frontier
plt.figure(figsize=(10, 6))
plt.plot(frontier_volatilities, frontier_returns, 'b-', label='Efficient Frontier')

# Plot maximum Sharpe ratio portfolio
plt.scatter(portfolio_volatility, portfolio_return, color='red', marker='*', s=200, 
            label=f'Maximum Sharpe Ratio Portfolio (SR={sharpe_ratio:.2f})')

# Plot risk-free rate
plt.axhline(y=risk_free_rate, color='green', linestyle='--', 
            label=f'Risk-Free Rate ({risk_free_rate:.2f})')

# Plot capital allocation line
max_volatility = max(frontier_volatilities) * 1.2
cal_volatilities = np.linspace(0, max_volatility, 100)
cal_returns = risk_free_rate + (portfolio_return - risk_free_rate) / portfolio_volatility * cal_volatilities
plt.plot(cal_volatilities, cal_returns, 'r--', label='Capital Allocation Line')

plt.xlabel('Volatility (Standard Deviation)')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier and Maximum Sharpe Ratio Portfolio')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Complete Example

Here's the complete code for the maximum Sharpe ratio portfolio optimization:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from portopt.core.problem import PortfolioOptProblem
from portopt.core.objective import MaximumSharpeRatioObjective, MeanVarianceObjective
from portopt.constraints.basic import FullInvestmentConstraint, LongOnlyConstraint
from portopt.solvers.classical import ClassicalSolver
from portopt.metrics.performance import calculate_portfolio_return, calculate_sharpe_ratio
from portopt.metrics.risk import calculate_portfolio_volatility
from portopt.utils.data import TestDataGenerator

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
generator = TestDataGenerator(seed=42)
problem = generator.generate_realistic_problem(
    n_assets=20,
    n_periods=252,
    n_factors=5,
    n_industries=8
)

# Define constraints
constraints = [
    FullInvestmentConstraint(),
    LongOnlyConstraint()
]

# Define objective
risk_free_rate = 0.02
objective = MaximumSharpeRatioObjective(risk_free_rate=risk_free_rate)

# Create solver
solver = ClassicalSolver(max_iterations=100, tolerance=1e-6)

# Solve the problem
result = solver.solve(problem, constraints=constraints, objective=objective)
weights = result.weights

# Analyze results
expected_returns = problem.get_expected_returns()
covariance_matrix = problem.get_covariance_matrix()

portfolio_return = calculate_portfolio_return(weights, expected_returns)
portfolio_volatility = calculate_portfolio_volatility(weights, covariance_matrix)
sharpe_ratio = calculate_sharpe_ratio(
    portfolio_return, 
    portfolio_volatility, 
    risk_free_rate
)

print(f"Portfolio Expected Return: {portfolio_return:.4f}")
print(f"Portfolio Volatility: {portfolio_volatility:.4f}")
print(f"Portfolio Sharpe Ratio: {sharpe_ratio:.4f}")

# Plot asset weights
plt.figure(figsize=(12, 6))
plt.bar(range(len(weights)), weights)
plt.xlabel('Asset')
plt.ylabel('Weight')
plt.title('Maximum Sharpe Ratio Portfolio Weights')
plt.xticks(range(len(weights)), [f'Asset {i+1}' for i in range(len(weights))])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('max_sharpe_weights.png')
plt.show()

# Generate points on the efficient frontier
n_points = 50
min_return = np.min(expected_returns)
max_return = np.max(expected_returns)
target_returns = np.linspace(min_return, max_return, n_points)

# Calculate efficient frontier
frontier_returns = []
frontier_volatilities = []

for target_return in target_returns:
    # Define mean-variance objective with target return
    mv_objective = MeanVarianceObjective(target_return=target_return)
    
    # Solve the problem
    mv_result = solver.solve(problem, constraints=constraints, objective=mv_objective)
    
    # Calculate portfolio return and volatility
    mv_weights = mv_result.weights
    mv_return = calculate_portfolio_return(mv_weights, expected_returns)
    mv_volatility = calculate_portfolio_volatility(mv_weights, covariance_matrix)
    
    frontier_returns.append(mv_return)
    frontier_volatilities.append(mv_volatility)

# Plot efficient frontier
plt.figure(figsize=(10, 6))
plt.plot(frontier_volatilities, frontier_returns, 'b-', label='Efficient Frontier')

# Plot maximum Sharpe ratio portfolio
plt.scatter(portfolio_volatility, portfolio_return, color='red', marker='*', s=200, 
            label=f'Maximum Sharpe Ratio Portfolio (SR={sharpe_ratio:.2f})')

# Plot risk-free rate
plt.axhline(y=risk_free_rate, color='green', linestyle='--', 
            label=f'Risk-Free Rate ({risk_free_rate:.2f})')

# Plot capital allocation line
max_volatility = max(frontier_volatilities) * 1.2
cal_volatilities = np.linspace(0, max_volatility, 100)
cal_returns = risk_free_rate + (portfolio_return - risk_free_rate) / portfolio_volatility * cal_volatilities
plt.plot(cal_volatilities, cal_returns, 'r--', label='Capital Allocation Line')

plt.xlabel('Volatility (Standard Deviation)')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier and Maximum Sharpe Ratio Portfolio')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('max_sharpe_efficient_frontier.png')
plt.show()
```

## Expected Output

When you run this example, you should see output similar to:

```
Portfolio Expected Return: 0.1245
Portfolio Volatility: 0.1876
Portfolio Sharpe Ratio: 0.5572
```

And two plots:
1. A bar chart showing the weights of each asset in the maximum Sharpe ratio portfolio
2. A plot of the efficient frontier with the maximum Sharpe ratio portfolio highlighted

## Variations and Extensions

### Adding Sector Constraints

You can add sector constraints to limit exposure to specific sectors:

```python
from portopt.constraints.sector import SectorConstraint

# Assuming industry_classes is an array of industry classifications for each asset
industry_classes = problem.industry_classes

# Define sector constraint with maximum 20% exposure to any sector
sector_constraint = SectorConstraint(
    sector_ids=industry_classes,
    max_weight=0.2
)

# Add to constraints list
constraints.append(sector_constraint)
```

### Adding Position Limits

You can add position limits to constrain individual asset weights:

```python
from portopt.constraints.basic import PositionLimitConstraint

# Limit each asset to between 1% and 10% of the portfolio
position_limit_constraint = PositionLimitConstraint(
    min_weight=0.01,
    max_weight=0.1
)

# Add to constraints list
constraints.append(position_limit_constraint)
```

### Using Factor Models for Covariance Estimation

You can use factor models for more robust covariance estimation:

```python
# Get covariance matrix using factor model
covariance_matrix = problem.get_covariance_matrix(method="factor")
```

## Conclusion

This example demonstrated how to construct a maximum Sharpe ratio portfolio using the Portfolio Optimization Testbed. The maximum Sharpe ratio portfolio represents the optimal trade-off between risk and return, providing the highest risk-adjusted return among all portfolios on the efficient frontier.

## Related Resources

- [Portfolio Tutorial](../concepts/portfolio-tutorial.md)
- [Risk Metrics](../concepts/risk-metrics.md)
- [Minimum Variance Portfolio Example](./minimum-variance-portfolio.md)
- [API Reference](../reference/api-reference.md)
- [Glossary](../reference/glossary.md)
