# Risk Parity Portfolio Example

This example demonstrates how to construct a risk parity portfolio using the Portfolio Optimization Testbed.

## Introduction

Risk parity is a portfolio allocation strategy that focuses on allocating risk, rather than capital, equally among assets or asset classes. The core principle is that each asset should contribute equally to the overall portfolio risk. This approach aims to create a more balanced portfolio that is not dominated by the risk of any single asset or asset class.

Unlike traditional portfolio optimization techniques that focus on maximizing returns for a given level of risk, risk parity focuses on risk allocation, making it particularly useful in environments where expected returns are difficult to estimate accurately.

## Mathematical Formulation

In a risk parity portfolio, the risk contribution of each asset should be equal:

$$
\text{RC}_i = \text{RC}_j \quad \forall i, j
$$

where $\text{RC}_i$ is the risk contribution of asset $i$, defined as:

$$
\text{RC}_i = w_i \cdot \frac{\partial \sigma_p}{\partial w_i} = w_i \cdot \frac{(\Sigma w)_i}{\sigma_p}
$$

where:
- $w_i$ is the weight of asset $i$
- $\Sigma$ is the covariance matrix of asset returns
- $\sigma_p$ is the portfolio standard deviation
- $(\Sigma w)_i$ is the $i$-th element of the vector $\Sigma w$

The risk parity portfolio can be formulated as an optimization problem:

$$
\begin{align}
\min_w \quad & \sum_{i=1}^{n} \sum_{j=1}^{n} (w_i (\Sigma w)_i - w_j (\Sigma w)_j)^2 \\
\text{subject to} \quad & \sum_{i=1}^{n} w_i = 1 \\
& w_i \geq 0, \quad i = 1, 2, \ldots, n
\end{align}
$$

## Implementation

### Step 1: Import Required Modules

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from portopt.core.problem import PortfolioOptProblem
from portopt.core.objective import RiskParityObjective
from portopt.constraints.basic import FullInvestmentConstraint, LongOnlyConstraint
from portopt.solvers.classical import ClassicalSolver
from portopt.metrics.risk import calculate_portfolio_volatility, calculate_risk_contribution
from portopt.utils.data import TestDataGenerator
```

### Step 2: Create or Load Data

For this example, we'll generate synthetic data using the `TestDataGenerator`:

```python
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

For a basic risk parity portfolio, we'll use two constraints:
1. Full investment constraint (weights sum to 1)
2. Long-only constraint (no short selling)

```python
constraints = [
    FullInvestmentConstraint(),
    LongOnlyConstraint()
]
```

### Step 4: Define the Objective

We'll use the `RiskParityObjective`:

```python
objective = RiskParityObjective()
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
# Calculate portfolio volatility
covariance_matrix = problem.get_covariance_matrix()
portfolio_volatility = calculate_portfolio_volatility(weights, covariance_matrix)

# Calculate risk contribution of each asset
risk_contributions = calculate_risk_contribution(weights, covariance_matrix)

# Calculate percentage risk contribution
percentage_risk_contribution = risk_contributions / portfolio_volatility

print(f"Portfolio Volatility: {portfolio_volatility:.4f}")
print("\nRisk Contributions:")
for i, rc in enumerate(risk_contributions):
    print(f"Asset {i+1}: {rc:.4f}")

print("\nPercentage Risk Contributions:")
for i, prc in enumerate(percentage_risk_contribution):
    print(f"Asset {i+1}: {prc:.4f}")
```

### Step 7: Visualize the Results

#### Plot Asset Weights vs. Risk Contributions

```python
plt.figure(figsize=(12, 6))

# Create a bar chart with two groups: weights and risk contributions
bar_width = 0.35
index = np.arange(len(weights))

plt.bar(index, weights, bar_width, label='Weight')
plt.bar(index + bar_width, percentage_risk_contribution, bar_width, label='Risk Contribution')

plt.xlabel('Asset')
plt.ylabel('Value')
plt.title('Risk Parity Portfolio: Weights vs. Risk Contributions')
plt.xticks(index + bar_width / 2, [f'Asset {i+1}' for i in range(len(weights))])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

#### Compare with Equal Weight Portfolio

```python
# Create equal weight portfolio
equal_weights = np.ones(len(weights)) / len(weights)

# Calculate risk contribution for equal weight portfolio
equal_weight_risk_contributions = calculate_risk_contribution(equal_weights, covariance_matrix)
equal_weight_percentage_risk_contribution = equal_weight_risk_contributions / np.sum(equal_weight_risk_contributions)

# Plot comparison
plt.figure(figsize=(12, 6))

# Create a bar chart with two groups: risk parity and equal weight
bar_width = 0.35
index = np.arange(len(weights))

plt.bar(index, percentage_risk_contribution, bar_width, label='Risk Parity')
plt.bar(index + bar_width, equal_weight_percentage_risk_contribution, bar_width, label='Equal Weight')

plt.xlabel('Asset')
plt.ylabel('Risk Contribution')
plt.title('Risk Contribution: Risk Parity vs. Equal Weight')
plt.xticks(index + bar_width / 2, [f'Asset {i+1}' for i in range(len(weights))])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Complete Example

Here's the complete code for the risk parity portfolio optimization:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from portopt.core.problem import PortfolioOptProblem
from portopt.core.objective import RiskParityObjective
from portopt.constraints.basic import FullInvestmentConstraint, LongOnlyConstraint
from portopt.solvers.classical import ClassicalSolver
from portopt.metrics.risk import calculate_portfolio_volatility, calculate_risk_contribution
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
objective = RiskParityObjective()

# Create solver
solver = ClassicalSolver(max_iterations=100, tolerance=1e-6)

# Solve the problem
result = solver.solve(problem, constraints=constraints, objective=objective)
weights = result.weights

# Analyze results
covariance_matrix = problem.get_covariance_matrix()
portfolio_volatility = calculate_portfolio_volatility(weights, covariance_matrix)
risk_contributions = calculate_risk_contribution(weights, covariance_matrix)
percentage_risk_contribution = risk_contributions / portfolio_volatility

print(f"Portfolio Volatility: {portfolio_volatility:.4f}")
print("\nRisk Contributions:")
for i, rc in enumerate(risk_contributions):
    print(f"Asset {i+1}: {rc:.4f}")

print("\nPercentage Risk Contributions:")
for i, prc in enumerate(percentage_risk_contribution):
    print(f"Asset {i+1}: {prc:.4f}")

# Plot asset weights vs. risk contributions
plt.figure(figsize=(12, 6))

# Create a bar chart with two groups: weights and risk contributions
bar_width = 0.35
index = np.arange(len(weights))

plt.bar(index, weights, bar_width, label='Weight')
plt.bar(index + bar_width, percentage_risk_contribution, bar_width, label='Risk Contribution')

plt.xlabel('Asset')
plt.ylabel('Value')
plt.title('Risk Parity Portfolio: Weights vs. Risk Contributions')
plt.xticks(index + bar_width / 2, [f'Asset {i+1}' for i in range(len(weights))])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('risk_parity_weights_vs_risk.png')
plt.show()

# Create equal weight portfolio
equal_weights = np.ones(len(weights)) / len(weights)

# Calculate risk contribution for equal weight portfolio
equal_weight_risk_contributions = calculate_risk_contribution(equal_weights, covariance_matrix)
equal_weight_percentage_risk_contribution = equal_weight_risk_contributions / np.sum(equal_weight_risk_contributions)

# Plot comparison
plt.figure(figsize=(12, 6))

# Create a bar chart with two groups: risk parity and equal weight
bar_width = 0.35
index = np.arange(len(weights))

plt.bar(index, percentage_risk_contribution, bar_width, label='Risk Parity')
plt.bar(index + bar_width, equal_weight_percentage_risk_contribution, bar_width, label='Equal Weight')

plt.xlabel('Asset')
plt.ylabel('Risk Contribution')
plt.title('Risk Contribution: Risk Parity vs. Equal Weight')
plt.xticks(index + bar_width / 2, [f'Asset {i+1}' for i in range(len(weights))])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('risk_parity_vs_equal_weight.png')
plt.show()

# Calculate standard deviation of risk contributions
risk_parity_std = np.std(percentage_risk_contribution)
equal_weight_std = np.std(equal_weight_percentage_risk_contribution)

print(f"\nStandard Deviation of Risk Contributions:")
print(f"Risk Parity: {risk_parity_std:.6f}")
print(f"Equal Weight: {equal_weight_std:.6f}")
```

## Expected Output

When you run this example, you should see output similar to:

```
Portfolio Volatility: 0.1542

Risk Contributions:
Asset 1: 0.0077
Asset 2: 0.0077
...
Asset 20: 0.0077

Percentage Risk Contributions:
Asset 1: 0.0500
Asset 2: 0.0500
...
Asset 20: 0.0500

Standard Deviation of Risk Contributions:
Risk Parity: 0.000012
Equal Weight: 0.015678
```

And two plots:
1. A bar chart comparing the weights and risk contributions for the risk parity portfolio
2. A bar chart comparing the risk contributions of the risk parity portfolio and an equal weight portfolio

## Variations and Extensions

### Risk Budgeting

Risk budgeting is a generalization of risk parity where the risk contribution of each asset is proportional to a predefined risk budget:

```python
# Define risk budget (in this case, allocate 40% of risk to the first 5 assets)
risk_budget = np.zeros(20)
risk_budget[:5] = 0.08  # 8% risk contribution for each of the first 5 assets
risk_budget[5:] = 0.04  # 4% risk contribution for each of the remaining 15 assets

# Define objective with risk budget
objective = RiskParityObjective(risk_budget=risk_budget)
```

### Sector Risk Parity

You can also apply risk parity at the sector level:

```python
from portopt.core.objective import SectorRiskParityObjective

# Assuming industry_classes is an array of industry classifications for each asset
industry_classes = problem.industry_classes

# Define sector risk parity objective
objective = SectorRiskParityObjective(sector_ids=industry_classes)
```

### Constrained Risk Parity

You can add additional constraints to the risk parity portfolio:

```python
from portopt.constraints.basic import PositionLimitConstraint
from portopt.constraints.sector import SectorConstraint

# Limit each asset to between 1% and 10% of the portfolio
position_limit_constraint = PositionLimitConstraint(
    min_weight=0.01,
    max_weight=0.1
)

# Limit sector exposure
sector_constraint = SectorConstraint(
    sector_ids=problem.industry_classes,
    max_weight=0.3
)

# Add to constraints list
constraints.extend([position_limit_constraint, sector_constraint])
```

## Conclusion

This example demonstrated how to construct a risk parity portfolio using the Portfolio Optimization Testbed. Risk parity is a robust portfolio construction technique that focuses on risk allocation rather than capital allocation, making it particularly useful in environments where expected returns are difficult to estimate accurately.

The key advantage of risk parity is that it creates a more balanced portfolio where each asset contributes equally to the overall portfolio risk, reducing the impact of estimation errors in expected returns.

## Related Resources

- [Portfolio Tutorial](../concepts/portfolio-tutorial.md)
- [Risk Metrics](../concepts/risk-metrics.md)
- [Minimum Variance Portfolio Example](./minimum-variance-portfolio.md)
- [Maximum Sharpe Ratio Portfolio Example](./maximum-sharpe-ratio-portfolio.md)
- [API Reference](../reference/api-reference.md)
- [Glossary](../reference/glossary.md)
