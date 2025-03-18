# API Reference

This document provides a comprehensive reference for the Portfolio Optimization Testbed API. It covers all the main modules, classes, and functions available in the library.

## Table of Contents

- [Core Components](#core-components)
  - [Problem Definition](#problem-definition)
  - [Objectives](#objectives)
  - [Constraints](#constraints)
- [Solvers](#solvers)
  - [Classical Solver](#classical-solver)
  - [Risk Parity Solver](#risk-parity-solver)
- [Metrics](#metrics)
  - [Performance Metrics](#performance-metrics)
  - [Risk Metrics](#risk-metrics)
- [Utilities](#utilities)
  - [Data Utilities](#data-utilities)
  - [Visualization](#visualization)
  - [Backtest](#backtest)

## Core Components

### Problem Definition

#### `portopt.core.problem.PortfolioOptProblem`

The central class for defining a portfolio optimization problem.

```python
class PortfolioOptProblem:
    """
    Portfolio optimization problem definition.
    
    This class encapsulates all the data needed for a portfolio optimization problem,
    including historical returns, volumes, and other optional parameters.
    """
    
    def __init__(self, returns, volumes=None, initial_weights=None, 
                 risk_free_rate=0, transaction_costs=0, market_impact_model=None):
        """
        Initialize a portfolio optimization problem.
        
        Parameters
        ----------
        returns : numpy.ndarray
            Historical returns matrix of shape (n_periods, n_assets).
        volumes : numpy.ndarray, optional
            Historical volumes matrix of shape (n_periods, n_assets).
        initial_weights : numpy.ndarray, optional
            Initial portfolio weights of shape (n_assets,).
        risk_free_rate : float, optional
            Risk-free rate, by default 0.
        transaction_costs : float, optional
            Transaction costs as a fraction of traded value, by default 0.
        market_impact_model : MarketImpactModel, optional
            Model for market impact of trades.
        """
        pass
    
    def get_covariance(self):
        """
        Calculate the covariance matrix of returns.
        
        Returns
        -------
        numpy.ndarray
            Covariance matrix of shape (n_assets, n_assets).
        """
        pass
    
    def get_expected_returns(self):
        """
        Calculate the expected returns.
        
        Returns
        -------
        numpy.ndarray
            Expected returns of shape (n_assets,).
        """
        pass
    
    def validate(self):
        """
        Validate the problem definition.
        
        Raises
        ------
        ValueError
            If the problem definition is invalid.
        """
        pass
```

### Objectives

#### `portopt.core.objective.Objective`

Base class for all objective functions.

```python
class Objective:
    """
    Base class for all objective functions.
    
    Objective functions define the goal of the optimization problem,
    such as minimizing risk or maximizing return.
    """
    
    def evaluate(self, weights, problem):
        """
        Evaluate the objective function for the given weights.
        
        Parameters
        ----------
        weights : numpy.ndarray
            Portfolio weights.
        problem : PortfolioOptProblem
            The portfolio optimization problem.
            
        Returns
        -------
        float
            Objective function value.
        """
        pass
    
    def gradient(self, weights, problem):
        """
        Calculate the gradient of the objective function.
        
        Parameters
        ----------
        weights : numpy.ndarray
            Portfolio weights.
        problem : PortfolioOptProblem
            The portfolio optimization problem.
            
        Returns
        -------
        numpy.ndarray
            Gradient of the objective function.
        """
        pass
```

#### `portopt.core.objective.MinimumVarianceObjective`

Objective function to minimize portfolio variance.

```python
class MinimumVarianceObjective(Objective):
    """
    Objective function to minimize portfolio variance.
    """
    pass
```

#### `portopt.core.objective.MaximumSharpeRatioObjective`

Objective function to maximize the Sharpe ratio.

```python
class MaximumSharpeRatioObjective(Objective):
    """
    Objective function to maximize the Sharpe ratio.
    
    Parameters
    ----------
    risk_free_rate : float, optional
        Risk-free rate, by default 0.
    """
    pass
```

#### `portopt.core.objective.MeanVarianceObjective`

Objective function for mean-variance optimization.

```python
class MeanVarianceObjective(Objective):
    """
    Objective function for mean-variance optimization.
    
    Parameters
    ----------
    risk_aversion : float
        Risk aversion parameter. Higher values prioritize risk reduction.
    """
    pass
```

### Constraints

#### `portopt.constraints.Constraint`

Base class for all constraints.

```python
class Constraint:
    """
    Base class for all constraints.
    
    Constraints define the feasible region for the optimization problem.
    """
    
    def evaluate(self, weights, problem):
        """
        Evaluate the constraint for the given weights.
        
        Parameters
        ----------
        weights : numpy.ndarray
            Portfolio weights.
        problem : PortfolioOptProblem
            The portfolio optimization problem.
            
        Returns
        -------
        float
            Constraint violation (0 if satisfied).
        """
        pass
    
    def gradient(self, weights, problem):
        """
        Calculate the gradient of the constraint.
        
        Parameters
        ----------
        weights : numpy.ndarray
            Portfolio weights.
        problem : PortfolioOptProblem
            The portfolio optimization problem.
            
        Returns
        -------
        numpy.ndarray
            Gradient of the constraint.
        """
        pass
    
    def validate(self, problem):
        """
        Validate the constraint for the given problem.
        
        Parameters
        ----------
        problem : PortfolioOptProblem
            The portfolio optimization problem.
            
        Raises
        ------
        ValueError
            If the constraint is invalid for the problem.
        """
        pass
```

#### `portopt.constraints.basic.FullInvestmentConstraint`

Constraint that the sum of weights equals 1.

```python
class FullInvestmentConstraint(Constraint):
    """
    Constraint that the sum of weights equals 1.
    """
    pass
```

#### `portopt.constraints.basic.LongOnlyConstraint`

Constraint that all weights are non-negative.

```python
class LongOnlyConstraint(Constraint):
    """
    Constraint that all weights are non-negative.
    """
    pass
```

#### `portopt.constraints.basic.MaxWeightConstraint`

Constraint that no weight exceeds a maximum value.

```python
class MaxWeightConstraint(Constraint):
    """
    Constraint that no weight exceeds a maximum value.
    
    Parameters
    ----------
    max_weight : float
        Maximum allowed weight for any asset.
    """
    pass
```

#### `portopt.constraints.basic.MinWeightConstraint`

Constraint that all weights are at least a minimum value.

```python
class MinWeightConstraint(Constraint):
    """
    Constraint that all weights are at least a minimum value.
    
    Parameters
    ----------
    min_weight : float
        Minimum allowed weight for any asset.
    """
    pass
```

#### `portopt.constraints.basic.SectorConstraint`

Constraint on sector allocations.

```python
class SectorConstraint(Constraint):
    """
    Constraint on sector allocations.
    
    Parameters
    ----------
    sector_mapping : dict
        Mapping from asset indices to sector names.
    min_weights : dict, optional
        Minimum weights for each sector.
    max_weights : dict, optional
        Maximum weights for each sector.
    """
    pass
```

#### `portopt.constraints.basic.TargetReturnConstraint`

Constraint to achieve a target expected return.

```python
class TargetReturnConstraint(Constraint):
    """
    Constraint to achieve a target expected return.
    
    Parameters
    ----------
    target_return : float
        Target expected return.
    """
    pass
```

## Solvers

### Classical Solver

#### `portopt.solvers.classical.ClassicalSolver`

Solver using classical optimization techniques.

```python
class ClassicalSolver:
    """
    Solver using classical optimization techniques.
    
    This solver uses sequential quadratic programming to solve
    portfolio optimization problems.
    
    Parameters
    ----------
    max_iterations : int, optional
        Maximum number of iterations, by default 100.
    tolerance : float, optional
        Convergence tolerance, by default 1e-8.
    verbose : bool, optional
        Whether to print progress information, by default False.
    """
    
    def solve(self, problem, constraints=None, objective=None, initial_weights=None):
        """
        Solve the portfolio optimization problem.
        
        Parameters
        ----------
        problem : PortfolioOptProblem
            The portfolio optimization problem.
        constraints : list, optional
            List of constraints.
        objective : Objective, optional
            Objective function.
        initial_weights : numpy.ndarray, optional
            Initial weights to start the optimization from.
            
        Returns
        -------
        OptimizationResult
            Result of the optimization.
        """
        pass
```

### Risk Parity Solver

#### `portopt.solvers.risk_parity.RiskParitySolver`

Specialized solver for risk parity portfolios.

```python
class RiskParitySolver:
    """
    Specialized solver for risk parity portfolios.
    
    This solver constructs portfolios where each asset contributes
    equally to the total portfolio risk.
    
    Parameters
    ----------
    max_iterations : int, optional
        Maximum number of iterations, by default 100.
    tolerance : float, optional
        Convergence tolerance, by default 1e-8.
    verbose : bool, optional
        Whether to print progress information, by default False.
    """
    
    def solve(self, problem, risk_budget=None):
        """
        Solve for a risk parity portfolio.
        
        Parameters
        ----------
        problem : PortfolioOptProblem
            The portfolio optimization problem.
        risk_budget : numpy.ndarray, optional
            Target risk budget for each asset. If None, equal risk contribution.
            
        Returns
        -------
        OptimizationResult
            Result of the optimization.
        """
        pass
```

## Metrics

### Performance Metrics

#### `portopt.metrics.performance.calculate_sharpe_ratio`

Calculate the Sharpe ratio of a portfolio.

```python
def calculate_sharpe_ratio(returns, risk_free_rate=0):
    """
    Calculate the Sharpe ratio of a portfolio.
    
    Parameters
    ----------
    returns : numpy.ndarray
        Portfolio returns.
    risk_free_rate : float, optional
        Risk-free rate, by default 0.
        
    Returns
    -------
    float
        Sharpe ratio.
    """
    pass
```

#### `portopt.metrics.performance.calculate_sortino_ratio`

Calculate the Sortino ratio of a portfolio.

```python
def calculate_sortino_ratio(returns, risk_free_rate=0, target_return=0):
    """
    Calculate the Sortino ratio of a portfolio.
    
    Parameters
    ----------
    returns : numpy.ndarray
        Portfolio returns.
    risk_free_rate : float, optional
        Risk-free rate, by default 0.
    target_return : float, optional
        Target return, by default 0.
        
    Returns
    -------
    float
        Sortino ratio.
    """
    pass
```

#### `portopt.metrics.performance.calculate_maximum_drawdown`

Calculate the maximum drawdown of a portfolio.

```python
def calculate_maximum_drawdown(cumulative_returns):
    """
    Calculate the maximum drawdown of a portfolio.
    
    Parameters
    ----------
    cumulative_returns : numpy.ndarray
        Cumulative portfolio returns.
        
    Returns
    -------
    float
        Maximum drawdown.
    """
    pass
```

### Risk Metrics

#### `portopt.metrics.risk.calculate_portfolio_volatility`

Calculate the volatility of a portfolio.

```python
def calculate_portfolio_volatility(weights, covariance_matrix):
    """
    Calculate the volatility of a portfolio.
    
    Parameters
    ----------
    weights : numpy.ndarray
        Portfolio weights.
    covariance_matrix : numpy.ndarray
        Covariance matrix of asset returns.
        
    Returns
    -------
    float
        Portfolio volatility.
    """
    pass
```

#### `portopt.metrics.risk.calculate_value_at_risk`

Calculate the Value at Risk (VaR) of a portfolio.

```python
def calculate_value_at_risk(returns, confidence_level=0.95):
    """
    Calculate the Value at Risk (VaR) of a portfolio.
    
    Parameters
    ----------
    returns : numpy.ndarray
        Portfolio returns.
    confidence_level : float, optional
        Confidence level, by default 0.95.
        
    Returns
    -------
    float
        Value at Risk.
    """
    pass
```

#### `portopt.metrics.risk.calculate_conditional_value_at_risk`

Calculate the Conditional Value at Risk (CVaR) of a portfolio.

```python
def calculate_conditional_value_at_risk(returns, confidence_level=0.95):
    """
    Calculate the Conditional Value at Risk (CVaR) of a portfolio.
    
    Parameters
    ----------
    returns : numpy.ndarray
        Portfolio returns.
    confidence_level : float, optional
        Confidence level, by default 0.95.
        
    Returns
    -------
    float
        Conditional Value at Risk.
    """
    pass
```

#### `portopt.metrics.risk.calculate_risk_contribution`

Calculate the risk contribution of each asset in a portfolio.

```python
def calculate_risk_contribution(weights, covariance_matrix):
    """
    Calculate the risk contribution of each asset in a portfolio.
    
    Parameters
    ----------
    weights : numpy.ndarray
        Portfolio weights.
    covariance_matrix : numpy.ndarray
        Covariance matrix of asset returns.
        
    Returns
    -------
    numpy.ndarray
        Risk contribution of each asset.
    """
    pass
```

## Utilities

### Data Utilities

#### `portopt.utils.data.TestDataGenerator`

Generate synthetic data for testing and examples.

```python
class TestDataGenerator:
    """
    Generate synthetic data for testing and examples.
    
    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility, by default None.
    """
    
    def generate_returns(self, n_assets, n_periods, volatility=0.2, correlation=0.5):
        """
        Generate synthetic returns.
        
        Parameters
        ----------
        n_assets : int
            Number of assets.
        n_periods : int
            Number of time periods.
        volatility : float or numpy.ndarray, optional
            Asset volatilities, by default 0.2.
        correlation : float or numpy.ndarray, optional
            Asset correlations, by default 0.5.
            
        Returns
        -------
        numpy.ndarray
            Synthetic returns of shape (n_periods, n_assets).
        """
        pass
    
    def generate_realistic_problem(self, n_assets, n_periods, n_factors=3, n_industries=5):
        """
        Generate a realistic portfolio optimization problem.
        
        Parameters
        ----------
        n_assets : int
            Number of assets.
        n_periods : int
            Number of time periods.
        n_factors : int, optional
            Number of risk factors, by default 3.
        n_industries : int, optional
            Number of industries, by default 5.
            
        Returns
        -------
        PortfolioOptProblem
            A realistic portfolio optimization problem.
        """
        pass
```

### Visualization

#### `portopt.utils.visualization.plot_efficient_frontier`

Plot the efficient frontier.

```python
def plot_efficient_frontier(problem, constraints, n_points=20, highlight_portfolio=None, 
                           highlight_label=None, risk_free_rate=0, ax=None):
    """
    Plot the efficient frontier.
    
    Parameters
    ----------
    problem : PortfolioOptProblem
        The portfolio optimization problem.
    constraints : list
        List of constraints.
    n_points : int, optional
        Number of points on the frontier, by default 20.
    highlight_portfolio : numpy.ndarray, optional
        Portfolio weights to highlight on the frontier.
    highlight_label : str, optional
        Label for the highlighted portfolio.
    risk_free_rate : float, optional
        Risk-free rate for the capital market line, by default 0.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None.
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    pass
```

#### `portopt.utils.visualization.plot_weights`

Plot portfolio weights.

```python
def plot_weights(weights, labels=None, title="Portfolio Weights", ax=None):
    """
    Plot portfolio weights.
    
    Parameters
    ----------
    weights : numpy.ndarray
        Portfolio weights.
    labels : list, optional
        Asset labels, by default None.
    title : str, optional
        Plot title, by default "Portfolio Weights".
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None.
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    pass
```

#### `portopt.utils.visualization.plot_risk_contribution`

Plot risk contribution of each asset.

```python
def plot_risk_contribution(weights, covariance_matrix, labels=None, 
                          title="Risk Contribution", ax=None):
    """
    Plot risk contribution of each asset.
    
    Parameters
    ----------
    weights : numpy.ndarray
        Portfolio weights.
    covariance_matrix : numpy.ndarray
        Covariance matrix of asset returns.
    labels : list, optional
        Asset labels, by default None.
    title : str, optional
        Plot title, by default "Risk Contribution".
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None.
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    pass
```

### Backtest

#### `portopt.utils.backtest.backtest_portfolio`

Backtest a portfolio strategy.

```python
def backtest_portfolio(returns, rebalance_strategy, rebalance_frequency=21, 
                      transaction_costs=0, initial_weights=None):
    """
    Backtest a portfolio strategy.
    
    Parameters
    ----------
    returns : numpy.ndarray
        Historical returns of shape (n_periods, n_assets).
    rebalance_strategy : callable
        Function that returns new weights given historical data and current index.
    rebalance_frequency : int, optional
        Number of periods between rebalancing, by default 21 (monthly).
    transaction_costs : float, optional
        Transaction costs as a fraction of traded value, by default 0.
    initial_weights : numpy.ndarray, optional
        Initial portfolio weights, by default None (equal weights).
        
    Returns
    -------
    dict
        Backtest results including performance metrics.
    """
    pass
```

## Full API Documentation

For complete API documentation with all classes, methods, and parameters, refer to the auto-generated API documentation at [https://neuralground.github.io/portopt/api/](https://neuralground.github.io/portopt/api/).
