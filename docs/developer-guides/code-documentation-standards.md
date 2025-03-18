# Code Documentation Standards

This guide outlines the standards for documenting code in the Portfolio Optimization Testbed project. Following these standards ensures that the codebase is maintainable, understandable, and accessible to new contributors.

## Docstrings

All modules, classes, and functions should have docstrings that follow the [NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html).

### Module Docstrings

Module docstrings should appear at the top of the file and describe the purpose of the module.

```python
"""
This module provides functionality for portfolio optimization using classical methods.

It includes implementations of various optimization algorithms, objective functions,
and constraint handlers designed for portfolio construction.
"""
```

### Class Docstrings

Class docstrings should describe the purpose of the class, its behavior, and important attributes.

```python
class PortfolioOptProblem:
    """
    A class representing a portfolio optimization problem.
    
    This class encapsulates all the data and parameters needed to define a
    portfolio optimization problem, including historical returns, constraints,
    and optional market impact data.
    
    Parameters
    ----------
    returns : numpy.ndarray
        Historical returns matrix with shape (n_periods, n_assets).
    volumes : numpy.ndarray, optional
        Trading volumes for each asset over time with shape (n_periods, n_assets).
    market_impact_model : MarketImpactModel, optional
        Model for estimating market impact of trades.
    
    Attributes
    ----------
    n_assets : int
        Number of assets in the portfolio.
    n_periods : int
        Number of time periods in the historical data.
    covariance_matrix : numpy.ndarray
        Covariance matrix of asset returns.
    
    Notes
    -----
    The returns matrix should contain asset returns, not prices. If you have
    price data, you should convert it to returns before creating a problem instance.
    
    Examples
    --------
    >>> import numpy as np
    >>> from portopt.core.problem import PortfolioOptProblem
    >>> 
    >>> # Create returns data (10 periods, 5 assets)
    >>> returns = np.random.normal(0.001, 0.05, (10, 5))
    >>> 
    >>> # Create a problem instance
    >>> problem = PortfolioOptProblem(returns=returns)
    >>> print(f"Number of assets: {problem.n_assets}")
    Number of assets: 5
    """
```

### Function Docstrings

Function docstrings should describe what the function does, its parameters, return values, and include examples.

```python
def calculate_sharpe_ratio(returns, risk_free_rate=0):
    """
    Calculate the Sharpe ratio of a series of returns.
    
    The Sharpe ratio is a measure of risk-adjusted return, calculated as
    the excess return (over the risk-free rate) per unit of volatility.
    
    Parameters
    ----------
    returns : numpy.ndarray
        Array of returns.
    risk_free_rate : float, optional
        Risk-free rate, by default 0.
    
    Returns
    -------
    float
        Sharpe ratio.
    
    Raises
    ------
    ValueError
        If the returns array is empty or if the volatility is zero.
    
    Examples
    --------
    >>> import numpy as np
    >>> from portopt.metrics.performance import calculate_sharpe_ratio
    >>> 
    >>> returns = np.array([0.05, 0.03, 0.04, -0.02, 0.01])
    >>> sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.01)
    >>> print(f"Sharpe ratio: {sharpe:.4f}")
    Sharpe ratio: 0.7071
    """
```

## Type Hints

Use type hints to specify the expected types of function parameters and return values.

```python
def calculate_portfolio_volatility(weights: np.ndarray, covariance_matrix: np.ndarray) -> float:
    """
    Calculate the volatility (standard deviation) of a portfolio.
    
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
    return np.sqrt(weights.T @ covariance_matrix @ weights)
```

## Comments

Use comments to explain complex code sections, but prefer self-explanatory code when possible.

```python
# Bad: Unclear what this calculation does
x = (a * b) / (c * d)

# Good: Clear explanation of a complex calculation
# Calculate the information ratio as the active return divided by tracking error
information_ratio = active_return / tracking_error
```

## Code Examples

Include code examples in docstrings to demonstrate how to use the function or class. Examples should be concise but complete enough to be run independently.

```python
def create_equal_weight_portfolio(n_assets: int) -> np.ndarray:
    """
    Create an equal weight portfolio.
    
    Parameters
    ----------
    n_assets : int
        Number of assets in the portfolio.
    
    Returns
    -------
    numpy.ndarray
        Array of equal weights.
    
    Examples
    --------
    >>> from portopt.utils.portfolio import create_equal_weight_portfolio
    >>> 
    >>> # Create an equal weight portfolio with 5 assets
    >>> weights = create_equal_weight_portfolio(5)
    >>> print(weights)
    [0.2 0.2 0.2 0.2 0.2]
    """
    return np.ones(n_assets) / n_assets
```

## Documenting Complex Algorithms

For complex algorithms, include:
1. A high-level description of the algorithm
2. References to relevant papers or resources
3. Explanation of key steps
4. Time and space complexity information

```python
def solve_risk_parity(covariance_matrix: np.ndarray, max_iterations: int = 100, 
                      tolerance: float = 1e-8) -> np.ndarray:
    """
    Solve for risk parity weights using the Newton-Raphson method.
    
    This implementation follows the approach described in:
    Spinu, F. (2013). "An Algorithm for Computing Risk Parity Weights."
    
    The algorithm iteratively adjusts weights to equalize risk contribution
    across all assets.
    
    Parameters
    ----------
    covariance_matrix : numpy.ndarray
        Covariance matrix of asset returns.
    max_iterations : int, optional
        Maximum number of iterations, by default 100.
    tolerance : float, optional
        Convergence tolerance, by default 1e-8.
    
    Returns
    -------
    numpy.ndarray
        Risk parity weights.
    
    Notes
    -----
    Time complexity: O(n^3 * iterations) where n is the number of assets
    Space complexity: O(n^2) for storing the covariance matrix
    
    The algorithm may not converge for certain covariance matrices. In such
    cases, consider using a different initial guess or regularizing the
    covariance matrix.
    """
```

## File Headers

Include a header at the top of each file with copyright information and a brief description.

```python
# Copyright (c) 2023 Neural Ground
# 
# This file is part of the Portfolio Optimization Testbed.
#
# This module implements risk metrics for portfolio optimization.
```

## Documenting Changes

When making significant changes to code, update the docstrings to reflect the changes and add a note in the docstring about when and why the change was made.

```python
def calculate_expected_shortfall(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Calculate the Expected Shortfall (ES) of a series of returns.
    
    Parameters
    ----------
    returns : numpy.ndarray
        Array of returns.
    alpha : float, optional
        Confidence level, by default 0.05 (95% confidence).
    
    Returns
    -------
    float
        Expected Shortfall.
    
    Notes
    -----
    2023-06-15: Updated to use a more numerically stable algorithm.
    """
```

## Documentation Testing

All code examples in docstrings should be testable using [doctest](https://docs.python.org/3/library/doctest.html). This ensures that examples remain accurate as the codebase evolves.

To run doctests:

```bash
python -m doctest -v path/to/module.py
```

## Best Practices

1. **Keep docstrings up to date**: Update documentation when you change code.
2. **Document why, not what**: Code shows what is happening; documentation should explain why.
3. **Be consistent**: Follow the same format throughout the codebase.
4. **Use simple language**: Avoid jargon and complex sentences.
5. **Include examples**: Practical examples help users understand how to use your code.
6. **Document edge cases**: Explain how your code handles unusual inputs or error conditions.
7. **Link to related documentation**: Reference related classes, functions, or external resources.

## Tools for Documentation

- [Sphinx](https://www.sphinx-doc.org/): Documentation generator
- [NumPy docstring format](https://numpydoc.readthedocs.io/): Docstring style guide
- [Black](https://black.readthedocs.io/): Code formatter
- [isort](https://pycqa.github.io/isort/): Import sorter
- [mypy](http://mypy-lang.org/): Static type checker
- [doctest](https://docs.python.org/3/library/doctest.html): Test code examples in docstrings

## Conclusion

Following these documentation standards will make the Portfolio Optimization Testbed codebase more accessible, maintainable, and user-friendly. Good documentation is a key aspect of software quality and helps both users and developers understand and use the code effectively.
