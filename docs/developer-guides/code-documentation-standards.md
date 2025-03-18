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

Class docstrings should describe the purpose of the class, its behavior, and any important attributes or methods.

```python
class ClassicalSolver(BaseSolver):
    """Classical portfolio optimization solver using sequential relaxation.

    This solver implements a sophisticated approach to portfolio optimization that:
    1. Uses sequential relaxation to handle non-linear constraints
    2. Supports warm starting from previous solutions
    3. Implements adaptive penalty adjustment
    4. Handles market impact and transaction costs

    The solver works in multiple stages:
    1. Find minimum variance portfolio without turnover constraints
    2. Gradually relax from current position to minimum variance target
    3. Fine-tune solution with multiple random starts

    Attributes:
        max_iterations: Maximum number of relaxation steps
        initial_penalty: Initial penalty for constraint violations
        penalty_multiplier: Factor to increase penalties
        perturbation_size: Size of random perturbations for multiple starts
    """
```

### Method and Function Docstrings

Method and function docstrings should describe what the function does, its parameters, return values, and any exceptions raised.

```python
def calculate_var_cvar(self, weights: np.ndarray, 
                      confidence_level: float = 0.95) -> Tuple[float, float]:
    """Calculate Value at Risk and Conditional Value at Risk.
    
    Uses historical simulation approach to estimate tail risk.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    confidence_level : float, optional
        Confidence level for VaR calculation, by default 0.95
        
    Returns
    -------
    Tuple[float, float]
        (VaR, CVaR) tuple at the specified confidence level
        
    Raises
    ------
    ValueError
        If weights do not sum to 1 or confidence level is not in (0,1)
    """
```

## Inline Comments

Inline comments should be used to explain complex or non-obvious code sections. They should focus on explaining "why" rather than "what" the code is doing.

### Good Inline Comments

```python
# Use Ledoit-Wolf shrinkage to handle ill-conditioned covariance matrix
shrinkage_factor = 0.2
cov_matrix = (1 - shrinkage_factor) * empirical_cov + shrinkage_factor * target_matrix

# Apply sequential relaxation to gradually enforce constraints
for iteration in range(max_iterations):
    # Increase penalty weights to enforce constraints more strictly in later iterations
    penalty = initial_penalty * (penalty_multiplier ** iteration)
```

### Poor Inline Comments (Avoid)

```python
# Calculate the mean
mean = np.mean(returns, axis=1)  # Obvious what the code does

# Loop through assets
for i in range(n_assets):  # Redundant comment
```

## Code Organization

### Import Order

Imports should be organized in the following order:

1. Standard library imports
2. Third-party library imports
3. Local application imports

Each group should be separated by a blank line.

```python
import numpy as np
import time
import logging
from typing import Dict, List, Optional

import pandas as pd
from scipy.optimize import minimize

from portopt.core.problem import PortfolioOptProblem
from portopt.utils.logging import setup_logging
```

### Class and Function Order

Within a module, classes and functions should be organized in a logical order:

1. Module-level constants and variables
2. Helper functions and classes
3. Main classes
4. Main functions
5. Conditional main block (`if __name__ == "__main__":`)

## Type Annotations

Use type annotations for all function parameters and return values to improve code readability and enable static type checking.

```python
def optimize_portfolio(
    returns: np.ndarray,
    constraints: Dict[str, Any],
    initial_weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """Optimize portfolio weights."""
```

## Examples

Include usage examples in docstrings for complex classes and functions:

```python
def calculate_efficient_frontier(
    returns: np.ndarray,
    cov_matrix: np.ndarray,
    min_return: float,
    max_return: float,
    points: int = 50
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Calculate the efficient frontier for a set of assets.
    
    Examples
    --------
    >>> returns = np.array([0.05, 0.1, 0.15, 0.2])
    >>> cov_matrix = np.array([[0.1, 0.01, 0.02, 0.03],
    ...                        [0.01, 0.2, 0.03, 0.04],
    ...                        [0.02, 0.03, 0.3, 0.05],
    ...                        [0.03, 0.04, 0.05, 0.4]])
    >>> risk, ret, weights = calculate_efficient_frontier(returns, cov_matrix, 0.05, 0.15, 10)
    >>> print(f"Number of portfolios: {len(weights)}")
    Number of portfolios: 10
    """
```

## Documentation Maintenance

### When to Update Documentation

- When adding new functionality
- When modifying existing functionality
- When fixing bugs that change behavior
- When refactoring code structure

### Documentation Review

All pull requests should include a review of documentation changes to ensure they meet these standards.

## Tools for Documentation

- Use [pydocstyle](http://www.pydocstyle.org/) for checking docstring style
- Use [mypy](http://mypy-lang.org/) for checking type annotations
- Use [sphinx](https://www.sphinx-doc.org/) for generating API documentation

## Implementation Checklist

When implementing or updating documentation, ensure:

- [ ] Module docstrings describe the purpose and functionality
- [ ] Class docstrings explain behavior and attributes
- [ ] Method/function docstrings include parameters, return values, and exceptions
- [ ] Complex code sections have explanatory inline comments
- [ ] Type annotations are used consistently
- [ ] Examples are provided for non-trivial functionality
- [ ] Documentation is kept up-to-date with code changes

## Best Practices

1. **Keep docstrings up to date**: Update documentation when you change code.
2. **Document why, not what**: Code shows what is happening; documentation should explain why.
3. **Be consistent**: Follow the same format throughout the codebase.
4. **Use simple language**: Avoid jargon and complex sentences.
5. **Include examples**: Practical examples help users understand how to use your code.
6. **Document edge cases**: Explain how your code handles unusual inputs or error conditions.
7. **Link to related documentation**: Reference related classes, functions, or external resources.

## Conclusion

Following these documentation standards will make the Portfolio Optimization Testbed codebase more accessible, maintainable, and user-friendly. Good documentation is a key aspect of software quality and helps both users and developers understand and use the code effectively.
