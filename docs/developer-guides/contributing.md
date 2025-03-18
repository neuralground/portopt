# Contributing Guide

Thank you for your interest in contributing to the Portfolio Optimization Testbed! This guide will help you get started with contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)
- [Issue Tracking](#issue-tracking)
- [Release Process](#release-process)

## Code of Conduct

This project follows a [Code of Conduct](../CODE_OF_CONDUCT.md) to ensure a welcoming and inclusive environment for all contributors.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- Git
- Node.js and npm (for frontend development)

### Setting Up the Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/portopt.git
   cd portopt
   ```

3. Set up the upstream remote:
   ```bash
   git remote add upstream https://github.com/originalowner/portopt.git
   ```

4. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

6. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

7. For frontend development, install dependencies:
   ```bash
   cd frontend
   npm install
   ```

## Development Workflow

### Branching Strategy

We use a simplified Git flow with the following branches:
- `main`: The main branch containing stable code
- `develop`: The development branch for integrating features
- Feature branches: Created from `develop` for new features
- Bugfix branches: Created from `develop` for bug fixes
- Release branches: Created from `develop` for preparing releases

### Creating a Feature Branch

1. Ensure you're on the latest `develop` branch:
   ```bash
   git checkout develop
   git pull upstream develop
   ```

2. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Make your changes, commit them, and push to your fork:
   ```bash
   git add .
   git commit -m "Add your feature"
   git push origin feature/your-feature-name
   ```

### Keeping Your Branch Updated

Regularly update your branch with changes from the upstream `develop` branch:

```bash
git checkout develop
git pull upstream develop
git checkout feature/your-feature-name
git merge develop
```

Resolve any conflicts that arise during the merge.

## Pull Request Process

1. Ensure your code passes all tests:
   ```bash
   pytest
   ```

2. Ensure your code follows the coding standards:
   ```bash
   black portopt tests
   isort portopt tests
   flake8 portopt tests
   mypy portopt
   ```

3. Update the documentation to reflect any changes.

4. Create a pull request from your feature branch to the `develop` branch of the upstream repository.

5. In your pull request description:
   - Describe the changes you've made
   - Reference any related issues
   - Mention any breaking changes
   - Include screenshots for UI changes

6. Wait for code review and address any feedback.

7. Once approved, your changes will be merged into the `develop` branch.

## Coding Standards

We follow strict coding standards to ensure code quality and consistency:

### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use [flake8](https://flake8.pycqa.org/) for linting
- Use [mypy](https://mypy.readthedocs.io/) for type checking

### TypeScript/JavaScript Code Style

- Follow the [Airbnb JavaScript Style Guide](https://github.com/airbnb/javascript)
- Use [ESLint](https://eslint.org/) for linting
- Use [Prettier](https://prettier.io/) for code formatting

### Naming Conventions

- **Python**:
  - Classes: `CamelCase`
  - Functions and variables: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
  - Private methods/variables: `_leading_underscore`

- **TypeScript/JavaScript**:
  - Classes and components: `CamelCase`
  - Functions and variables: `camelCase`
  - Constants: `UPPER_SNAKE_CASE`
  - Private methods/variables: `_leadingUnderscore`

### Code Organization

- Keep functions and methods small and focused
- Follow the single responsibility principle
- Use meaningful names for variables, functions, and classes
- Add comments for complex logic
- Use docstrings for all public functions, classes, and methods

## Testing Guidelines

We use [pytest](https://docs.pytest.org/) for testing Python code and [Jest](https://jestjs.io/) for testing JavaScript/TypeScript code.

### Writing Python Tests

1. Create test files in the `tests` directory
2. Name test files with the prefix `test_`
3. Name test functions with the prefix `test_`
4. Use descriptive test names that explain what is being tested
5. Use fixtures for common setup
6. Use parameterized tests for testing multiple inputs
7. Aim for high test coverage

Example test:

```python
def test_classical_solver_converges():
    # Arrange
    problem = create_test_problem()
    solver = ClassicalSolver(max_iterations=100)
    
    # Act
    result = solver.solve(problem)
    
    # Assert
    assert result.converged
    assert result.iterations < 100
    assert result.objective_value > 0
```

### Writing JavaScript/TypeScript Tests

1. Create test files with the `.test.ts` or `.test.tsx` extension
2. Use descriptive test names
3. Use Jest's mocking capabilities for external dependencies
4. Test components in isolation

Example test:

```typescript
describe('RiskMetricsCard', () => {
  it('renders risk metrics correctly', () => {
    // Arrange
    const metrics = {
      volatility: 0.15,
      var: 0.05,
      cvar: 0.08,
      sharpe: 1.2
    };
    
    // Act
    const { getByText } = render(<RiskMetricsCard metrics={metrics} />);
    
    // Assert
    expect(getByText('Volatility: 15.00%')).toBeInTheDocument();
    expect(getByText('VaR (95%): 5.00%')).toBeInTheDocument();
    expect(getByText('CVaR (95%): 8.00%')).toBeInTheDocument();
    expect(getByText('Sharpe Ratio: 1.20')).toBeInTheDocument();
  });
});
```

### Running Tests

Run all tests:

```bash
# Python tests
pytest

# JavaScript/TypeScript tests
cd frontend
npm test
```

Run tests with coverage:

```bash
# Python tests
pytest --cov=portopt

# JavaScript/TypeScript tests
cd frontend
npm test -- --coverage
```

## Documentation Guidelines

Good documentation is crucial for the project's usability and maintainability.

### Code Documentation

- Add docstrings to all public classes, methods, and functions
- Follow the [NumPy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html)
- Include type hints for parameters and return values
- Document parameters, return values, and exceptions
- Provide examples for complex functions

Example docstring:

```python
def calculate_var(
    weights: np.ndarray,
    returns: np.ndarray,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Value at Risk (VaR) for a portfolio.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights of shape (n_assets,)
    returns : np.ndarray
        Historical returns of shape (n_periods, n_assets)
    confidence_level : float, optional
        Confidence level for VaR calculation, by default 0.95
        
    Returns
    -------
    float
        Value at Risk at the specified confidence level
        
    Examples
    --------
    >>> weights = np.array([0.5, 0.5])
    >>> returns = np.array([[0.01, 0.02], [0.02, 0.01], [-0.01, -0.02]])
    >>> calculate_var(weights, returns, 0.95)
    0.015
    """
```

### User Documentation

- Write clear, concise, and accurate documentation
- Use markdown for formatting
- Include examples and code snippets
- Organize documentation in a logical structure
- Keep documentation up-to-date with code changes

### Documentation Structure

Follow the documentation structure outlined in the [Documentation Plan](../documentation-plan.md).

### Documentation Contributions

We welcome improvements to the [documentation](../). If you find any gaps or errors in the documentation, please submit a pull request with your proposed changes.

## Issue Tracking

We use GitHub Issues for tracking bugs, features, and other tasks.

### Creating Issues

When creating an issue:
1. Use a clear and descriptive title
2. Provide a detailed description
3. Include steps to reproduce for bugs
4. Add relevant labels
5. Assign to a milestone if applicable

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `question`: Further information is requested

### Working on Issues

1. Comment on the issue to express your interest
2. Wait for assignment or confirmation
3. Create a branch for the issue
4. Reference the issue in your pull request

## Release Process

We follow [Semantic Versioning](https://semver.org/) for releases.

### Version Numbering

- **Major version**: Incompatible API changes
- **Minor version**: Backwards-compatible functionality
- **Patch version**: Backwards-compatible bug fixes

### Release Steps

1. Create a release branch from `develop`:
   ```bash
   git checkout develop
   git pull
   git checkout -b release/vX.Y.Z
   ```

2. Update version numbers in:
   - `pyproject.toml`
   - `portopt/__init__.py`
   - `package.json` (for frontend)

3. Update the changelog:
   - Add a new section for the release
   - List all notable changes
   - Categorize changes (Added, Changed, Deprecated, Removed, Fixed, Security)

4. Create a pull request from the release branch to `main`

5. After approval and merge, tag the release:
   ```bash
   git checkout main
   git pull
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```

6. Merge `main` back into `develop`:
   ```bash
   git checkout develop
   git merge main
   git push origin develop
   ```

7. Create a GitHub Release with release notes

8. Publish to PyPI:
   ```bash
   python -m build
   twine upload dist/*
   ```

## Getting Help

If you need help with contributing:
- Check the [documentation](../)
- Ask questions in GitHub Issues
- Contact the maintainers

Thank you for contributing to the Portfolio Optimization Testbed!
