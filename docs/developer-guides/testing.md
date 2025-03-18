# Testing Guide

This guide explains how to run tests and write new tests for the Portfolio Optimization Testbed.

## Testing Philosophy

Testing is a critical part of our development process. We follow these testing principles:

1. **Test-Driven Development**: Write tests before implementing features
2. **Comprehensive Coverage**: Aim for high test coverage
3. **Fast Feedback**: Tests should run quickly
4. **Isolation**: Tests should be independent of each other
5. **Readability**: Tests should be easy to understand

## Test Types

We use several types of tests:

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test interactions between components
3. **Functional Tests**: Test end-to-end workflows
4. **Performance Tests**: Test performance characteristics
5. **Property-Based Tests**: Test invariants and properties

## Testing Tools

### Python Testing

We use the following tools for Python testing:

- [pytest](https://docs.pytest.org/): Test framework
- [pytest-cov](https://pytest-cov.readthedocs.io/): Coverage reporting
- [pytest-benchmark](https://pytest-benchmark.readthedocs.io/): Performance testing
- [hypothesis](https://hypothesis.readthedocs.io/): Property-based testing
- [mock](https://docs.python.org/3/library/unittest.mock.html): Mocking

### JavaScript/TypeScript Testing

For frontend testing, we use:

- [Jest](https://jestjs.io/): Test framework
- [React Testing Library](https://testing-library.com/docs/react-testing-library/intro/): Component testing
- [Cypress](https://www.cypress.io/): End-to-end testing

## Running Tests

### Running Python Tests

To run all Python tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=portopt

# Generate HTML coverage report
pytest --cov=portopt --cov-report=html

# Run specific test file
pytest tests/test_solvers/test_classical_solver.py

# Run specific test
pytest tests/test_solvers/test_classical_solver.py::test_solve_minimum_variance
```

### Running with Different Configurations

We provide several test configurations:

```bash
# Run with quick configuration (fast tests)
pytest --config config/test_configs/quick.ini

# Run with thorough configuration (comprehensive tests)
pytest --config config/test_configs/thorough.ini

# Run with stress configuration (performance tests)
pytest --config config/test_configs/stress.ini
```

### Running JavaScript/TypeScript Tests

To run frontend tests:

```bash
# Navigate to frontend directory
cd frontend

# Run all tests
npm test

# Run with coverage
npm test -- --coverage

# Run specific test file
npm test -- src/components/RiskMetrics.test.tsx

# Run in watch mode
npm test -- --watch
```

## Writing Tests

### Writing Python Tests

#### Unit Test Example

Here's an example of a unit test for the `ClassicalSolver`:

```python
import numpy as np
import pytest
from portopt import TestDataGenerator
from portopt.solvers.classical import ClassicalSolver
from portopt.constraints.basic import FullInvestmentConstraint
from portopt.core.objective import MinimumVarianceObjective

def test_classical_solver_minimum_variance():
    # Arrange
    generator = TestDataGenerator(seed=42)
    problem = generator.generate_realistic_problem(n_assets=10, n_periods=100)
    solver = ClassicalSolver(max_iterations=50, tolerance=1e-6)
    constraints = [FullInvestmentConstraint()]
    objective = MinimumVarianceObjective()
    
    # Act
    result = solver.solve(problem, constraints=constraints, objective=objective)
    
    # Assert
    assert result.converged
    assert result.iterations < 50
    assert np.isclose(np.sum(result.weights), 1.0, atol=1e-6)
    
    # Calculate portfolio variance
    cov_matrix = problem.get_covariance_matrix()
    variance = result.weights.T @ cov_matrix @ result.weights
    
    # Variance should be positive
    assert variance > 0
```

#### Integration Test Example

Here's an example of an integration test for the benchmarking system:

```python
import os
import pytest
from portopt.benchmark.runner import BenchmarkRunner
from portopt.solvers.classical import ClassicalSolver

@pytest.fixture
def benchmark_dir(tmpdir):
    return str(tmpdir.mkdir("benchmark_results"))

def test_benchmark_runner_integration(benchmark_dir):
    # Arrange
    runner = BenchmarkRunner(output_dir=benchmark_dir)
    
    # Act
    results = runner.run_size_scaling_benchmark(
        solver_classes=[ClassicalSolver],
        n_assets_range=[10, 20],
        n_periods_range=[50, 100],
        n_runs=2
    )
    
    # Assert
    assert results is not None
    assert len(results.configurations) == 4  # 2 asset sizes × 2 period sizes
    assert os.path.exists(os.path.join(benchmark_dir, "size_scaling_results.json"))
    
    # Check that results contain expected data
    for config in results.configurations:
        assert "n_assets" in config
        assert "n_periods" in config
        assert "solver_class" in config
        assert "mean_time" in config
        assert "mean_iterations" in config
```

#### Performance Test Example

Here's an example of a performance test:

```python
import pytest
from portopt import TestDataGenerator
from portopt.solvers.classical import ClassicalSolver

@pytest.mark.benchmark
def test_classical_solver_performance(benchmark):
    # Arrange
    generator = TestDataGenerator(seed=42)
    problem = generator.generate_realistic_problem(n_assets=50, n_periods=252)
    solver = ClassicalSolver()
    
    # Act & Assert
    # Benchmark the solve method
    result = benchmark(solver.solve, problem)
    
    # Additional assertions
    assert result.converged
```

#### Property-Based Test Example

Here's an example of a property-based test using Hypothesis:

```python
import numpy as np
from hypothesis import given, strategies as st
from portopt.metrics.risk import calculate_portfolio_volatility

@given(
    weights=st.lists(st.floats(min_value=0, max_value=1), min_size=2, max_size=10).map(
        lambda x: np.array(x) / sum(x) if sum(x) > 0 else np.array(x)
    ),
    returns=st.lists(
        st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=2, max_size=10),
        min_size=10,
        max_size=100,
    ).map(lambda x: np.array(x))
)
def test_portfolio_volatility_properties(weights, returns):
    # Ensure dimensions match
    if returns.shape[1] != len(weights):
        returns = returns[:, :len(weights)]
    
    # Calculate covariance matrix
    cov_matrix = np.cov(returns.T)
    
    # Calculate portfolio volatility
    volatility = calculate_portfolio_volatility(weights, cov_matrix)
    
    # Properties to test
    assert volatility >= 0  # Volatility should be non-negative
    
    # For zero weights, volatility should be zero
    if np.allclose(weights, 0):
        assert np.isclose(volatility, 0)
    
    # Scaling weights should not change volatility if sum is preserved
    if np.sum(weights) > 0:
        scaled_weights = weights / np.sum(weights)
        scaled_volatility = calculate_portfolio_volatility(scaled_weights, cov_matrix)
        assert np.isclose(volatility, scaled_volatility)
```

### Writing JavaScript/TypeScript Tests

#### Component Test Example

Here's an example of a React component test:

```typescript
import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import RiskMetricsCard from '../components/RiskMetricsCard';

describe('RiskMetricsCard', () => {
  const defaultProps = {
    volatility: 0.15,
    var95: 0.05,
    cvar95: 0.08,
    sharpeRatio: 1.2,
    onTimeRangeChange: jest.fn(),
  };

  it('renders risk metrics correctly', () => {
    render(<RiskMetricsCard {...defaultProps} />);
    
    expect(screen.getByText('Volatility')).toBeInTheDocument();
    expect(screen.getByText('15.00%')).toBeInTheDocument();
    
    expect(screen.getByText('VaR (95%)')).toBeInTheDocument();
    expect(screen.getByText('5.00%')).toBeInTheDocument();
    
    expect(screen.getByText('CVaR (95%)')).toBeInTheDocument();
    expect(screen.getByText('8.00%')).toBeInTheDocument();
    
    expect(screen.getByText('Sharpe Ratio')).toBeInTheDocument();
    expect(screen.getByText('1.20')).toBeInTheDocument();
  });

  it('calls onTimeRangeChange when time range is changed', async () => {
    render(<RiskMetricsCard {...defaultProps} />);
    
    const timeRangeSelect = screen.getByLabelText('Time Range');
    await userEvent.selectOptions(timeRangeSelect, '1Y');
    
    expect(defaultProps.onTimeRangeChange).toHaveBeenCalledWith('1Y');
  });
});
```

#### API Integration Test Example

Here's an example of a test for API integration:

```typescript
import { rest } from 'msw';
import { setupServer } from 'msw/node';
import { renderHook, act } from '@testing-library/react-hooks';
import { useOptimizationResults } from '../hooks/useOptimizationResults';

// Mock API server
const server = setupServer(
  rest.get('/api/results/123', (req, res, ctx) => {
    return res(
      ctx.json({
        id: '123',
        weights: [0.4, 0.3, 0.2, 0.1],
        objective_value: 0.05,
        iterations: 15,
        converged: true
      })
    );
  })
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('useOptimizationResults', () => {
  it('fetches optimization results successfully', async () => {
    const { result, waitForNextUpdate } = renderHook(() => 
      useOptimizationResults('123')
    );
    
    // Initial state
    expect(result.current.loading).toBe(true);
    expect(result.current.error).toBe(null);
    expect(result.current.data).toBe(null);
    
    // Wait for API response
    await waitForNextUpdate();
    
    // After API response
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBe(null);
    expect(result.current.data).toEqual({
      id: '123',
      weights: [0.4, 0.3, 0.2, 0.1],
      objective_value: 0.05,
      iterations: 15,
      converged: true
    });
  });

  it('handles API errors gracefully', async () => {
    // Override the default handler for this test
    server.use(
      rest.get('/api/results/456', (req, res, ctx) => {
        return res(ctx.status(500), ctx.json({ error: 'Server error' }));
      })
    );
    
    const { result, waitForNextUpdate } = renderHook(() => 
      useOptimizationResults('456')
    );
    
    // Wait for API response
    await waitForNextUpdate();
    
    // After API error
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBe('Failed to fetch optimization results');
    expect(result.current.data).toBe(null);
  });
});
```

## Test Organization

### Python Test Organization

Python tests are organized in a structure that mirrors the main package:

```
tests/
├── __init__.py
├── conftest.py                    # Common fixtures
├── test_core/                     # Tests for core module
│   ├── __init__.py
│   ├── test_problem.py
│   └── test_objective.py
├── test_solvers/                  # Tests for solvers module
│   ├── __init__.py
│   ├── test_classical_solver.py
│   └── test_multithreaded_solver.py
├── test_constraints/              # Tests for constraints module
│   ├── __init__.py
│   ├── test_basic_constraints.py
│   └── test_sector_constraints.py
├── test_metrics/                  # Tests for metrics module
│   ├── __init__.py
│   ├── test_performance_metrics.py
│   └── test_risk_metrics.py
├── test_benchmark/                # Tests for benchmark module
│   ├── __init__.py
│   └── test_runner.py
└── test_integration/              # Integration tests
    ├── __init__.py
    └── test_end_to_end.py
```

### JavaScript/TypeScript Test Organization

Frontend tests are organized alongside the components they test:

```
frontend/src/
├── components/
│   ├── RiskMetricsCard.tsx
│   └── RiskMetricsCard.test.tsx
├── hooks/
│   ├── useOptimizationResults.ts
│   └── useOptimizationResults.test.ts
├── pages/
│   ├── Dashboard.tsx
│   └── Dashboard.test.tsx
└── utils/
    ├── formatters.ts
    └── formatters.test.ts
```

## Test Fixtures

### Python Fixtures

We use pytest fixtures for common setup:

```python
# conftest.py
import pytest
import numpy as np
from portopt import TestDataGenerator
from portopt.core.problem import PortfolioOptProblem

@pytest.fixture
def small_test_problem():
    """Create a small test problem for unit tests."""
    generator = TestDataGenerator(seed=42)
    return generator.generate_realistic_problem(n_assets=10, n_periods=100)

@pytest.fixture
def medium_test_problem():
    """Create a medium test problem for integration tests."""
    generator = TestDataGenerator(seed=42)
    return generator.generate_realistic_problem(n_assets=50, n_periods=252)

@pytest.fixture
def large_test_problem():
    """Create a large test problem for performance tests."""
    generator = TestDataGenerator(seed=42)
    return generator.generate_realistic_problem(n_assets=200, n_periods=504)
```

### JavaScript/TypeScript Fixtures

For frontend tests, we use factory functions:

```typescript
// testUtils.ts
import { OptimizationResult } from '../types';

export const createMockResult = (overrides = {}): OptimizationResult => ({
  id: '123',
  weights: [0.4, 0.3, 0.2, 0.1],
  objective_value: 0.05,
  iterations: 15,
  converged: true,
  ...overrides
});

export const createMockProblem = (overrides = {}) => ({
  id: '123',
  n_assets: 4,
  n_periods: 252,
  ...overrides
});
```

## Mocking

### Python Mocking

We use the `unittest.mock` library for mocking:

```python
from unittest.mock import Mock, patch

def test_benchmark_runner_with_mock_solver():
    # Create a mock solver class
    mock_solver = Mock()
    mock_solver.solve.return_value = Mock(
        weights=np.array([0.25, 0.25, 0.25, 0.25]),
        objective_value=0.1,
        iterations=10,
        converged=True
    )
    
    # Patch the solver class
    with patch('portopt.solvers.classical.ClassicalSolver', return_value=mock_solver):
        # Test code that uses ClassicalSolver
        # ...
```

### JavaScript/TypeScript Mocking

We use Jest's mocking capabilities:

```typescript
// Mock API module
jest.mock('../api', () => ({
  fetchOptimizationResults: jest.fn().mockResolvedValue({
    id: '123',
    weights: [0.4, 0.3, 0.2, 0.1],
    objective_value: 0.05,
    iterations: 15,
    converged: true
  })
}));

// In test
import { fetchOptimizationResults } from '../api';

test('component uses API correctly', async () => {
  // Component renders and uses the API
  // ...
  
  // Check that API was called correctly
  expect(fetchOptimizationResults).toHaveBeenCalledWith('123');
});
```

## Continuous Integration

We use GitHub Actions for continuous integration:

```yaml
# .github/workflows/tests.yml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    - name: Run tests
      run: |
        pytest --cov=portopt
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## Test Coverage

We aim for high test coverage:

```bash
# Generate coverage report
pytest --cov=portopt --cov-report=html

# View coverage report
open htmlcov/index.html
```

Coverage goals:
- Overall coverage: >90%
- Core modules: >95%
- Utility modules: >85%

## Performance Testing

We use pytest-benchmark for performance testing:

```python
def test_solver_performance(benchmark):
    # Arrange
    problem = create_large_test_problem()
    solver = ClassicalSolver()
    
    # Act & Assert
    result = benchmark(solver.solve, problem)
    
    # Additional assertions
    assert result.converged
```

Run performance tests:

```bash
# Run all benchmark tests
pytest tests/test_performance/ --benchmark-only

# Compare against previous runs
pytest tests/test_performance/ --benchmark-compare=0001
```

## Testing Best Practices

1. **Write Tests First**: Follow test-driven development
2. **Keep Tests Simple**: Each test should test one thing
3. **Use Descriptive Names**: Test names should describe what is being tested
4. **Isolate Tests**: Tests should not depend on each other
5. **Test Edge Cases**: Include tests for boundary conditions
6. **Use Fixtures**: Reuse common setup code
7. **Mock External Dependencies**: Isolate the code being tested
8. **Test Failure Cases**: Test how code handles errors
9. **Keep Tests Fast**: Slow tests discourage testing
10. **Maintain Tests**: Update tests when code changes

## Troubleshooting

### Common Test Issues

1. **Flaky Tests**
   
   Tests that sometimes pass and sometimes fail are often due to:
   - Random data generation without fixed seeds
   - Time-dependent code
   - Order-dependent tests
   - External dependencies
   
   Solution: Use fixed seeds, mock time-dependent code, ensure test isolation.

2. **Slow Tests**
   
   Tests that take too long to run:
   - Use smaller problem sizes for unit tests
   - Mock expensive computations
   - Use the `--config config/test_configs/quick.ini` option
   
3. **Test Dependencies**
   
   Tests that depend on each other:
   - Refactor tests to be independent
   - Use fixtures for common setup
   - Reset state between tests

## Related Resources

- [Contributing Guide](./contributing.md)
- [Architecture Overview](./architecture.md)
- [Frontend Development Guide](./frontend-guide.md)
- [API Reference](../reference/api-reference.md)
