# Quantum Portfolio Optimization Testbed

A Python package for comparing classical and quantum approaches to portfolio optimization.

## Installation

```bash
pip install -e .
```

## Usage

```python
from quantum_portfolio_opt import TestDataGenerator, ClassicalSolver

# Generate a test problem
generator = TestDataGenerator()
problem = generator.generate_realistic_problem(n_assets=50, n_periods=252)

# Solve using classical optimizer
solver = ClassicalSolver()
result = solver.solve(problem)
print(result)
```

## Project Structure

- `quantum_portfolio_opt/`: Main package directory
  - `core/`: Core problem and result definitions
  - `solvers/`: Optimization solvers (classical and quantum)
  - `data/`: Data generation and handling
  - `utils/`: Utility functions

## Development

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Run tests:
```bash
pytest
```

