# Portfolio Optimization Testbed

A Python package for comparing classical and quantum approaches to portfolio optimization.

## Overview
This package provides a testbed for experimenting with different portfolio optimization approaches, with particular focus on:
- Classical optimization methods
- Risk-based portfolio construction
- Constraint handling
- Performance benchmarking

## Documentation

Comprehensive documentation is available in the [docs](./docs) directory:

- [Documentation Index](./docs/index.md) - Complete overview of all documentation resources
- [Getting Started](./docs/getting-started/quick-start.md) - Quick introduction to using the testbed
- [Concepts](./docs/concepts/portfolio-tutorial.md) - Learn about portfolio optimization concepts
- [Workflow Guide](./docs/user-guides/workflow-guide.md) - Step-by-step workflow for portfolio optimization
- [Examples](./docs/examples/) - Practical examples of using the testbed
- [API Reference](./docs/reference/) - Detailed API documentation
- [Developer Guides](./docs/developer-guides/) - Resources for developers

### Interactive Examples

The repository includes interactive Jupyter notebooks for hands-on learning:

- [Minimum Variance Portfolio](./docs/examples/notebooks/minimum_variance_portfolio.ipynb) - Interactive example of minimum variance portfolio optimization

## Installation

### Development Installation
```bash
# Clone the repository
git clone <repository-url>
cd portopt

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all extras
pip install -e ".[dev]"
```

## Usage

### Basic Usage
```python
from portopt import TestDataGenerator, ClassicalSolver
from portopt.utils.logging import setup_logging, OptimizationLogger

# Set up logging
setup_logging(level="INFO")
logger = OptimizationLogger("example")

# Generate test data
generator = TestDataGenerator()
problem = generator.generate_realistic_problem(
    n_assets=50,
    n_periods=252
)

# Create and run solver
solver = ClassicalSolver(
    max_iterations=20,
    initial_penalty=100.0
)
result = solver.solve(problem)

print(result)
```

### Configuration
The package uses configuration files to control optimization parameters and test settings. Three preset configurations are provided:

- `quick.ini`: Fast tests with small problem sizes
- `thorough.ini`: Comprehensive tests with medium problem sizes
- `stress.ini`: Stress tests with large problem sizes

Example configuration usage:
```bash
# Run tests with quick configuration
pytest tests/test_solver_performance.py -s --config config/test_configs/quick.ini

# Run tests with thorough configuration
pytest tests/test_solver_performance.py -s --config config/test_configs/thorough.ini
```

### Debugging Tools
The package includes utilities for debugging optimization problems:

```python
from portopt.utils.debug import OptimizationDebugger

# Initialize debugger
debugger = OptimizationDebugger(debug_dir="debug_output")

# Solve with debugging enabled
solver = ClassicalSolver()
result = solver.solve(problem, debugger=debugger)

# Generate debug report
debugger.save_report()
```

The debug report includes:
- Convergence analysis
- Constraint violation tracking
- Iteration-by-iteration metrics
- Performance statistics

## Examples

For more detailed examples, see the [examples directory](./docs/examples):

- [Minimum Variance Portfolio](./docs/examples/minimum-variance-portfolio.md) - How to construct a minimum variance portfolio
- [Maximum Sharpe Ratio Portfolio](./docs/examples/maximum-sharpe-ratio-portfolio.md) - How to construct a maximum Sharpe ratio portfolio
- [Risk Parity Portfolio](./docs/examples/risk-parity-portfolio.md) - How to construct a risk parity portfolio

### Interactive Notebooks

For interactive learning, explore our Jupyter notebooks:

- [Minimum Variance Portfolio Notebook](./docs/examples/notebooks/minimum_variance_portfolio.ipynb) - Interactive example with visualizations and detailed explanations

## Development

### Developer Documentation

Resources for developers who want to contribute to or extend the testbed:

- [Architecture Overview](./docs/developer-guides/architecture.md) - Overview of the system architecture
- [Architecture Diagrams](./docs/developer-guides/architecture-diagram.md) - Visual representations of the system architecture
- [Code Documentation Standards](./docs/developer-guides/code-documentation-standards.md) - Standards for documenting code
- [Documentation Testing Guide](./docs/developer-guides/documentation-testing.md) - Guide for testing documentation
- [Contributing Guide](./docs/developer-guides/contributing.md) - How to contribute to the project

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=portopt

# Run specific test configuration
pytest tests/test_solver_performance.py --config config/test_configs/thorough.ini
```

### Code Style
The project uses:
- Black for code formatting
- isort for import sorting
- mypy for type checking
- flake8 for linting

Run code quality checks:
```bash
# Format code
black portopt tests

# Sort imports
isort portopt tests

# Type checking
mypy portopt

# Linting
flake8 portopt tests
```

## Contributing
Contributions are welcome! Please ensure:
1. Tests pass and coverage is maintained
2. Code is formatted with black and passes linting
3. Documentation is updated as needed
4. Type hints are used consistently

For more details, see the [Contributing Guide](./docs/developer-guides/contributing.md).

## License
This project is licensed under the MIT License - see the LICENSE file for details.
