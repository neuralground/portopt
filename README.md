# Portfolio Optimization Testbed

A Python package for comparing classical and quantum approaches to portfolio optimization.

## Overview
This package provides a testbed for experimenting with different portfolio optimization approaches, with particular focus on:
- Classical optimization methods
- Approximate/heuristic optimization methods
- Quantum optimization algorithms
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
from portopt import TestDataGenerator
from portopt.solvers import SolverFactory
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

# Create a solver factory
factory = SolverFactory()

# Create a classical solver
solver = factory.create_solver('classical', 
    max_iterations=20,
    initial_penalty=100.0
)

# Solve the problem
result = solver.solve(problem)

print(result)
```

### Using Multiple Solver Types
The package supports various solver types through a unified interface:

```python
from portopt.solvers import SolverFactory

# Create a solver factory
factory = SolverFactory()

# List available solvers
available_solvers = factory.get_available_solvers()
print(available_solvers)

# Create different solver types
classical_solver = factory.create_solver('classical')
genetic_solver = factory.create_solver('genetic', population_size=100)
annealing_solver = factory.create_solver('annealing', initial_temp=100.0)

# Quantum solvers
qaoa_solver = factory.create_solver('qaoa', depth=1, shots=1024)
vqe_solver = factory.create_solver('vqe', ansatz_type='RealAmplitudes')
```

### Quantum Solver Features

The package provides quantum computing approaches to portfolio optimization using Qiskit:

- **Quantum Approximate Optimization Algorithm (QAOA)**:
  - Designed for combinatorial optimization problems
  - Configurable circuit depth for control over solution quality
  - Automatic problem conversion to QUBO format
  - Handles both small problems (full quantum) and large problems (hybrid approach)

- **Variational Quantum Eigensolver (VQE)**:
  - Finds the minimum eigenvalue of the problem Hamiltonian
  - Multiple ansatz options: 'RealAmplitudes' (default) and 'TwoLocal'
  - Configurable circuit depth and shots

- **Hybrid Quantum-Classical Approach**:
  - Automatically divides large problems into smaller subproblems
  - Solves subproblems using quantum algorithms
  - Combines solutions for the complete portfolio

Example of using quantum solvers:

```python
# Create and configure a QAOA solver
qaoa_solver = factory.create_solver(
    'qaoa',
    depth=2,                  # Number of QAOA layers
    shots=1024,               # Number of measurement shots
    backend_name='aer_simulator',  # Quantum backend
    optimizer_name='COBYLA',  # Classical optimizer
    max_iterations=100        # Maximum optimizer iterations
)

# Create a VQE solver with TwoLocal ansatz
vqe_solver = factory.create_solver(
    'vqe',
    ansatz_type='TwoLocal',   # Type of ansatz
    depth=2,                  # Circuit depth
    shots=1024,               # Measurement shots
    backend_name='aer_simulator'
)

# Solve a portfolio optimization problem
result = qaoa_solver.solve(problem)
print(f"Optimal weights: {result.weights}")
print(f"Objective value: {result.objective}")
```

See the [quantum_optimization.py](./examples/quantum_optimization.py) example for a complete demonstration of quantum solvers.

### Solver Types

The package currently includes the following solver types:

1. **Classical Solvers**
   - `classical`: Sequential Least Squares Programming (SLSQP) solver

2. **Approximate/Heuristic Solvers**
   - `genetic`: Genetic Algorithm solver
   - `advanced_genetic`: Advanced Genetic Algorithm solver with island model and multi-objective capabilities
   - `advanced_genetic_multi`: Advanced Genetic Algorithm solver with multi-objective optimization
   - `annealing`: Simulated Annealing solver

3. **Quantum Solvers**
   - `qaoa`: Quantum Approximate Optimization Algorithm
   - `vqe`: Variational Quantum Eigensolver

4. **Statistical Models**
   - `black_litterman`: Black-Litterman model for blending market equilibrium with investor views
   - `black_litterman_conservative`: Black-Litterman model with higher risk aversion

See the [black_litterman_example.py](./examples/black_litterman_example.py) example for a demonstration of the Black-Litterman model.

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
solver = factory.create_solver('classical')
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
