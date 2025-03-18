# Architecture Overview

This document provides a comprehensive overview of the Portfolio Optimization Testbed architecture, helping developers understand the system design and component interactions.

## System Architecture Diagram

![System Architecture](../assets/images/system-architecture.png)

### Core Components

1. **Problem Definition**: Encapsulates portfolio optimization problems
2. **Solvers**: Implements various optimization algorithms
3. **Constraints**: Defines portfolio constraints
4. **Metrics**: Calculates performance and risk metrics
5. **Benchmarking**: Evaluates solver performance
6. **Visualization**: Provides interactive dashboards

## Component Interactions

The following diagram illustrates how the components interact:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Test Data      │────▶│  Problem        │────▶│  Solver         │
│  Generator      │     │  Definition     │     │  Framework      │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Dashboard      │◀────│  Performance    │◀────│  Optimization   │
│  Visualization  │     │  Metrics        │     │  Result         │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Directory Structure

The codebase is organized into the following directory structure:

```
portopt/
├── __init__.py                 # Package initialization
├── benchmark/                  # Benchmarking framework
│   ├── __init__.py
│   └── runner.py               # Benchmark execution
├── config/                     # Configuration management
│   ├── __init__.py
│   └── test_configs/           # Preset configurations
├── constraints/                # Portfolio constraints
│   ├── __init__.py
│   └── base.py                 # Base constraint classes
├── core/                       # Core functionality
│   ├── __init__.py
│   ├── problem.py              # Problem definition
│   └── objective.py            # Optimization objectives
├── data/                       # Data handling
│   ├── __init__.py
│   └── generator.py            # Test data generation
├── impact/                     # Market impact modeling
│   ├── __init__.py
│   └── models.py               # Impact models
├── metrics/                    # Performance metrics
│   ├── __init__.py
│   ├── performance.py          # Return metrics
│   └── risk.py                 # Risk metrics
├── solvers/                    # Optimization solvers
│   ├── __init__.py
│   ├── base.py                 # Base solver class
│   ├── classical.py            # Classical solver
│   └── multithreaded.py        # Multithreaded solver
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── debug.py                # Debugging tools
│   └── logging.py              # Logging utilities
└── visualization/              # Visualization tools
    ├── __init__.py
    ├── dashboard.py            # Dashboard creation
    └── plots.py                # Plotting functions

frontend/                       # Frontend implementation
├── src/                        # Source code
│   ├── components/             # React components
│   ├── hooks/                  # Custom React hooks
│   ├── pages/                  # Page definitions
│   └── utils/                  # Utility functions
├── public/                     # Static assets
└── package.json                # Dependencies

tests/                          # Test suite
├── __init__.py
├── test_solvers/               # Solver tests
├── test_constraints/           # Constraint tests
└── test_metrics/               # Metrics tests
```

## Key Design Patterns

The system implements several design patterns to promote maintainability and extensibility:

### 1. Strategy Pattern

The solver framework uses the Strategy pattern to allow different optimization algorithms to be used interchangeably:

```python
# Base strategy
class BaseSolver(ABC):
    @abstractmethod
    def solve(self, problem, constraints, objective):
        pass

# Concrete strategies
class ClassicalSolver(BaseSolver):
    def solve(self, problem, constraints, objective):
        # Implementation

class MultiThreadedSolver(BaseSolver):
    def solve(self, problem, constraints, objective):
        # Implementation
```

### 2. Factory Pattern

The test data generator uses the Factory pattern to create problem instances:

```python
class TestDataGenerator:
    def generate_realistic_problem(self, n_assets, n_periods):
        # Create and return a problem instance
        
    def generate_random_problem(self, n_assets, n_periods):
        # Create and return a problem instance
```

### 3. Composite Pattern

The constraint system uses the Composite pattern to combine multiple constraints:

```python
class CompositeConstraint(Constraint):
    def __init__(self, constraints):
        self.constraints = constraints
        
    def evaluate(self, weights):
        return all(c.evaluate(weights) for c in self.constraints)
```

### 4. Observer Pattern

The optimization process uses the Observer pattern to track progress:

```python
class OptimizationDebugger:
    def __init__(self):
        self.iterations = []
        
    def update(self, iteration, weights, objective_value):
        self.iterations.append({
            'iteration': iteration,
            'weights': weights.copy(),
            'objective_value': objective_value
        })
```

## Data Flow

The following diagram illustrates the data flow through the system:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │     │             │
│  Historical │────▶│  Problem    │────▶│  Optimizer  │────▶│  Portfolio  │
│  Data       │     │  Definition │     │  Engine     │     │  Weights    │
│             │     │             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                           │                                       │
                           │                                       │
                           ▼                                       ▼
                    ┌─────────────┐                        ┌─────────────┐
                    │             │                        │             │
                    │  Constraint │                        │  Performance │
                    │  Definitions│                        │  Metrics    │
                    │             │                        │             │
                    └─────────────┘                        └─────────────┘
```

## Core Classes

### PortfolioOptProblem

The central data structure for portfolio optimization problems:

```python
class PortfolioOptProblem:
    def __init__(
        self,
        returns,            # Historical returns
        volumes=None,       # Trading volumes
        spreads=None,       # Bid-ask spreads
        factor_returns=None,  # Factor returns
        factor_exposures=None,  # Factor exposures
        industry_classes=None,  # Industry classifications
        asset_classes=None,  # Asset class classifications
        currencies=None,    # Currency classifications
        credit_ratings=None,  # Credit rating classifications
        validate=True       # Whether to validate the problem
    ):
        # Initialize problem
```

### BaseSolver

Abstract base class for portfolio optimization solvers:

```python
class BaseSolver(ABC):
    @abstractmethod
    def solve(
        self,
        problem,            # Portfolio optimization problem
        initial_weights=None,  # Initial portfolio weights
        constraints=None,   # List of constraints
        objective=None,     # Optimization objective
        debugger=None       # Debugger for tracking progress
    ):
        # Solve the optimization problem
```

### Constraint

Abstract base class for portfolio constraints:

```python
class Constraint(ABC):
    @abstractmethod
    def evaluate(self, weights):
        # Evaluate constraint satisfaction
        
    @abstractmethod
    def gradient(self, weights):
        # Compute constraint gradient
        
    @abstractmethod
    def penalty(self, weights):
        # Compute penalty for constraint violation
```

### BenchmarkRunner

Runs benchmarks for portfolio optimization algorithms:

```python
class BenchmarkRunner:
    def __init__(self, output_dir="benchmark_results", config_file=None):
        # Initialize benchmark runner
        
    def run_size_scaling_benchmark(
        self,
        solver_classes,     # List of solver classes
        n_assets_range,     # Range of asset counts
        n_periods_range,    # Range of time periods
        n_runs=3            # Number of runs per configuration
    ):
        # Run benchmark and return results
```

## Frontend Architecture

The dashboard frontend is built using React and follows a component-based architecture:

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│                      App Container                      │
│                                                         │
├─────────────┬─────────────┬─────────────┬─────────────┐│
││            ││            ││            ││            ││
││  Risk      ││  Market    ││  Performance│  Constraints││
││  Analysis  ││  Impact    ││  Tab       ││  Tab       ││
││  Tab       ││  Tab       ││            ││            ││
│└────────────┘└────────────┘└────────────┘└────────────┘│
│                                                         │
│┌─────────────────────────────────────────────────────┐ │
││                                                     │ │
││                  Chart Components                   │ │
││                                                     │ │
│└─────────────────────────────────────────────────────┘ │
│                                                         │
│┌─────────────────────────────────────────────────────┐ │
││                                                     │ │
││                  Control Components                 │ │
││                                                     │ │
│└─────────────────────────────────────────────────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

The frontend communicates with the backend API to retrieve optimization results and metrics.

## API Endpoints

The backend API provides the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/problems` | GET | List available problems |
| `/api/problems/<id>` | GET | Get problem details |
| `/api/solve` | POST | Solve optimization problem |
| `/api/results/<id>` | GET | Get optimization results |
| `/api/metrics/<id>` | GET | Get performance metrics |
| `/api/benchmarks` | GET | List available benchmarks |
| `/api/benchmarks/<id>` | GET | Get benchmark results |

## Extension Points

The system is designed to be extended in several ways:

### 1. Adding New Solvers

To add a new solver:

1. Create a new class that inherits from `BaseSolver`
2. Implement the `solve()` method
3. Register the solver in the solver registry

```python
class NewSolver(BaseSolver):
    def solve(self, problem, initial_weights=None, constraints=None, objective=None, debugger=None):
        # Implementation
        return result
```

### 2. Adding New Constraints

To add a new constraint:

1. Create a new class that inherits from `Constraint`
2. Implement the required methods
3. Use the constraint in optimization

```python
class NewConstraint(Constraint):
    def evaluate(self, weights):
        # Implementation
        
    def gradient(self, weights):
        # Implementation
        
    def penalty(self, weights):
        # Implementation
```

### 3. Adding New Metrics

To add a new metric:

1. Create a new function in the appropriate metrics module
2. Implement the metric calculation
3. Update the dashboard to display the metric

```python
def calculate_new_metric(weights, returns, parameters):
    # Implementation
    return metric_value
```

### 4. Adding New Visualizations

To add a new visualization:

1. Create a new function in the visualization module
2. Implement the visualization
3. Add the visualization to the dashboard

```python
def plot_new_visualization(result, problem, parameters):
    # Implementation
    return figure
```

## Performance Considerations

The system is designed with performance in mind:

1. **Memory Management**: Large matrices are handled efficiently
2. **Parallelization**: Computationally intensive tasks can be parallelized
3. **Caching**: Intermediate results are cached when appropriate
4. **Lazy Evaluation**: Some calculations are deferred until needed

## Security Considerations

While the system is primarily for research and analysis, security is still important:

1. **Input Validation**: All inputs are validated before processing
2. **Error Handling**: Errors are caught and handled gracefully
3. **Logging**: Activities are logged for debugging and auditing
4. **Configuration**: Sensitive configuration is separated from code

## Related Resources

- [Contributing Guide](./contributing.md)
- [Testing Guide](./testing.md)
- [Frontend Development Guide](./frontend-guide.md)
- [API Reference](../reference/api-reference.md)
