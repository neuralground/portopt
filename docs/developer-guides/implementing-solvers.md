# Implementing New Solvers

This guide provides instructions for implementing new solver types in the Portfolio Optimization framework.

## Table of Contents

- [Overview](#overview)
- [Solver Architecture](#solver-architecture)
- [Implementing a New Solver](#implementing-a-new-solver)
- [Registering with the SolverFactory](#registering-with-the-solverfactory)
- [Testing Your Solver](#testing-your-solver)
- [Examples](#examples)

## Overview

The Portfolio Optimization framework is designed to support multiple solver approaches, including:

1. **Classical solvers**: Traditional optimization algorithms like SLSQP
2. **Approximate/heuristic solvers**: Metaheuristic approaches like Genetic Algorithms
3. **Quantum solvers**: Quantum computing algorithms like QAOA and VQE

This modular design allows researchers and developers to experiment with different optimization approaches and compare their performance on portfolio optimization problems.

## Solver Architecture

The solver architecture consists of several key components:

1. **BaseSolver**: Abstract base class that all solvers must inherit from
2. **ConstraintAdapter**: Utility for converting constraints between different formats
3. **SolverFactory**: Factory for creating and configuring solver instances

### BaseSolver

The `BaseSolver` class provides common functionality for all solvers, including:

- Problem validation and preprocessing
- Constraint handling and validation
- Weight processing and normalization
- Performance metrics calculation

All solvers must inherit from `BaseSolver` and implement the `solve` method.

### ConstraintAdapter

The `ConstraintAdapter` class provides methods for converting constraints between different formats:

- `to_scipy_constraints`: Converts to SciPy format for classical solvers
- `to_penalty_functions`: Converts to penalty functions for heuristic solvers
- `create_bounds`: Creates bounds for optimization variables

### SolverFactory

The `SolverFactory` class provides a unified interface for creating and configuring solvers:

- `register_solver`: Registers a new solver type with default parameters
- `create_solver`: Creates a solver instance with specified parameters
- `get_available_solvers`: Returns a list of available solver types
- `get_solver_parameters`: Returns default parameters for a solver type

## Implementing a New Solver

To implement a new solver, follow these steps:

1. Create a new class that inherits from `BaseSolver`
2. Implement the `solve` method
3. Register the solver with the `SolverFactory`

### Step 1: Create a New Solver Class

Create a new class that inherits from `BaseSolver`:

```python
from portopt.solvers.base import BaseSolver
from portopt.core.problem import PortfolioOptProblem
from portopt.core.result import PortfolioOptResult

class MySolver(BaseSolver):
    """My custom solver implementation."""
    
    def __init__(self, **kwargs):
        """Initialize the solver.
        
        Args:
            **kwargs: Solver-specific parameters
        """
        super().__init__(**kwargs)
        self.my_param = kwargs.get('my_param', 'default_value')
        self.iterations = kwargs.get('iterations', 100)
    
    def solve(self, problem: PortfolioOptProblem) -> PortfolioOptResult:
        """Solve the portfolio optimization problem.
        
        Args:
            problem: The portfolio optimization problem to solve
            
        Returns:
            PortfolioOptResult containing the optimized weights
        """
        # Preprocess the problem
        processed_problem = self.preprocess_problem(problem)
        
        # Implement your solver logic here
        # ...
        
        # Create and return result
        return PortfolioOptResult(
            weights=weights,
            objective_value=objective_value,
            solve_time=solve_time,
            feasible=feasible,
            iterations_used=iterations
        )
```

### Step 2: Implement the Solve Method

The `solve` method should:

1. Preprocess the problem using `self.preprocess_problem`
2. Implement your solver-specific logic
3. Return a `PortfolioOptResult` with the optimized weights

For example, a simple random search solver might look like:

```python
def solve(self, problem: PortfolioOptProblem) -> PortfolioOptResult:
    """Solve using random search."""
    import time
    import numpy as np
    
    start_time = time.time()
    
    # Preprocess the problem
    processed_problem = self.preprocess_problem(problem)
    
    # Get penalty functions for constraints
    penalties = ConstraintAdapter.to_penalty_functions(processed_problem)
    
    # Get bounds for the weights
    bounds = ConstraintAdapter.create_bounds(processed_problem)
    
    # Initialize with random weights
    best_weights = np.random.random(processed_problem.n_assets)
    best_weights = best_weights / np.sum(best_weights)
    best_objective = float('inf')
    
    # Random search
    for i in range(self.iterations):
        # Generate random weights
        weights = np.random.random(processed_problem.n_assets)
        weights = weights / np.sum(weights)
        
        # Calculate objective value
        objective = self.calculate_objective(weights, processed_problem)
        
        # Apply penalties
        for penalty_func, penalty_weight in penalties:
            objective += penalty_weight * penalty_func(weights)
        
        # Update best solution
        if objective < best_objective:
            best_objective = objective
            best_weights = weights.copy()
    
    # Process weights to handle minimum weight threshold
    min_weight = processed_problem.constraints.get('min_weight', 0.0)
    best_weights = self.process_weights(best_weights, min_weight)
    
    # Check if solution is feasible
    constraint_results = self.check_constraints(best_weights, processed_problem)
    feasible = all(constraint_results.values())
    
    # Calculate final objective value
    objective_value = self.calculate_objective(best_weights, processed_problem)
    
    # Create result
    solve_time = time.time() - start_time
    result = PortfolioOptResult(
        weights=best_weights,
        objective_value=objective_value,
        solve_time=solve_time,
        feasible=feasible,
        iterations_used=self.iterations
    )
    
    return result
```

## Registering with the SolverFactory

To make your solver available through the `SolverFactory`, you need to register it:

```python
from portopt.solvers.factory import SolverFactory
from my_module import MySolver

# Create a factory
factory = SolverFactory()

# Register your solver
factory.register_solver('my_solver', MySolver, {
    'my_param': 'default_value',
    'iterations': 100
})

# Now you can create instances of your solver
solver = factory.create_solver('my_solver', iterations=200)
```

To make your solver available by default, add it to the `__init__` method of the `SolverFactory` class in `portopt/solvers/factory.py`:

```python
def __init__(self):
    """Initialize the solver factory with default solver registrations."""
    self._solvers = {}
    self._default_params = {}
    
    # Register default solvers
    # ...
    
    # Register your solver
    self.register_solver('my_solver', MySolver, {
        'my_param': 'default_value',
        'iterations': 100
    })
```

## Testing Your Solver

Create a test file for your solver in the `tests` directory:

```python
import pytest
import numpy as np

from portopt.core.problem import PortfolioOptProblem
from my_module import MySolver

class TestMySolver:
    """Tests for MySolver."""
    
    @pytest.fixture
    def sample_problem(self) -> PortfolioOptProblem:
        """Create a sample portfolio optimization problem."""
        # Create a sample problem for testing
        # ...
    
    def test_my_solver_initialization(self):
        """Test solver initialization."""
        solver = MySolver()
        assert solver.my_param == 'default_value'
        assert solver.iterations == 100
    
    def test_my_solver_solve(self, sample_problem):
        """Test that the solver can solve a basic problem."""
        solver = MySolver()
        result = solver.solve(sample_problem)
        
        # Check that weights sum to 1
        assert np.isclose(np.sum(result.weights), 1.0)
        
        # Check that weights are within bounds
        assert np.all(result.weights >= 0.0)
        
        # Check that solution is feasible
        assert result.feasible
```

Run the tests using pytest:

```bash
python -m pytest -xvs tests/test_my_solver.py
```

## Examples

For examples of different solver implementations, see:

- `portopt/solvers/classical.py`: Classical SLSQP solver
- `portopt/solvers/approximate.py`: Basic Genetic Algorithm and Simulated Annealing solvers
- `portopt/solvers/advanced_genetic.py`: Advanced Genetic Algorithm with island model and multi-objective capabilities
- `portopt/solvers/quantum.py`: QAOA and VQE quantum solvers

These examples demonstrate different approaches to implementing solvers within the framework.

### Advanced Genetic Solver Features

The `AdvancedGeneticSolver` in `advanced_genetic.py` provides an excellent example of implementing advanced metaheuristic features:

1. **Island Model**: Maintains multiple sub-populations that evolve independently and occasionally exchange individuals, improving exploration of the solution space.

2. **Adaptive Rates**: Dynamically adjusts mutation and crossover rates based on population diversity, preventing premature convergence.

3. **Multi-Objective Optimization**: Balances risk and return objectives with configurable weights, allowing for more nuanced portfolio construction.

4. **Diversity Preservation**: Implements niching and other techniques to maintain population diversity.

5. **Early Stopping**: Monitors improvement over generations and can stop early if no progress is detected.

When implementing your own advanced solvers, consider these techniques for improving performance and solution quality.
