# Portfolio Optimization Tutorial

This tutorial provides an overview of portfolio optimization concepts and how to use the `portopt` library to solve portfolio optimization problems.

## Table of Contents

- [Introduction to Portfolio Optimization](#introduction-to-portfolio-optimization)
- [Mathematical Formulation](#mathematical-formulation)
- [Constraints in Portfolio Optimization](#constraints-in-portfolio-optimization)
- [Solver Approaches](#solver-approaches)
- [Examples](#examples)
- [Quantum Solvers](#quantum-solvers)

## Introduction to Portfolio Optimization

Portfolio optimization is the process of selecting the best asset allocation that maximizes expected returns for a given level of risk, or minimizes risk for a given level of expected return. This concept was introduced by Harry Markowitz in 1952 and forms the foundation of Modern Portfolio Theory (MPT).

The key insight of MPT is that an asset's risk and return should not be assessed individually, but by how it contributes to a portfolio's overall risk and return. By combining assets with different correlations, investors can construct portfolios that have better risk-return profiles than any individual asset.

## Mathematical Formulation

The standard portfolio optimization problem can be formulated as:

**Objective**: Minimize portfolio risk (variance) for a given level of expected return

```
minimize    w^T Σ w
subject to  w^T μ ≥ r_target
            sum(w) = 1
            w_i ≥ 0 for all i
```

Where:
- w is the vector of portfolio weights
- Σ is the covariance matrix of asset returns
- μ is the vector of expected returns
- r_target is the target return

Alternatively, we can maximize the Sharpe ratio (return per unit of risk):

```
maximize    (w^T μ - r_f) / sqrt(w^T Σ w)
subject to  sum(w) = 1
            w_i ≥ 0 for all i
```

Where r_f is the risk-free rate.

## Constraints in Portfolio Optimization

Real-world portfolio optimization often involves various constraints:

1. **Budget Constraint**: The sum of all weights must equal 1 (fully invested portfolio)
2. **Long-only Constraint**: All weights must be non-negative (no short selling)
3. **Position Limits**: Minimum and maximum weights for each asset
4. **Cardinality Constraints**: Limit on the number of assets with non-zero weights
5. **Group Constraints**: Limits on exposure to certain sectors, industries, or asset classes
6. **Turnover Constraints**: Limits on how much the portfolio can change from a previous allocation

Some of these constraints (like cardinality constraints) make the problem non-convex and much harder to solve using traditional optimization methods.

## Solver Approaches

The `portopt` library provides multiple solver approaches to tackle portfolio optimization problems:

### 1. Classical Solvers

Classical solvers use traditional optimization algorithms like Sequential Least Squares Programming (SLSQP) to find the optimal solution. These methods work well for convex problems with linear and non-linear constraints, but may struggle with non-convex constraints like cardinality constraints.

```python
from portopt.solvers import SolverFactory

factory = SolverFactory()
solver = factory.create_solver('classical')
result = solver.solve(problem)
```

### 2. Approximate/Heuristic Solvers

For more complex problems, especially those with non-convex constraints, heuristic methods can be more effective:

#### Genetic Algorithm

Genetic algorithms mimic the process of natural selection to evolve a population of potential solutions toward better solutions. They are particularly effective for problems with complex constraints and discrete variables.

```python
from portopt.solvers import SolverFactory

factory = SolverFactory()
solver = factory.create_solver('genetic', population_size=200, generations=100)
result = solver.solve(problem)
```

#### Simulated Annealing

Simulated annealing is inspired by the annealing process in metallurgy. It starts with a high "temperature" allowing for large random moves, and gradually "cools down" to refine the solution. This approach is good at avoiding local optima.

```python
from portopt.solvers import SolverFactory

factory = SolverFactory()
solver = factory.create_solver('annealing', initial_temp=200, iterations=2000)
result = solver.solve(problem)
```

### 3. Quantum Solvers

Quantum computing approaches leverage quantum mechanics to solve optimization problems. These are still experimental but show promise for certain types of problems:

- **QAOA** (Quantum Approximate Optimization Algorithm)
- **VQE** (Variational Quantum Eigensolver)

*Note: Quantum solvers are planned for future releases.*

## Quantum Solvers

Quantum computing offers a promising approach to solving complex optimization problems, including portfolio optimization. The `portopt` framework includes implementations of two quantum algorithms using Qiskit:

### Quantum Approximate Optimization Algorithm (QAOA)

QAOA is a hybrid quantum-classical algorithm designed for combinatorial optimization problems. It was first introduced by Farhi, Goldstone, and Gutmann in 2014 [1]. QAOA works by:

1. Encoding the portfolio optimization problem as a Quadratic Unconstrained Binary Optimization (QUBO) problem
2. Creating a parameterized quantum circuit with alternating problem and mixing Hamiltonians
3. Optimizing the circuit parameters using classical optimization techniques
4. Sampling from the optimized circuit to obtain portfolio weights

For portfolio optimization, QAOA is particularly well-suited for problems with discrete constraints, such as cardinality constraints (limiting the number of assets in the portfolio). The algorithm's performance typically improves with increasing circuit depth (number of QAOA layers).

### Variational Quantum Eigensolver (VQE)

VQE is another hybrid quantum-classical algorithm that finds the ground state of a Hamiltonian. It was first proposed by Peruzzo et al. in 2014 [2]. For portfolio optimization:

1. The objective function and constraints are encoded as a Hamiltonian
2. A parameterized quantum circuit (ansatz) is created
3. The circuit parameters are optimized to minimize the Hamiltonian's expectation value
4. The final state represents the optimal portfolio weights

VQE offers flexibility in the choice of ansatz, allowing for different circuit architectures to be explored. Our implementation supports:

- **RealAmplitudes**: A hardware-efficient ansatz using only RY rotations and CX gates
- **TwoLocal**: A more expressive ansatz with configurable entanglement patterns

### Hybrid Approach for Large Problems

Current quantum computers have limited qubit counts, making it challenging to solve large portfolio optimization problems directly. To address this, the framework implements a hybrid approach:

1. Divide the portfolio into smaller subproblems based on asset correlations
2. Solve each subproblem using quantum algorithms (QAOA or VQE)
3. Combine the solutions to form the complete portfolio

This approach allows quantum solvers to be applied to portfolios with many assets, while still leveraging quantum advantages for the subproblems. It's based on the divide-and-conquer principle similar to that described by Brandhofer et al. [3].

### Implementation Details

The quantum solver implementations in `portopt` use the latest Qiskit API and follow these key design principles:

1. **Modular Architecture**: Separating problem formulation, quantum circuit creation, and optimization
2. **Adaptability**: Automatic switching between full quantum and hybrid approaches based on problem size
3. **Extensibility**: Easy addition of new quantum algorithms or improvements to existing ones

The solvers use `SamplingVQE` from Qiskit Algorithms to handle the quantum processing and include proper error handling and result validation.

### Using Quantum Solvers

To use the quantum solvers, you can create them through the SolverFactory:

```python
from portopt.solvers.factory import SolverFactory

# Create a solver factory
factory = SolverFactory()

# Create a QAOA solver
qaoa_solver = factory.create_solver('qaoa', depth=2, shots=1024)

# Create a VQE solver
vqe_solver = factory.create_solver('vqe', ansatz_type='RealAmplitudes', depth=2)

# Solve a portfolio optimization problem
result = qaoa_solver.solve(problem)
```

The framework provides several pre-configured quantum solver types:

- `qaoa`: Basic QAOA implementation with depth=1
- `qaoa_deep`: QAOA with higher circuit depth (depth=3)
- `vqe`: VQE with RealAmplitudes ansatz
- `vqe_twolocal`: VQE with TwoLocal ansatz (more expressive)

Each solver can be further customized with parameters such as:

- `shots`: Number of measurement shots (default: 1024)
- `backend_name`: Quantum backend to use (default: 'aer_simulator')
- `optimizer_name`: Classical optimizer for parameter optimization ('COBYLA', 'SPSA', 'SLSQP')
- `max_iterations`: Maximum number of classical optimization iterations (default: 100)
- `max_assets_per_subproblem`: Maximum assets to include in each quantum subproblem (default: 5)

### Performance Considerations

When using quantum solvers, consider the following:

1. **Circuit Depth**: Higher QAOA depths or more complex ansatze improve solution quality but increase runtime and are more susceptible to quantum noise
2. **Shot Count**: More shots improve measurement statistics but increase runtime
3. **Optimizer Choice**: COBYLA is generally more stable but slower, while SPSA can converge faster but may be less reliable
4. **Problem Size**: The hybrid approach is automatically used for problems with more assets than `max_assets_per_subproblem`

### Example

See the `examples/quantum_optimization.py` script for a complete example of using quantum solvers for portfolio optimization and comparing them with classical approaches.

### References

[1] Farhi, E., Goldstone, J., & Gutmann, S. (2014). A quantum approximate optimization algorithm. arXiv preprint arXiv:1411.4028.

[2] Peruzzo, A., McClean, J., Shadbolt, P., Yung, M. H., Zhou, X. Q., Love, P. J., ... & O'Brien, J. L. (2014). A variational eigenvalue solver on a photonic quantum processor. Nature communications, 5(1), 4213.

[3] Brandhofer, N., Braun, D., & Johri, S. (2022). Quantum algorithms for portfolio optimization. Journal of Finance and Data Science, 8, 71-83.

[4] Herman, D., Googin, C., Liu, X., Galda, A., & Safro, I. (2022). A survey of quantum computing for finance. ACM Computing Surveys, 55(9), 1-37.

[5] Egger, D. J., Gutiérrez, R. G., Mestre, J. C., & Woerner, S. (2020). Credit risk analysis using quantum computers. IEEE Transactions on Computers, 70(12), 2136-2145.

## Examples

### Basic Portfolio Optimization

```python
import numpy as np
from portopt.core.problem import PortfolioOptProblem
from portopt.solvers import SolverFactory

# Create sample data
n_assets = 10
n_periods = 100
returns = np.random.normal(0.001, 0.02, (n_assets, n_periods))

# Create problem with basic constraints
problem = PortfolioOptProblem(
    returns=returns,
    constraints={
        'min_weight': 0.01,
        'max_weight': 0.3,
        'sum_to_one': True
    }
)

# Solve using classical solver
factory = SolverFactory()
solver = factory.create_solver('classical')
result = solver.solve(problem)

print(f"Optimal weights: {result.weights}")
print(f"Expected return: {result.objective_value}")
print(f"Solve time: {result.solve_time} seconds")
```

### Portfolio Optimization with Cardinality Constraints

```python
# Create problem with cardinality constraint
problem = PortfolioOptProblem(
    returns=returns,
    constraints={
        'min_weight': 0.05,
        'max_weight': 0.4,
        'sum_to_one': True,
        'cardinality': 5  # Only 5 assets can have non-zero weights
    }
)

# Solve using genetic algorithm
solver = factory.create_solver('genetic', population_size=200, generations=100)
result = solver.solve(problem)

print(f"Optimal weights: {result.weights}")
print(f"Number of assets used: {np.sum(result.weights > 0.01)}")
```

For more detailed examples, please refer to the examples directory in the repository.
