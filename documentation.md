# Portfolio Optimization Test Harness Documentation

## Overview
The Portfolio Optimization Test Harness provides a comprehensive framework for experimenting with and evaluating portfolio optimization algorithms. The system is designed to be modular, extensible, and focused on thorough testing across multiple dimensions of portfolio optimization challenges.

## Core Components

### Problem Definition (`portopt/core/problem.py`)
The `PortfolioOptProblem` class serves as the central data structure for portfolio optimization problems. Key features:
- Historical returns matrix
- Market impact data (volumes, spreads)
- Factor model data (returns, exposures)
- Classification data (industry, asset class, currency, credit)
- Built-in validation of problem instances

### Solvers
The system implements multiple solver approaches:

#### Classical Solver (`portopt/solvers/classical.py`)
- Sequential relaxation approach
- Handles non-linear constraints
- Supports warm starting
- Built-in penalty adjustment mechanism

#### Multithreaded Solver (`portopt/solvers/MultiThreadedSolver.py`)
- Parallel optimization attempts
- Enhanced exploration of solution space
- Suitable for larger problem instances

### Benchmarking (`portopt/benchmark/runner.py`)
The `BenchmarkRunner` class provides:
- Systematic performance evaluation
- Size scaling analysis
- Stress scenario testing
- Comprehensive metrics collection
- Automated reporting

### Market Impact (`portopt/impact/`)
Sophisticated market impact modeling with:
- Separate permanent and temporary impact
- Volume-based participation limits
- Spread cost modeling
- Decay effects

### Risk and Performance Metrics
Comprehensive metric calculation including:
- Standard risk metrics (VaR, CVaR)
- Factor exposures and attribution
- Transaction cost analysis
- Liquidity scoring

## Key Features

### Configuration Management
- Hierarchical configuration system
- Environment-specific settings
- Test-specific parameter sets
- Default fallbacks

### Visualization
- Interactive dashboard
- Risk decomposition plots
- Impact analysis visualization
- Performance attribution charts

### Constraint Handling
- Position limits
- Sector constraints
- Turnover controls
- Market impact limits
- Factor exposure bounds

## Usage Examples

### Basic Usage
```python
from portopt import TestDataGenerator, ClassicalSolver

# Generate test problem
generator = TestDataGenerator()
problem = generator.generate_realistic_problem(
    n_assets=50,
    n_periods=252
)

# Create and run solver
solver = ClassicalSolver(max_iterations=20)
result = solver.solve(problem)
```

### Running Benchmarks
```python
from portopt.benchmark.runner import BenchmarkRunner

runner = BenchmarkRunner()
results = runner.run_size_scaling_benchmark(
    solver_classes=[ClassicalSolver],
    n_assets_range=[50, 100, 200],
    n_periods_range=[252, 504]
)
```

## Development Guidelines

### Adding New Solvers
1. Inherit from `BaseSolver`
2. Implement `solve()` method
3. Add solver-specific configuration
4. Include comprehensive tests
5. Document performance characteristics

### Testing
- Unit tests for all components
- Integration tests for workflows
- Performance benchmarks
- Stress testing scenarios

### Documentation Standards
- Docstrings for all public methods
- Type hints throughout
- README updates for new features
- Example updates as needed

## Future Extensions
The system is designed to be extended in several directions:

1. Additional Solvers
   - Quantum algorithms
   - Machine learning approaches
   - Hybrid methods

2. Enhanced Metrics
   - ESG constraints
   - Alternative risk measures
   - Custom objective functions

3. Market Impact
   - More sophisticated decay models
   - Cross-impact effects
   - Adaptive parameters

4. Visualization
   - Real-time monitoring
   - Interactive analysis
   - Custom reporting

## Troubleshooting

### Common Issues
1. Memory Usage
   - Use appropriate problem sizes
   - Monitor solver memory patterns
   - Consider multithreaded solver for large problems

2. Convergence
   - Adjust solver parameters
   - Check constraint feasibility
   - Monitor iteration progress

3. Performance
   - Use appropriate configuration
   - Consider problem preprocessing
   - Monitor system resources

## Contributing
Guidelines for contributing to the test harness:

1. Code Style
   - Follow PEP 8
   - Use type hints
   - Maintain documentation

2. Testing
   - Add tests for new features
   - Maintain coverage
   - Include performance tests

3. Documentation
   - Update relevant docs
   - Include examples
   - Describe performance characteristics