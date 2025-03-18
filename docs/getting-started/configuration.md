# Configuration Guide

This guide explains how to configure the Portfolio Optimization Testbed for different use cases and environments.

## Configuration System

The Portfolio Optimization Testbed uses a hierarchical configuration system that allows you to customize various aspects of the optimization process, benchmarking, and visualization.

### Configuration Files

Configuration files use the INI format and can be placed in several locations:

1. **System-wide configuration**: `/etc/portopt/config.ini`
2. **User configuration**: `~/.config/portopt/config.ini`
3. **Project configuration**: `./config/config.ini`
4. **Custom configuration**: Specified via command-line or API

Configuration files are loaded in the order listed above, with later files overriding earlier ones.

## Basic Configuration Structure

A typical configuration file has the following structure:

```ini
[general]
log_level = INFO
output_dir = results

[solver]
max_iterations = 100
tolerance = 1e-6
initial_penalty = 10.0
penalty_growth_factor = 2.0
use_warm_start = true
solver_backend = osqp

[constraints]
position_lower_bound = 0.0
position_upper_bound = 0.05
sector_deviation = 0.1
max_turnover = 0.2

[benchmark]
n_runs = 3
timeout = 300
save_results = true
```

## Preset Configurations

The package includes several preset configurations for different use cases:

### Quick Configuration

For fast tests with small problem sizes:

```ini
# config/test_configs/quick.ini
[general]
log_level = INFO

[test_data]
n_assets = 20
n_periods = 100
n_factors = 3

[solver]
max_iterations = 20
tolerance = 1e-4
```

### Thorough Configuration

For comprehensive tests with medium problem sizes:

```ini
# config/test_configs/thorough.ini
[general]
log_level = INFO

[test_data]
n_assets = 100
n_periods = 252
n_factors = 5

[solver]
max_iterations = 100
tolerance = 1e-6
```

### Stress Configuration

For stress tests with large problem sizes:

```ini
# config/test_configs/stress.ini
[general]
log_level = INFO

[test_data]
n_assets = 500
n_periods = 504
n_factors = 10

[solver]
max_iterations = 200
tolerance = 1e-8
```

## Configuration Sections

### General Configuration

```ini
[general]
# Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
log_level = INFO

# Directory for output files
output_dir = results

# Random seed for reproducibility
random_seed = 42

# Enable/disable progress bars
show_progress = true
```

### Solver Configuration

```ini
[solver]
# Maximum number of iterations
max_iterations = 100

# Convergence tolerance
tolerance = 1e-6

# Initial penalty parameter
initial_penalty = 10.0

# Penalty growth factor
penalty_growth_factor = 2.0

# Whether to use warm starting
use_warm_start = true

# Backend solver: osqp, cvxopt, scipy
solver_backend = osqp

# Number of threads for parallel solvers
n_threads = 4
```

### Constraints Configuration

```ini
[constraints]
# Position limits
position_lower_bound = 0.0
position_upper_bound = 0.05

# Sector constraints
sector_deviation = 0.1

# Turnover constraints
max_turnover = 0.2

# Factor exposure constraints
factor_exposure_bound = 0.2
```

### Benchmark Configuration

```ini
[benchmark]
# Number of runs per configuration
n_runs = 3

# Timeout in seconds
timeout = 300

# Whether to save results
save_results = true

# Output format: csv, json, pickle
output_format = json
```

### Visualization Configuration

```ini
[visualization]
# Dashboard port
port = 8050

# Theme: light, dark
theme = light

# Default chart type: line, bar, scatter
default_chart_type = line

# Enable/disable animations
animations = true
```

## Using Configuration Files

### Command-Line Usage

You can specify a configuration file when running tests or benchmarks:

```bash
# Run tests with quick configuration
pytest tests/test_solver_performance.py -s --config config/test_configs/quick.ini

# Run tests with thorough configuration
pytest tests/test_solver_performance.py -s --config config/test_configs/thorough.ini
```

### Programmatic Usage

You can load configuration files in your code:

```python
from portopt.config import load_config

# Load a specific configuration file
config = load_config("config/test_configs/thorough.ini")

# Access configuration values
max_iterations = config.get("solver", "max_iterations", fallback=100)
tolerance = config.getfloat("solver", "tolerance", fallback=1e-6)

# Use in solver
solver = ClassicalSolver(
    max_iterations=max_iterations,
    tolerance=tolerance
)
```

### Environment Variables

You can also override configuration values using environment variables:

```bash
# Set environment variables
export PORTOPT_SOLVER_MAX_ITERATIONS=200
export PORTOPT_SOLVER_TOLERANCE=1e-8

# Run with environment-based configuration
python -m portopt.benchmark.runner
```

Environment variables follow the pattern `PORTOPT_SECTION_KEY`, where `SECTION` and `KEY` correspond to the section and key in the configuration file.

## Creating Custom Configurations

You can create custom configurations for specific use cases:

1. Create a new INI file:
   ```bash
   touch config/my_custom_config.ini
   ```

2. Add your configuration settings:
   ```ini
   [general]
   log_level = INFO
   output_dir = custom_results

   [solver]
   max_iterations = 150
   tolerance = 1e-7
   ```

3. Use your custom configuration:
   ```bash
   pytest tests/test_solver_performance.py -s --config config/my_custom_config.ini
   ```

## Configuration Best Practices

1. **Start with a preset**: Begin with one of the preset configurations and customize as needed.

2. **Version control**: Include configuration files in version control to ensure reproducibility.

3. **Documentation**: Document any custom configuration settings in your project.

4. **Environment-specific configs**: Create separate configurations for development, testing, and production.

5. **Sensitive information**: Do not store API keys or sensitive information in configuration files. Use environment variables instead.

## Troubleshooting

### Common Issues

1. **Configuration Not Found**
   
   If you get an error about a missing configuration file:
   ```
   FileNotFoundError: No such file or directory: 'config/config.ini'
   ```
   
   Make sure the file exists and the path is correct. You can specify an absolute path if needed.

2. **Invalid Configuration Values**
   
   If you get an error about invalid configuration values:
   ```
   ValueError: Invalid value for 'max_iterations': must be a positive integer
   ```
   
   Check the type and range of the configuration value. Some values have specific requirements.

3. **Configuration Not Applied**
   
   If your configuration changes don't seem to take effect, check the loading order. Later configurations override earlier ones.

## Related Resources

- [Installation Guide](./installation.md)
- [Quick Start Guide](./quick-start.md)
- [API Reference](../reference/api-reference.md)
