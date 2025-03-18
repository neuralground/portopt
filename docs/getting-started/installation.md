# Installation Guide

This guide will walk you through the process of installing the Portfolio Optimization Testbed on your system.

## Prerequisites

Before installing the package, ensure you have the following prerequisites:

- Python 3.8 or higher
- pip (Python package installer)
- Git (for development installation)
- Node.js and npm (for frontend development)

## Installation Options

### Option 1: Development Installation (Recommended)

For development or the latest features, install directly from the repository:

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

This installs the package in development mode, allowing you to modify the code and see changes immediately without reinstalling.

### Option 2: Install from PyPI

For stable releases, you can install directly from PyPI:

```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install from PyPI
pip install portopt
```

### Option 3: Install with Specific Features

You can install the package with specific feature sets:

```bash
# Install with visualization support
pip install "portopt[viz]"

# Install with development tools
pip install "portopt[dev]"

# Install with all features
pip install "portopt[all]"
```

## Frontend Installation

To use the dashboard frontend:

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

## Verifying Installation

To verify that the installation was successful:

```python
# Start Python interpreter
python

# Import the package
>>> import portopt
>>> print(portopt.__version__)
```

You should see the version number printed without any errors.

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   
   If you encounter errors about missing dependencies, try installing with all extras:
   ```bash
   pip install -e ".[all]"
   ```

2. **Version Conflicts**
   
   If you have version conflicts with existing packages, consider using a virtual environment:
   ```bash
   python -m venv fresh_env
   source fresh_env/bin/activate
   pip install portopt
   ```

3. **Build Errors**
   
   Some numerical packages might require additional system libraries. On Ubuntu/Debian:
   ```bash
   sudo apt-get install build-essential libopenblas-dev
   ```
   
   On macOS with Homebrew:
   ```bash
   brew install openblas
   ```

### Getting Help

If you continue to experience issues with installation:

1. Check the [GitHub Issues](https://github.com/example/portopt/issues) for similar problems
2. Consult the [Troubleshooting Guide](../developer-guides/troubleshooting.md)
3. Reach out to the community through [Discussions](https://github.com/example/portopt/discussions)

## Next Steps

Now that you have installed the Portfolio Optimization Testbed, you can:

- Follow the [Quick Start Guide](./quick-start.md) to run your first optimization
- Explore the [API Reference](../reference/api-reference.md) to learn about available functions
- Try out the [Dashboard](../user-guides/dashboard-guide.md) for visualization

## System Requirements

For optimal performance, we recommend:

- 4+ CPU cores
- 8GB+ RAM
- 1GB free disk space
- Modern web browser (for dashboard)
