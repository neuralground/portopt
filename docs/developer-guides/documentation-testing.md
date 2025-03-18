# Documentation Testing Guide

This guide outlines the processes and tools for testing documentation in the Portfolio Optimization Testbed project. Ensuring that documentation is accurate, up-to-date, and functional is critical for providing a good user experience.

## Why Test Documentation?

Documentation testing helps to:
- Ensure code examples work as expected
- Verify that API references match the actual implementation
- Check that links are valid and point to the correct resources
- Confirm that installation and usage instructions are correct
- Identify outdated information

## Types of Documentation Tests

### 1. Code Example Testing

#### Doctest

Python's built-in `doctest` module can be used to test code examples in docstrings:

```python
def calculate_sharpe_ratio(returns, risk_free_rate=0):
    """
    Calculate the Sharpe ratio of a series of returns.
    
    Parameters
    ----------
    returns : numpy.ndarray
        Array of returns.
    risk_free_rate : float, optional
        Risk-free rate, by default 0.
    
    Returns
    -------
    float
        Sharpe ratio.
    
    Examples
    --------
    >>> import numpy as np
    >>> returns = np.array([0.05, 0.03, 0.04, -0.02, 0.01])
    >>> calculate_sharpe_ratio(returns, risk_free_rate=0.01)
    0.7071067811865475
    """
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std()
```

To run doctests:

```bash
python -m doctest -v portopt/metrics/performance.py
```

#### Jupyter Notebook Testing

Jupyter notebooks can be tested using `nbconvert` and `pytest`:

```bash
# Install required packages
pip install nbconvert pytest pytest-notebook

# Run notebook tests
pytest --nbval docs/examples/notebooks/
```

### 2. API Reference Testing

Verify that the API reference documentation matches the actual implementation:

```python
import inspect
import portopt

def test_api_reference_completeness():
    """Test that all public modules, classes, and functions are documented."""
    # Get all public modules
    modules = [m for m in dir(portopt) if not m.startswith('_')]
    
    for module_name in modules:
        module = getattr(portopt, module_name)
        
        # Check if module has docstring
        assert module.__doc__ is not None, f"Module {module_name} is missing docstring"
        
        # Get all public classes and functions in the module
        members = inspect.getmembers(module)
        public_members = [m for m in members if not m[0].startswith('_') and 
                         (inspect.isclass(m[1]) or inspect.isfunction(m[1]))]
        
        for name, obj in public_members:
            # Check if class/function has docstring
            assert obj.__doc__ is not None, f"{module_name}.{name} is missing docstring"
            
            # If it's a class, check its methods
            if inspect.isclass(obj):
                methods = inspect.getmembers(obj, predicate=inspect.isfunction)
                public_methods = [m for m in methods if not m[0].startswith('_')]
                
                for method_name, method in public_methods:
                    assert method.__doc__ is not None, f"{module_name}.{name}.{method_name} is missing docstring"
```

### 3. Link Validation

Check that all links in the documentation are valid:

```python
import os
import re
import requests
from urllib.parse import urljoin

def test_markdown_links():
    """Test that all links in markdown files are valid."""
    docs_dir = 'docs'
    
    # Regular expression to find markdown links
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    
    broken_links = []
    
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith('.md'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    content = f.read()
                
                # Find all links in the file
                links = link_pattern.findall(content)
                
                for text, url in links:
                    # Skip anchor links within the same page
                    if url.startswith('#'):
                        continue
                    
                    # Check if it's a relative link to another markdown file
                    if url.endswith('.md'):
                        target_path = os.path.normpath(os.path.join(os.path.dirname(filepath), url))
                        if not os.path.exists(target_path):
                            broken_links.append((filepath, text, url))
                    
                    # Check external URLs
                    elif url.startswith(('http://', 'https://')):
                        try:
                            response = requests.head(url, timeout=5)
                            if response.status_code >= 400:
                                broken_links.append((filepath, text, url))
                        except requests.RequestException:
                            broken_links.append((filepath, text, url))
    
    assert not broken_links, f"Found {len(broken_links)} broken links: {broken_links}"
```

### 4. Installation and Setup Testing

Test installation and setup instructions in a clean environment:

```bash
# Create a test environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Follow installation instructions
git clone https://github.com/neuralground/portopt.git
cd portopt
pip install -e .

# Run a simple test to verify installation
python -c "import portopt; print(portopt.__version__)"

# Deactivate and clean up
deactivate
rm -rf test_env
```

### 5. Documentation Coverage Testing

Measure documentation coverage to identify undocumented code:

```python
import inspect
import portopt

def calculate_doc_coverage():
    """Calculate documentation coverage percentage."""
    total_items = 0
    documented_items = 0
    
    # Get all modules
    modules = [m for m in dir(portopt) if not m.startswith('_')]
    
    for module_name in modules:
        module = getattr(portopt, module_name)
        
        # Count module
        total_items += 1
        if module.__doc__:
            documented_items += 1
        
        # Get all classes and functions in the module
        members = inspect.getmembers(module)
        public_members = [m for m in members if not m[0].startswith('_') and 
                         (inspect.isclass(m[1]) or inspect.isfunction(m[1]))]
        
        for name, obj in public_members:
            # Count class/function
            total_items += 1
            if obj.__doc__:
                documented_items += 1
            
            # If it's a class, count its methods
            if inspect.isclass(obj):
                methods = inspect.getmembers(obj, predicate=inspect.isfunction)
                public_methods = [m for m in methods if not m[0].startswith('_')]
                
                for method_name, method in public_methods:
                    total_items += 1
                    if method.__doc__:
                        documented_items += 1
    
    coverage = (documented_items / total_items) * 100 if total_items > 0 else 0
    return coverage, documented_items, total_items

coverage, documented, total = calculate_doc_coverage()
print(f"Documentation coverage: {coverage:.2f}% ({documented}/{total} items)")
```

## Automated Documentation Testing

To automate documentation testing, create a script that combines these tests:

```python
#!/usr/bin/env python
"""
Documentation test script for Portfolio Optimization Testbed.
"""

import os
import sys
import subprocess
import doctest
import unittest
import inspect
import re
import requests
from urllib.parse import urljoin

import portopt

class DocumentationTests(unittest.TestCase):
    def test_doctests(self):
        """Run doctests on all modules."""
        failures = 0
        for root, _, files in os.walk('portopt'):
            for file in files:
                if file.endswith('.py'):
                    module_path = os.path.join(root, file)
                    result = doctest.testfile(module_path, module_relative=False)
                    failures += result.failed
        
        self.assertEqual(failures, 0, f"Found {failures} doctest failures")
    
    def test_api_reference(self):
        """Test API reference completeness."""
        # Implementation as shown above
        pass
    
    def test_markdown_links(self):
        """Test markdown links."""
        # Implementation as shown above
        pass
    
    def test_notebook_execution(self):
        """Test Jupyter notebook execution."""
        result = subprocess.run(
            ["pytest", "--nbval", "docs/examples/notebooks/"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0, f"Notebook tests failed: {result.stderr}")
    
    def test_doc_coverage(self):
        """Test documentation coverage."""
        coverage, documented, total = calculate_doc_coverage()
        min_coverage = 80  # Set minimum acceptable coverage
        self.assertGreaterEqual(
            coverage, min_coverage,
            f"Documentation coverage too low: {coverage:.2f}% < {min_coverage}%"
        )

if __name__ == "__main__":
    unittest.main()
```

Add this to your CI/CD pipeline to run documentation tests automatically.

## Documentation Review Process

In addition to automated testing, implement a manual review process:

1. **Pre-release review**: Before each release, conduct a thorough review of all documentation.
2. **Peer review**: All documentation changes should be reviewed by at least one other team member.
3. **User feedback**: Collect and address user feedback on documentation.

## Documentation Testing Checklist

Use this checklist for manual documentation reviews:

- [ ] All code examples run without errors
- [ ] API reference matches the actual implementation
- [ ] Installation instructions work in a clean environment
- [ ] All links are valid
- [ ] Screenshots and diagrams are up-to-date
- [ ] No typos or grammatical errors
- [ ] Consistent formatting and style
- [ ] Documentation coverage meets minimum threshold
- [ ] Cross-references between related documents are correct
- [ ] Version-specific information is clearly marked

## Conclusion

Documentation testing is an essential part of maintaining high-quality documentation. By implementing both automated tests and manual review processes, you can ensure that your documentation remains accurate, up-to-date, and useful to users.

For more information on documentation standards, see the [Code Documentation Standards](./code-documentation-standards.md) guide.
