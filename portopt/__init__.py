"""Portfolio Optimization Testbed"""

from portopt.core.problem import PortfolioOptProblem
from portopt.core.result import PortfolioOptResult
from portopt.solvers.classical import ClassicalSolver
from portopt.data.generator import EnhancedTestDataGenerator as TestDataGenerator  # Alias for backward compatibility

__version__ = "0.1.0"

