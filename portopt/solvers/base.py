"""Base solver interface."""

from abc import ABC, abstractmethod
from portopt.core.problem import PortfolioOptProblem
from portopt.core.result import PortfolioOptResult

class BaseSolver(ABC):
    """Abstract base class for portfolio optimization solvers."""
    
    @abstractmethod
    def solve(self, problem: PortfolioOptProblem) -> PortfolioOptResult:
        """Solve the given portfolio optimization problem."""
        pass

