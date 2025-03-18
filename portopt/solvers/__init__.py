"""Portfolio optimization solvers package.

This package provides a collection of solvers for portfolio optimization problems:

1. Classical solvers: Traditional optimization algorithms
   - SLSQP (Sequential Least Squares Programming)

2. Approximate/heuristic solvers: Metaheuristic approaches
   - Genetic Algorithm
   - Advanced Genetic Algorithm (with island model and multi-objective capabilities)
   - Simulated Annealing
   - Particle Swarm Optimization
   - Frank-Wolfe Algorithm

3. Quantum solvers: Quantum computing algorithms
   - QAOA (Quantum Approximate Optimization Algorithm)
   - VQE (Variational Quantum Eigensolver)

The package also includes utilities for constraint handling, solver factory,
and common interfaces for all solver types.
"""

from .base import BaseSolver
from .classical import ClassicalSolver
from .constraint_adapter import ConstraintAdapter
from .factory import SolverFactory
from .approximate import GeneticSolver, SimulatedAnnealingSolver
from .quantum import QAOASolver, VQESolver
from .advanced_genetic import AdvancedGeneticSolver

__all__ = [
    'BaseSolver',
    'ClassicalSolver',
    'ConstraintAdapter',
    'SolverFactory',
    'GeneticSolver',
    'SimulatedAnnealingSolver',
    'QAOASolver',
    'VQESolver',
    'AdvancedGeneticSolver'
]
