"""Benchmarking utilities."""

import pandas as pd
from typing import List
from portopt.core.problem import PortfolioOptProblem
from portopt.solvers.base import BaseSolver

def run_benchmark(solver: BaseSolver, problems: List[PortfolioOptProblem]) -> pd.DataFrame:
    """Run benchmark tests on a solver with multiple problem instances."""
    results = []
    
    for i, problem in enumerate(problems):
        try:
            result = solver.solve(problem)
            results.append({
                'problem_id': i,
                'n_assets': problem.n_assets,
                'n_periods': problem.n_periods,
                'objective_value': result.objective_value,
                'solve_time': result.solve_time,
                'feasible': result.feasible
            })
        except Exception as e:
            results.append({
                'problem_id': i,
                'n_assets': problem.n_assets,
                'n_periods': problem.n_periods,
                'error': str(e)
            })
    
    return pd.DataFrame(results)

