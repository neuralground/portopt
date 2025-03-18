"""Example script to compare different solver types for portfolio optimization."""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

from portopt.core.problem import PortfolioOptProblem
from portopt.core.result import PortfolioOptResult
from portopt.solvers import SolverFactory


def generate_test_problem(n_assets=10, n_periods=100, seed=42):
    """Generate a test portfolio optimization problem.
    
    Args:
        n_assets: Number of assets in the portfolio
        n_periods: Number of time periods for returns
        seed: Random seed for reproducibility
        
    Returns:
        PortfolioOptProblem instance
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate random returns with some correlation structure
    returns = np.random.normal(0.001, 0.02, (n_assets, n_periods))
    
    # Add some correlation between assets
    correlation = np.random.normal(0, 0.7, (n_assets, n_assets))
    correlation = correlation @ correlation.T
    np.fill_diagonal(correlation, 1.0)
    
    # Apply correlation to returns
    correlated_returns = np.zeros_like(returns)
    for t in range(n_periods):
        correlated_returns[:, t] = correlation @ returns[:, t]
    
    # Create problem with basic constraints
    problem = PortfolioOptProblem(
        returns=correlated_returns,
        constraints={
            'min_weight': 0.01,
            'max_weight': 0.3,
            'sum_to_one': True
        }
    )
    
    return problem


def compare_solvers(problem: PortfolioOptProblem) -> Dict[str, PortfolioOptResult]:
    """Compare different solver types on the same problem.
    
    Args:
        problem: The portfolio optimization problem to solve
        
    Returns:
        Dictionary mapping solver names to results
    """
    factory = SolverFactory()
    results = {}
    
    # Configure and run classical solver
    print("Running classical solver...")
    classical_solver = factory.create_solver('classical', max_iterations=10)
    results['Classical (SLSQP)'] = classical_solver.solve(problem)
    
    # Configure and run genetic algorithm solver
    print("Running genetic algorithm solver...")
    genetic_solver = factory.create_solver('genetic', population_size=100, generations=50)
    results['Genetic Algorithm'] = genetic_solver.solve(problem)
    
    # Configure and run simulated annealing solver
    print("Running simulated annealing solver...")
    annealing_solver = factory.create_solver('annealing', iterations=1000)
    results['Simulated Annealing'] = annealing_solver.solve(problem)
    
    # Configure and run QAOA solver (placeholder implementation)
    print("Running QAOA solver (placeholder)...")
    qaoa_solver = factory.create_solver('qaoa')
    results['QAOA (Placeholder)'] = qaoa_solver.solve(problem)
    
    # Configure and run VQE solver (placeholder implementation)
    print("Running VQE solver (placeholder)...")
    vqe_solver = factory.create_solver('vqe')
    results['VQE (Placeholder)'] = vqe_solver.solve(problem)
    
    return results


def print_results(results: Dict[str, PortfolioOptResult]):
    """Print results from different solvers.
    
    Args:
        results: Dictionary mapping solver names to results
    """
    print("\n" + "=" * 80)
    print(f"{'Solver':<25} {'Objective':<15} {'Feasible':<10} {'Time (s)':<10}")
    print("-" * 80)
    
    for solver_name, result in results.items():
        print(f"{solver_name:<25} {result.objective_value:<15.6f} {str(result.feasible):<10} {result.solve_time:<10.4f}")
    
    print("=" * 80 + "\n")


def plot_weights(results: Dict[str, PortfolioOptResult], n_assets: int):
    """Plot portfolio weights from different solvers.
    
    Args:
        results: Dictionary mapping solver names to results
        n_assets: Number of assets in the portfolio
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up bar positions
    bar_width = 0.15
    positions = np.arange(n_assets)
    
    # Plot bars for each solver
    for i, (solver_name, result) in enumerate(results.items()):
        ax.bar(
            positions + i * bar_width - (len(results) - 1) * bar_width / 2,
            result.weights,
            width=bar_width,
            label=solver_name
        )
    
    # Configure plot
    ax.set_xlabel('Asset')
    ax.set_ylabel('Weight')
    ax.set_title('Portfolio Weights by Solver')
    ax.set_xticks(positions)
    ax.set_xticklabels([f'Asset {i+1}' for i in range(n_assets)])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('solver_comparison.png')
    print("Plot saved as 'solver_comparison.png'")


def main():
    """Main function to run the solver comparison."""
    # Generate test problem
    n_assets = 10
    problem = generate_test_problem(n_assets=n_assets)
    
    # Compare solvers
    results = compare_solvers(problem)
    
    # Print results
    print_results(results)
    
    # Plot weights
    plot_weights(results, n_assets)


if __name__ == "__main__":
    main()
