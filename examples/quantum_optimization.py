#!/usr/bin/env python
"""
Example of using quantum solvers for portfolio optimization.

This script demonstrates how to use the quantum solvers (QAOA and VQE)
for portfolio optimization. It creates a sample portfolio optimization problem,
solves it using different solvers, and compares the results.

The script showcases:
1. How to create and configure quantum solvers
2. Performance comparison between classical and quantum approaches
3. Visualization of portfolio weights from different solvers
4. Impact of different quantum solver parameters

References:
- Brandhofer, N., et al. (2022). Quantum algorithms for portfolio optimization.
  Journal of Finance and Data Science, 8, 71-83.
- Herman, D., et al. (2022). A survey of quantum computing for finance.
  ACM Computing Surveys, 55(9), 1-37.
"""

import numpy as np
import matplotlib.pyplot as plt
from portopt.core.problem import PortfolioOptProblem
from portopt.solvers.factory import SolverFactory
import time
import argparse

# Create a simple portfolio optimization problem
def create_sample_problem(n_assets=5, n_periods=252, seed=42):
    """Create a sample portfolio optimization problem.
    
    Args:
        n_assets: Number of assets in the portfolio
        n_periods: Number of time periods for returns data
        seed: Random seed for reproducibility
        
    Returns:
        A PortfolioOptProblem instance
    """
    # Generate random returns with a small positive drift
    np.random.seed(seed)
    returns = np.random.normal(0.0005, 0.01, (n_assets, n_periods))
    
    # Add correlation structure to make the problem more realistic
    correlation = np.zeros((n_assets, n_assets))
    for i in range(n_assets):
        for j in range(n_assets):
            # Assets have higher correlation with nearby assets (sector-like)
            correlation[i, j] = 0.2 + 0.6 * np.exp(-0.5 * (i - j)**2 / 2)
    
    # Apply correlation to returns
    cholesky = np.linalg.cholesky(correlation)
    for t in range(n_periods):
        returns[:, t] = 0.0005 + 0.01 * (cholesky @ np.random.randn(n_assets))
    
    # Create constraints
    constraints = {
        'sum_to_one': True,
        'min_weight': 0.0,
        'max_weight': 1.0
    }
    
    # Create problem
    problem = PortfolioOptProblem(
        returns=returns,
        constraints=constraints
    )
    
    return problem

def main():
    """Run the quantum optimization example."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Portfolio optimization using quantum solvers')
    parser.add_argument('--assets', type=int, default=5, help='Number of assets (default: 5)')
    parser.add_argument('--qaoa-depth', type=int, default=1, help='QAOA circuit depth (default: 1)')
    parser.add_argument('--vqe-ansatz', type=str, default='RealAmplitudes', 
                        choices=['RealAmplitudes', 'TwoLocal'],
                        help='VQE ansatz type (default: RealAmplitudes)')
    parser.add_argument('--shots', type=int, default=1024, help='Number of quantum shots (default: 1024)')
    parser.add_argument('--optimizer', type=str, default='COBYLA',
                       choices=['COBYLA', 'SPSA', 'SLSQP'],
                       help='Classical optimizer to use (default: COBYLA)')
    
    args = parser.parse_args()
    
    print(f"Running portfolio optimization with {args.assets} assets")
    print(f"QAOA depth: {args.qaoa_depth}, VQE ansatz: {args.vqe_ansatz}")
    print(f"Shots: {args.shots}, Optimizer: {args.optimizer}")
    print("\n" + "="*80 + "\n")
    
    # Create a sample problem
    problem = create_sample_problem(n_assets=args.assets)
    print(f"Created portfolio problem with {args.assets} assets")
    print(f"Expected returns: {problem.expected_returns}")
    print(f"Risk (diagonal of covariance matrix): {np.diag(problem.covariance)}")
    print("\n" + "="*80 + "\n")
    
    # Create solvers
    factory = SolverFactory()
    
    # Classical solver for baseline comparison
    classical_solver = factory.create_solver('classical')
    
    # Configure quantum solvers with command-line parameters
    qaoa_solver = factory.create_solver(
        'qaoa',
        depth=args.qaoa_depth,
        shots=args.shots,
        optimizer_name=args.optimizer
    )
    
    vqe_solver = factory.create_solver(
        'vqe',
        ansatz_type=args.vqe_ansatz,
        depth=2,  # VQE typically works better with slightly deeper circuits
        shots=args.shots,
        optimizer_name=args.optimizer
    )
    
    # Dictionary to store all results for comparison
    results = {}
    times = {}
    
    # Solve with classical solver
    print("Solving with classical solver...")
    start_time = time.time()
    classical_result = classical_solver.solve(problem)
    classical_time = time.time() - start_time
    results['Classical'] = classical_result
    times['Classical'] = classical_time
    print(f"Classical solution: {classical_result.weights}")
    print(f"Classical objective: {classical_result.objective}")
    print(f"Classical time: {classical_time:.4f} seconds")
    print("\n" + "-"*80 + "\n")
    
    # Solve with QAOA solver
    print(f"Solving with QAOA solver (depth={args.qaoa_depth})...")
    start_time = time.time()
    qaoa_result = qaoa_solver.solve(problem)
    qaoa_time = time.time() - start_time
    results['QAOA'] = qaoa_result
    times['QAOA'] = qaoa_time
    print(f"QAOA solution: {qaoa_result.weights}")
    print(f"QAOA objective: {qaoa_result.objective}")
    print(f"QAOA time: {qaoa_time:.4f} seconds")
    print("\n" + "-"*80 + "\n")
    
    # Solve with VQE solver
    print(f"Solving with VQE solver (ansatz={args.vqe_ansatz})...")
    start_time = time.time()
    vqe_result = vqe_solver.solve(problem)
    vqe_time = time.time() - start_time
    results['VQE'] = vqe_result
    times['VQE'] = vqe_time
    print(f"VQE solution: {vqe_result.weights}")
    print(f"VQE objective: {vqe_result.objective}")
    print(f"VQE time: {vqe_time:.4f} seconds")
    print("\n" + "-"*80 + "\n")
    
    # Compare solutions
    print("Comparing solutions:")
    print(f"Classical vs QAOA correlation: {np.corrcoef(classical_result.weights, qaoa_result.weights)[0, 1]:.4f}")
    print(f"Classical vs VQE correlation: {np.corrcoef(classical_result.weights, vqe_result.weights)[0, 1]:.4f}")
    print(f"QAOA vs VQE correlation: {np.corrcoef(qaoa_result.weights, vqe_result.weights)[0, 1]:.4f}")
    
    # Calculate relative error in objective value compared to classical
    print("\nRelative objective difference from classical solution:")
    print(f"QAOA: {(qaoa_result.objective - classical_result.objective) / classical_result.objective:.4f}")
    print(f"VQE: {(vqe_result.objective - classical_result.objective) / classical_result.objective:.4f}")
    print("\n" + "="*80 + "\n")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot portfolio weights comparison
    plt.subplot(2, 2, 1)
    width = 0.25
    x = np.arange(len(classical_result.weights))
    plt.bar(x - width, classical_result.weights, width, label='Classical')
    plt.bar(x, qaoa_result.weights, width, label='QAOA')
    plt.bar(x + width, vqe_result.weights, width, label='VQE')
    plt.xlabel('Asset')
    plt.ylabel('Weight')
    plt.title('Portfolio Weights Comparison')
    plt.xticks(x)
    plt.legend()
    
    # Plot execution time comparison
    plt.subplot(2, 2, 2)
    solver_names = list(times.keys())
    execution_times = list(times.values())
    plt.bar(solver_names, execution_times)
    plt.ylabel('Execution Time (seconds)')
    plt.title('Solver Performance Comparison')
    
    # Plot objective values
    plt.subplot(2, 2, 3)
    solver_names = list(results.keys())
    objectives = [results[name].objective for name in solver_names]
    plt.bar(solver_names, objectives)
    plt.ylabel('Objective Value (Lower is Better)')
    plt.title('Solution Quality Comparison')
    
    # Plot expected returns vs risk for each solution
    plt.subplot(2, 2, 4)
    for name, result in results.items():
        weights = result.weights
        expected_return = weights @ problem.expected_returns
        risk = np.sqrt(weights @ problem.covariance @ weights)
        plt.scatter(risk, expected_return, label=name, s=100)
    
    plt.xlabel('Risk (Standard Deviation)')
    plt.ylabel('Expected Return')
    plt.title('Risk-Return Profile')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('quantum_optimization_results.png')
    print(f"Results saved to 'quantum_optimization_results.png'")
    plt.show()

if __name__ == "__main__":
    main()
