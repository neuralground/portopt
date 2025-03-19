#!/usr/bin/env python
"""Example of using the Black-Litterman model for portfolio optimization.

This script demonstrates how to use the Black-Litterman model to optimize a portfolio
with investor views. It compares the results with traditional mean-variance optimization.

The Black-Litterman model is particularly useful when:
1. You want to incorporate subjective views into the optimization process
2. You want to avoid extreme allocations that often result from mean-variance optimization
3. You want to start from market equilibrium and make targeted adjustments

The example shows:
- Setting up a portfolio optimization problem
- Creating investor views (both absolute and relative)
- Running the Black-Litterman solver
- Comparing results with classical mean-variance optimization
- Visualizing the impact of different views
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from portopt.core.problem import PortfolioOptProblem
from portopt.solvers.factory import SolverFactory
from portopt.models.black_litterman import InvestorView


def create_sample_problem(n_assets: int = 5, seed: int = 42) -> PortfolioOptProblem:
    """Create a sample portfolio optimization problem.
    
    Args:
        n_assets: Number of assets in the portfolio
        seed: Random seed for reproducibility
        
    Returns:
        PortfolioOptProblem instance
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate random returns (mean between 0.05 and 0.15)
    returns = np.random.uniform(0.05, 0.15, n_assets)
    
    # Generate random covariance matrix (positive definite)
    A = np.random.randn(n_assets, n_assets)
    cov_matrix = np.dot(A, A.T) / n_assets
    # Scale down the covariance to realistic levels
    cov_matrix *= 0.04
    
    # Generate random market caps (for equilibrium weights)
    market_caps = np.random.uniform(1e9, 1e11, n_assets)
    
    # Create asset names
    asset_names = [f"Asset_{i+1}" for i in range(n_assets)]
    
    # Create problem instance
    problem = PortfolioOptProblem(
        returns=returns,
        cov_matrix=cov_matrix,
        market_caps=market_caps,
        constraints={
            'asset_names': asset_names,
            'min_weight': 0.0,  # Allow zero weights
            'max_weight': 1.0,  # No single asset can be more than 100%
        }
    )
    
    return problem


def add_views_to_problem(
    problem: PortfolioOptProblem,
    views: List[InvestorView]
) -> PortfolioOptProblem:
    """Add investor views to the problem.
    
    Args:
        problem: Portfolio optimization problem
        views: List of investor views
        
    Returns:
        Updated problem with views
    """
    # Convert views to the format expected by the solver
    view_dicts = []
    for view in views:
        view_dict = {
            'assets': view.assets,
            'weights': view.weights,
            'value': view.value,
            'confidence': view.confidence,
            'is_relative': view.is_relative
        }
        view_dicts.append(view_dict)
    
    # Create a new problem with views
    new_constraints = problem.constraints.copy()
    new_constraints['views'] = view_dicts
    
    new_problem = PortfolioOptProblem(
        returns=problem.returns,
        cov_matrix=problem.cov_matrix,
        market_caps=problem.market_caps,
        constraints=new_constraints
    )
    
    return new_problem


def run_comparison(problem: PortfolioOptProblem, views: List[InvestorView] = None):
    """Run a comparison between classical and Black-Litterman optimization.
    
    Args:
        problem: Portfolio optimization problem
        views: Optional list of investor views
    """
    # Create solver factory
    factory = SolverFactory()
    
    # Create classical solver
    classical_solver = factory.create_solver('classical')
    
    # Create Black-Litterman solver
    bl_solver = factory.create_solver('black_litterman')
    
    # Solve using classical approach
    classical_result = classical_solver.solve(problem)
    
    # If views are provided, add them to the problem for BL
    if views:
        bl_problem = add_views_to_problem(problem, views)
    else:
        bl_problem = problem
    
    # Solve using Black-Litterman approach
    bl_result = bl_solver.solve(bl_problem)
    
    # Get asset names
    asset_names = problem.constraints.get('asset_names', [f"Asset_{i+1}" for i in range(problem.n_assets)])
    
    # Create a DataFrame for comparison
    results = pd.DataFrame({
        'Asset': asset_names,
        'Expected Return': problem.returns,
        'Market Weight': problem.market_caps / np.sum(problem.market_caps),
        'Classical Weight': classical_result.weights,
        'Black-Litterman Weight': bl_result.weights
    })
    
    # Print results
    print("\nPortfolio Optimization Results:")
    print("-" * 80)
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("-" * 80)
    
    # Calculate portfolio statistics
    classical_return = np.dot(classical_result.weights, problem.returns)
    classical_risk = np.sqrt(classical_result.weights @ problem.cov_matrix @ classical_result.weights)
    classical_sharpe = classical_return / classical_risk
    
    bl_return = np.dot(bl_result.weights, problem.returns)
    bl_risk = np.sqrt(bl_result.weights @ problem.cov_matrix @ bl_result.weights)
    bl_sharpe = bl_return / bl_risk
    
    market_weights = problem.market_caps / np.sum(problem.market_caps)
    market_return = np.dot(market_weights, problem.returns)
    market_risk = np.sqrt(market_weights @ problem.cov_matrix @ market_weights)
    market_sharpe = market_return / market_risk
    
    print("\nPortfolio Statistics:")
    print(f"{'Strategy':<20} {'Return':<10} {'Risk':<10} {'Sharpe':<10}")
    print("-" * 50)
    print(f"{'Market Cap Weighted':<20} {market_return:.4f}    {market_risk:.4f}    {market_sharpe:.4f}")
    print(f"{'Classical MVO':<20} {classical_return:.4f}    {classical_risk:.4f}    {classical_sharpe:.4f}")
    print(f"{'Black-Litterman':<20} {bl_return:.4f}    {bl_risk:.4f}    {bl_sharpe:.4f}")
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    
    # Bar chart of weights
    bar_width = 0.25
    x = np.arange(len(asset_names))
    
    plt.bar(x - bar_width, market_weights, bar_width, label='Market Weights')
    plt.bar(x, classical_result.weights, bar_width, label='Classical MVO')
    plt.bar(x + bar_width, bl_result.weights, bar_width, label='Black-Litterman')
    
    plt.xlabel('Assets')
    plt.ylabel('Portfolio Weight')
    plt.title('Portfolio Weights Comparison')
    plt.xticks(x, asset_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bl_comparison.png')
    plt.show()
    
    return results


def main():
    """Run the Black-Litterman example."""
    print("Black-Litterman Portfolio Optimization Example")
    print("=" * 50)
    
    # Create a sample problem
    problem = create_sample_problem(n_assets=5)
    
    # Print problem details
    print("\nProblem Details:")
    print(f"Number of assets: {problem.n_assets}")
    print(f"Asset names: {problem.constraints['asset_names']}")
    print("\nExpected Returns:")
    for i, r in enumerate(problem.returns):
        print(f"  {problem.constraints['asset_names'][i]}: {r:.4f}")
    
    print("\nCovariance Matrix:")
    print(pd.DataFrame(
        problem.cov_matrix,
        index=problem.constraints['asset_names'],
        columns=problem.constraints['asset_names']
    ).to_string(float_format=lambda x: f"{x:.4f}"))
    
    # Example 1: No views (should be similar to market weights)
    print("\n\nExample 1: No Views")
    run_comparison(problem)
    
    # Example 2: Single absolute view
    print("\n\nExample 2: Single Absolute View")
    # View: Asset_1 will return 20%
    views = [
        InvestorView(
            assets=[0],  # Asset_1
            weights=[1.0],
            value=0.20,  # 20% return
            confidence=0.8,  # High confidence
            is_relative=False
        )
    ]
    run_comparison(problem, views)
    
    # Example 3: Multiple views (absolute and relative)
    print("\n\nExample 3: Multiple Views (Absolute and Relative)")
    views = [
        # View 1: Asset_1 will return 20%
        InvestorView(
            assets=[0],  # Asset_1
            weights=[1.0],
            value=0.20,  # 20% return
            confidence=0.8,  # High confidence
            is_relative=False
        ),
        # View 2: Asset_3 will outperform Asset_5 by 5%
        InvestorView(
            assets=[2, 4],  # Asset_3 and Asset_5
            weights=[1.0, -1.0],
            value=0.05,  # 5% outperformance
            confidence=0.6,  # Medium confidence
            is_relative=True
        )
    ]
    run_comparison(problem, views)
    
    # Example 4: Low confidence views
    print("\n\nExample 4: Low Confidence Views")
    views = [
        # View 1: Asset_1 will return 20% (low confidence)
        InvestorView(
            assets=[0],  # Asset_1
            weights=[1.0],
            value=0.20,  # 20% return
            confidence=0.2,  # Low confidence
            is_relative=False
        ),
        # View 2: Asset_3 will outperform Asset_5 by 5% (low confidence)
        InvestorView(
            assets=[2, 4],  # Asset_3 and Asset_5
            weights=[1.0, -1.0],
            value=0.05,  # 5% outperformance
            confidence=0.2,  # Low confidence
            is_relative=True
        )
    ]
    run_comparison(problem, views)
    
    print("\nBlack-Litterman example completed. Results saved to 'bl_comparison.png'.")


if __name__ == "__main__":
    main()
