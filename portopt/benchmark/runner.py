import numpy as np
import pandas as pd
from typing import Dict, List, Type, Optional
import time
from datetime import datetime
import json
from pathlib import Path

from ..solvers.base import BaseSolver
from ..data.generator import TestDataGenerator
from ..utils.logging import setup_logging

class BenchmarkRunner:
    """Runs performance benchmarks across different solvers and problem sizes."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(level="INFO")
        self.logger = setup_logging(level="INFO")
    
    def run_size_scaling_benchmark(self,
                                 solver_class: Type[BaseSolver],
                                 solver_params: Dict,
                                 n_assets_range: List[int],
                                 n_periods_range: List[int],
                                 n_trials: int = 3) -> pd.DataFrame:
        """Run benchmarks across different problem sizes."""
        results = []
        
        for n_assets in n_assets_range:
            for n_periods in n_periods_range:
                print(f"\nBenchmarking n_assets={n_assets}, n_periods={n_periods}")
                
                for trial in range(n_trials):
                    # Generate problem
                    problem = self._generate_test_problem(n_assets, n_periods)
                    
                    # Create and run solver
                    solver = solver_class(**solver_params)
                    
                    start_time = time.perf_counter()
                    result = solver.solve(problem)
                    solve_time = time.perf_counter() - start_time
                    
                    # Calculate metrics
                    portfolio_metrics = self._calculate_metrics(result, problem)
                    
                    # Record results
                    results.append({
                        'n_assets': n_assets,
                        'n_periods': n_periods,
                        'trial': trial,
                        'solve_time': solve_time,
                        'objective_value': result.objective_value,
                        'feasible': result.feasible,
                        **portfolio_metrics
                    })
                    
                    print(f"  Trial {trial + 1}: {solve_time:.2f}s")
        
        # Create DataFrame and save results
        df = pd.DataFrame(results)
        self._save_benchmark_results(df, 'size_scaling')
        return df
    
    def run_constraint_sensitivity(self,
                                 solver_class: Type[BaseSolver],
                                 solver_params: Dict,
                                 base_constraints: Dict,
                                 param_ranges: Dict[str, List[float]],
                                 n_trials: int = 3) -> pd.DataFrame:
        """Run benchmarks with varying constraint parameters."""
        results = []
        
        for param, values in param_ranges.items():
            print(f"\nTesting sensitivity to {param}")
            
            for value in values:
                # Update constraints
                constraints = base_constraints.copy()
                constraints[param] = value
                
                for trial in range(n_trials):
                    # Generate problem
                    problem = self._generate_test_problem(
                        constraints.get('n_assets', 100),
                        constraints.get('n_periods', 252),
                        constraints
                    )
                    
                    # Create and run solver
                    solver = solver_class(**solver_params)
                    
                    start_time = time.perf_counter()
                    result = solver.solve(problem)
                    solve_time = time.perf_counter() - start_time
                    
                    # Calculate metrics
                    portfolio_metrics = self._calculate_metrics(result, problem)
                    
                    # Record results
                    results.append({
                        'parameter': param,
                        'value': value,
                        'trial': trial,
                        'solve_time': solve_time,
                        'objective_value': result.objective_value,
                        'feasible': result.feasible,
                        **portfolio_metrics
                    })
                    
                    print(f"  {param}={value}, Trial {trial + 1}: {solve_time:.2f}s")
        
        # Create DataFrame and save results
        df = pd.DataFrame(results)
        self._save_benchmark_results(df, 'constraint_sensitivity')
        return df
    
    def _generate_test_problem(self, n_assets: int, n_periods: int, 
                             constraints: Optional[Dict] = None) -> 'PortfolioOptProblem':
        """Generate a test problem with given dimensions."""
        generator = TestDataGenerator()
        problem = generator.generate_realistic_problem(
            n_assets=n_assets,
            n_periods=n_periods
        )
        
        if constraints is None:
            constraints = {
                'min_weight': 0.005,
                'max_weight': 0.15,
                'max_sector_weight': 0.25,
                'min_stocks_held': max(30, int(n_assets * 0.1)),
                'turnover_limit': 0.15
            }
        
        # Add sector map if not provided
        if 'sector_map' not in constraints:
            sector_map = np.random.randint(0, 11, size=n_assets)  # 11 GICS sectors
            constraints['sector_map'] = sector_map
        
        # Add previous weights if not provided
        if 'prev_weights' not in constraints and 'turnover_limit' in constraints:
            constraints['prev_weights'] = np.random.dirichlet(np.ones(n_assets))
        
        problem.constraints.update(constraints)
        return problem
    
    def _calculate_metrics(self, result: 'PortfolioOptResult', 
                         problem: 'PortfolioOptProblem') -> Dict[str, float]:
        """Calculate portfolio metrics."""
        weights = result.weights
        portfolio_return = np.dot(weights, np.mean(problem.returns, axis=1))
        portfolio_vol = np.sqrt(weights.T @ problem.cov_matrix @ weights)
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': portfolio_return / portfolio_vol if portfolio_vol > 0 else 0,
            'active_positions': np.sum(weights > 0),
            'concentration': np.sum(weights ** 2)  # Herfindahl index
        }
    
    def _save_benchmark_results(self, df: pd.DataFrame, benchmark_type: str) -> None:
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{benchmark_type}_{timestamp}"
        
        # Save detailed results
        df.to_csv(self.output_dir / f"{base_name}.csv", index=False)
        
        # Generate and save summary
        summary = {
            'mean_solve_time': df.groupby(['n_assets', 'n_periods'])['solve_time'].mean().to_dict(),
            'success_rate': df.groupby(['n_assets', 'n_periods'])['feasible'].mean().to_dict(),
            'timestamp': timestamp,
            'total_trials': len(df)
        }
        
        with open(self.output_dir / f"{base_name}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

