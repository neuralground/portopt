import numpy as np
import pandas as pd
from typing import Dict, List, Type, Optional, Any
import time
from datetime import datetime
import json
from pathlib import Path
from tqdm import tqdm
import logging

from ..solvers.base import BaseSolver
from ..data.generator import EnhancedTestDataGenerator
from ..utils.logging import setup_logging

class BenchmarkRunner:
    """Runs performance benchmarks across different solvers and problem sizes."""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Set up file handler for warnings
        fh = logging.FileHandler(self.output_dir / 'benchmark_warnings.log')
        fh.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        return obj

    def run_size_scaling_benchmark(self,
                                 solver_classes: List[Type[BaseSolver]],
                                 solver_params: Dict,
                                 n_assets_range: List[int],
                                 n_periods_range: List[int],
                                 n_trials: int = 3,
                                 log_level: str = "INFO") -> pd.DataFrame:
        """Run benchmarks across different problem sizes."""
        setup_logging(level=log_level)
        results = []
        
        # Calculate total problems
        total_problems = len(solver_classes) * len(n_assets_range) * len(n_periods_range) * n_trials
        
        with tqdm(total=total_problems, desc="Running benchmark cases", ncols=80, 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            
            for solver_class in solver_classes:
                for n_assets in n_assets_range:
                    for n_periods in n_periods_range:
                        for trial in range(n_trials):
                            # Generate and solve problem
                            problem = self._generate_test_problem(n_assets, n_periods)
                            solver = solver_class(**solver_params)
                            
                            start_time = time.perf_counter()
                            result = solver.solve(problem)
                            solve_time = time.perf_counter() - start_time
                            
                            # Record results
                            results.append({
                                'solver_class': solver_class.__name__,
                                'n_assets': n_assets,
                                'n_periods': n_periods,
                                'trial': trial,
                                'solve_time': solve_time,
                                'objective_value': result.objective_value,
                                'feasible': result.feasible,
                                **self._calculate_metrics(result, problem)
                            })
                            
                            pbar.update(1)
        
        # Create DataFrame and save results
        df = pd.DataFrame(results)
        self._save_benchmark_results(df, 'size_scaling')
        return df

    def _generate_test_problem(self, n_assets: int, n_periods: int,
                             constraints: Optional[Dict] = None) -> 'PortfolioOptProblem':
        """Generate a test problem with given dimensions."""
        generator = EnhancedTestDataGenerator()
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
            sector_map = np.random.randint(0, 11, size=n_assets)
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
            'return': float(portfolio_return),
            'volatility': float(portfolio_vol),
            'sharpe_ratio': float(portfolio_return / portfolio_vol if portfolio_vol > 0 else 0),
            'active_positions': int(np.sum(weights > 0)),
            'concentration': float(np.sum(weights ** 2))
        }

    def _save_benchmark_results(self, df: pd.DataFrame, benchmark_type: str) -> None:
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{benchmark_type}_{timestamp}"

        # Save detailed results
        df.to_csv(self.output_dir / f"{base_name}.csv", index=False)
        
        # Generate and save summary
        summary = self._generate_summary(df, benchmark_type)
        
        # Convert to JSON-serializable format
        serializable_summary = self._convert_to_serializable(summary)
        
        with open(self.output_dir / f"{base_name}_summary.json", 'w') as f:
            json.dump(serializable_summary, f, indent=2)

    def _generate_summary(self, df: pd.DataFrame, benchmark_type: str) -> Dict:
        """Generate summary statistics from benchmark results."""
        summary = {
            'benchmark_type': benchmark_type,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_trials': len(df),
            'size_ranges': {
                'n_assets': df['n_assets'].unique().tolist(),
                'n_periods': df['n_periods'].unique().tolist()
            },
            'metrics': {}
        }
        
        # Calculate statistics for relevant metrics
        metrics_to_summarize = ['solve_time', 'objective_value', 'return', 
                              'volatility', 'sharpe_ratio', 'active_positions']
        
        for metric in metrics_to_summarize:
            if metric in df.columns:
                summary['metrics'][metric] = {
                    'mean': float(df[metric].mean()),
                    'std': float(df[metric].std()),
                    'min': float(df[metric].min()),
                    'max': float(df[metric].max()),
                    'median': float(df[metric].median())
                }
        
        # Calculate success rates
        if 'feasible' in df.columns:
            summary['success_rate'] = float(df['feasible'].mean())
            
            # Success rate by problem size
            size_success = df.groupby(['n_assets', 'n_periods'])['feasible'].mean()
            summary['size_success_rates'] = {
                f"{assets}_{periods}": float(rate) 
                for (assets, periods), rate in size_success.items()
            }
        
        # Performance scaling analysis
        if 'solve_time' in df.columns:
            size_times = df.groupby(['n_assets', 'n_periods'])['solve_time'].agg(['mean', 'std'])
            summary['performance_scaling'] = {
                f"{assets}_{periods}": {
                    'mean_time': float(stats['mean']),
                    'std_time': float(stats['std'])
                }
                for (assets, periods), stats in size_times.iterrows()
            }
        
        return summary

