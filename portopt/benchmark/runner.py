"""Enhanced benchmark runner with comprehensive metrics and analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Type, Optional, Any, Tuple
import time
from datetime import datetime
import json
from pathlib import Path
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns

from ..solvers.base import BaseSolver
from ..data.generator import EnhancedTestDataGenerator
from ..utils.logging import setup_logging
from ..metrics import EnhancedRiskMetrics
from ..impact import MarketImpactModel, MarketImpactParams
from ..visualization.plots import (
    create_risk_plots,
    create_impact_plots,
    create_performance_plots,
    create_constraint_plots
)
from ..core.problem import PortfolioOptProblem
from ..core.result import PortfolioOptResult

class BenchmarkRunner:
    """Comprehensive benchmark runner for portfolio optimization solvers.
    
    This class provides a systematic framework for:
    1. Performance evaluation across different problem sizes
    2. Stress testing under various market scenarios
    3. Comprehensive metric collection and analysis
    4. Automated result reporting and visualization
    
    The runner supports:
    - Multiple solver comparison
    - Size scaling analysis
    - Stress scenario evaluation
    - Detailed performance metrics
    - Result persistence and visualization
    
    Example usage:
        runner = BenchmarkRunner(output_dir="benchmark_results")
        results = runner.run_size_scaling_benchmark(
            solver_classes=[ClassicalSolver],
            solver_params={'max_iterations': 20},
            n_assets_range=[50, 100],
            n_periods_range=[252],
            n_trials=3
        )
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize benchmark runner with output directory setup.
        
        Args:
            output_dir: Directory for storing benchmark results and plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Create organized subdirectories for different output types
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        # Set up warning logging
        fh = logging.FileHandler(self.output_dir / 'benchmark_warnings.log')
        fh.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def _save_plots(self, figures: List['Figure'], prefix: str) -> None:
        """Save generated plots to files."""
        for i, fig in enumerate(figures):
            filename = self.plots_dir / f"{prefix}_{i}.png"
            fig.savefig(filename)
            plt.close(fig)

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
                                 stress_scenarios: Optional[List[str]] = None,
                                 log_level: str = "INFO") -> pd.DataFrame:
        """Run comprehensive benchmarks across different problem sizes.
        
        This method:
        1. Generates test problems of various sizes
        2. Tests multiple solver configurations
        3. Applies stress scenarios if specified
        4. Collects comprehensive metrics
        5. Generates detailed reports
        
        Args:
            solver_classes: List of solver classes to benchmark
            solver_params: Parameters for solver initialization
            n_assets_range: List of portfolio sizes to test
            n_periods_range: List of time periods to test
            n_trials: Number of trials per configuration
            stress_scenarios: Optional list of stress scenarios
            log_level: Logging verbosity level
            
        Returns:
            DataFrame containing benchmark results
            
        The results include:
        - Solve time and success rate
        - Risk and return metrics
        - Transaction cost analysis
        - Constraint satisfaction
        - Stress scenario impact
        """
        setup_logging(level=log_level)
        results = []
        
        # Calculate total number of benchmark cases
        total_problems = len(solver_classes) * len(n_assets_range) * \
                        len(n_periods_range) * n_trials
        if stress_scenarios:
            total_problems *= len(stress_scenarios)
        
        # Run benchmarks with progress tracking
        with tqdm(total=total_problems, desc="Running benchmark cases", ncols=80) as pbar:
            for solver_class in solver_classes:
                for n_assets in n_assets_range:
                    for n_periods in n_periods_range:
                        for trial in range(n_trials):
                            # Generate base test problem
                            problem, market_data = self._generate_test_problem(
                                n_assets, n_periods
                            )
                            
                            # Initialize metrics calculators
                            risk_metrics = EnhancedRiskMetrics(
                                returns=market_data.returns,
                                factor_returns=market_data.factor_returns,
                                factor_exposures=market_data.factor_exposures
                            )
                            
                            # Test base case and stress scenarios
                            scenarios = ['base'] + (stress_scenarios or [])
                            for scenario in scenarios:
                                # Apply stress scenario if applicable
                                if scenario != 'base':
                                    market_data = self._apply_stress_scenario(
                                        market_data, scenario
                                    )
                                    problem = self._update_problem_data(
                                        problem, market_data
                                    )
                                
                                # Create impact model with current market data
                                impact_model = MarketImpactModel(
                                    volumes=market_data.volumes,
                                    spreads=market_data.spreads,
                                    volatility=market_data.volatility if hasattr(market_data, 'volatility') and market_data.volatility is not None else np.std(market_data.returns, axis=1)
                                )
                                
                                # Solve problem and analyze results
                                solver = solver_class(**solver_params)
                                result = self._solve_and_analyze(
                                    solver, problem, market_data,
                                    risk_metrics, impact_model
                                )
                                
                                # Record results with metadata
                                result_data = {
                                    'solver_class': solver_class.__name__,
                                    'n_assets': n_assets,
                                    'n_periods': n_periods,
                                    'trial': trial,
                                    'scenario': scenario,
                                    **result
                                }
                                results.append(result_data)
                                pbar.update(1)
        
        # Create DataFrame and save results
        df = pd.DataFrame(results)
        self._save_benchmark_results(df, 'size_scaling')
        return df

    def _generate_test_problem(self, n_assets: int, n_periods: int,
                             constraints: Optional[Dict] = None) -> Tuple[PortfolioOptProblem, Any]:
        """Generate a test problem with given dimensions."""
        generator = EnhancedTestDataGenerator()
        market_data = generator.generate_market_data(
            n_assets=n_assets,
            n_periods=n_periods
        )
        
        if constraints is None:
            constraints = {
                'min_weight': 0.005,
                'max_weight': 0.15,
                'max_sector_weight': 0.25,
                'min_stocks_held': max(30, int(n_assets * 0.1)),
                'turnover_limit': 0.15,
                'prev_weights': np.ones(n_assets) / n_assets,  # Equal weight portfolio as starting point
                'benchmark_weights': np.ones(n_assets) / n_assets  # Use equal weight as benchmark
            }

        problem = PortfolioOptProblem(
            returns=market_data.returns,
            constraints=constraints,
            volumes=market_data.volumes,
            spreads=market_data.spreads,
            factor_returns=market_data.factor_returns,
            factor_exposures=market_data.factor_exposures,
            market_caps=market_data.market_caps,
            classifications=market_data.classifications,
            asset_classes=market_data.asset_classes,
            currencies=market_data.currencies,
            credit_profiles=market_data.credit_profiles
        )
        
        return problem, market_data

    def _solve_and_analyze(self, solver: BaseSolver,
                          problem: PortfolioOptProblem,
                          market_data: Any,
                          risk_metrics: EnhancedRiskMetrics,
                          impact_model: MarketImpactModel) -> Dict[str, Any]:
        """Run solver and calculate comprehensive metrics.
        
        This method:
        1. Times the solver execution
        2. Calculates risk metrics
        3. Estimates market impact
        4. Generates visualization plots
        
        Args:
            solver: Portfolio optimization solver instance
            problem: Problem definition
            market_data: Market data for analysis
            risk_metrics: Risk metrics calculator
            impact_model: Market impact model
            
        Returns:
            Dictionary of metrics and analysis results
        """
        # Solve problem with timing
        start_time = time.perf_counter()
        result = solver.solve(problem)
        solve_time = time.perf_counter() - start_time

        # Calculate comprehensive risk metrics
        risk_results = risk_metrics.calculate_all_metrics(
            weights=result.weights,
            benchmark_weights=problem.constraints.get('benchmark_weights'),
            volumes=market_data.volumes,
            spreads=market_data.spreads
        )

        # Calculate market impact costs
        impact_results = impact_model.estimate_total_costs(
            weights=result.weights,
            prev_weights=problem.constraints.get('prev_weights')
        )

        # Generate analysis plots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create and save various analysis plots
        risk_plots = create_risk_plots(
            risk_results,
            factor_exposures=problem.factor_exposures,
            benchmark_comparison=problem.constraints.get('benchmark_weights')
        )
        self._save_plots(risk_plots, f"{timestamp}_risk")

        impact_plots = create_impact_plots(
            impact_model,
            result.weights,
            prev_weights=problem.constraints.get('prev_weights')
        )
        self._save_plots(impact_plots, f"{timestamp}_impact")

        perf_plots = create_performance_plots(
            problem.returns,
            result.weights
        )
        self._save_plots(perf_plots, f"{timestamp}_performance")

        # Compile comprehensive metrics
        portfolio_metrics = {
            'solve_time': solve_time,
            'objective_value': result.objective_value,
            'feasible': result.feasible,
            'active_positions': int(np.sum(result.weights > 0)),
            'concentration': float(np.sum(result.weights ** 2)),
            **risk_results,
            **impact_results
        }

        return portfolio_metrics
    
    def _apply_stress_scenario(self, market_data: Any, scenario: str) -> Any:
        """Apply stress scenario to market data."""
        generator = EnhancedTestDataGenerator()
        return generator.create_stress_scenario(market_data, scenario)

    def _update_problem_data(self, problem: PortfolioOptProblem, 
                           market_data: Any) -> PortfolioOptProblem:
        """Update problem with new market data."""
        problem.returns = market_data.returns
        problem.volumes = market_data.volumes
        problem.spreads = market_data.spreads
        problem.factor_returns = market_data.factor_returns
        problem.factor_exposures = market_data.factor_exposures
        return problem

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
        """Generate comprehensive summary statistics from benchmark results."""
        summary = {
            'benchmark_type': benchmark_type,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_trials': len(df),
            'size_ranges': {
                'n_assets': df['n_assets'].unique().tolist(),
                'n_periods': df['n_periods'].unique().tolist()
            },
            'metrics': {},
            'stress_analysis': {}
        }

        # Calculate statistics for all numeric metrics
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for metric in numeric_columns:
            summary['metrics'][metric] = {
                'mean': float(df[metric].mean()),
                'std': float(df[metric].std()),
                'min': float(df[metric].min()),
                'max': float(df[metric].max()),
                'median': float(df[metric].median())
            }

        # Generate summary plots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Size scaling plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='n_assets', y='solve_time')
        plt.title('Solve Time by Problem Size')
        plt.savefig(self.plots_dir / f"{timestamp}_size_scaling.png")
        plt.close()

        # Metric correlation plot
        plt.figure(figsize=(12, 12))
        sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm')
        plt.title('Metric Correlations')
        plt.savefig(self.plots_dir / f"{timestamp}_metric_correlations.png")
        plt.close()

        # Analyze stress scenarios if present
        if 'scenario' in df.columns and len(df['scenario'].unique()) > 1:
            for scenario in df['scenario'].unique():
                scenario_data = df[df['scenario'] == scenario]
                summary['stress_analysis'][scenario] = {
                    'solve_success_rate': float(scenario_data['feasible'].mean()),
                    'avg_solve_time': float(scenario_data['solve_time'].mean()),
                    'avg_cost_increase': float(
                        (scenario_data['total_cost'].mean() /
                         df[df['scenario'] == 'base']['total_cost'].mean() - 1) * 100
                    )
                }

            # Stress scenario comparison plot
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df, x='scenario', y='total_cost')
            plt.title('Trading Costs by Scenario')
            plt.xticks(rotation=45)
            plt.savefig(self.plots_dir / f"{timestamp}_stress_comparison.png")
            plt.close()

        return summary
