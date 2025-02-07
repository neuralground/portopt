import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from .test_types import TestResult

class TestDataHandler:
    """Handles test data generation and validation."""
    
    @staticmethod
    def generate_sector_map(n_assets: int, n_sectors: int = 11) -> np.ndarray:
        """Generate random sector assignments for assets."""
        return np.random.randint(0, n_sectors, size=n_assets)
    
    @staticmethod
    def calculate_sector_weights(weights: np.ndarray, sector_map: np.ndarray) -> np.ndarray:
        """Calculate total weight per sector."""
        n_sectors = len(np.unique(sector_map))
        sector_weights = np.zeros(n_sectors)
        for i in range(n_sectors):
            sector_weights[i] = np.sum(weights[sector_map == i])
        return sector_weights
    
    @staticmethod
    def check_constraints(weights: np.ndarray, 
                         params: Dict[str, Any],
                         sector_map: Optional[np.ndarray] = None,
                         prev_weights: Optional[np.ndarray] = None) -> Dict[str, bool]:
        """Check if all portfolio constraints are satisfied."""
        constraints_satisfied = {
            'sum_to_one': np.isclose(np.sum(weights), 1.0, rtol=1e-5),
            'min_weight': np.all(weights[weights > 0] >= params['min_weight']),
            'max_weight': np.all(weights <= params['max_weight']),
            'min_stocks_held': np.sum(weights > 0) >= params['min_stocks_held']
        }
        
        if sector_map is not None:
            sector_weights = TestDataHandler.calculate_sector_weights(weights, sector_map)
            constraints_satisfied['sector_limits'] = np.all(
                sector_weights <= params['max_sector_weight']
            )
        
        if prev_weights is not None:
            turnover = np.sum(np.abs(weights - prev_weights))
            constraints_satisfied['turnover'] = turnover <= params['turnover_limit']
        
        return constraints_satisfied

class TestMetricsCalculator:
    """Calculates and aggregates test metrics."""
    
    @staticmethod
    def calculate_portfolio_metrics(weights: np.ndarray,
                                  returns: np.ndarray,
                                  cov_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        portfolio_return = np.dot(weights, np.mean(returns, axis=1))
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': portfolio_return / portfolio_vol if portfolio_vol > 0 else 0,
            'active_positions': np.sum(weights > 0),
            'concentration': np.sum(weights ** 2)  # Herfindahl index
        }
    
    @staticmethod
    def aggregate_results(results: List[TestResult]) -> pd.DataFrame:
        """Aggregate multiple test results into a DataFrame."""
        rows = []
        for result in results:
            row = {
                'test_name': result.test_name,
                'duration': (result.end_time - result.start_time).total_seconds(),
                **result.parameters,
                **result.metrics,
                'all_constraints_satisfied': all(result.constraints_satisfied.values())
            }
            rows.append(row)
        return pd.DataFrame(rows)
    
    @staticmethod
    def generate_summary_stats(results: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Generate summary statistics for test results."""
        numeric_cols = results.select_dtypes(include=[np.number]).columns
        summary = {}
        
        for col in numeric_cols:
            summary[col] = {
                'mean': results[col].mean(),
                'std': results[col].std(),
                'min': results[col].min(),
                'max': results[col].max(),
                'median': results[col].median()
            }
        
        return summary

def print_test_report(result: TestResult) -> None:
    """Print formatted test results."""
    print("\nTest Results:")
    print("=" * 50)
    
    print(f"\nTest: {result.test_name}")
    print(f"Duration: {(result.end_time - result.start_time).total_seconds():.2f}s")
    
    print("\nParameters:")
    for key, value in result.parameters.items():
        print(f"  {key}: {value}")
    
    print("\nMetrics:")
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nConstraints:")
    for constraint, satisfied in result.constraints_satisfied.items():
        print(f"  {constraint}: {'✓' if satisfied else '✗'}")
    
    if result.additional_info:
        print("\nAdditional Information:")
        for key, value in result.additional_info.items():
            if isinstance(value, list) and len(value) > 10:
                print(f"  {key}: [array of size {len(value)}]")
            else:
                print(f"  {key}: {value}")

