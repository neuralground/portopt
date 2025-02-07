import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from .test_types import TestResult

class TestDataHandler:
    """Handles test data generation and validation."""
    
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
                         market_data: Optional['MarketData'] = None) -> Dict[str, bool]:
        """Check if all portfolio constraints are satisfied."""
        constraints_satisfied = {
            'sum_to_one': np.isclose(np.sum(weights), 1.0, rtol=1e-5),
            'min_weight': np.all(weights[weights > 0] >= params['min_weight']),
            'max_weight': np.all(weights <= params['max_weight']),
            'min_stocks_held': np.sum(weights > 0) >= params['min_stocks_held']
        }
        
        if market_data is not None:
            # Check sector constraints
            sector_weights = TestDataHandler.calculate_sector_weights(
                weights, market_data.sector_map
            )
            constraints_satisfied['sector_limits'] = np.all(
                sector_weights <= params['max_sector_weight']
            )
            
            # Check turnover if previous weights exist
            if 'prev_weights' in params:
                turnover = np.sum(np.abs(weights - params['prev_weights']))
                constraints_satisfied['turnover'] = turnover <= params['turnover_limit']
            
            # Check market impact constraints if volumes present
            if hasattr(market_data, 'volumes'):
                participation_rates = weights / market_data.volumes.mean(axis=1)
                constraints_satisfied['market_impact'] = np.all(
                    participation_rates <= params.get('max_participation', 0.3)
                )
        
        return constraints_satisfied

class TestMetricsCalculator:
    """Calculates and aggregates test metrics."""
    
    @staticmethod
    def calculate_portfolio_metrics(weights: np.ndarray,
                                  problem: 'PortfolioOptProblem') -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        metrics = {
            'return': np.dot(weights, problem.exp_returns),
            'volatility': np.sqrt(weights.T @ problem.cov_matrix @ weights),
            'active_positions': np.sum(weights > 0),
            'concentration': np.sum(weights ** 2)  # Herfindahl index
        }
        
        # Add factor model metrics if available
        if problem.factor_returns is not None and problem.factor_exposures is not None:
            portfolio_exposures = problem.factor_exposures.T @ weights
            metrics.update({
                'market_beta': portfolio_exposures[0],  # Assuming first factor is market
                'factor_r2': TestMetricsCalculator._calculate_factor_r2(
                    weights, problem
                )
            })
            
        # Add transaction cost metrics if available
        if problem.volumes is not None and problem.spreads is not None:
            metrics.update(
                TestMetricsCalculator._calculate_cost_metrics(
                    weights, problem
                )
            )
            
        return metrics
    
    @staticmethod
    def _calculate_factor_r2(weights: np.ndarray,
                           problem: 'PortfolioOptProblem') -> float:
        """Calculate R-squared from factor model."""
        portfolio_returns = problem.returns.T @ weights
        factor_returns = problem.factor_returns.T
        portfolio_exposures = problem.factor_exposures.T @ weights
        
        predicted_returns = factor_returns @ portfolio_exposures
        residual_var = np.var(portfolio_returns - predicted_returns)
        total_var = np.var(portfolio_returns)
        
        return 1 - (residual_var / total_var)
    
    @staticmethod
    def _calculate_cost_metrics(weights: np.ndarray,
                              problem: 'PortfolioOptProblem') -> Dict[str, float]:
        """Calculate transaction cost related metrics."""
        # Average daily volume participation
        participation = weights / problem.volumes.mean(axis=1)
        
        # Spread costs
        spread_cost = np.sum(weights * problem.spreads.mean(axis=1))
        
        # Market impact estimate (simple square-root model)
        impact_cost = np.sum(
            0.1 * weights * np.sqrt(weights / problem.volumes.mean(axis=1))
        )
        
        return {
            'max_participation': np.max(participation),
            'spread_cost': spread_cost,
            'impact_cost': impact_cost,
            'total_cost': spread_cost + impact_cost
        }

