"""Plotting utilities for portfolio optimization analysis."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple
import seaborn as sns
from ..metrics import EnhancedRiskMetrics
from ..impact import MarketImpactModel

def create_risk_plots(risk_metrics: Dict[str, float], 
                     factor_exposures: Optional[np.ndarray] = None,
                     benchmark_comparison: Optional[Dict[str, float]] = None) -> List[Figure]:
    """
    Create comprehensive risk analysis plots.
    
    Args:
        risk_metrics: Dictionary of risk metrics
        factor_exposures: Optional factor exposures array
        benchmark_comparison: Optional benchmark metrics for comparison
    
    Returns:
        List of matplotlib figures
    """
    figures = []
    
    # Risk decomposition plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    risk_types = ['Total Risk', 'Factor Risk', 'Specific Risk']
    risk_values = [
        risk_metrics.get('total_risk', 0),
        risk_metrics.get('factor_risk', 0),
        risk_metrics.get('specific_risk', 0)
    ]
    ax1.bar(risk_types, risk_values)
    ax1.set_title('Risk Decomposition')
    ax1.set_ylabel('Annualized Risk (%)')
    plt.xticks(rotation=45)
    figures.append(fig1)
    
    # Factor exposures plot
    if factor_exposures is not None:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        factor_names = ['Market', 'Size', 'Value', 'Momentum', 'Quality']
        ax2.barh(factor_names, factor_exposures)
        ax2.set_title('Factor Exposures')
        ax2.set_xlabel('Exposure')
        figures.append(fig2)
    
    # Risk metrics comparison
    if benchmark_comparison:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        metrics = ['Volatility', 'VaR', 'CVaR', 'Tracking Error']
        portfolio_values = [risk_metrics.get(m.lower(), 0) for m in metrics]
        benchmark_values = [benchmark_comparison.get(m.lower(), 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax3.bar(x - width/2, portfolio_values, width, label='Portfolio')
        ax3.bar(x + width/2, benchmark_values, width, label='Benchmark')
        
        ax3.set_ylabel('Value')
        ax3.set_title('Risk Metrics Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        figures.append(fig3)
    
    return figures

def create_impact_plots(impact_model: MarketImpactModel,
                       weights: np.ndarray,
                       prev_weights: Optional[np.ndarray] = None,
                       trade_schedule: Optional[np.ndarray] = None) -> List[Figure]:
    """
    Create market impact analysis plots.
    
    Args:
        impact_model: Market impact model
        weights: Target portfolio weights
        prev_weights: Previous portfolio weights
        trade_schedule: Optional trading schedule
    
    Returns:
        List of matplotlib figures
    """
    figures = []
    
    # Cost breakdown plot
    costs = impact_model.estimate_total_costs(weights, prev_weights)
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    cost_components = ['Spread', 'Temporary Impact', 'Permanent Impact']
    cost_values = [
        costs['spread_costs']['total_spread_cost'],
        costs['impact_costs']['temporary_impact'],
        costs['impact_costs']['permanent_impact']
    ]
    ax1.pie(cost_values, labels=cost_components, autopct='%1.1f%%')
    ax1.set_title('Trading Cost Breakdown')
    figures.append(fig1)
    
    # Participation analysis
    if trade_schedule is not None:
        fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Daily participation rates
        participation = impact_model.estimate_market_impact(
            weights, prev_weights
        )['max_participation']
        days = np.arange(len(trade_schedule))
        
        ax2a.plot(days, participation * 100)
        ax2a.axhline(y=impact_model.params.participation_limit * 100,
                     color='r', linestyle='--', label='Limit')
        ax2a.set_title('Daily Market Participation')
        ax2a.set_xlabel('Day')
        ax2a.set_ylabel('Participation Rate (%)')
        ax2a.legend()
        
        # Cumulative execution
        cumulative_trade = np.cumsum(trade_schedule)
        ax2b.plot(days, cumulative_trade / cumulative_trade[-1] * 100)
        ax2b.set_title('Cumulative Execution')
        ax2b.set_xlabel('Day')
        ax2b.set_ylabel('Completion (%)')
        
        figures.append(fig2)
    
    return figures

def create_performance_plots(returns: np.ndarray,
                           weights: np.ndarray,
                           benchmark_returns: Optional[np.ndarray] = None) -> List[Figure]:
    """
    Create performance analysis plots.
    
    Args:
        returns: Asset returns matrix
        weights: Portfolio weights
        benchmark_returns: Optional benchmark returns
    
    Returns:
        List of matplotlib figures
    """
    figures = []
    
    # Portfolio returns distribution
    portfolio_returns = returns.T @ weights
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(portfolio_returns, stat='density', ax=ax1)
    sns.kdeplot(portfolio_returns, ax=ax1, color='red')
    ax1.set_title('Portfolio Returns Distribution')
    ax1.set_xlabel('Return')
    figures.append(fig1)
    
    # Cumulative returns comparison
    if benchmark_returns is not None:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        cum_portfolio = np.cumprod(1 + portfolio_returns) - 1
        cum_benchmark = np.cumprod(1 + benchmark_returns) - 1
        
        ax2.plot(cum_portfolio, label='Portfolio')
        ax2.plot(cum_benchmark, label='Benchmark')
        ax2.set_title('Cumulative Returns')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Cumulative Return')
        ax2.legend()
        figures.append(fig2)
        
        # Rolling metrics
        window = 63  # ~3 months
        fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Rolling volatility
        roll_vol_p = np.sqrt(252) * pd.Series(portfolio_returns).rolling(window).std()
        roll_vol_b = np.sqrt(252) * pd.Series(benchmark_returns).rolling(window).std()
        
        ax3a.plot(roll_vol_p, label='Portfolio')
        ax3a.plot(roll_vol_b, label='Benchmark')
        ax3a.set_title('Rolling Volatility')
        ax3a.legend()
        
        # Rolling Sharpe ratio
        rf_daily = 0.02/252  # Assume 2% risk-free rate
        excess_p = portfolio_returns - rf_daily
        excess_b = benchmark_returns - rf_daily
        
        roll_sharpe_p = np.sqrt(252) * pd.Series(excess_p).rolling(window).mean() / \
                        pd.Series(excess_p).rolling(window).std()
        roll_sharpe_b = np.sqrt(252) * pd.Series(excess_b).rolling(window).mean() / \
                        pd.Series(excess_b).rolling(window).std()
        
        ax3b.plot(roll_sharpe_p, label='Portfolio')
        ax3b.plot(roll_sharpe_b, label='Benchmark')
        ax3b.set_title('Rolling Sharpe Ratio')
        ax3b.legend()
        
        figures.append(fig3)
    
    return figures

def create_constraint_plots(weights: np.ndarray,
                          constraints: Dict,
                          sector_map: Optional[np.ndarray] = None) -> List[Figure]:
    """
    Create constraint analysis plots.
    
    Args:
        weights: Portfolio weights
        constraints: Dictionary of constraints
        sector_map: Optional sector assignments
    
    Returns:
        List of matplotlib figures
    """
    figures = []
    
    # Weight distribution
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    active_weights = weights[weights > 0]
    sns.histplot(active_weights, ax=ax1)
    ax1.axvline(constraints.get('min_weight', 0), color='r', linestyle='--',
                label='Min Weight')
    ax1.axvline(constraints.get('max_weight', 1), color='r', linestyle='--',
                label='Max Weight')
    ax1.set_title('Weight Distribution')
    ax1.legend()
    figures.append(fig1)
    
    # Sector analysis
    if sector_map is not None:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sector_weights = np.zeros(len(np.unique(sector_map)))
        for i, sector in enumerate(np.unique(sector_map)):
            sector_weights[i] = np.sum(weights[sector_map == sector])
        
        ax2.bar(range(len(sector_weights)), sector_weights)
        ax2.axhline(y=constraints.get('max_sector_weight', 1),
                    color='r', linestyle='--', label='Sector Limit')
        ax2.set_title('Sector Weights')
        ax2.set_xlabel('Sector')
        ax2.set_ylabel('Weight')
        ax2.legend()
        figures.append(fig2)
    
    return figures

