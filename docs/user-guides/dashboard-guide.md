# Portfolio Optimization Dashboard Guide

This comprehensive guide explains how to use the Portfolio Optimization Dashboard to analyze and visualize optimization results.

## Introduction

The Portfolio Optimization Dashboard provides an interactive interface for exploring optimization results, analyzing risk metrics, and visualizing portfolio characteristics. This guide will help you navigate the dashboard effectively.

![Dashboard Overview](../assets/images/dashboard-overview.png)

## Getting Started

### Launching the Dashboard

To start the dashboard:

```bash
# Navigate to the project directory
cd portopt

# Start the backend API server
python -m portopt.visualization.server

# In a separate terminal, start the frontend
cd frontend
npm run dev
```

The dashboard will be available at `http://localhost:3000` by default.

### Dashboard Layout

The main interface consists of:
- Tab navigation at the top
- Cards containing specific metrics and visualizations
- Interactive charts and graphs
- Real-time updates as optimization runs

## Main Dashboard Tabs

The dashboard is organized into four main tabs, each focusing on different aspects of portfolio optimization:

### 1. Risk Analysis Tab

This tab provides comprehensive risk metrics and visualizations.

#### Risk Metrics Card

![Risk Metrics Card](../assets/images/risk-metrics-card.png)

- **Value at Risk (VaR)**: Potential loss at a given confidence level
- **Conditional VaR (CVaR)**: Expected loss beyond VaR
- **Volatility**: Standard deviation of returns
- **Tracking Error**: Deviation from benchmark

```javascript
// Example: Accessing risk metrics programmatically
fetch('/api/risk-metrics?portfolio_id=123')
  .then(response => response.json())
  .then(data => console.log(data));
```

#### Factor Exposures

Visualizes exposure to major risk factors:
- Market Beta
- Size
- Value
- Momentum
- Quality
- Low Volatility

**Interactive Features:**
- Click on factors to highlight their contribution
- Toggle between absolute and relative exposures
- Adjust confidence intervals

#### Risk Decomposition

Breaks down portfolio risk by source:
- Systematic risk
- Specific risk
- Factor contributions

**Best Practice:** Focus on the largest risk contributors first when optimizing your portfolio.

### 2. Market Impact Tab

This tab helps analyze trading costs and liquidity considerations.

#### Trading Cost Analysis

![Trading Cost Analysis](../assets/images/trading-cost-analysis.png)

Provides detailed cost breakdowns:
- Spread costs
- Market impact
- Total implementation shortfall
- Trading schedule recommendations

#### Impact Visualization

Interactive charts showing:
- Cost curves
- Participation rates
- Volume profiles
- Price impact estimates

**Advanced Usage:** Use the participation rate slider to see how different trading speeds affect market impact.

#### Liquidation Profile

Analysis of portfolio liquidity:
- Time to liquidate
- Cost of liquidation
- Optimal trading schedules
- Capacity constraints

### 3. Performance Tab

This tab focuses on return metrics and performance analysis.

#### Cumulative Returns

![Cumulative Returns](../assets/images/cumulative-returns.png)

Shows portfolio performance over time:
- Absolute returns
- Benchmark-relative returns
- Risk-adjusted metrics
- Attribution analysis

**Interactive Feature:** Hover over the chart to see point-in-time metrics.

#### Rolling Metrics

Displays rolling window statistics:
- Sharpe Ratio
- Information Ratio
- Sortino Ratio
- Tracking Error

**Customization:** Adjust the rolling window period using the dropdown menu.

#### Return Distribution

Visualizes return characteristics:
- Histogram of returns
- Normal distribution fit
- Skewness and kurtosis
- Tail risk analysis

### 4. Constraints Tab

This tab helps monitor portfolio constraints and limits.

#### Position Analysis

Shows portfolio composition:
- Active positions
- Weight distribution
- Concentration metrics
- Position limits

**Filtering:** Use the search box to find specific assets in your portfolio.

#### Sector Exposure

Provides sector breakdown:
- Sector weights
- Benchmark differences
- Limit monitoring
- Risk contributions

#### Constraint Violations

Monitors constraint compliance:
- Position limits
- Sector constraints
- Turnover limits
- Risk bounds

**Alert Feature:** Violations are highlighted in red for easy identification.

## Interactive Features

The dashboard provides rich interactive capabilities to explore your data.

### Chart Interaction

1. **Hover Information**
   - Mouse over chart elements for detailed data
   - Tooltips provide precise values
   - Compare multiple series

2. **Zoom and Pan**
   - Click and drag to zoom
   - Use toolbar controls for navigation
   - Reset view option available

3. **Data Selection**
   - Click legend items to show/hide series
   - Select date ranges
   - Filter by categories

### Customization Options

1. **Display Settings**
   - Adjust chart types (line, bar, area)
   - Modify color schemes
   - Change time periods
   - Select metrics to display

2. **Analysis Parameters**
   - Set confidence levels
   - Adjust rolling windows
   - Choose benchmark comparisons
   - Modify constraints

## Common Workflows

### 1. Daily Portfolio Monitoring

Follow this workflow for regular portfolio checks:

1. Check Risk Metrics → Risk Analysis tab
2. Review Constraints → Constraints tab
3. Monitor Costs → Market Impact tab
4. Track Performance → Performance tab

### 2. Trading Analysis

When planning trades:

1. Check Liquidity → Market Impact tab
2. Review Costs → Trading Cost Analysis
3. Plan Schedule → Liquidation Profile
4. Monitor Impact → Impact Timeline

### 3. Performance Review

For periodic performance evaluation:

1. Open Performance tab
2. Check Returns chart
3. Review Risk Metrics
4. Compare to Benchmark

### 4. Constraint Monitoring

To ensure compliance:

1. Open Constraints tab
2. Check position limits
3. Review sector exposures
4. Monitor turnover
5. Identify potential violations

## Keyboard Shortcuts

Increase your productivity with these shortcuts:

```
Tab Navigation:
- Alt + R: Risk Analysis
- Alt + M: Market Impact
- Alt + P: Performance
- Alt + C: Constraints

Chart Controls:
- +/-: Zoom In/Out
- →←: Pan Left/Right
- Home: Reset View
- Esc: Close Expanded View
```

## Export and Sharing

The dashboard provides several options for exporting data:

### Data Export

- **CSV**: Raw data for further analysis
- **Excel**: Formatted report with multiple sheets
- **PDF**: Chart images and summary tables
- **PNG**: Screenshots of specific visualizations

### Report Generation

Generate standardized reports:
- Summary Report
- Detailed Analysis
- Custom Report
- Data Extract

## Troubleshooting

### Common Issues

1. **Slow Loading**
   - Reduce date range
   - Disable real-time updates
   - Close unused browser tabs
   - Check network connection

2. **Visualization Errors**
   - Clear browser cache
   - Check for browser compatibility
   - Verify data integrity
   - Refresh the page

3. **Performance Issues**
   - Reduce data range
   - Limit active charts
   - Close unused tabs
   - Check network connection

## Advanced Features

### API Integration

The dashboard can be integrated with external systems:

```javascript
// Example: Embedding a dashboard chart in another application
const chartConfig = {
  portfolioId: '123',
  metric: 'cumulative_returns',
  timeRange: '1Y'
};

fetch('/api/embed-chart', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(chartConfig)
})
.then(response => response.json())
.then(data => {
  document.getElementById('embedded-chart').innerHTML = data.embedCode;
});
```

### Custom Visualizations

Create custom visualizations:
1. Navigate to the "Custom" tab
2. Select data sources
3. Choose visualization type
4. Configure parameters
5. Save for future use

## Related Resources

- [Dashboard Quick Reference](./dashboard-quickref.md) - Condensed reference guide
- [API Reference](../reference/api-reference.md) - Backend API documentation
- [Frontend Development Guide](../developer-guides/frontend-guide.md) - For developers
- [Data Dictionary](./data-dictionary.md) - Explanation of metrics and terms

## Feedback and Support

We welcome your feedback on the dashboard:
- Use the feedback button in the dashboard
- Report issues on GitHub
- Suggest features through our community forum
