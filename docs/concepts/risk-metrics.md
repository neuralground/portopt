# Risk Metrics

This document explains the various risk metrics used in portfolio optimization and how they are implemented in the Portfolio Optimization Testbed.

## Introduction to Risk Metrics

Risk metrics are quantitative measures that help investors understand and manage the uncertainty associated with investment returns. These metrics are essential for portfolio optimization, risk management, and performance evaluation.

## Volatility-Based Risk Metrics

### Standard Deviation (Volatility)

Standard deviation, commonly referred to as volatility in finance, measures the dispersion of returns around the mean. It is the most widely used risk metric in portfolio optimization.

**Mathematical Definition:**

$$\sigma_p = \sqrt{\frac{1}{T} \sum_{t=1}^{T} (r_{p,t} - \bar{r}_p)^2}$$

where:
- $\sigma_p$ is the portfolio standard deviation
- $r_{p,t}$ is the portfolio return at time $t$
- $\bar{r}_p$ is the mean portfolio return
- $T$ is the number of time periods

**Implementation in Portfolio Optimization Testbed:**

```python
def calculate_portfolio_volatility(weights, covariance_matrix):
    """
    Calculate the portfolio volatility (standard deviation of returns).
    
    Parameters
    ----------
    weights : numpy.ndarray
        Portfolio weights of shape (n_assets,)
    covariance_matrix : numpy.ndarray
        Covariance matrix of asset returns of shape (n_assets, n_assets)
        
    Returns
    -------
    float
        Portfolio volatility
    """
    portfolio_variance = weights.T @ covariance_matrix @ weights
    return np.sqrt(portfolio_variance)
```

### Variance

Variance is the square of standard deviation and represents the average squared deviation from the mean.

**Mathematical Definition:**

$$\sigma_p^2 = \frac{1}{T} \sum_{t=1}^{T} (r_{p,t} - \bar{r}_p)^2$$

**Implementation in Portfolio Optimization Testbed:**

```python
def calculate_portfolio_variance(weights, covariance_matrix):
    """
    Calculate the portfolio variance.
    
    Parameters
    ----------
    weights : numpy.ndarray
        Portfolio weights of shape (n_assets,)
    covariance_matrix : numpy.ndarray
        Covariance matrix of asset returns of shape (n_assets, n_assets)
        
    Returns
    -------
    float
        Portfolio variance
    """
    return weights.T @ covariance_matrix @ weights
```

### Annualized Volatility

Annualized volatility scales the volatility calculated from higher-frequency data (e.g., daily) to an annual basis.

**Mathematical Definition:**

$$\sigma_{p,annual} = \sigma_p \times \sqrt{f}$$

where:
- $\sigma_{p,annual}$ is the annualized portfolio volatility
- $\sigma_p$ is the portfolio volatility for the frequency of the data
- $f$ is the number of periods in a year (e.g., 252 for daily data, 52 for weekly data, 12 for monthly data)

**Implementation in Portfolio Optimization Testbed:**

```python
def annualize_volatility(volatility, frequency='daily'):
    """
    Annualize volatility from a higher frequency.
    
    Parameters
    ----------
    volatility : float
        Volatility at the given frequency
    frequency : str, optional
        Frequency of the input volatility, by default 'daily'
        Must be one of: 'daily', 'weekly', 'monthly', 'quarterly'
        
    Returns
    -------
    float
        Annualized volatility
    """
    frequency_factors = {
        'daily': 252,
        'weekly': 52,
        'monthly': 12,
        'quarterly': 4
    }
    
    if frequency not in frequency_factors:
        raise ValueError(f"Frequency must be one of: {list(frequency_factors.keys())}")
    
    return volatility * np.sqrt(frequency_factors[frequency])
```

## Downside Risk Metrics

Downside risk metrics focus only on negative deviations from a target return, addressing the asymmetric nature of risk preferences.

### Semi-Variance (Downside Variance)

Semi-variance measures the average squared deviation below a target return.

**Mathematical Definition:**

$$\sigma_{down}^2 = \frac{1}{T} \sum_{t=1}^{T} \min(r_{p,t} - \tau, 0)^2$$

where:
- $\sigma_{down}^2$ is the semi-variance
- $r_{p,t}$ is the portfolio return at time $t$
- $\tau$ is the target return (often set to the mean return or zero)
- $T$ is the number of time periods

**Implementation in Portfolio Optimization Testbed:**

```python
def calculate_semi_variance(returns, target_return=0):
    """
    Calculate the semi-variance (downside variance) of returns.
    
    Parameters
    ----------
    returns : numpy.ndarray
        Array of returns of shape (n_periods,)
    target_return : float, optional
        Target return, by default 0
        
    Returns
    -------
    float
        Semi-variance
    """
    downside_returns = np.minimum(returns - target_return, 0)
    return np.mean(downside_returns ** 2)
```

### Semi-Deviation (Downside Risk)

Semi-deviation is the square root of semi-variance and represents the standard deviation of returns below a target return.

**Mathematical Definition:**

$$\sigma_{down} = \sqrt{\frac{1}{T} \sum_{t=1}^{T} \min(r_{p,t} - \tau, 0)^2}$$

**Implementation in Portfolio Optimization Testbed:**

```python
def calculate_semi_deviation(returns, target_return=0):
    """
    Calculate the semi-deviation (downside risk) of returns.
    
    Parameters
    ----------
    returns : numpy.ndarray
        Array of returns of shape (n_periods,)
    target_return : float, optional
        Target return, by default 0
        
    Returns
    -------
    float
        Semi-deviation
    """
    return np.sqrt(calculate_semi_variance(returns, target_return))
```

### Maximum Drawdown

Maximum drawdown measures the largest peak-to-trough decline in the value of a portfolio.

**Mathematical Definition:**

$$\text{MDD} = \max_{t \in (0,T)} \left( \frac{\max_{s \in (0,t)} V_s - V_t}{\max_{s \in (0,t)} V_s} \right)$$

where:
- $V_t$ is the portfolio value at time $t$
- $T$ is the number of time periods

**Implementation in Portfolio Optimization Testbed:**

```python
def calculate_maximum_drawdown(returns):
    """
    Calculate the maximum drawdown of returns.
    
    Parameters
    ----------
    returns : numpy.ndarray
        Array of returns of shape (n_periods,)
        
    Returns
    -------
    float
        Maximum drawdown
    """
    # Convert returns to cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cum_returns)
    
    # Calculate drawdown
    drawdown = (running_max - cum_returns) / running_max
    
    # Return maximum drawdown
    return np.max(drawdown)
```

## Tail Risk Metrics

Tail risk metrics focus on extreme events in the tails of the return distribution.

### Value at Risk (VaR)

Value at Risk (VaR) measures the potential loss in value of a portfolio over a defined period for a given confidence interval.

**Mathematical Definition:**

$$\text{VaR}_\alpha = -\inf\{x \in \mathbb{R} : P(X \leq x) > \alpha\}$$

where:
- $\alpha$ is the confidence level (e.g., 95%)
- $X$ is the random variable representing portfolio returns

**Implementation in Portfolio Optimization Testbed:**

```python
def calculate_var(returns, confidence_level=0.95, method='historical'):
    """
    Calculate the Value at Risk (VaR) of returns.
    
    Parameters
    ----------
    returns : numpy.ndarray
        Array of returns of shape (n_periods,)
    confidence_level : float, optional
        Confidence level for VaR calculation, by default 0.95
    method : str, optional
        Method for VaR calculation, by default 'historical'
        Must be one of: 'historical', 'parametric', 'monte_carlo'
        
    Returns
    -------
    float
        Value at Risk at the specified confidence level
    """
    if method == 'historical':
        # Historical VaR
        return -np.percentile(returns, 100 * (1 - confidence_level))
    elif method == 'parametric':
        # Parametric VaR (assuming normal distribution)
        mu = np.mean(returns)
        sigma = np.std(returns)
        return -mu - sigma * stats.norm.ppf(confidence_level)
    elif method == 'monte_carlo':
        # Monte Carlo VaR (simplified example)
        mu = np.mean(returns)
        sigma = np.std(returns)
        simulated_returns = np.random.normal(mu, sigma, 10000)
        return -np.percentile(simulated_returns, 100 * (1 - confidence_level))
    else:
        raise ValueError("Method must be one of: 'historical', 'parametric', 'monte_carlo'")
```

### Conditional Value at Risk (CVaR)

Conditional Value at Risk (CVaR), also known as Expected Shortfall, measures the expected loss given that the loss exceeds the VaR.

**Mathematical Definition:**

$$\text{CVaR}_\alpha = -\mathbb{E}[X | X \leq -\text{VaR}_\alpha]$$

**Implementation in Portfolio Optimization Testbed:**

```python
def calculate_cvar(returns, confidence_level=0.95, method='historical'):
    """
    Calculate the Conditional Value at Risk (CVaR) of returns.
    
    Parameters
    ----------
    returns : numpy.ndarray
        Array of returns of shape (n_periods,)
    confidence_level : float, optional
        Confidence level for CVaR calculation, by default 0.95
    method : str, optional
        Method for CVaR calculation, by default 'historical'
        Must be one of: 'historical', 'parametric'
        
    Returns
    -------
    float
        Conditional Value at Risk at the specified confidence level
    """
    if method == 'historical':
        # Calculate VaR
        var = calculate_var(returns, confidence_level, 'historical')
        
        # Find returns below VaR
        tail_returns = returns[returns <= -var]
        
        # Calculate CVaR
        return -np.mean(tail_returns)
    elif method == 'parametric':
        # Parametric CVaR (assuming normal distribution)
        mu = np.mean(returns)
        sigma = np.std(returns)
        var = -mu - sigma * stats.norm.ppf(confidence_level)
        
        # Calculate CVaR
        return -mu - sigma * stats.norm.pdf(stats.norm.ppf(confidence_level)) / (1 - confidence_level)
    else:
        raise ValueError("Method must be one of: 'historical', 'parametric'")
```

### Expected Shortfall (ES)

Expected Shortfall is another name for CVaR and measures the expected loss in the worst α% of cases.

**Mathematical Definition:**

$$\text{ES}_\alpha = -\mathbb{E}[X | X \leq -\text{VaR}_\alpha]$$

**Implementation in Portfolio Optimization Testbed:**

```python
def calculate_expected_shortfall(returns, confidence_level=0.95):
    """
    Calculate the Expected Shortfall (ES) of returns.
    This is an alias for CVaR.
    
    Parameters
    ----------
    returns : numpy.ndarray
        Array of returns of shape (n_periods,)
    confidence_level : float, optional
        Confidence level for ES calculation, by default 0.95
        
    Returns
    -------
    float
        Expected Shortfall at the specified confidence level
    """
    return calculate_cvar(returns, confidence_level, 'historical')
```

## Relative Risk Metrics

Relative risk metrics measure risk relative to a benchmark.

### Tracking Error

Tracking error measures the standard deviation of the difference between portfolio returns and benchmark returns.

**Mathematical Definition:**

$$\text{TE} = \sqrt{\frac{1}{T} \sum_{t=1}^{T} (r_{p,t} - r_{b,t})^2}$$

where:
- $r_{p,t}$ is the portfolio return at time $t$
- $r_{b,t}$ is the benchmark return at time $t$
- $T$ is the number of time periods

**Implementation in Portfolio Optimization Testbed:**

```python
def calculate_tracking_error(portfolio_returns, benchmark_returns):
    """
    Calculate the tracking error of portfolio returns relative to benchmark returns.
    
    Parameters
    ----------
    portfolio_returns : numpy.ndarray
        Array of portfolio returns of shape (n_periods,)
    benchmark_returns : numpy.ndarray
        Array of benchmark returns of shape (n_periods,)
        
    Returns
    -------
    float
        Tracking error
    """
    active_returns = portfolio_returns - benchmark_returns
    return np.std(active_returns)
```

### Information Ratio

Information ratio measures the active return divided by the active risk (tracking error).

**Mathematical Definition:**

$$\text{IR} = \frac{\bar{r}_p - \bar{r}_b}{\text{TE}}$$

where:
- $\bar{r}_p$ is the mean portfolio return
- $\bar{r}_b$ is the mean benchmark return
- $\text{TE}$ is the tracking error

**Implementation in Portfolio Optimization Testbed:**

```python
def calculate_information_ratio(portfolio_returns, benchmark_returns):
    """
    Calculate the information ratio of portfolio returns relative to benchmark returns.
    
    Parameters
    ----------
    portfolio_returns : numpy.ndarray
        Array of portfolio returns of shape (n_periods,)
    benchmark_returns : numpy.ndarray
        Array of benchmark returns of shape (n_periods,)
        
    Returns
    -------
    float
        Information ratio
    """
    active_return = np.mean(portfolio_returns - benchmark_returns)
    tracking_error = calculate_tracking_error(portfolio_returns, benchmark_returns)
    
    # Avoid division by zero
    if tracking_error == 0:
        return 0
    
    return active_return / tracking_error
```

### Beta

Beta measures the sensitivity of portfolio returns to benchmark returns.

**Mathematical Definition:**

$$\beta = \frac{\text{Cov}(r_p, r_b)}{\text{Var}(r_b)}$$

where:
- $r_p$ is the portfolio return
- $r_b$ is the benchmark return

**Implementation in Portfolio Optimization Testbed:**

```python
def calculate_beta(portfolio_returns, benchmark_returns):
    """
    Calculate the beta of portfolio returns relative to benchmark returns.
    
    Parameters
    ----------
    portfolio_returns : numpy.ndarray
        Array of portfolio returns of shape (n_periods,)
    benchmark_returns : numpy.ndarray
        Array of benchmark returns of shape (n_periods,)
        
    Returns
    -------
    float
        Beta
    """
    covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    
    # Avoid division by zero
    if benchmark_variance == 0:
        return 0
    
    return covariance / benchmark_variance
```

## Risk-Adjusted Return Metrics

Risk-adjusted return metrics measure return per unit of risk.

### Sharpe Ratio

Sharpe ratio measures the excess return per unit of total risk.

**Mathematical Definition:**

$$\text{Sharpe} = \frac{\bar{r}_p - r_f}{\sigma_p}$$

where:
- $\bar{r}_p$ is the mean portfolio return
- $r_f$ is the risk-free rate
- $\sigma_p$ is the portfolio standard deviation

**Implementation in Portfolio Optimization Testbed:**

```python
def calculate_sharpe_ratio(returns, risk_free_rate=0):
    """
    Calculate the Sharpe ratio of returns.
    
    Parameters
    ----------
    returns : numpy.ndarray
        Array of returns of shape (n_periods,)
    risk_free_rate : float, optional
        Risk-free rate, by default 0
        
    Returns
    -------
    float
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(returns)
```

### Sortino Ratio

Sortino ratio measures the excess return per unit of downside risk.

**Mathematical Definition:**

$$\text{Sortino} = \frac{\bar{r}_p - r_f}{\sigma_{down}}$$

where:
- $\bar{r}_p$ is the mean portfolio return
- $r_f$ is the risk-free rate
- $\sigma_{down}$ is the downside deviation

**Implementation in Portfolio Optimization Testbed:**

```python
def calculate_sortino_ratio(returns, risk_free_rate=0, target_return=0):
    """
    Calculate the Sortino ratio of returns.
    
    Parameters
    ----------
    returns : numpy.ndarray
        Array of returns of shape (n_periods,)
    risk_free_rate : float, optional
        Risk-free rate, by default 0
    target_return : float, optional
        Target return for downside deviation calculation, by default 0
        
    Returns
    -------
    float
        Sortino ratio
    """
    excess_returns = np.mean(returns) - risk_free_rate
    downside_deviation = calculate_semi_deviation(returns, target_return)
    
    # Avoid division by zero
    if downside_deviation == 0:
        return 0
    
    return excess_returns / downside_deviation
```

### Calmar Ratio

Calmar ratio measures the excess return per unit of maximum drawdown.

**Mathematical Definition:**

$$\text{Calmar} = \frac{\bar{r}_p - r_f}{\text{MDD}}$$

where:
- $\bar{r}_p$ is the mean portfolio return
- $r_f$ is the risk-free rate
- $\text{MDD}$ is the maximum drawdown

**Implementation in Portfolio Optimization Testbed:**

```python
def calculate_calmar_ratio(returns, risk_free_rate=0):
    """
    Calculate the Calmar ratio of returns.
    
    Parameters
    ----------
    returns : numpy.ndarray
        Array of returns of shape (n_periods,)
    risk_free_rate : float, optional
        Risk-free rate, by default 0
        
    Returns
    -------
    float
        Calmar ratio
    """
    excess_returns = np.mean(returns) - risk_free_rate
    max_drawdown = calculate_maximum_drawdown(returns)
    
    # Avoid division by zero
    if max_drawdown == 0:
        return 0
    
    return excess_returns / max_drawdown
```

## Factor-Based Risk Metrics

Factor-based risk metrics decompose portfolio risk into factor exposures.

### Factor Exposures

Factor exposures measure the sensitivity of portfolio returns to specific risk factors.

**Mathematical Definition:**

$$r_p = \alpha + \sum_{j=1}^{k} \beta_j F_j + \epsilon$$

where:
- $r_p$ is the portfolio return
- $\alpha$ is the intercept
- $\beta_j$ is the exposure to factor $j$
- $F_j$ is the return of factor $j$
- $\epsilon$ is the residual return
- $k$ is the number of factors

**Implementation in Portfolio Optimization Testbed:**

```python
def calculate_factor_exposures(returns, factor_returns):
    """
    Calculate factor exposures using linear regression.
    
    Parameters
    ----------
    returns : numpy.ndarray
        Array of returns of shape (n_periods,)
    factor_returns : numpy.ndarray
        Array of factor returns of shape (n_periods, n_factors)
        
    Returns
    -------
    numpy.ndarray
        Array of factor exposures of shape (n_factors,)
    """
    # Add constant for intercept
    X = sm.add_constant(factor_returns)
    
    # Fit linear regression
    model = sm.OLS(returns, X).fit()
    
    # Return factor exposures (excluding intercept)
    return model.params[1:]
```

### Factor Contribution to Risk

Factor contribution to risk measures how much each factor contributes to the total portfolio risk.

**Mathematical Definition:**

$$\text{Risk Contribution}_j = \beta_j \times \text{Cov}(r_p, F_j) / \sigma_p^2$$

where:
- $\beta_j$ is the exposure to factor $j$
- $\text{Cov}(r_p, F_j)$ is the covariance between portfolio returns and factor $j$
- $\sigma_p^2$ is the portfolio variance

**Implementation in Portfolio Optimization Testbed:**

```python
def calculate_factor_risk_contribution(returns, factor_returns, factor_exposures=None):
    """
    Calculate factor contribution to risk.
    
    Parameters
    ----------
    returns : numpy.ndarray
        Array of returns of shape (n_periods,)
    factor_returns : numpy.ndarray
        Array of factor returns of shape (n_periods, n_factors)
    factor_exposures : numpy.ndarray, optional
        Array of factor exposures of shape (n_factors,), by default None
        If None, factor exposures will be calculated
        
    Returns
    -------
    numpy.ndarray
        Array of factor risk contributions of shape (n_factors,)
    """
    # Calculate factor exposures if not provided
    if factor_exposures is None:
        factor_exposures = calculate_factor_exposures(returns, factor_returns)
    
    # Calculate covariance matrix of factor returns
    factor_cov = np.cov(factor_returns, rowvar=False)
    
    # Calculate portfolio variance
    portfolio_variance = np.var(returns)
    
    # Calculate factor contribution to risk
    factor_risk_contribution = np.zeros(len(factor_exposures))
    for i, beta in enumerate(factor_exposures):
        factor_risk_contribution[i] = beta * np.sum(factor_cov[i] * factor_exposures) / portfolio_variance
    
    return factor_risk_contribution
```

## Risk Decomposition

Risk decomposition breaks down portfolio risk into its components.

### Marginal Contribution to Risk

Marginal contribution to risk measures the change in portfolio risk for a small change in the weight of an asset.

**Mathematical Definition:**

$$\text{MCR}_i = \frac{\partial \sigma_p}{\partial w_i} = \frac{(\Sigma w)_i}{\sigma_p}$$

where:
- $\text{MCR}_i$ is the marginal contribution to risk of asset $i$
- $\sigma_p$ is the portfolio standard deviation
- $w_i$ is the weight of asset $i$
- $\Sigma$ is the covariance matrix
- $(\Sigma w)_i$ is the $i$-th element of the vector $\Sigma w$

**Implementation in Portfolio Optimization Testbed:**

```python
def calculate_marginal_risk_contribution(weights, covariance_matrix):
    """
    Calculate marginal contribution to risk for each asset.
    
    Parameters
    ----------
    weights : numpy.ndarray
        Portfolio weights of shape (n_assets,)
    covariance_matrix : numpy.ndarray
        Covariance matrix of asset returns of shape (n_assets, n_assets)
        
    Returns
    -------
    numpy.ndarray
        Array of marginal risk contributions of shape (n_assets,)
    """
    # Calculate portfolio volatility
    portfolio_volatility = calculate_portfolio_volatility(weights, covariance_matrix)
    
    # Calculate marginal contribution to risk
    return (covariance_matrix @ weights) / portfolio_volatility
```

### Risk Contribution

Risk contribution measures how much each asset contributes to the total portfolio risk.

**Mathematical Definition:**

$$\text{RC}_i = w_i \times \text{MCR}_i = w_i \times \frac{(\Sigma w)_i}{\sigma_p}$$

where:
- $\text{RC}_i$ is the risk contribution of asset $i$
- $w_i$ is the weight of asset $i$
- $\text{MCR}_i$ is the marginal contribution to risk of asset $i$

**Implementation in Portfolio Optimization Testbed:**

```python
def calculate_risk_contribution(weights, covariance_matrix):
    """
    Calculate risk contribution for each asset.
    
    Parameters
    ----------
    weights : numpy.ndarray
        Portfolio weights of shape (n_assets,)
    covariance_matrix : numpy.ndarray
        Covariance matrix of asset returns of shape (n_assets, n_assets)
        
    Returns
    -------
    numpy.ndarray
        Array of risk contributions of shape (n_assets,)
    """
    # Calculate marginal contribution to risk
    mcr = calculate_marginal_risk_contribution(weights, covariance_matrix)
    
    # Calculate risk contribution
    return weights * mcr
```

### Percentage Contribution to Risk

Percentage contribution to risk measures the percentage of total portfolio risk contributed by each asset.

**Mathematical Definition:**

$$\text{PCR}_i = \frac{\text{RC}_i}{\sigma_p} = \frac{w_i \times (\Sigma w)_i}{\sigma_p^2}$$

where:
- $\text{PCR}_i$ is the percentage contribution to risk of asset $i$
- $\text{RC}_i$ is the risk contribution of asset $i$
- $\sigma_p$ is the portfolio standard deviation

**Implementation in Portfolio Optimization Testbed:**

```python
def calculate_percentage_risk_contribution(weights, covariance_matrix):
    """
    Calculate percentage contribution to risk for each asset.
    
    Parameters
    ----------
    weights : numpy.ndarray
        Portfolio weights of shape (n_assets,)
    covariance_matrix : numpy.ndarray
        Covariance matrix of asset returns of shape (n_assets, n_assets)
        
    Returns
    -------
    numpy.ndarray
        Array of percentage risk contributions of shape (n_assets,)
    """
    # Calculate risk contribution
    rc = calculate_risk_contribution(weights, covariance_matrix)
    
    # Calculate portfolio volatility
    portfolio_volatility = calculate_portfolio_volatility(weights, covariance_matrix)
    
    # Calculate percentage contribution to risk
    return rc / portfolio_volatility
```

## Risk Parity

Risk parity is a portfolio construction technique that allocates weights such that each asset contributes equally to the total portfolio risk.

**Mathematical Definition:**

$$w_i \times (\Sigma w)_i = w_j \times (\Sigma w)_j, \quad \forall i, j$$

**Implementation in Portfolio Optimization Testbed:**

```python
def calculate_risk_parity_weights(covariance_matrix, risk_budget=None, max_iterations=100, tolerance=1e-6):
    """
    Calculate risk parity weights.
    
    Parameters
    ----------
    covariance_matrix : numpy.ndarray
        Covariance matrix of asset returns of shape (n_assets, n_assets)
    risk_budget : numpy.ndarray, optional
        Risk budget for each asset of shape (n_assets,), by default None
        If None, equal risk budget is used
    max_iterations : int, optional
        Maximum number of iterations, by default 100
    tolerance : float, optional
        Convergence tolerance, by default 1e-6
        
    Returns
    -------
    numpy.ndarray
        Array of risk parity weights of shape (n_assets,)
    """
    n_assets = covariance_matrix.shape[0]
    
    # Initialize weights
    weights = np.ones(n_assets) / n_assets
    
    # Set risk budget
    if risk_budget is None:
        risk_budget = np.ones(n_assets) / n_assets
    else:
        risk_budget = risk_budget / np.sum(risk_budget)
    
    # Iterative algorithm
    for i in range(max_iterations):
        # Calculate portfolio volatility
        portfolio_volatility = calculate_portfolio_volatility(weights, covariance_matrix)
        
        # Calculate risk contribution
        rc = calculate_risk_contribution(weights, covariance_matrix)
        
        # Calculate percentage risk contribution
        pcr = rc / portfolio_volatility
        
        # Check convergence
        if np.max(np.abs(pcr - risk_budget)) < tolerance:
            break
        
        # Update weights
        weights = weights * (risk_budget / pcr)
        weights = weights / np.sum(weights)
    
    return weights
```

## Conclusion

Risk metrics are essential tools for portfolio optimization and risk management. The Portfolio Optimization Testbed provides a comprehensive set of risk metrics to help investors understand and manage portfolio risk.

## Related Resources

- [Portfolio Tutorial](./portfolio-tutorial.md)
- [Optimization Challenges](./optimization-challenges.md)
- [API Reference](../reference/api-reference.md)
- [Glossary](../reference/glossary.md)
