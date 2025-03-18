# Glossary of Portfolio Optimization Terms

This glossary provides definitions for key terms used throughout the Portfolio Optimization Testbed documentation.

## A

### Active Return
The difference between a portfolio's return and the benchmark return.

### Active Risk (Tracking Error)
The standard deviation of the active returns, measuring how consistently a portfolio tracks its benchmark.

### Alpha
Excess return of an investment relative to the return of a benchmark index.

### Asset Allocation
The process of dividing investments among different asset classes to optimize risk-return characteristics.

### Asset Class
A group of securities that have similar financial characteristics, behave similarly in the marketplace, and are subject to the same laws and regulations.

## B

### Backtest
A simulation of a trading strategy using historical data to evaluate its performance.

### Benchmark
A standard against which the performance of a portfolio can be measured.

### Beta
A measure of a stock's volatility in relation to the market.

## C

### Capital Asset Pricing Model (CAPM)
A model that describes the relationship between systematic risk and expected return for assets.

### Cardinality Constraint
A constraint that limits the number of assets in a portfolio.

### Covariance Matrix
A matrix that captures the correlations and volatilities of a set of assets. Used extensively in portfolio optimization to quantify the risk relationships between assets.

### Conditional Value at Risk (CVaR)
The expected loss given that the loss exceeds the Value at Risk (VaR). Also known as Expected Shortfall, it's considered a more coherent risk measure than VaR.

### Constraints
Restrictions placed on a portfolio optimization problem, such as minimum or maximum weights for specific assets.

## D

### Diversification
The practice of spreading investments across various assets to reduce risk.

### Drawdown
The peak-to-trough decline during a specific period for an investment or fund.

## E

### Efficient Frontier
The set of optimal portfolios that offer the highest expected return for a defined level of risk. A key concept in Modern Portfolio Theory.

### Equal-Weight Portfolio
A portfolio where each asset is given the same weight. Often used as a benchmark for more complex allocation strategies.

### Expected Return
The anticipated return of an investment based on historical data or forecasts.

## F

### Factor Exposure
The sensitivity of a portfolio to specific risk factors.

### Factor Model
A model that uses factors to explain asset returns.

### Full Investment Constraint
A constraint requiring that all available capital is invested (sum of weights equals 1).

## G

### Global Minimum Variance Portfolio
The portfolio on the efficient frontier with the lowest possible risk. See [Minimum Variance Portfolio Example](../examples/minimum-variance-portfolio.md) for implementation details.

## H

### Hessian Matrix
A square matrix of second-order partial derivatives of a scalar-valued function, used in optimization algorithms.

## I

### Information Ratio
A measure of risk-adjusted return, calculated as active return divided by tracking error.

### Integer Programming
A mathematical optimization technique where some or all variables are restricted to be integers.

## K

### Kelly Criterion
A formula that determines the optimal size of a series of bets to maximize wealth growth.

## L

### Lagrangian Multiplier
A method used to find the extrema of a function subject to constraints.

### Long-Only Constraint
A constraint requiring that all asset weights are non-negative (no short selling allowed).

## M

### Market Impact
The effect that a market participant has when buying or selling an asset.

### Markowitz Portfolio Theory
A theory that investors aim to maximize returns for a given level of risk. Also known as Modern Portfolio Theory (MPT), developed by Harry Markowitz.

### Maximum Drawdown
The largest peak-to-trough decline in the value of a portfolio.

### Maximum Sharpe Ratio Portfolio
The portfolio that maximizes the Sharpe ratio, offering the highest risk-adjusted return. See [Maximum Sharpe Ratio Portfolio Example](../examples/maximum-sharpe-ratio-portfolio.md) for implementation details.

### Mean-Variance Optimization
An approach to portfolio construction that considers both expected returns and risk.

### Minimum Variance Portfolio
A portfolio constructed to have the lowest possible risk without regard to expected returns. See [Minimum Variance Portfolio Example](../examples/minimum-variance-portfolio.md) for implementation details.

## N

### Non-Convex Optimization
Optimization problems where the objective function or constraints are not convex.

## O

### Objective Function
The function to be maximized or minimized in an optimization problem.

### Optimization Problem
A mathematical problem that seeks to find the best solution from all feasible solutions.

## P

### Portfolio
A collection of financial investments like stocks, bonds, commodities, cash, and cash equivalents.

### Portfolio Optimization
The process of selecting the best portfolio out of the set of all portfolios being considered.

### Portfolio Optimization Problem
A mathematical formulation that represents the portfolio selection process, typically including an objective function, constraints, and decision variables.

### Portfolio Turnover
The percentage of a portfolio that is sold and replaced over a period.

### Portfolio Weight
The percentage of a portfolio invested in a particular asset.

## Q

### Quadratic Programming
A type of optimization problem where the objective function is quadratic and the constraints are linear. Commonly used in mean-variance optimization.

## R

### Rebalancing
The process of realigning the weightings of a portfolio of assets.

### Return
The gain or loss of an investment over a specified period, expressed as a percentage.

### Risk
The uncertainty or variability of returns.

### Risk Aversion
The reluctance of an investor to accept higher risk for higher returns.

### Risk Budgeting
An approach to portfolio construction where risk is allocated according to a predefined budget. A generalization of risk parity.

### Risk Contribution
The amount of risk that each asset contributes to the total portfolio risk. Used in risk parity portfolio construction.

### Risk Parity
An approach to portfolio management that focuses on allocating risk rather than capital. Each asset contributes equally to the total portfolio risk. See [Risk Parity Portfolio Example](../examples/risk-parity-portfolio.md) for implementation details.

### Risk-Free Rate
The theoretical rate of return of an investment with zero risk.

### Risk-Return Tradeoff
The principle that potential return rises with an increase in risk.

### Robust Optimization
An approach to optimization that takes into account uncertainty in the input data.

## S

### Sector Constraint
A constraint that limits the exposure to specific market sectors.

### Sharpe Ratio
A measure of risk-adjusted return, calculated as excess return divided by standard deviation. Named after William F. Sharpe. See [Maximum Sharpe Ratio Portfolio Example](../examples/maximum-sharpe-ratio-portfolio.md) for implementation details.

### Short Selling
The practice of selling assets that are not currently owned.

### Slippage
The difference between the expected price of a trade and the actual price at which the trade is executed.

### Solver
An algorithm or method used to solve optimization problems.

### Sortino Ratio
A variation of the Sharpe ratio that uses downside deviation instead of standard deviation.

### Stochastic Optimization
Optimization methods that generate and use random variables.

### Systematic Risk
Risk that affects the entire market or market segment.

## T

### Target Return
The desired level of return for a portfolio.

### Transaction Costs
The costs incurred when buying or selling assets.

### Turnover Constraint
A constraint that limits the amount of trading in a portfolio.

## U

### Utility Function
A function that measures an investor's preferences regarding risk and return.

## V

### Value at Risk (VaR)
A statistical technique used to measure the level of financial risk within a portfolio over a specific time frame.

### Volatility
A statistical measure of the dispersion of returns for a given security or market index. Typically measured as the standard deviation of returns.

## W

### Weight Constraint
A constraint on the minimum or maximum weight of an asset in a portfolio.

## Related Resources

- [Portfolio Tutorial](../concepts/portfolio-tutorial.md)
- [Risk Metrics](../concepts/risk-metrics.md)
- [Optimization Challenges](../concepts/optimization-challenges.md)
- [Minimum Variance Portfolio Example](../examples/minimum-variance-portfolio.md)
- [Maximum Sharpe Ratio Portfolio Example](../examples/maximum-sharpe-ratio-portfolio.md)
- [Risk Parity Portfolio Example](../examples/risk-parity-portfolio.md)
- [API Reference](../reference/api-reference.md)
