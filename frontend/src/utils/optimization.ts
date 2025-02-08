import { OptimizationResult, RiskMetrics, ImpactMetrics } from '../types';
import { formatters } from './formatters';

export const optimizationUtils = {
  // Calculate summary statistics
  calculateSummaryStats: (results: OptimizationResult): Record<string, string> => {
    return {
      activePositions: formatters.number(results.constraints.activePositions),
      turnover: formatters.percent(results.constraints.turnover),
      totalCost: formatters.bps(results.impactMetrics.totalCost),
      riskLevel: formatters.percent(results.riskMetrics.volatility)
    };
  },

  // Check constraint violations
  checkConstraintViolations: (results: OptimizationResult): Array<{
    constraint: string;
    current: number;
    limit: number;
    violation: boolean;
  }> => {
    return [
      {
        constraint: 'Max Position',
        current: results.constraints.maxPosition,
        limit: 0.05, // Example limit
        violation: results.constraints.maxPosition > 0.05
      },
      {
        constraint: 'Sector Exposure',
        current: results.constraints.maxSector,
        limit: 0.25,
        violation: results.constraints.maxSector > 0.25
      },
      {
        constraint: 'Turnover',
        current: results.constraints.turnover,
        limit: 0.20,
        violation: results.constraints.turnover > 0.20
      }
    ];
  },

  // Calculate risk decomposition
  calculateRiskDecomposition: (riskMetrics: RiskMetrics): Array<{
    name: string;
    value: number;
  }> => {
    const totalRisk = riskMetrics.volatility;
    const factorRisk = totalRisk * Math.sqrt(riskMetrics.beta);
    const specificRisk = Math.sqrt(Math.max(0, totalRisk * totalRisk - factorRisk * factorRisk));

    return [
      { name: 'Factor Risk', value: factorRisk },
      { name: 'Specific Risk', value: specificRisk }
    ];
  },

  // Calculate implementation shortfall
  calculateImplementationShortfall: (impactMetrics: ImpactMetrics): {
    total: number;
    breakdown: Array<{ name: string; value: number }>;
  } => {
    return {
      total: impactMetrics.totalCost,
      breakdown: [
        { name: 'Spread Cost', value: impactMetrics.spreadCost },
        { name: 'Market Impact', value: impactMetrics.impactCost }
      ]
    };
  },

  // Calculate liquidity score
  calculateLiquidityScore: (results: OptimizationResult): number => {
    const participationScore = Math.min(1, results.impactMetrics.avgParticipation / 0.3);
    const turnoverScore = Math.min(1, results.constraints.turnover / 0.2);
    const positionScore = Math.min(1, results.constraints.maxPosition / 0.05);

    return (participationScore + turnoverScore + positionScore) / 3;
  },

  // Generate optimization summary
  generateSummary: (results: OptimizationResult): string => {
    const violations = optimizationUtils.checkConstraintViolations(results);
    const hasViolations = violations.some(v => v.violation);
    const liquidityScore = optimizationUtils.calculateLiquidityScore(results);

    return `Portfolio optimization ${hasViolations ? 'completed with violations' : 'successful'}.
    Active positions: ${results.constraints.activePositions}
    Turnover: ${formatters.percent(results.constraints.turnover)}
    Implementation cost: ${formatters.bps(results.impactMetrics.totalCost)}
    Liquidity score: ${formatters.percent(liquidityScore)}`;
  }
};
