export interface RiskMetrics {
    volatility: number;
    var95: number;
    cvar95: number;
    beta: number;
    trackingError: number;
  }
  
  export interface ImpactMetrics {
    totalCost: number;
    spreadCost: number;
    impactCost: number;
    avgParticipation: number;
  }
  
  export interface PerformanceMetrics {
    sharpeRatio: number;
    informationRatio: number;
    sortinoRatio: number;
    returns: number[];
  }
  
  export interface ConstraintMetrics {
    activePositions: number;
    maxPosition: number;
    maxSector: number;
    turnover: number;
    sectorWeights: {
      [key: string]: number;
    };
  }
  
  export interface OptimizationResult {
    weights: number[];
    riskMetrics: RiskMetrics;
    impactMetrics: ImpactMetrics;
    performanceMetrics: PerformanceMetrics;
    constraints: ConstraintMetrics;
  }
  
  export interface ChartConfig {
    colors: {
      primary: string;
      secondary: string;
      success: string;
      warning: string;
      error: string;
      muted: string;
      sectors: string[];
    };
    formatters: {
      percent: (value: number) => string;
      decimal: (value: number) => string;
      currency: (value: number) => string;
      basis: (value: number) => string;
    };
    chartDefaults: {
      margin: {
        top: number;
        right: number;
        bottom: number;
        left: number;
      };
      animate: boolean;
    };
  }
  