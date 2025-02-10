import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, 
         PieChart, Pie, Cell, BarChart, Bar } from 'recharts';
import { useOptimizationData } from '../../hooks/useOptimizationData';
import { useChartConfig } from '../../hooks/useChartConfig';

interface RiskChartProps {
  type: 'decomposition' | 'factors' | 'metrics';
}

const RiskChart: React.FC<RiskChartProps> = ({ type }) => {
  const { data, loading } = useOptimizationData();
  const { colors, formatters } = useChartConfig();

  if (loading || !data) {
    return <div className="h-64 flex items-center justify-center">Loading...</div>;
  }

  if (type === 'decomposition') {
    const riskData = [
      { name: 'Market Risk', value: data.riskMetrics.beta * data.riskMetrics.volatility },
      { name: 'Specific Risk', value: Math.sqrt(Math.pow(data.riskMetrics.volatility, 2) - 
        Math.pow(data.riskMetrics.beta * data.riskMetrics.volatility, 2)) },
      { name: 'Tracking Error', value: data.riskMetrics.trackingError }
    ];

    return (
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={riskData}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="50%"
            outerRadius={80}
            label={formatters.percentLabel}
          >
            {riskData.map((entry, index) => (
              <Cell key={entry.name} fill={colors.sectors[index]} />
            ))}
          </Pie>
          <Tooltip formatter={formatters.percent} />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    );
  }

  if (type === 'factors') {
    const factorData = [
      { factor: 'Market', exposure: data.riskMetrics.beta },
      { factor: 'Size', exposure: -0.15 },
      { factor: 'Value', exposure: 0.45 },
      { factor: 'Momentum', exposure: -0.22 },
      { factor: 'Quality', exposure: 0.33 }
    ];

    return (
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={factorData} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" />
          <YAxis type="category" dataKey="factor" />
          <Tooltip formatter={formatters.decimal} />
          <Bar dataKey="exposure" fill={colors.primary} />
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (type === 'metrics') {
    // Create time series of risk metrics
    const metricsData = data.performanceMetrics.returns.map((_, i) => ({
      period: i + 1,
      volatility: data.riskMetrics.volatility,
      var: data.riskMetrics.var95,
      cvar: data.riskMetrics.cvar95
    }));

    return (
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={metricsData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="period" />
          <YAxis tickFormatter={formatters.percent} />
          <Tooltip formatter={formatters.percent} />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="volatility" 
            stroke={colors.primary} 
            name="Volatility" 
          />
          <Line 
            type="monotone" 
            dataKey="var" 
            stroke={colors.secondary} 
            name="VaR" 
          />
          <Line 
            type="monotone" 
            dataKey="cvar" 
            stroke={colors.warning} 
            name="CVaR" 
          />
        </LineChart>
      </ResponsiveContainer>
    );
  }

  return null;
};

export default RiskChart;
