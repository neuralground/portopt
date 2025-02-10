import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, 
         AreaChart, Area } from 'recharts';
import { useOptimizationData } from '../../hooks/useOptimizationData';
import { useChartConfig } from '../../hooks/useChartConfig';

interface ImpactChartProps {
  type: 'costs' | 'liquidation' | 'timeline';
}

const ImpactChart: React.FC<ImpactChartProps> = ({ type }) => {
  const { data, loading } = useOptimizationData();
  const { colors, formatters } = useChartConfig();

  if (loading || !data) {
    return <div className="h-64 flex items-center justify-center">Loading...</div>;
  }

  if (type === 'costs') {
    const costData = Array.from({ length: 10 }, (_, i) => {
      const size = (i + 1) * 0.1;
      return {
        size: `${(i + 1) * 10}%`,
        spread: data.impactMetrics.spreadCost * Math.sqrt(size),
        impact: data.impactMetrics.impactCost * Math.pow(size, 0.6),
        total: (data.impactMetrics.spreadCost * Math.sqrt(size)) + 
               (data.impactMetrics.impactCost * Math.pow(size, 0.6))
      };
    });

    return (
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={costData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="size" />
          <YAxis tickFormatter={formatters.basis} />
          <Tooltip formatter={formatters.basis} />
          <Legend />
          <Line type="monotone" dataKey="spread" stroke={colors.primary} name="Spread Cost" />
          <Line type="monotone" dataKey="impact" stroke={colors.secondary} name="Impact Cost" />
          <Line type="monotone" dataKey="total" stroke={colors.warning} name="Total Cost" />
        </LineChart>
      </ResponsiveContainer>
    );
  }

  if (type === 'liquidation') {
    const liquidationData = Array.from({ length: 20 }, (_, i) => ({
      day: i + 1,
      completion: 100 * (1 - Math.exp(-i * data.impactMetrics.avgParticipation)),
      participation: data.impactMetrics.avgParticipation * 100 * Math.exp(-i * 0.1)
    }));

    return (
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={liquidationData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="day" />
          <YAxis yAxisId="left" />
          <YAxis yAxisId="right" orientation="right" />
          <Tooltip formatter={formatters.percent} />
          <Legend />
          <Line 
            yAxisId="left"
            type="monotone" 
            dataKey="completion" 
            stroke={colors.primary} 
            name="Completion %" 
          />
          <Line 
            yAxisId="right"
            type="monotone" 
            dataKey="participation" 
            stroke={colors.secondary} 
            name="Participation %" 
          />
        </LineChart>
      </ResponsiveContainer>
    );
  }

  if (type === 'timeline') {
    const timelineData = Array.from({ length: 20 }, (_, i) => ({
      day: i + 1,
      volume: data.impactMetrics.avgParticipation * 1000000 * Math.exp(-i * 0.1),
      impact: data.impactMetrics.impactCost * 10000 * Math.exp(-i * 0.15)
    }));

    return (
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={timelineData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="day" />
          <YAxis yAxisId="left" />
          <YAxis yAxisId="right" orientation="right" />
          <Tooltip />
          <Legend />
          <Area
            yAxisId="left"
            type="monotone"
            dataKey="volume"
            stackId="1"
            stroke={colors.primary}
            fill={colors.primary}
            fillOpacity={0.3}
            name="Trading Volume"
          />
          <Area
            yAxisId="right"
            type="monotone"
            dataKey="impact"
            stackId="2"
            stroke={colors.secondary}
            fill={colors.secondary}
            fillOpacity={0.3}
            name="Price Impact (bps)"
          />
        </AreaChart>
      </ResponsiveContainer>
    );
  }

  return null;
};

export default ImpactChart;
