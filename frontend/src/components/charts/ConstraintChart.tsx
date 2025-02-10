import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, 
         PieChart, Pie, Cell } from 'recharts';
import { useOptimizationData } from '../../hooks/useOptimizationData';
import { useChartConfig } from '../../hooks/useChartConfig';

interface ConstraintChartProps {
  type: 'weights' | 'sectors' | 'violations';
}

const ConstraintChart: React.FC<ConstraintChartProps> = ({ type }) => {
  const { data, loading } = useOptimizationData();
  const { colors, formatters } = useChartConfig();

  if (loading || !data) {
    return <div className="h-64 flex items-center justify-center">Loading...</div>;
  }

  if (type === 'weights') {
    // Create weight distribution data from portfolio weights
    const weights = data.weights;
    const buckets = [0.01, 0.02, 0.03, 0.04, 0.05];
    const weightData = buckets.map((threshold, i) => {
      const lowerBound = i === 0 ? 0 : buckets[i - 1];
      const count = weights.filter(w => w >= lowerBound && w < threshold).length;
      return {
        range: `${(lowerBound * 100).toFixed(0)}-${(threshold * 100).toFixed(0)}%`,
        count
      };
    });

    return (
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={weightData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="range" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="count" fill={colors.primary} />
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (type === 'sectors') {
    const sectorData = Object.entries(data.constraints.sectorWeights).map(([name, value]) => ({
      name,
      value: value * 100
    }));

    return (
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={sectorData}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="50%"
            outerRadius="70%"
            label={formatters.percentLabel}
          >
            {sectorData.map((entry, index) => (
              <Cell key={entry.name} fill={colors.sectors[index % colors.sectors.length]} />
            ))}
          </Pie>
          <Tooltip formatter={formatters.percent} />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    );
  }

  if (type === 'violations') {
    const violationData = [
      { 
        constraint: 'Max Position', 
        current: data.constraints.maxPosition,
        limit: 0.05,
        distance: 0.05 - data.constraints.maxPosition
      },
      {
        constraint: 'Max Sector',
        current: data.constraints.maxSector,
        limit: 0.25,
        distance: 0.25 - data.constraints.maxSector
      },
      {
        constraint: 'Turnover',
        current: data.constraints.turnover,
        limit: 0.20,
        distance: 0.20 - data.constraints.turnover
      }
    ];

    return (
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={violationData} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" domain={[0, 'dataMax']} tickFormatter={formatters.percent} />
          <YAxis type="category" dataKey="constraint" />
          <Tooltip formatter={formatters.percent} />
          <Bar dataKey="current" fill={colors.primary} name="Current">
            {violationData.map((entry, index) => (
              <Cell 
                key={`cell-${index}`}
                fill={entry.distance < 0.01 ? colors.warning : colors.primary}
              />
            ))}
          </Bar>
          <Bar dataKey="limit" fill={colors.muted} name="Limit" />
        </BarChart>
      </ResponsiveContainer>
    );
  }

  return null;
};

export default ConstraintChart;
