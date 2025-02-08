import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, 
         PieChart, Pie, Cell } from 'recharts';
import { useOptimizationData } from '@/hooks/useOptimizationData';
import { useChartConfig } from '@/hooks/useChartConfig';

interface ConstraintChartProps {
  type: 'weights' | 'sectors' | 'violations';
}

const ConstraintChart: React.FC<ConstraintChartProps> = ({ type }) => {
  const { data } = useOptimizationData();
  const { colors, formatters } = useChartConfig();

  if (type === 'weights') {
    const weightData = [
      { range: '0-1%', count: 25 },
      { range: '1-2%', count: 15 },
      { range: '2-3%', count: 8 },
      { range: '3-4%', count: 4 },
      { range: '4-5%', count: 2 }
    ];

    return (
      <ResponsiveContainer width="100%" height="100%">
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
    const sectorData = [
      { name: 'Technology', value: 24.5 },
      { name: 'Financials', value: 22.3 },
      { name: 'Healthcare', value: 18.7 },
      { name: 'Consumer', value: 15.2 },
      { name: 'Industrials', value: 12.8 },
      { name: 'Others', value: 6.5 }
    ];

    return (
      <ResponsiveContainer width="100%" height="100%">
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
      { constraint: 'Min Weight', current: 0.01, limit: 0.01, distance: 0 },
      { constraint: 'Max Weight', current: 0.042, limit: 0.05, distance: 0.008 },
      { constraint: 'Tech Sector', current: 0.245, limit: 0.25, distance: 0.005 },
      { constraint: 'Turnover', current: 0.182, limit: 0.20, distance: 0.018 }
    ];

    return (
      <ResponsiveContainer width="100%" height={200}>
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
