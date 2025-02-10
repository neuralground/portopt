import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { useOptimizationData } from '../../hooks/useOptimizationData';
import { useChartConfig } from '../../hooks/useChartConfig';

interface PerformanceChartProps {
  type: 'cumulative' | 'rolling' | 'distribution';
}

const PerformanceChart: React.FC<PerformanceChartProps> = ({ type }) => {
  const { data } = useOptimizationData();
  const { colors, formatters } = useChartConfig();

  if (type === 'cumulative') {
    // Generate sample cumulative return data
    const cumulativeData = Array.from({ length: 252 }, (_, i) => ({
      day: i + 1,
      portfolio: Math.exp(i * 0.001) - 1,
      benchmark: Math.exp(i * 0.0008) - 1
    }));

    return (
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={cumulativeData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="day" />
          <YAxis tickFormatter={formatters.percent} />
          <Tooltip formatter={formatters.percent} />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="portfolio" 
            stroke={colors.primary} 
            name="Portfolio" 
          />
          <Line 
            type="monotone" 
            dataKey="benchmark" 
            stroke={colors.secondary} 
            name="Benchmark" 
          />
        </LineChart>
      </ResponsiveContainer>
    );
  }

  if (type === 'rolling') {
    // Generate sample rolling metrics data
    const rollingData = Array.from({ length: 12 }, (_, i) => ({
      month: `Month ${i + 1}`,
      sharpe: 1.5 + Math.sin(i * 0.5),
      information: 0.8 + Math.cos(i * 0.5)
    }));

    return (
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rollingData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="month" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="sharpe" 
            stroke={colors.primary} 
            name="Sharpe Ratio" 
          />
          <Line 
            type="monotone" 
            dataKey="information" 
            stroke={colors.secondary} 
            name="Information Ratio" 
          />
        </LineChart>
      </ResponsiveContainer>
    );
  }

  if (type === 'distribution') {
    // Generate sample return distribution data
    const distributionData = Array.from({ length: 10 }, (_, i) => ({
      range: `${(i - 5) * 0.5}% to ${(i - 4) * 0.5}%`,
      frequency: Math.exp(-(i - 5) * (i - 5) / 8) * 100
    }));

    return (
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={distributionData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="range" />
          <YAxis />
          <Tooltip />
          <Bar 
            dataKey="frequency" 
            fill={colors.primary} 
            name="Return Frequency" 
          />
        </BarChart>
      </ResponsiveContainer>
    );
  }

  return null;
};

export default PerformanceChart;
