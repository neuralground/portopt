import { useMemo } from 'react';

export const useChartConfig = () => {
  const config = useMemo(() => ({
    colors: {
      primary: '#2563eb',    // blue-600
      secondary: '#9333ea',  // purple-600
      warning: '#d97706',    // amber-600
      error: '#dc2626',      // red-600
      muted: '#9ca3af',      // gray-400
      sectors: [
        '#2563eb',  // blue-600
        '#9333ea',  // purple-600
        '#16a34a',  // green-600
        '#d97706',  // amber-600
        '#dc2626',  // red-600
        '#9ca3af',  // gray-400
        '#0891b2',  // cyan-600
        '#4f46e5'   // indigo-600
      ]
    },
    formatters: {
      percent: (value: number) => `${(value * 100).toFixed(1)}%`,
      decimal: (value: number) => value.toFixed(4),
      currency: (value: number) => `$${value.toFixed(2)}`,
      percentLabel: (entry: { value: number }) => `${entry.value.toFixed(1)}%`,
      basis: (value: number) => `${(value * 10000).toFixed(1)} bps`
    }
  }), []);

  return config;
};

export default useChartConfig;
