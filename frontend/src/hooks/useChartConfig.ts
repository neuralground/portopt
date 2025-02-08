import { useMemo } from 'react';

export const useChartConfig = () => {
  const config = useMemo(() => ({
    colors: {
      primary: '#2563eb',
      secondary: '#9333ea',
      success: '#16a34a',
      warning: '#d97706',
      error: '#dc2626',
      muted: '#9ca3af',
      sectors: [
        '#2563eb',
        '#9333ea',
        '#16a34a',
        '#d97706',
        '#dc2626',
        '#9ca3af',
        '#0891b2',
        '#4f46e5'
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
