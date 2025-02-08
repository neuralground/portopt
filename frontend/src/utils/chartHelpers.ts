import { formatters } from './formatters';

export const chartHelpers = {
  // Calculate suitable axis intervals
  calculateAxisInterval: (min: number, max: number, targetSteps: number = 5): number => {
    const range = max - min;
    const unroundedInterval = range / targetSteps;
    const magnitude = Math.pow(10, Math.floor(Math.log10(unroundedInterval)));
    const possibleIntervals = [1, 2, 2.5, 5, 10].map(x => x * magnitude);
    return possibleIntervals.find(i => i >= unroundedInterval) || possibleIntervals[possibleIntervals.length - 1];
  },

  // Generate tick values for axis
  generateTicks: (min: number, max: number, interval: number): number[] => {
    const ticks = [];
    for (let tick = Math.ceil(min / interval) * interval; tick <= max; tick += interval) {
      ticks.push(tick);
    }
    return ticks;
  },

  // Format tooltip values based on data type
  formatTooltipValue: (value: any, type: 'number' | 'percent' | 'currency' | 'bps' = 'number'): string => {
    switch (type) {
      case 'percent':
        return formatters.percent(value);
      case 'currency':
        return formatters.currency(value);
      case 'bps':
        return formatters.bps(value);
      default:
        return formatters.number(value);
    }
  },

  // Calculate domain padding for charts
  calculateDomainPadding: (data: number[], paddingPercent: number = 0.1): [number, number] => {
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min;
    const padding = range * paddingPercent;
    return [min - padding, max + padding];
  },

  // Generate gradient definitions for charts
  generateGradientDef: (id: string, startColor: string, endColor: string) => ({
    id,
    x1: '0',
    y1: '0',
    x2: '0',
    y2: '1',
    stops: [
      { offset: '0%', color: startColor, opacity: 1 },
      { offset: '100%', color: endColor, opacity: 0.5 }
    ]
  }),

  // Calculate moving averages for smoothing
  calculateMovingAverage: (data: number[], window: number = 5): number[] => {
    const result = [];
    for (let i = 0; i < data.length; i++) {
      const start = Math.max(0, i - window + 1);
      const end = i + 1;
      const windowData = data.slice(start, end);
      const average = windowData.reduce((sum, val) => sum + val, 0) / windowData.length;
      result.push(average);
    }
    return result;
  },

  // Generate custom dot styles for scatter plots
  generateDotStyle: (data: any) => {
    // Example custom dot style based on value
    const radius = Math.max(3, Math.min(8, Math.abs(data.value) * 10));
    return {
      r: radius,
      fill: data.value >= 0 ? '#16a34a' : '#dc2626',
      fillOpacity: 0.7,
      stroke: 'white',
      strokeWidth: 1
    };
  },

  // Format axis labels
  formatAxisLabel: (value: number, type: 'number' | 'percent' | 'currency' | 'bps' = 'number'): string => {
    switch (type) {
      case 'percent':
        return formatters.percent(value, 0);
      case 'currency':
        return formatters.compact(value);
      case 'bps':
        return formatters.bps(value, 0);
      default:
        return formatters.compact(value);
    }
  },

  // Custom legend formatter
  formatLegendValue: (value: number, type: 'number' | 'percent' | 'currency' | 'bps' = 'number'): string => {
    return chartHelpers.formatTooltipValue(value, type);
  }
};
