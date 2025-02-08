export const formatters = {
    // Number formatters
    number: (value: number, decimals: number = 2): string => {
      return value.toLocaleString(undefined, {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
      });
    },
  
    // Percentage formatters
    percent: (value: number, decimals: number = 1): string => {
      return `${(value * 100).toLocaleString(undefined, {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
      })}%`;
    },
  
    // Basis point formatters
    bps: (value: number, decimals: number = 1): string => {
      return `${(value * 10000).toLocaleString(undefined, {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
      })} bps`;
    },
  
    // Currency formatters
    currency: (value: number, currency: string = 'USD', decimals: number = 2): string => {
      return value.toLocaleString(undefined, {
        style: 'currency',
        currency: currency,
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
      });
    },
  
    // Compact number formatter (e.g., 1.2M, 45K)
    compact: (value: number): string => {
      const formatter = Intl.NumberFormat('en', { notation: 'compact' });
      return formatter.format(value);
    },
  
    // Duration formatter
    duration: (milliseconds: number): string => {
      const seconds = Math.floor(milliseconds / 1000);
      const minutes = Math.floor(seconds / 60);
      const hours = Math.floor(minutes / 60);
  
      if (hours > 0) {
        return `${hours}h ${minutes % 60}m`;
      }
      if (minutes > 0) {
        return `${minutes}m ${seconds % 60}s`;
      }
      return `${seconds}s`;
    },
  
    // Date formatters
    date: (date: Date): string => {
      return date.toLocaleDateString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
      });
    },
  
    dateTime: (date: Date): string => {
      return date.toLocaleString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    }
  };
  