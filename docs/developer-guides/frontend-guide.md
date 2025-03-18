# Frontend Development Guide

This guide provides information for developers working on the Portfolio Optimization Testbed dashboard frontend.

## Overview

The dashboard frontend is built using modern web technologies to provide an interactive visualization and control interface for the Portfolio Optimization Testbed. It allows users to:

1. Configure and run optimization problems
2. Visualize optimization results
3. Compare different solver algorithms
4. Analyze portfolio performance metrics
5. Export results for further analysis

## Technology Stack

The frontend is built with the following technologies:

- **React**: A JavaScript library for building user interfaces
- **TypeScript**: A typed superset of JavaScript
- **Redux**: A state management library
- **React Router**: For navigation between different views
- **D3.js**: For custom data visualizations
- **Plotly.js**: For interactive charts
- **Material-UI**: For UI components
- **Axios**: For API communication
- **Jest**: For testing
- **Webpack**: For bundling

## Directory Structure

The frontend code is organized as follows:

```
frontend/
├── public/                # Static assets
│   ├── index.html         # HTML template
│   ├── favicon.ico        # Favicon
│   └── assets/            # Static assets like images
├── src/                   # Source code
│   ├── components/        # React components
│   │   ├── common/        # Shared components
│   │   ├── dashboard/     # Dashboard components
│   │   ├── optimization/  # Optimization components
│   │   └── visualization/ # Visualization components
│   ├── hooks/             # Custom React hooks
│   ├── pages/             # Page components
│   ├── services/          # API and other services
│   ├── store/             # Redux store
│   │   ├── actions/       # Redux actions
│   │   ├── reducers/      # Redux reducers
│   │   └── selectors/     # Redux selectors
│   ├── types/             # TypeScript type definitions
│   ├── utils/             # Utility functions
│   ├── App.tsx            # Root component
│   ├── index.tsx          # Entry point
│   └── routes.tsx         # Route definitions
├── tests/                 # Test files
├── .eslintrc.js           # ESLint configuration
├── .prettierrc            # Prettier configuration
├── jest.config.js         # Jest configuration
├── package.json           # Dependencies and scripts
├── tsconfig.json          # TypeScript configuration
└── webpack.config.js      # Webpack configuration
```

## Getting Started

### Prerequisites

- Node.js 14.x or higher
- npm 7.x or higher

### Installation

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

### Development Server

Start the development server:

```bash
npm start
```

This will start the development server at http://localhost:3000.

### Building for Production

Build the frontend for production:

```bash
npm run build
```

This will create a production-ready build in the `build` directory.

### Running Tests

Run the test suite:

```bash
npm test
```

Run tests with coverage:

```bash
npm test -- --coverage
```

## Key Components

### Dashboard Layout

The dashboard layout is defined in `src/components/dashboard/DashboardLayout.tsx`:

```tsx
import React from 'react';
import { Box, Grid, Paper } from '@material-ui/core';
import Header from '../common/Header';
import Sidebar from '../common/Sidebar';
import Footer from '../common/Footer';

interface DashboardLayoutProps {
  children: React.ReactNode;
}

const DashboardLayout: React.FC<DashboardLayoutProps> = ({ children }) => {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <Header />
      <Box sx={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <Sidebar />
        <Box component="main" sx={{ flex: 1, p: 3, overflow: 'auto' }}>
          <Paper sx={{ p: 2 }}>
            {children}
          </Paper>
        </Box>
      </Box>
      <Footer />
    </Box>
  );
};

export default DashboardLayout;
```

### Optimization Form

The optimization form allows users to configure and run optimization problems:

```tsx
import React, { useState } from 'react';
import { useDispatch } from 'react-redux';
import { Button, FormControl, Grid, InputLabel, MenuItem, Select, TextField } from '@material-ui/core';
import { runOptimization } from '../../store/actions/optimizationActions';

const OptimizationForm: React.FC = () => {
  const dispatch = useDispatch();
  const [formData, setFormData] = useState({
    problemType: 'minimum_variance',
    nAssets: 20,
    nPeriods: 252,
    constraints: ['full_investment', 'long_only'],
    maxIterations: 100,
    tolerance: 1e-6
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | { name?: string; value: unknown }>) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name as string]: value
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    dispatch(runOptimization(formData));
  };

  return (
    <form onSubmit={handleSubmit}>
      <Grid container spacing={3}>
        {/* Form fields */}
        <Grid item xs={12}>
          <Button type="submit" variant="contained" color="primary">
            Run Optimization
          </Button>
        </Grid>
      </Grid>
    </form>
  );
};

export default OptimizationForm;
```

### Portfolio Chart

The portfolio chart visualizes the optimized portfolio weights:

```tsx
import React from 'react';
import { useSelector } from 'react-redux';
import { Pie } from 'react-chartjs-2';
import { Box, Typography } from '@material-ui/core';
import { selectOptimizationResult } from '../../store/selectors/optimizationSelectors';

const PortfolioChart: React.FC = () => {
  const result = useSelector(selectOptimizationResult);

  if (!result) {
    return <Typography>No optimization result available</Typography>;
  }

  const { weights } = result;

  const data = {
    labels: weights.map((_, i) => `Asset ${i + 1}`),
    datasets: [
      {
        data: weights,
        backgroundColor: [
          '#FF6384',
          '#36A2EB',
          '#FFCE56',
          '#4BC0C0',
          '#9966FF',
          '#FF9F40',
          // Add more colors as needed
        ]
      }
    ]
  };

  return (
    <Box>
      <Typography variant="h6">Portfolio Allocation</Typography>
      <Pie data={data} />
    </Box>
  );
};

export default PortfolioChart;
```

### Performance Metrics

The performance metrics component displays key portfolio metrics:

```tsx
import React from 'react';
import { useSelector } from 'react-redux';
import { Grid, Paper, Typography } from '@material-ui/core';
import { selectPerformanceMetrics } from '../../store/selectors/metricsSelectors';

const PerformanceMetrics: React.FC = () => {
  const metrics = useSelector(selectPerformanceMetrics);

  if (!metrics) {
    return <Typography>No metrics available</Typography>;
  }

  const { expectedReturn, volatility, sharpeRatio, maxDrawdown } = metrics;

  return (
    <Grid container spacing={3}>
      <Grid item xs={12} sm={6} md={3}>
        <Paper sx={{ p: 2, textAlign: 'center' }}>
          <Typography variant="h6">Expected Return</Typography>
          <Typography variant="h4">{(expectedReturn * 100).toFixed(2)}%</Typography>
        </Paper>
      </Grid>
      <Grid item xs={12} sm={6} md={3}>
        <Paper sx={{ p: 2, textAlign: 'center' }}>
          <Typography variant="h6">Volatility</Typography>
          <Typography variant="h4">{(volatility * 100).toFixed(2)}%</Typography>
        </Paper>
      </Grid>
      <Grid item xs={12} sm={6} md={3}>
        <Paper sx={{ p: 2, textAlign: 'center' }}>
          <Typography variant="h6">Sharpe Ratio</Typography>
          <Typography variant="h4">{sharpeRatio.toFixed(2)}</Typography>
        </Paper>
      </Grid>
      <Grid item xs={12} sm={6} md={3}>
        <Paper sx={{ p: 2, textAlign: 'center' }}>
          <Typography variant="h6">Max Drawdown</Typography>
          <Typography variant="h4">{(maxDrawdown * 100).toFixed(2)}%</Typography>
        </Paper>
      </Grid>
    </Grid>
  );
};

export default PerformanceMetrics;
```

## State Management

The frontend uses Redux for state management. The store is organized as follows:

### Store Structure

```typescript
interface RootState {
  optimization: OptimizationState;
  problems: ProblemsState;
  results: ResultsState;
  metrics: MetricsState;
  ui: UIState;
}

interface OptimizationState {
  loading: boolean;
  error: string | null;
  currentJobId: string | null;
}

interface ProblemsState {
  problems: Problem[];
  loading: boolean;
  error: string | null;
}

interface ResultsState {
  results: { [key: string]: OptimizationResult };
  loading: boolean;
  error: string | null;
}

interface MetricsState {
  metrics: { [key: string]: PerformanceMetrics };
  loading: boolean;
  error: string | null;
}

interface UIState {
  sidebarOpen: boolean;
  currentTab: string;
  theme: 'light' | 'dark';
}
```

### Actions

Actions are defined in the `src/store/actions` directory:

```typescript
// optimizationActions.ts
export const RUN_OPTIMIZATION_REQUEST = 'RUN_OPTIMIZATION_REQUEST';
export const RUN_OPTIMIZATION_SUCCESS = 'RUN_OPTIMIZATION_SUCCESS';
export const RUN_OPTIMIZATION_FAILURE = 'RUN_OPTIMIZATION_FAILURE';

export const runOptimization = (params) => async (dispatch) => {
  dispatch({ type: RUN_OPTIMIZATION_REQUEST });
  try {
    const response = await api.runOptimization(params);
    dispatch({ type: RUN_OPTIMIZATION_SUCCESS, payload: response.data });
  } catch (error) {
    dispatch({ type: RUN_OPTIMIZATION_FAILURE, payload: error.message });
  }
};
```

### Reducers

Reducers are defined in the `src/store/reducers` directory:

```typescript
// optimizationReducer.ts
import {
  RUN_OPTIMIZATION_REQUEST,
  RUN_OPTIMIZATION_SUCCESS,
  RUN_OPTIMIZATION_FAILURE
} from '../actions/optimizationActions';

const initialState: OptimizationState = {
  loading: false,
  error: null,
  currentJobId: null
};

export default function optimizationReducer(state = initialState, action) {
  switch (action.type) {
    case RUN_OPTIMIZATION_REQUEST:
      return {
        ...state,
        loading: true,
        error: null
      };
    case RUN_OPTIMIZATION_SUCCESS:
      return {
        ...state,
        loading: false,
        currentJobId: action.payload.jobId
      };
    case RUN_OPTIMIZATION_FAILURE:
      return {
        ...state,
        loading: false,
        error: action.payload
      };
    default:
      return state;
  }
}
```

### Selectors

Selectors are defined in the `src/store/selectors` directory:

```typescript
// optimizationSelectors.ts
import { createSelector } from 'reselect';
import { RootState } from '../types';

export const selectOptimizationState = (state: RootState) => state.optimization;
export const selectResultsState = (state: RootState) => state.results;

export const selectCurrentJobId = createSelector(
  selectOptimizationState,
  (optimization) => optimization.currentJobId
);

export const selectOptimizationResult = createSelector(
  selectResultsState,
  selectCurrentJobId,
  (results, jobId) => jobId ? results.results[jobId] : null
);
```

## API Integration

The frontend communicates with the backend API using Axios. API services are defined in the `src/services` directory:

```typescript
// api.ts
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  }
});

export const runOptimization = (params) => {
  return api.post('/optimize', params);
};

export const getOptimizationResult = (jobId) => {
  return api.get(`/results/${jobId}`);
};

export const getProblems = () => {
  return api.get('/problems');
};

export const getPerformanceMetrics = (resultId) => {
  return api.get(`/metrics/${resultId}`);
};

export default api;
```

## Custom Hooks

Custom React hooks are defined in the `src/hooks` directory:

```typescript
// useOptimizationResult.ts
import { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { fetchOptimizationResult } from '../store/actions/resultsActions';
import { selectOptimizationResult } from '../store/selectors/optimizationSelectors';

export const useOptimizationResult = (jobId) => {
  const dispatch = useDispatch();
  const result = useSelector(selectOptimizationResult);

  useEffect(() => {
    if (jobId) {
      dispatch(fetchOptimizationResult(jobId));
    }
  }, [dispatch, jobId]);

  return result;
};
```

## Routing

Routes are defined in `src/routes.tsx`:

```tsx
import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import DashboardPage from './pages/DashboardPage';
import OptimizationPage from './pages/OptimizationPage';
import ResultsPage from './pages/ResultsPage';
import ComparePage from './pages/ComparePage';
import SettingsPage from './pages/SettingsPage';
import NotFoundPage from './pages/NotFoundPage';

const Routes: React.FC = () => {
  return (
    <Router>
      <Switch>
        <Route exact path="/" component={DashboardPage} />
        <Route path="/optimization" component={OptimizationPage} />
        <Route path="/results/:id" component={ResultsPage} />
        <Route path="/compare" component={ComparePage} />
        <Route path="/settings" component={SettingsPage} />
        <Route component={NotFoundPage} />
      </Switch>
    </Router>
  );
};

export default Routes;
```

## Theming

The frontend supports light and dark themes using Material-UI's theming system:

```tsx
// src/theme.ts
import { createTheme } from '@material-ui/core/styles';

export const lightTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

export const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
  },
});
```

## Testing

The frontend uses Jest and React Testing Library for testing:

```tsx
// src/components/PortfolioChart.test.tsx
import React from 'react';
import { render, screen } from '@testing-library/react';
import { Provider } from 'react-redux';
import configureStore from 'redux-mock-store';
import PortfolioChart from './PortfolioChart';

const mockStore = configureStore([]);

describe('PortfolioChart', () => {
  test('renders no data message when result is not available', () => {
    const store = mockStore({
      results: { results: {} },
      optimization: { currentJobId: null }
    });

    render(
      <Provider store={store}>
        <PortfolioChart />
      </Provider>
    );

    expect(screen.getByText('No optimization result available')).toBeInTheDocument();
  });

  test('renders chart when result is available', () => {
    const store = mockStore({
      results: {
        results: {
          'job-123': {
            weights: [0.4, 0.3, 0.2, 0.1]
          }
        }
      },
      optimization: { currentJobId: 'job-123' }
    });

    render(
      <Provider store={store}>
        <PortfolioChart />
      </Provider>
    );

    expect(screen.getByText('Portfolio Allocation')).toBeInTheDocument();
  });
});
```

## Best Practices

### Code Style

- Follow the [Airbnb JavaScript Style Guide](https://github.com/airbnb/javascript)
- Use ESLint and Prettier for code formatting
- Use TypeScript for type safety

### Component Structure

- Use functional components with hooks
- Keep components small and focused
- Use prop types or TypeScript interfaces for component props
- Use React.memo for performance optimization when appropriate

### State Management

- Use Redux for global state
- Use React Context for theme and localization
- Use local state for component-specific state
- Use selectors for derived data

### Performance

- Use React.memo for expensive components
- Use useCallback for event handlers
- Use useMemo for expensive calculations
- Use virtualization for long lists

### Accessibility

- Use semantic HTML elements
- Add ARIA attributes when necessary
- Ensure keyboard navigation works
- Test with screen readers

## Deployment

The frontend can be deployed using various methods:

### Static Hosting

Build the frontend and deploy to a static hosting service:

```bash
npm run build
```

The build output in the `build` directory can be deployed to services like:
- GitHub Pages
- Netlify
- Vercel
- AWS S3 + CloudFront

### Docker

The frontend can be deployed using Docker:

```dockerfile
# Dockerfile
FROM node:14-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

Build and run the Docker image:

```bash
docker build -t portopt-frontend .
docker run -p 80:80 portopt-frontend
```

## Troubleshooting

### Common Issues

1. **API Connection Issues**
   
   If the frontend can't connect to the API:
   - Check that the API server is running
   - Verify the API_BASE_URL environment variable
   - Check for CORS issues

2. **Build Errors**
   
   If you encounter build errors:
   - Check for TypeScript errors
   - Ensure all dependencies are installed
   - Clear the node_modules directory and reinstall

3. **Performance Issues**
   
   If the dashboard is slow:
   - Use React DevTools to identify performance bottlenecks
   - Optimize expensive components with memoization
   - Use virtualization for long lists

## Related Resources

- [Architecture Overview](./architecture.md)
- [Contributing Guide](./contributing.md)
- [Testing Guide](./testing.md)
- [API Reference](../reference/api-reference.md)
- [Dashboard Guide](../user-guides/dashboard-guide.md)
