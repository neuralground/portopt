// src/tests/mocks/handlers.js
const { rest } = require('msw');

exports.handlers = [
  // Mock GET /api/benchmark/configs
  rest.get('/api/benchmark/configs', (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json([
        {
          id: '1',
          name: 'Test Configuration',
          description: 'A test benchmark configuration',
          problemParams: {
            nAssets: 50,
            nPeriods: 252,
            nTrials: 3
          },
          solverParams: {
            solverType: 'classical',
            maxIterations: 20,
            initialPenalty: 100,
            penaltyMultiplier: 2
          },
          constraints: {
            minWeight: 0.01,
            maxWeight: 0.05,
            maxSectorWeight: 0.25,
            turnoverLimit: 0.15,
            targetReturn: undefined,
            maxVolatility: undefined
          },
          objectives: {
            minimizeRisk: true,
            maximizeReturn: true,
            minimizeCosts: false,
            weights: {
              risk: 0.6,
              return: 0.4,
              costs: 0.0
            }
          }
        }
      ])
    );
  }),
  
  // Mock POST /api/benchmark/configs
  rest.post('/api/benchmark/configs', (req, res, ctx) => {
    const config = req.body;
    return res(
      ctx.status(200),
      ctx.json({
        ...config,
        id: config.id || 'new-config-id'
      })
    );
  }),
  
  // Mock POST /api/benchmark/runs/:id
  rest.post('/api/benchmark/runs/:id', (req, res, ctx) => {
    const { id } = req.params;
    return res(
      ctx.status(200),
      ctx.json({
        id: 'run-123',
        configId: id,
        status: 'running',
        progress: 0,
        currentTrial: 0,
        totalTrials: 3,
        metrics: {
          solveTime: [],
          objectiveValue: [],
          riskMetrics: {
            volatility: [],
            var: [],
            cvar: []
          },
          performanceMetrics: {
            return: [],
            sharpeRatio: [],
            turnover: []
          }
        }
      })
    );
  }),
  
  // Mock GET /api/benchmark/runs/:id
  rest.get('/api/benchmark/runs/:id', (req, res, ctx) => {
    const { id } = req.params;
    return res(
      ctx.status(200),
      ctx.json({
        id: id,
        configId: '1',
        status: 'completed',
        progress: 100,
        currentTrial: 3,
        totalTrials: 3,
        metrics: {
          solveTime: [0.5, 0.6, 0.7],
          objectiveValue: [0.1, 0.09, 0.11],
          riskMetrics: {
            volatility: [0.15, 0.14, 0.16],
            var: [0.25, 0.24, 0.26],
            cvar: [0.35, 0.34, 0.36]
          },
          performanceMetrics: {
            return: [0.08, 0.09, 0.07],
            sharpeRatio: [0.5, 0.6, 0.4],
            turnover: [0.12, 0.11, 0.13]
          }
        }
      })
    );
  })
];
