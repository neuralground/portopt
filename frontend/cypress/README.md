# Cypress E2E Testing

This directory contains Cypress end-to-end tests for the portfolio optimization frontend.

## Setup Instructions

1. Install Cypress:

```bash
npm install --save-dev cypress
```

2. Add Cypress scripts to package.json:

```json
{
  "scripts": {
    "cypress:open": "cypress open",
    "cypress:run": "cypress run"
  }
}
```

3. Create fixture files:

Create the following fixture files in the `cypress/fixtures` directory:

**configs.json**:
```json
[
  {
    "id": "1",
    "name": "Test Configuration",
    "description": "A test benchmark configuration",
    "problemParams": {
      "nAssets": 50,
      "nPeriods": 252,
      "nTrials": 3
    },
    "solverParams": {
      "solverType": "classical",
      "maxIterations": 20,
      "initialPenalty": 100,
      "penaltyMultiplier": 2
    },
    "constraints": {
      "minWeight": 0.01,
      "maxWeight": 0.05,
      "maxSectorWeight": 0.25,
      "turnoverLimit": 0.15
    },
    "objectives": {
      "minimizeRisk": true,
      "maximizeReturn": true,
      "minimizeCosts": false,
      "weights": {
        "risk": 0.6,
        "return": 0.4,
        "costs": 0.0
      }
    }
  }
]
```

**run.json**:
```json
{
  "id": "run-123",
  "configId": "1",
  "status": "running",
  "progress": 0,
  "currentTrial": 0,
  "totalTrials": 3,
  "metrics": {
    "solveTime": [],
    "objectiveValue": [],
    "riskMetrics": {
      "volatility": [],
      "var": [],
      "cvar": []
    },
    "performanceMetrics": {
      "return": [],
      "sharpeRatio": [],
      "turnover": []
    }
  }
}
```

## Running Tests

To open Cypress Test Runner:
```bash
npm run cypress:open
```

To run Cypress tests in headless mode:
```bash
npm run cypress:run
```
