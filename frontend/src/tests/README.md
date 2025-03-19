# Frontend Testing Setup

This directory contains the testing setup for the portfolio optimization frontend.

## Setup Instructions

1. Install the necessary dependencies:

```bash
npm install --save-dev jest @testing-library/react @testing-library/jest-dom @testing-library/user-event jest-environment-jsdom @testing-library/react-hooks msw identity-obj-proxy babel-jest @types/jest
```

2. Add the following scripts to your package.json:

```json
{
  "scripts": {
    "test": "jest",
    "test:watch": "jest --watch",
    "test:coverage": "jest --coverage"
  }
}
```

3. Make sure your jest.config.js is properly configured (already done in this project).

4. Run the tests:

```bash
npm test
```

## Test Structure

- `setupTests.js`: Jest setup file that includes global test configuration
- `mocks/`: Directory containing MSW (Mock Service Worker) setup for API mocking
  - `handlers.js`: API endpoint mock handlers
  - `server.js`: MSW server setup
- `types.d.ts`: TypeScript declarations for testing

## Component Tests

Component tests are located alongside the components they test, with a `.test.tsx` extension.

## Hook Tests

Hook tests are located alongside the hooks they test, with a `.test.tsx` extension.

## Running Tests

To run all tests:
```bash
npm test
```

To run tests in watch mode:
```bash
npm run test:watch
```

To run tests with coverage:
```bash
npm run test:coverage
```
