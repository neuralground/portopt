// Import jest-dom matchers
require('@testing-library/jest-dom');

// This will be used later for MSW setup
const { server } = require('./mocks/server');

// Establish API mocking before all tests
beforeAll(() => server.listen());

// Reset any request handlers that we may add during the tests
afterEach(() => server.resetHandlers());

// Clean up after the tests are finished
afterAll(() => server.close());
