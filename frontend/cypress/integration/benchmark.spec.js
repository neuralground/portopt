// cypress/integration/benchmark.spec.js

describe('Benchmark Configuration', () => {
  beforeEach(() => {
    // Intercept API calls
    cy.intercept('GET', '/api/benchmark/configs', { fixture: 'configs.json' }).as('getConfigs');
    cy.intercept('POST', '/api/benchmark/configs', {}).as('saveConfig');
    
    // Visit the benchmark page
    cy.visit('/benchmark');
  });
  
  it('should display the benchmark configuration component', () => {
    // Wait for configs to load
    cy.wait('@getConfigs');
    
    // Check that the component is rendered
    cy.contains('Benchmark Configuration').should('be.visible');
  });
  
  it('should allow selecting different solver types', () => {
    // Wait for configs to load
    cy.wait('@getConfigs');
    
    // Click on the solver tab
    cy.contains('Solver').click();
    
    // Check that the solver type dropdown is visible
    cy.contains('Solver Type').should('be.visible');
    
    // Select the factor model solver
    cy.get('select').contains('Solver Type').parent().select('Factor Model');
    
    // Save the configuration
    cy.contains('Save').click();
    
    // Verify the API was called with the correct solver type
    cy.wait('@saveConfig').its('request.body.solverParams.solverType').should('eq', 'factor');
  });
  
  it('should run a benchmark with the selected solver type', () => {
    // Mock the run API
    cy.intercept('POST', '/api/benchmark/runs/*', { fixture: 'run.json' }).as('startRun');
    
    // Wait for configs to load
    cy.wait('@getConfigs');
    
    // Click on the solver tab
    cy.contains('Solver').click();
    
    // Select the factor model solver
    cy.get('select').contains('Solver Type').parent().select('Factor Model');
    
    // Run the benchmark
    cy.contains('Run').click();
    
    // Verify the run was started
    cy.wait('@startRun');
    
    // Check that the run status is displayed
    cy.contains('Running benchmark...').should('be.visible');
  });
});
