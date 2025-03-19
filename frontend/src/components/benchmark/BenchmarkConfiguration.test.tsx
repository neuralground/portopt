import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import BenchmarkConfiguration from './BenchmarkConfiguration';

// Mock the useBenchmark hook
jest.mock('../../hooks/useBenchmark', () => ({
  useBenchmark: () => ({
    configs: [],
    loading: false,
    error: null,
    saveConfig: jest.fn(),
    loadPresets: jest.fn(),
    startRun: jest.fn()
  })
}));

// Mock the UI components
jest.mock('@/components/ui/card', () => ({
  Card: ({ children }: { children: React.ReactNode }) => <div data-testid="card">{children}</div>,
  CardHeader: ({ children }: { children: React.ReactNode }) => <div data-testid="card-header">{children}</div>,
  CardTitle: ({ children }: { children: React.ReactNode }) => <div data-testid="card-title">{children}</div>,
  CardContent: ({ children }: { children: React.ReactNode }) => <div data-testid="card-content">{children}</div>
}));

jest.mock('@/components/ui/input', () => ({
  Input: (props: any) => <input data-testid="input" {...props} />
}));

jest.mock('@/components/ui/button', () => ({
  Button: (props: any) => <button data-testid="button" {...props} />
}));

jest.mock('@/components/ui/alert', () => ({
  Alert: ({ children }: { children: React.ReactNode }) => <div data-testid="alert">{children}</div>,
  AlertTitle: ({ children }: { children: React.ReactNode }) => <div data-testid="alert-title">{children}</div>,
  AlertDescription: ({ children }: { children: React.ReactNode }) => <div data-testid="alert-description">{children}</div>
}));

jest.mock('@/components/ui/tabs', () => ({
  Tabs: ({ children }: { children: React.ReactNode }) => <div data-testid="tabs">{children}</div>,
  TabsContent: ({ children, value }: { children: React.ReactNode, value: string }) => (
    <div data-testid={`tabs-content-${value}`}>{children}</div>
  ),
  TabsList: ({ children }: { children: React.ReactNode }) => <div data-testid="tabs-list">{children}</div>,
  TabsTrigger: ({ children, value, onClick }: { children: React.ReactNode, value: string, onClick?: () => void }) => (
    <button data-testid={`tabs-trigger-${value}`} onClick={onClick}>{children}</button>
  )
}));

describe('BenchmarkConfiguration', () => {
  beforeEach(() => {
    // Clear all mocks before each test
    jest.clearAllMocks();
  });

  test('renders the component', () => {
    render(<BenchmarkConfiguration />);
    
    // Check that the main elements are rendered
    expect(screen.getByTestId('card')).toBeInTheDocument();
    expect(screen.getByTestId('card-title')).toBeInTheDocument();
    expect(screen.getByText('Benchmark Configuration')).toBeInTheDocument();
  });

  test('renders solver tab with solver type dropdown', () => {
    render(<BenchmarkConfiguration />);
    
    // Find and click the solver tab
    const solverTab = screen.getByTestId('tabs-trigger-solver');
    fireEvent.click(solverTab);
    
    // Check that the solver content is rendered
    const solverContent = screen.getByTestId('tabs-content-solver');
    expect(solverContent).toBeInTheDocument();
    
    // Check that the solver type dropdown is rendered
    const solverTypeLabel = screen.getByText('Solver Type');
    expect(solverTypeLabel).toBeInTheDocument();
    
    // Find the select element near the label
    const solverTypeSelect = solverTypeLabel.parentElement?.querySelector('select');
    expect(solverTypeSelect).toBeInTheDocument();
    
    // Check that the factor model options are available
    expect(screen.getByText('Factor Model')).toBeInTheDocument();
    expect(screen.getByText('Factor Model (Conservative)')).toBeInTheDocument();
    expect(screen.getByText('Factor Model (Aggressive)')).toBeInTheDocument();
  });

  test('changes solver type when a new option is selected', () => {
    render(<BenchmarkConfiguration />);
    
    // Find and click the solver tab
    const solverTab = screen.getByTestId('tabs-trigger-solver');
    fireEvent.click(solverTab);
    
    // Find the solver type select
    const solverTypeLabel = screen.getByText('Solver Type');
    const solverTypeSelect = solverTypeLabel.parentElement?.querySelector('select');
    
    // Change the value to 'factor'
    fireEvent.change(solverTypeSelect!, { target: { value: 'factor' } });
    
    // Check that the value has changed
    expect(solverTypeSelect!.value).toBe('factor');
  });
});
