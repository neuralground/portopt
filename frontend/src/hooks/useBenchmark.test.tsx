import { renderHook, act } from '@testing-library/react';
import { useBenchmark } from './useBenchmark';
import { benchmarkApi } from '../components/api/benchmarkApi';

// Mock the benchmarkApi
jest.mock('../components/api/benchmarkApi', () => ({
  benchmarkApi: {
    loadConfigs: jest.fn(),
    saveConfig: jest.fn(),
    loadPresets: jest.fn(),
    startRun: jest.fn(),
    listenToRunStatus: jest.fn()
  }
}));

describe('useBenchmark hook', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('loads configurations on mount', async () => {
    const mockConfigs = [
      {
        id: '1',
        name: 'Test Config',
        description: 'Test description',
        problemParams: { nAssets: 50, nPeriods: 252, nTrials: 3 },
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
          turnoverLimit: 0.15
        },
        objectives: {
          minimizeRisk: true,
          maximizeReturn: true,
          minimizeCosts: false
        }
      }
    ];

    (benchmarkApi.loadConfigs as jest.Mock).mockResolvedValue(mockConfigs);

    const { result } = renderHook(() => useBenchmark());

    // Initially loading should be true
    expect(result.current.loading).toBe(true);
    expect(result.current.configs).toEqual([]);

    // Wait for the hook to update
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0));
    });

    // After loading, configs should be populated
    expect(result.current.loading).toBe(false);
    expect(result.current.configs).toEqual(mockConfigs);
    expect(benchmarkApi.loadConfigs).toHaveBeenCalledTimes(1);
  });

  test('saves configuration', async () => {
    const mockConfig = {
      id: '1',
      name: 'Test Config',
      description: 'Test description',
      problemParams: { nAssets: 50, nPeriods: 252, nTrials: 3 },
      solverParams: { 
        solverType: 'factor',
        maxIterations: 20, 
        initialPenalty: 100, 
        penaltyMultiplier: 2 
      },
      constraints: {
        minWeight: 0.01,
        maxWeight: 0.05,
        maxSectorWeight: 0.25,
        turnoverLimit: 0.15
      },
      objectives: {
        minimizeRisk: true,
        maximizeReturn: true,
        minimizeCosts: false
      }
    };

    (benchmarkApi.saveConfig as jest.Mock).mockResolvedValue(mockConfig);
    (benchmarkApi.loadConfigs as jest.Mock).mockResolvedValue([mockConfig]);

    const { result } = renderHook(() => useBenchmark());

    // Wait for initial load
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0));
    });

    // Save the config
    await act(async () => {
      await result.current.saveConfig(mockConfig);
    });

    expect(benchmarkApi.saveConfig).toHaveBeenCalledWith(mockConfig);
    expect(benchmarkApi.loadConfigs).toHaveBeenCalledTimes(2); // Initial + after save
  });

  test('starts a benchmark run', async () => {
    const mockRunId = 'run-123';
    (benchmarkApi.startRun as jest.Mock).mockResolvedValue({ id: mockRunId });
    (benchmarkApi.loadConfigs as jest.Mock).mockResolvedValue([]);

    const { result } = renderHook(() => useBenchmark());

    // Wait for initial load
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0));
    });

    // Start a run
    let runId;
    await act(async () => {
      runId = await result.current.startRun('config-1');
    });

    expect(benchmarkApi.startRun).toHaveBeenCalledWith('config-1');
    expect(runId).toBe(mockRunId);
  });
});
