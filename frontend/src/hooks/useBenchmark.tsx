import { useState, useEffect, useCallback } from 'react';
import { benchmarkApi, BenchmarkConfig, BenchmarkStatus } from '../components/api/benchmarkApi';

export const useBenchmark = () => {
  const [configs, setConfigs] = useState<BenchmarkConfig[]>([]);
  const [activeRuns, setActiveRuns] = useState<BenchmarkStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load saved configurations
  useEffect(() => {
    const loadConfigs = async () => {
      try {
        const savedConfigs = await benchmarkApi.loadConfigs();
        setConfigs(savedConfigs);
      } catch (err) {
        setError('Failed to load configurations');
      } finally {
        setLoading(false);
      }
    };

    loadConfigs();
  }, []);

  // Save a configuration
  const saveConfig = useCallback(async (config: BenchmarkConfig) => {
    try {
      const savedConfig = await benchmarkApi.saveConfig(config);
      setConfigs(prev => [...prev.filter(c => c.id !== config.id), savedConfig]);
      return savedConfig;
    } catch (err) {
      setError('Failed to save configuration');
      throw err;
    }
  }, []);

  // Start a benchmark run
  const startRun = useCallback(async (configId: string) => {
    try {
      const runId = await benchmarkApi.startRun(configId);
      
      // Subscribe to updates
      const unsubscribe = benchmarkApi.subscribeToRunUpdates(runId, (status) => {
        setActiveRuns(prev => [...prev.filter(r => r.id !== runId), status]);
      });

      // Initial status
      const initialStatus = await benchmarkApi.getRunStatus(runId);
      setActiveRuns(prev => [...prev, initialStatus]);

      return unsubscribe;
    } catch (err) {
      setError('Failed to start benchmark run');
      throw err;
    }
  }, []);

  // Stop a benchmark run
  const stopRun = useCallback(async (runId: string) => {
    try {
      await benchmarkApi.stopRun(runId);
      // Status update will come through subscription
    } catch (err) {
      setError('Failed to stop benchmark run');
      throw err;
    }
  }, []);

  // Export results
  const exportResults = useCallback(async (runId: string, format: 'json' | 'csv' | 'excel') => {
    try {
      const blob = await benchmarkApi.saveResults(runId, format);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `benchmark-results-${runId}.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      setError('Failed to export results');
      throw err;
    }
  }, []);

  // Load preset configurations
  const loadPresets = useCallback(async () => {
    try {
      const presets = await benchmarkApi.getPresetConfigs();
      return presets;
    } catch (err) {
      setError('Failed to load preset configurations');
      throw err;
    }
  }, []);

  return {
    configs,
    activeRuns,
    loading,
    error,
    saveConfig,
    startRun,
    stopRun,
    exportResults,
    loadPresets
  };
};

export default useBenchmark;