import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Save, Upload, Play, Settings, AlertCircle, 
  Download, Copy, RefreshCw 
} from 'lucide-react';
import { useBenchmark } from '@/hooks/useBenchmark';
import type { BenchmarkConfig } from '@/api/benchmarkApi';

const BenchmarkConfiguration = () => {
  const {
    configs,
    loading,
    error,
    saveConfig,
    loadPresets,
    startRun
  } = useBenchmark();

  const [selectedConfig, setSelectedConfig] = useState<BenchmarkConfig | null>(null);
  const [activeTab, setActiveTab] = useState('general');
  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    if (configs.length > 0 && !selectedConfig) {
      setSelectedConfig(configs[0]);
    }
  }, [configs]);

  const defaultConfig: BenchmarkConfig = {
    id: crypto.randomUUID(),
    name: "New Configuration",
    description: "",
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
      minimizeCosts: true
    }
  };

  const handleSave = async () => {
    if (!selectedConfig) return;
    setIsSaving(true);
    setSaveError(null);

    try {
      await saveConfig(selectedConfig);
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : 'Failed to save configuration');
    } finally {
      setIsSaving(false);
    }
  };

  const handleLoadPresets = async () => {
    try {
      const presets = await loadPresets();
      if (presets.length > 0) {
        setSelectedConfig(presets[0]);
      }
    } catch (err) {
      // Handle error
    }
  };

  const handleRun = async () => {
    if (!selectedConfig) return;
    try {
      await startRun(selectedConfig.id);
    } catch (err) {
      // Handle error
    }
  };

  const createNewConfig = () => {
    setSelectedConfig({ ...defaultConfig, id: crypto.randomUUID() });
  };

  if (loading) {
    return (
      <Card>
        <CardContent className="py-8">
          <div className="flex items-center justify-center space-x-2">
            <RefreshCw className="h-5 w-5 animate-spin" />
            <span>Loading configurations...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Benchmark Configuration</span>
            <div className="space-x-2">
              <Button onClick={createNewConfig} variant="outline" size="sm">
                <Settings className="h-4 w-4 mr-2" />
                New
              </Button>
              <Button onClick={handleLoadPresets} variant="outline" size="sm">
                <Download className="h-4 w-4 mr-2" />
                Load Preset
              </Button>
              <Button
                onClick={handleSave}
                variant="outline"
                size="sm"
                disabled={isSaving}
              >
                {isSaving ? (
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Save className="h-4 w-4 mr-2" />
                )}
                Save
              </Button>
              <Button
                onClick={handleRun}
                disabled={!selectedConfig}
                size="sm"
              >
                <Play className="h-4 w-4 mr-2" />
                Run
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {saveError && (
            <Alert variant="destructive" className="mb-4">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Save Failed</AlertTitle>
              <AlertDescription>{saveError}</AlertDescription>
            </Alert>
          )}

          {selectedConfig ? (
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList>
                <TabsTrigger value="general">General</TabsTrigger>
                <TabsTrigger value="solver">Solver</TabsTrigger>
                <TabsTrigger value="constraints">Constraints</TabsTrigger>
                <TabsTrigger value="objectives">Objectives</TabsTrigger>
              </TabsList>

              <TabsContent value="general" className="space-y-4">
                <div className="space-y-2">
                  <Input
                    placeholder="Configuration Name"
                    value={selectedConfig.name}
                    onChange={e => setSelectedConfig({
                      ...selectedConfig,
                      name: e.target.value
                    })}
                  />
                  <Input
                    placeholder="Description"
                    value={selectedConfig.description}
                    onChange={e => setSelectedConfig({
                      ...selectedConfig,
                      description: e.target.value
                    })}
                  />
                </div>

                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <label className="text-sm font-medium">Number of Assets</label>
                    <Input
                      type="number"
                      value={selectedConfig.problemParams.nAssets}
                      onChange={e => setSelectedConfig({
                        ...selectedConfig,
                        problemParams: {
                          ...selectedConfig.problemParams,
                          nAssets: parseInt(e.target.value)
                        }
                      })}
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Number of Periods</label>
                    <Input
                      type="number"
                      value={selectedConfig.problemParams.nPeriods}
                      onChange={e => setSelectedConfig({
                        ...selectedConfig,
                        problemParams: {
                          ...selectedConfig.problemParams,
                          nPeriods: parseInt(e.target.value)
                        }
                      })}
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Number of Trials</label>
                    <Input
                      type="number"
                      value={selectedConfig.problemParams.nTrials}
                      onChange={e => setSelectedConfig({
                        ...selectedConfig,
                        problemParams: {
                          ...selectedConfig.problemParams,
                          nTrials: parseInt(e.target.value)
                        }
                      })}
                    />
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="solver" className="space-y-4">
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <label className="text-sm font-medium">Solver Type</label>
                    <select
                      className="w-full p-2 border rounded-md"
                      value={selectedConfig.solverParams.solverType}
                      onChange={e => setSelectedConfig({
                        ...selectedConfig,
                        solverParams: {
                          ...selectedConfig.solverParams,
                          solverType: e.target.value
                        }
                      })}
                    >
                      <option value="classical">Classical (SLSQP)</option>
                      <option value="genetic">Genetic Algorithm</option>
                      <option value="advanced_genetic">Advanced Genetic</option>
                      <option value="annealing">Simulated Annealing</option>
                      <option value="black_litterman">Black-Litterman</option>
                      <option value="black_litterman_conservative">Black-Litterman (Conservative)</option>
                      <option value="factor">Factor Model</option>
                      <option value="factor_conservative">Factor Model (Conservative)</option>
                      <option value="factor_aggressive">Factor Model (Aggressive)</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-sm font-medium">Max Iterations</label>
                    <Input
                      type="number"
                      value={selectedConfig.solverParams.maxIterations}
                      onChange={e => setSelectedConfig({
                        ...selectedConfig,
                        solverParams: {
                          ...selectedConfig.solverParams,
                          maxIterations: parseInt(e.target.value)
                        }
                      })}
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Initial Penalty</label>
                    <Input
                      type="number"
                      value={selectedConfig.solverParams.initialPenalty}
                      onChange={e => setSelectedConfig({
                        ...selectedConfig,
                        solverParams: {
                          ...selectedConfig.solverParams,
                          initialPenalty: parseFloat(e.target.value)
                        }
                      })}
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Penalty Multiplier</label>
                    <Input
                      type="number"
                      value={selectedConfig.solverParams.penaltyMultiplier}
                      onChange={e => setSelectedConfig({
                        ...selectedConfig,
                        solverParams: {
                          ...selectedConfig.solverParams,
                          penaltyMultiplier: parseFloat(e.target.value)
                        }
                      })}
                    />
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="constraints" className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm font-medium">Min Weight</label>
                    <Input
                      type="number"
                      value={selectedConfig.constraints.minWeight}
                      onChange={e => setSelectedConfig({
                        ...selectedConfig,
                        constraints: {
                          ...selectedConfig.constraints,
                          minWeight: parseFloat(e.target.value)
                        }
                      })}
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Max Weight</label>
                    <Input
                      type="number"
                      value={selectedConfig.constraints.maxWeight}
                      onChange={e => setSelectedConfig({
                        ...selectedConfig,
                        constraints: {
                          ...selectedConfig.constraints,
                          maxWeight: parseFloat(e.target.value)
                        }
                      })}
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Max Sector Weight</label>
                    <Input
                      type="number"
                      value={selectedConfig.constraints.maxSectorWeight}
                      onChange={e => setSelectedConfig({
                        ...selectedConfig,
                        constraints: {
                          ...selectedConfig.constraints,
                          maxSectorWeight: parseFloat(e.target.value)
                        }
                      })}
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Turnover Limit</label>
                    <Input
                      type="number"
                      value={selectedConfig.constraints.turnoverLimit}
                      onChange={e => setSelectedConfig({
                        ...selectedConfig,
                        constraints: {
                          ...selectedConfig.constraints,
                          turnoverLimit: parseFloat(e.target.value)
                        }
                      })}
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Target Return</label>
                    <Input
                      type="number"
                      value={selectedConfig.constraints.targetReturn || ''}
                      onChange={e => setSelectedConfig({
                        ...selectedConfig,
                        constraints: {
                          ...selectedConfig.constraints,
                          targetReturn: e.target.value ? parseFloat(e.target.value) : undefined
                        }
                      })}
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Max Volatility</label>
                    <Input
                      type="number"
                      value={selectedConfig.constraints.maxVolatility || ''}
                      onChange={e => setSelectedConfig({
                        ...selectedConfig,
                        constraints: {
                          ...selectedConfig.constraints,
                          maxVolatility: e.target.value ? parseFloat(e.target.value) : undefined
                        }
                      })}
                    />
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="objectives" className="space-y-4">
                <div className="grid grid-cols-1 gap-4">
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={selectedConfig.objectives.minimizeRisk}
                      onChange={e => setSelectedConfig({
                        ...selectedConfig,
                        objectives: {
                          ...selectedConfig.objectives,
                          minimizeRisk: e.target.checked
                        }
                      })}
                    />
                    <label>Minimize Risk</label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={selectedConfig.objectives.maximizeReturn}
                      onChange={e => setSelectedConfig({
                        ...selectedConfig,
                        objectives: {
                          ...selectedConfig.objectives,
                          maximizeReturn: e.target.checked
                        }
                      })}
                    />
                    <label>Maximize Return</label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={selectedConfig.objectives.minimizeCosts}
                      onChange={e => setSelectedConfig({
                        ...selectedConfig,
                        objectives: {
                          ...selectedConfig.objectives,
                          minimizeCosts: e.target.checked
                        }
                      })}
                    />
                    <label>Minimize Costs</label>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          ) : (
            <Alert>
              <AlertTitle>No Configuration Selected</AlertTitle>
              <AlertDescription>
                Create a new configuration or select an existing one to begin.
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default BenchmarkConfiguration;
