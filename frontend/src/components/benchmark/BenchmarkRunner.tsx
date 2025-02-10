import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Play, StopCircle, RefreshCw, Save, AlertCircle } from 'lucide-react';
import { useBenchmark } from '@/hooks/useBenchmark';
import type { BenchmarkStatus } from '@/api/benchmarkApi';

const BenchmarkRunner = () => {
  const { activeRuns, exportResults, stopRun } = useBenchmark();
  const [selectedFormat, setSelectedFormat] = useState<'json' | 'csv' | 'excel'>('json');
  const [exportError, setExportError] = useState<string | null>(null);

  const handleStopRun = async (runId: string) => {
    try {
      await stopRun(runId);
    } catch (error) {
      console.error('Failed to stop run:', error);
    }
  };

  const handleExport = async (runId: string) => {
    setExportError(null);
    try {
      await exportResults(runId, selectedFormat);
    } catch (error) {
      setExportError(error instanceof Error ? error.message : 'Failed to export results');
    }
  };

  const renderMetricsChart = (run: BenchmarkStatus) => {
    const performanceData = run.metrics.solveTime.map((time, index) => ({
      iteration: index + 1,
      solveTime: time,
      objective: run.metrics.objectiveValue[index],
      return: run.metrics.performanceMetrics.return[index],
      sharpe: run.metrics.performanceMetrics.sharpeRatio[index]
    }));

    return (
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={performanceData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="iteration" />
            <YAxis yAxisId="left" orientation="left" stroke="#2563eb" />
            <YAxis yAxisId="right" orientation="right" stroke="#16a34a" />
            <Tooltip />
            <Legend />
            <Line 
              yAxisId="left"
              type="monotone" 
              dataKey="objective" 
              stroke="#2563eb" 
              name="Objective Value" 
            />
            <Line 
              yAxisId="right"
              type="monotone" 
              dataKey="sharpe" 
              stroke="#16a34a" 
              name="Sharpe Ratio" 
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const renderRiskMetrics = (run: BenchmarkStatus) => {
    const riskData = run.metrics.riskMetrics.volatility.map((vol, index) => ({
      iteration: index + 1,
      volatility: vol,
      var: run.metrics.riskMetrics.var[index],
      cvar: run.metrics.riskMetrics.cvar[index]
    }));

    return (
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={riskData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="iteration" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="volatility" 
              stroke="#2563eb" 
              name="Volatility" 
            />
            <Line 
              type="monotone" 
              dataKey="var" 
              stroke="#dc2626" 
              name="VaR" 
            />
            <Line 
              type="monotone" 
              dataKey="cvar" 
              stroke="#d97706" 
              name="CVaR" 
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  };

  return (
    <div className="space-y-4">
      {activeRuns.length === 0 ? (
        <Card>
          <CardContent className="py-8">
            <div className="text-center text-gray-500">
              No active benchmark runs. Configure and start a new run from the configuration panel.
            </div>
          </CardContent>
        </Card>
      ) : (
        activeRuns.map(run => (
          <Card key={run.id}>
            <CardHeader>
              <CardTitle className="flex justify-between items-center">
                <div className="flex items-center space-x-2">
                  <span>Benchmark Run {run.id}</span>
                  {run.status === 'running' && (
                    <RefreshCw className="h-4 w-4 animate-spin text-blue-500" />
                  )}
                  {run.status === 'completed' && (
                    <span className="text-green-500 text-sm">Completed</span>
                  )}
                  {run.status === 'failed' && (
                    <span className="text-red-500 text-sm">Failed</span>
                  )}
                </div>
                <div className="flex space-x-2">
                  {run.status === 'running' && (
                    <Button 
                      variant="destructive" 
                      size="sm"
                      onClick={() => handleStopRun(run.id)}
                    >
                      <StopCircle className="h-4 w-4 mr-2" />
                      Stop
                    </Button>
                  )}
                  {run.status === 'completed' && (
                    <>
                      <select
                        className="px-2 py-1 border rounded"
                        value={selectedFormat}
                        onChange={(e) => setSelectedFormat(e.target.value as any)}
                      >
                        <option value="json">JSON</option>
                        <option value="csv">CSV</option>
                        <option value="excel">Excel</option>
                      </select>
                      <Button 
                        variant="outline" 
                        size="sm"
                        onClick={() => handleExport(run.id)}
                      >
                        <Save className="h-4 w-4 mr-2" />
                        Export
                      </Button>
                    </>
                  )}
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {run.error && (
                <Alert variant="destructive" className="mb-4">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Error</AlertTitle>
                  <AlertDescription>{run.error}</AlertDescription>
                </Alert>
              )}

              {exportError && (
                <Alert variant="destructive" className="mb-4">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Export Failed</AlertTitle>
                  <AlertDescription>{exportError}</AlertDescription>
                </Alert>
              )}

              {run.status === 'running' && (
                <div className="space-y-2 mb-4">
                  <div className="flex justify-between text-sm">
                    <span>Progress</span>
                    <span>{run.progress.toFixed(1)}%</span>
                  </div>
                  <Progress value={run.progress} />
                  <div className="text-sm text-gray-500">
                    Trial {run.currentTrial} of {run.totalTrials}
                  </div>
                </div>
              )}

              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold mb-2">Performance Metrics</h3>
                  {renderMetricsChart(run)}
                </div>

                <div>
                  <h3 className="text-lg font-semibold mb-2">Risk Metrics</h3>
                  {renderRiskMetrics(run)}
                </div>

                <div className="grid grid-cols-4 gap-4">
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <div className="text-sm font-medium">Feasible Solutions</div>
                    <div className="text-2xl font-bold">
                      {run.metrics.constraintViolations.count.filter(c => c === 0).length} / {run.totalTrials}
                    </div>
                  </div>
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <div className="text-sm font-medium">Avg Solve Time</div>
                    <div className="text-2xl font-bold">
                      {(run.metrics.solveTime.reduce((a, b) => a + b, 0) / run.metrics.solveTime.length).toFixed(2)}s
                    </div>
                  </div>
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <div className="text-sm font-medium">Best Sharpe Ratio</div>
                    <div className="text-2xl font-bold">
                      {Math.max(...run.metrics.performanceMetrics.sharpeRatio).toFixed(2)}
                    </div>
                  </div>
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <div className="text-sm font-medium">Avg Turnover</div>
                    <div className="text-2xl font-bold">
                      {(run.metrics.performanceMetrics.turnover.reduce((a, b) => a + b, 0) / 
                        run.metrics.performanceMetrics.turnover.length * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))
      )}
    </div>
  );
};

export default BenchmarkRunner;
