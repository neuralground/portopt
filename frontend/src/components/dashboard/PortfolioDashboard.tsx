import React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import RiskMetrics from './RiskMetrics';
import MarketImpact from './MarketImpact';
import PerformanceMetrics from './PerformanceMetrics';
import ConstraintAnalysis from './ConstraintAnalysis';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

const PortfolioDashboard = () => {
  return (
    <div className="w-full p-4 space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Portfolio Optimization Dashboard</CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="risk" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="risk">Risk Analysis</TabsTrigger>
              <TabsTrigger value="impact">Market Impact</TabsTrigger>
              <TabsTrigger value="performance">Performance</TabsTrigger>
              <TabsTrigger value="constraints">Constraints</TabsTrigger>
              <TabsTrigger value="benchmark">Benchmarking</TabsTrigger>
            </TabsList>

            <TabsContent value="risk">
              <div className="grid gap-4 mt-4">
                <RiskMetrics />
              </div>
            </TabsContent>

            <TabsContent value="impact">
              <div className="grid gap-4 mt-4">
                <MarketImpact />
              </div>
            </TabsContent>

            <TabsContent value="performance">
              <div className="grid gap-4 mt-4">
                <PerformanceMetrics />
              </div>
            </TabsContent>

            <TabsContent value="constraints">
              <div className="grid gap-4 mt-4">
                <ConstraintAnalysis />
              </div>
            </TabsContent>

            <TabsContent value="benchmark">
              <div className="grid gap-4 mt-4">
                <BenchmarkConfiguration />
                <BenchmarkRunner />
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};

export default PortfolioDashboard;
