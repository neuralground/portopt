import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import PerformanceChart from '../charts/PerformanceChart';
import { LineChart, BarChart, PieChart } from 'lucide-react';

const PerformanceMetrics = () => {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {/* Cumulative Returns */}
      <Card className="md:col-span-2">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <LineChart className="h-4 w-4" />
            Cumulative Returns
          </CardTitle>
        </CardHeader>
        <CardContent className="h-80">
          <PerformanceChart type="cumulative" />
        </CardContent>
      </Card>

      {/* Rolling Metrics */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart className="h-4 w-4" />
            Rolling Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm font-medium">Sharpe Ratio</p>
                <p className="text-2xl font-bold">1.85</p>
              </div>
              <div>
                <p className="text-sm font-medium">Information Ratio</p>
                <p className="text-2xl font-bold">0.92</p>
              </div>
              <div>
                <p className="text-sm font-medium">Sortino Ratio</p>
                <p className="text-2xl font-bold">2.15</p>
              </div>
              <div>
                <p className="text-sm font-medium">Tracking Error</p>
                <p className="text-2xl font-bold">3.2%</p>
              </div>
            </div>
            <PerformanceChart type="rolling" />
          </div>
        </CardContent>
      </Card>

      {/* Return Distribution */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <PieChart className="h-4 w-4" />
            Return Distribution
          </CardTitle>
        </CardHeader>
        <CardContent>
          <PerformanceChart type="distribution" />
        </CardContent>
      </Card>
    </div>
  );
};

export default PerformanceMetrics;

