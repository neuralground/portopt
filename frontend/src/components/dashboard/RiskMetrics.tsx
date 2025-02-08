import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import RiskChart from '../charts/RiskChart';
import { Activity, Wallet, TrendingUp } from 'lucide-react';

const RiskMetrics = () => {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {/* Risk Decomposition */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Risk Decomposition
          </CardTitle>
        </CardHeader>
        <CardContent>
          <RiskChart type="decomposition" />
        </CardContent>
      </Card>

      {/* Factor Exposures */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4" />
            Factor Exposures
          </CardTitle>
        </CardHeader>
        <CardContent>
          <RiskChart type="factors" />
        </CardContent>
      </Card>

      {/* Risk Metrics */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Wallet className="h-4 w-4" />
            Risk Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm font-medium">VaR (95%)</p>
                <p className="text-2xl font-bold">-2.34%</p>
              </div>
              <div>
                <p className="text-sm font-medium">CVaR (95%)</p>
                <p className="text-2xl font-bold">-3.12%</p>
              </div>
              <div>
                <p className="text-sm font-medium">Volatility</p>
                <p className="text-2xl font-bold">15.6%</p>
              </div>
              <div>
                <p className="text-sm font-medium">Beta</p>
                <p className="text-2xl font-bold">1.05</p>
              </div>
            </div>
            <RiskChart type="metrics" />
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default RiskMetrics;

