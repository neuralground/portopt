import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { AlertCircle, TrendingUp, DollarSign, Activity } from 'lucide-react';

const PortfolioDashboard = () => {
  const [selectedMetric, setSelectedMetric] = useState('risk');

  // Simulated data - this would come from your test harness
  const performanceData = [
    { iteration: 1, var: -0.02, cvar: -0.03, tracking_error: 0.015, sharpe: 1.2 },
    { iteration: 2, var: -0.018, cvar: -0.028, tracking_error: 0.014, sharpe: 1.3 },
    // ... more data points
  ];

  const impactData = [
    { time: '1D', spread_cost: 0.0012, impact_cost: 0.0018, total_cost: 0.003 },
    { time: '5D', spread_cost: 0.0015, impact_cost: 0.0022, total_cost: 0.0037 },
    // ... more data points
  ];

  return (
    <div className="w-full space-y-4 p-4">
      <Tabs defaultValue="metrics" className="w-full">
        <TabsList>
          <TabsTrigger value="metrics">Performance Metrics</TabsTrigger>
          <TabsTrigger value="impact">Market Impact</TabsTrigger>
        </TabsList>

        <TabsContent value="metrics">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-4 w-4" />
                  Risk Metrics
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={performanceData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="iteration" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="var" stroke="#8884d8" name="VaR" />
                      <Line type="monotone" dataKey="cvar" stroke="#82ca9d" name="CVaR" />
                      <Line type="monotone" dataKey="tracking_error" stroke="#ffc658" name="Tracking Error" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-4 w-4" />
                  Factor Exposures
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={[
                      { factor: 'Market', exposure: 1.02 },
                      { factor: 'Size', exposure: -0.15 },
                      { factor: 'Value', exposure: 0.45 },
                      { factor: 'Momentum', exposure: -0.22 },
                      { factor: 'Quality', exposure: 0.33 }
                    ]}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="factor" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="exposure" fill="#8884d8" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="impact">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <DollarSign className="h-4 w-4" />
                  Trading Costs
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={impactData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="spread_cost" stroke="#8884d8" name="Spread Cost" />
                      <Line type="monotone" dataKey="impact_cost" stroke="#82ca9d" name="Impact Cost" />
                      <Line type="monotone" dataKey="total_cost" stroke="#ffc658" name="Total Cost" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertCircle className="h-4 w-4" />
                  Liquidity Metrics
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={[
                      { metric: 'ADV %', value: 2.5 },
                      { metric: 'Turnover', value: 15.2 },
                      { metric: 'Spread Impact', value: 0.12 },
                      { metric: 'Market Impact', value: 0.18 }
                    ]}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="metric" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="value" fill="#82ca9d" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default PortfolioDashboard;

