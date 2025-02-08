import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import ConstraintChart from '../charts/ConstraintChart';
import { AlertCircle, PieChart, BarChart, AlertTriangle } from 'lucide-react';

const ConstraintAnalysis = () => {
  return (
    <div className="grid gap-4">
      {/* Summary Statistics */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="p-4">
            <CardTitle className="text-sm font-medium">Active Positions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">45/100</div>
            <p className="text-xs text-muted-foreground">Min required: 40</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="p-4">
            <CardTitle className="text-sm font-medium">Max Position</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">4.2%</div>
            <p className="text-xs text-muted-foreground">Limit: 5.0%</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="p-4">
            <CardTitle className="text-sm font-medium">Max Sector</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">24.5%</div>
            <p className="text-xs text-muted-foreground">Limit: 25.0%</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="p-4">
            <CardTitle className="text-sm font-medium">Turnover</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">18.2%</div>
            <p className="text-xs text-muted-foreground">Limit: 20.0%</p>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Weight Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart className="h-4 w-4" />
              Weight Distribution
            </CardTitle>
          </CardHeader>
          <CardContent className="h-80">
            <ConstraintChart type="weights" />
          </CardContent>
        </Card>

        {/* Sector Exposure */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <PieChart className="h-4 w-4" />
              Sector Exposure
            </CardTitle>
          </CardHeader>
          <CardContent className="h-80">
            <ConstraintChart type="sectors" />
          </CardContent>
        </Card>
      </div>

      {/* Constraint Violations */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertCircle className="h-4 w-4" />
            Constraint Violations
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Alert variant="warning">
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Near Limits</AlertTitle>
              <AlertDescription>
                Technology sector exposure (24.5%) is approaching limit (25.0%)
              </AlertDescription>
            </Alert>
            <div className="mt-4">
              <ConstraintChart type="violations" />
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ConstraintAnalysis;

