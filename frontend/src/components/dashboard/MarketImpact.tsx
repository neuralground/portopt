import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import ImpactChart from '../charts/ImpactChart';
import { DollarSign, Clock, Activity } from 'lucide-react';

const MarketImpact = () => {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {/* Trading Cost Analysis */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <DollarSign className="h-4 w-4" />
            Trading Cost Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm font-medium">Total Cost</p>
                <p className="text-2xl font-bold">32.5 bps</p>
              </div>
              <div>
                <p className="text-sm font-medium">Spread Cost</p>
                <p className="text-2xl font-bold">12.3 bps</p>
              </div>
              <div>
                <p className="text-sm font-medium">Impact Cost</p>
                <p className="text-2xl font-bold">20.2 bps</p>
              </div>
              <div>
                <p className="text-sm font-medium">Avg Participation</p>
                <p className="text-2xl font-bold">15.4%</p>
              </div>
            </div>
            <ImpactChart type="costs" />
          </div>
        </CardContent>
      </Card>

      {/* Liquidation Profile */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-4 w-4" />
            Liquidation Profile
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ImpactChart type="liquidation" />
        </CardContent>
      </Card>

      {/* Market Impact Over Time */}
      <Card className="md:col-span-2">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Market Impact Over Time
          </CardTitle>
        </CardHeader>
        <CardContent className="h-80">
          <ImpactChart type="timeline" />
        </CardContent>
      </Card>
    </div>
  );
};

export default MarketImpact;

