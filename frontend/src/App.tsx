import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

function App() {
  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <Card className="mx-auto container">
        <CardHeader>
          <CardTitle>Portfolio Optimization Dashboard</CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="risk" className="w-full">
            <TabsList className="grid w-full grid-cols-5">
              <TabsTrigger value="risk">Risk Analysis</TabsTrigger>
              <TabsTrigger value="impact">Market Impact</TabsTrigger>
              <TabsTrigger value="performance">Performance</TabsTrigger>
              <TabsTrigger value="constraints">Constraints</TabsTrigger>
              <TabsTrigger value="benchmark">Benchmarking</TabsTrigger>
            </TabsList>

            <TabsContent value="risk">
              <div className="grid gap-4 mt-4">
                <Card>
                  <CardHeader>
                    <CardTitle>Risk Metrics</CardTitle>
                  </CardHeader>
                  <CardContent>
                    Risk analysis content will go here...
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="impact">
              <div className="grid gap-4 mt-4">
                <Card>
                  <CardHeader>
                    <CardTitle>Market Impact</CardTitle>
                  </CardHeader>
                  <CardContent>
                    Market impact content will go here...
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="performance">
              <div className="grid gap-4 mt-4">
                <Card>
                  <CardHeader>
                    <CardTitle>Performance Metrics</CardTitle>
                  </CardHeader>
                  <CardContent>
                    Performance content will go here...
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="constraints">
              <div className="grid gap-4 mt-4">
                <Card>
                  <CardHeader>
                    <CardTitle>Constraints</CardTitle>
                  </CardHeader>
                  <CardContent>
                    Constraints content will go here...
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="benchmark">
              <div className="grid gap-4 mt-4">
                <Card>
                  <CardHeader>
                    <CardTitle>Benchmarking</CardTitle>
                  </CardHeader>
                  <CardContent>
                    Benchmarking content will go here...
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}

export default App;
