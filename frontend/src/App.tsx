import React from 'react';
import PortfolioDashboard from '@/components/dashboard/PortfolioDashboard';

function App() {
  console.log('Rendering App');
  return (
    <div className="min-h-screen bg-background">
      <main className="container mx-auto py-4">
        <PortfolioDashboard />
      </main>
    </div>
  );
}

export default App;
