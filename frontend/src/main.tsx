import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';

// Debug logging
console.log('Script started');

try {
  console.log('Looking for root element');
  const rootElement = document.getElementById('root');
  console.log('Root element found:', !!rootElement);

  if (!rootElement) {
    throw new Error('Root element not found');
  }

  console.log('Creating React root');
  const root = createRoot(rootElement);
  
  console.log('Rendering app');
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
  console.log('Render complete');
} catch (error) {
  console.error('Error during initialization:', error);
}
