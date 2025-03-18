import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    strictPort: true, // Fail if port is in use
    host: '0.0.0.0', // Listen on all available network interfaces
    open: false, // Don't auto-open browser
    cors: true, // Enable CORS
    hmr: {
      overlay: true
    },
    watch: {
      usePolling: true // Use polling for file changes
    }
  },
});

