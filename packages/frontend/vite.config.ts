import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const usePolling = process.env.CHOKIDAR_USEPOLLING === 'true'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5183,
    host: true,
    strictPort: true,
    watch: usePolling ? { usePolling: true, interval: 300 } : undefined,
  },
})
