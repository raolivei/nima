import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true
  },
  preview: {
    allowedHosts: [
      'nima.eldertree.local',
      'pihole.eldertree.local',
      'grafana.eldertree.local',
      'prometheus.eldertree.local',
      'vault.eldertree.local',
      'flux-ui.eldertree.local',
      'localhost',
    ],
  },
})

