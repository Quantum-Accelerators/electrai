import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

const allowedHosts = process.env.VITE_ALLOWED_HOSTS?.split(',') ?? []

export default defineConfig({
  plugins: [react()],
  base: '/',
  resolve: {
    alias: {
      '@elvis/core': path.resolve(__dirname, '../core/src'),
    },
  },
  server: {
    port: 3150,
    host: true,
    allowedHosts,
  },
})
