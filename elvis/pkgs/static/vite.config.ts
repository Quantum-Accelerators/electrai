import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

const allowedHosts = process.env.VITE_ALLOWED_HOSTS?.split(',') ?? []

export default defineConfig({
  plugins: [react()],
  base: process.env.BASE_PATH || '/',

  define: {
    'process.env.AWS_STATIC_SSO_PROXY_URL': JSON.stringify(''),
  },

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
