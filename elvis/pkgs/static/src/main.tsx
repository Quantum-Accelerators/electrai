import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { HotkeysProvider } from 'use-kbd'
import './index.css'
import App from './App.tsx'

const queryClient = new QueryClient()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <HotkeysProvider config={{ storageKey: 'elvis-kbd' }}>
        <App />
      </HotkeysProvider>
    </QueryClientProvider>
  </StrictMode>,
)
