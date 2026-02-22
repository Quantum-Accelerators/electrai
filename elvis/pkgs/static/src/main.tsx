import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { HotkeysProvider } from 'use-kbd'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <HotkeysProvider config={{ storageKey: 'elvis-kbd' }}>
      <App />
    </HotkeysProvider>
  </StrictMode>,
)
