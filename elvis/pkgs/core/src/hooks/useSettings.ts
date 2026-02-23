import { useState, useCallback } from 'react'

export interface ElvisSettings {
  cacheInOPFS: boolean
  maxUploadSizeMB: number
  animationDuration: number
}

const STORAGE_KEY = 'elvis-settings'

const DEFAULTS: ElvisSettings = {
  cacheInOPFS: true,
  maxUploadSizeMB: 20,
  animationDuration: 1.0,
}

function load(): ElvisSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return { ...DEFAULTS }
    return { ...DEFAULTS, ...JSON.parse(raw) }
  } catch {
    return { ...DEFAULTS }
  }
}

function save(settings: ElvisSettings) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings))
}

export function useSettings() {
  const [settings, setSettingsState] = useState<ElvisSettings>(load)

  const update = useCallback((patch: Partial<ElvisSettings>) => {
    setSettingsState(prev => {
      const next = { ...prev, ...patch }
      save(next)
      return next
    })
  }, [])

  return { settings, update } as const
}
