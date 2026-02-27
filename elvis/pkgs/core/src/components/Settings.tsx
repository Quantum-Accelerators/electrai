import { useState, useCallback } from 'react'
import type { ElvisSettings } from '../hooks/useSettings.ts'

interface SettingsProps {
  settings: ElvisSettings
  onUpdate: (patch: Partial<ElvisSettings>) => void
  showCacheToggle?: boolean
  lineWidth?: number
  onLineWidthChange?: (v: number) => void
}

export function Settings({ settings, onUpdate, showCacheToggle = true, lineWidth, onLineWidthChange }: SettingsProps) {
  const [collapsed, setCollapsed] = useState(() => {
    return sessionStorage.getItem('elvis-settings-collapsed') === 'true'
  })
  const toggleCollapsed = useCallback(() => {
    setCollapsed(prev => {
      const next = !prev
      sessionStorage.setItem('elvis-settings-collapsed', String(next))
      return next
    })
  }, [])

  return (
    <div style={{ borderBottom: '1px solid #333' }}>
      <button
        onClick={toggleCollapsed}
        style={{
          width: '100%',
          padding: '8px 16px',
          background: 'transparent',
          border: 'none',
          color: '#aaa',
          fontSize: 13,
          fontWeight: 600,
          cursor: 'pointer',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <span>Settings</span>
        <span style={{ fontSize: 10 }}>{collapsed ? '\u25b6' : '\u25bc'}</span>
      </button>
      {!collapsed && <div style={{ padding: '0 16px 8px' }}>
      {showCacheToggle && (
        <label style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          fontSize: 13,
          color: '#ccc',
          cursor: 'pointer',
          marginBottom: 6,
        }}>
          <input
            type="checkbox"
            checked={settings.cacheInOPFS}
            onChange={e => onUpdate({ cacheInOPFS: e.target.checked })}
            style={{ accentColor: '#4a9eff' }}
          />
          Cache files in browser
        </label>
      )}
      <label style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        fontSize: 13,
        color: '#ccc',
      }}>
        Prompt for files over
        <input
          type="number"
          value={settings.maxUploadSizeMB}
          onChange={e => onUpdate({ maxUploadSizeMB: Math.max(1, parseInt(e.target.value) || 1) })}
          min={1}
          step={1}
          style={{
            width: 50,
            padding: '2px 4px',
            background: '#2a2a3e',
            border: '1px solid #444',
            borderRadius: 4,
            color: '#eee',
            fontSize: 12,
            textAlign: 'right',
          }}
        />
        MB
      </label>
      {onLineWidthChange && (
        <label style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          fontSize: 13,
          color: '#ccc',
          marginTop: 6,
        }}>
          Line width: {(lineWidth ?? 1).toFixed(1)}×
          <input
            type="range"
            min={0.5}
            max={5.0}
            step={0.1}
            value={lineWidth ?? 1}
            onChange={e => onLineWidthChange(parseFloat(e.target.value))}
            style={{ flex: 1, accentColor: '#4a9eff' }}
          />
        </label>
      )}
      </div>}
    </div>
  )
}
