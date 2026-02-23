import type { ElvisSettings } from '../hooks/useSettings.ts'

interface SettingsProps {
  settings: ElvisSettings
  onUpdate: (patch: Partial<ElvisSettings>) => void
  showCacheToggle?: boolean
}

export function Settings({ settings, onUpdate, showCacheToggle = true }: SettingsProps) {
  return (
    <div style={{ padding: '8px 16px', borderBottom: '1px solid #333' }}>
      <div style={{ fontSize: 13, color: '#aaa', fontWeight: 600, marginBottom: 8 }}>
        Settings
      </div>
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
      <label style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        fontSize: 13,
        color: '#ccc',
        marginTop: 6,
      }}>
        Animation: {settings.animationDuration.toFixed(1)}s
        <input
          type="range"
          min={0.1}
          max={2.0}
          step={0.1}
          value={settings.animationDuration}
          onChange={e => onUpdate({ animationDuration: parseFloat(e.target.value) })}
          style={{ flex: 1, accentColor: '#4a9eff' }}
        />
      </label>
    </div>
  )
}
