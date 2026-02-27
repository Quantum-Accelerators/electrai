import { useState, useCallback } from 'react'

interface URLInputProps {
  onSubmit: (url: string) => void
  loading?: boolean
}

export function URLInput({ onSubmit, loading }: URLInputProps) {
  const [value, setValue] = useState('')
  const [collapsed, setCollapsed] = useState(() => {
    return sessionStorage.getItem('elvis-url-input-collapsed') === 'true'
  })
  const toggleCollapsed = useCallback(() => {
    setCollapsed(prev => {
      const next = !prev
      sessionStorage.setItem('elvis-url-input-collapsed', String(next))
      return next
    })
  }, [])

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault()
    const url = value.trim()
    if (!url) return
    onSubmit(url)
  }, [value, onSubmit])

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
        <span>Load from URL</span>
        <span style={{ fontSize: 10 }}>{collapsed ? '\u25b6' : '\u25bc'}</span>
      </button>
      {!collapsed && <form onSubmit={handleSubmit} style={{ padding: '0 16px 8px' }}>
      <div style={{ display: 'flex', gap: 6 }}>
        <input
          type="text"
          value={value}
          onChange={e => setValue(e.target.value)}
          placeholder="https://... or s3://..."
          disabled={loading}
          style={{
            flex: 1,
            padding: '6px 8px',
            background: '#2a2a3e',
            border: '1px solid #444',
            borderRadius: 4,
            color: '#eee',
            fontSize: 12,
            outline: 'none',
          }}
        />
        <button
          type="submit"
          disabled={loading || !value.trim()}
          style={{
            padding: '6px 12px',
            background: loading ? '#333' : '#4a9eff',
            border: 'none',
            borderRadius: 4,
            color: '#fff',
            fontSize: 12,
            cursor: loading ? 'default' : 'pointer',
            opacity: loading || !value.trim() ? 0.5 : 1,
          }}
        >
          {loading ? 'Loading\u2026' : 'Load'}
        </button>
      </div>
      </form>}
    </div>
  )
}
