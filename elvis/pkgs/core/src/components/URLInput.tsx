import { useState, useCallback } from 'react'

interface URLInputProps {
  onSubmit: (url: string) => void
  loading?: boolean
}

export function URLInput({ onSubmit, loading }: URLInputProps) {
  const [value, setValue] = useState('')

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault()
    const url = value.trim()
    if (!url) return
    onSubmit(url)
  }, [value, onSubmit])

  return (
    <form onSubmit={handleSubmit} style={{ padding: '8px 16px' }}>
      <div style={{ fontSize: 13, color: '#aaa', fontWeight: 600, marginBottom: 6 }}>
        Load from URL
      </div>
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
    </form>
  )
}
