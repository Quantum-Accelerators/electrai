interface SizeConfirmModalProps {
  open: boolean
  filename: string
  fileSizeMB: number
  maxSizeMB: number
  storageUsage: { count: number; totalBytes: number } | null
  onConfirm: (cache: boolean) => void
  onCancel: () => void
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function SizeConfirmModal({
  open,
  filename,
  fileSizeMB,
  maxSizeMB,
  storageUsage,
  onConfirm,
  onCancel,
}: SizeConfirmModalProps) {
  if (!open) return null

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(0,0,0,0.6)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
    }} onClick={onCancel}>
      <div style={{
        background: '#1e1e30',
        borderRadius: 8,
        padding: 24,
        width: 380,
        maxWidth: '90vw',
      }} onClick={e => e.stopPropagation()}>
        <div style={{ fontSize: 16, fontWeight: 600, color: '#eee', marginBottom: 12 }}>
          Large file
        </div>
        <div style={{ fontSize: 13, color: '#ccc', lineHeight: 1.5 }}>
          <strong>{filename}</strong> is {fileSizeMB.toFixed(1)} MB,
          exceeding the {maxSizeMB} MB cache limit.
        </div>
        {storageUsage && (
          <div style={{ fontSize: 12, color: '#888', marginTop: 8 }}>
            Current cache: {storageUsage.count} files, {formatBytes(storageUsage.totalBytes)}
          </div>
        )}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 20 }}>
          <button
            onClick={onCancel}
            style={{
              padding: '6px 16px',
              background: 'transparent',
              border: '1px solid #555',
              borderRadius: 4,
              color: '#aaa',
              cursor: 'pointer',
              fontSize: 13,
            }}
          >
            Cancel
          </button>
          <button
            onClick={() => onConfirm(false)}
            style={{
              padding: '6px 16px',
              background: '#555',
              border: 'none',
              borderRadius: 4,
              color: '#eee',
              cursor: 'pointer',
              fontSize: 13,
            }}
          >
            Load (skip cache)
          </button>
          <button
            onClick={() => onConfirm(true)}
            style={{
              padding: '6px 16px',
              background: '#4a9eff',
              border: 'none',
              borderRadius: 4,
              color: '#fff',
              cursor: 'pointer',
              fontSize: 13,
            }}
          >
            Load + cache
          </button>
        </div>
      </div>
    </div>
  )
}
