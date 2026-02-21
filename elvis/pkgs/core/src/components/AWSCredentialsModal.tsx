import { useState, useCallback } from 'react'
import type { AWSCredentials } from './aws-credentials-types.ts'

interface AWSCredentialsModalProps {
  open: boolean
  onClose: () => void
  onSave: (creds: AWSCredentials) => void
  currentCreds: AWSCredentials | null
}

function parseCredentials(input: string): AWSCredentials | null {
  const trimmed = input.trim()

  // Try JSON format (e.g. from `aws configure export-credentials`)
  try {
    const json = JSON.parse(trimmed)
    if (json.AccessKeyId || json.accessKeyId) {
      return {
        accessKeyId: json.AccessKeyId ?? json.accessKeyId,
        secretAccessKey: json.SecretAccessKey ?? json.secretAccessKey,
        sessionToken: json.SessionToken ?? json.sessionToken,
        expiration: json.Expiration ?? json.expiration,
      }
    }
  } catch { /* not JSON */ }

  // Try shell export format
  const envMatch = {
    accessKeyId: trimmed.match(/AWS_ACCESS_KEY_ID[=\s]+["']?(\S+?)["']?\s*$/m)?.[1],
    secretAccessKey: trimmed.match(/AWS_SECRET_ACCESS_KEY[=\s]+["']?(\S+?)["']?\s*$/m)?.[1],
    sessionToken: trimmed.match(/AWS_SESSION_TOKEN[=\s]+["']?(\S+?)["']?\s*$/m)?.[1],
  }
  if (envMatch.accessKeyId && envMatch.secretAccessKey) {
    return {
      accessKeyId: envMatch.accessKeyId,
      secretAccessKey: envMatch.secretAccessKey,
      sessionToken: envMatch.sessionToken,
    }
  }

  // Try plain lines: key, secret, optional token
  const lines = trimmed.split('\n').map(l => l.trim()).filter(Boolean)
  if (lines.length >= 2) {
    return {
      accessKeyId: lines[0],
      secretAccessKey: lines[1],
      sessionToken: lines[2],
    }
  }

  return null
}

function maskKey(key: string): string {
  if (key.length <= 8) return '****'
  return key.slice(0, 4) + '****' + key.slice(-4)
}

function formatExpiration(exp: string | undefined): string | null {
  if (!exp) return null
  const d = new Date(exp)
  const remaining = d.getTime() - Date.now()
  if (remaining <= 0) return 'expired'
  const hours = Math.floor(remaining / (1000 * 60 * 60))
  const minutes = Math.floor((remaining % (1000 * 60 * 60)) / (1000 * 60))
  return `${hours}h ${minutes}m remaining`
}

export function AWSCredentialsModal({ open, onClose, onSave, currentCreds }: AWSCredentialsModalProps) {
  const [text, setText] = useState('')
  const [parsed, setParsed] = useState<AWSCredentials | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handlePaste = useCallback((value: string) => {
    setText(value)
    setError(null)
    const creds = parseCredentials(value)
    if (creds) {
      setParsed(creds)
    } else {
      setParsed(null)
      if (value.trim()) setError('Could not parse credentials')
    }
  }, [])

  const handleSave = useCallback(() => {
    if (parsed) {
      onSave(parsed)
      setText('')
      setParsed(null)
      onClose()
    }
  }, [parsed, onSave, onClose])

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
    }} onClick={onClose}>
      <div style={{
        background: '#1e1e30',
        borderRadius: 8,
        padding: 24,
        width: 440,
        maxWidth: '90vw',
      }} onClick={e => e.stopPropagation()}>
        <div style={{ fontSize: 16, fontWeight: 600, color: '#eee', marginBottom: 12 }}>
          AWS Credentials
        </div>

        {currentCreds && (
          <div style={{
            padding: '8px 12px',
            background: '#2a2a3e',
            borderRadius: 4,
            marginBottom: 12,
            fontSize: 12,
            color: '#aaa',
          }}>
            <div>Key: {maskKey(currentCreds.accessKeyId)}</div>
            {currentCreds.expiration && (
              <div>{formatExpiration(currentCreds.expiration)}</div>
            )}
          </div>
        )}

        <div style={{ fontSize: 12, color: '#888', marginBottom: 8 }}>
          Paste output of <code style={{ color: '#aaa' }}>aws configure export-credentials | pbcopy</code>
        </div>

        <textarea
          value={text}
          onChange={e => handlePaste(e.target.value)}
          placeholder={'{"AccessKeyId": "...", "SecretAccessKey": "...", ...}'}
          rows={5}
          style={{
            width: '100%',
            padding: 8,
            background: '#2a2a3e',
            border: '1px solid #444',
            borderRadius: 4,
            color: '#eee',
            fontSize: 12,
            fontFamily: 'monospace',
            resize: 'vertical',
            outline: 'none',
          }}
        />

        {error && (
          <div style={{ color: '#ff4444', fontSize: 12, marginTop: 6 }}>{error}</div>
        )}

        {parsed && (
          <div style={{
            marginTop: 8,
            padding: '6px 10px',
            background: 'rgba(74, 158, 255, 0.1)',
            borderRadius: 4,
            fontSize: 12,
            color: '#8bc',
          }}>
            Parsed: {maskKey(parsed.accessKeyId)}
            {parsed.expiration && ` \u00b7 ${formatExpiration(parsed.expiration)}`}
          </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 16 }}>
          <button
            onClick={onClose}
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
            onClick={handleSave}
            disabled={!parsed}
            style={{
              padding: '6px 16px',
              background: parsed ? '#4a9eff' : '#333',
              border: 'none',
              borderRadius: 4,
              color: '#fff',
              cursor: parsed ? 'pointer' : 'default',
              fontSize: 13,
              opacity: parsed ? 1 : 0.5,
            }}
          >
            Save
          </button>
        </div>
      </div>
    </div>
  )
}

export type { AWSCredentials }
