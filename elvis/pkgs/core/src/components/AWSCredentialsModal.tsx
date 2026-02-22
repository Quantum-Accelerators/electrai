import { useState, useCallback } from 'react'
import type { ReactNode } from 'react'
import type { AWSCredentials } from './aws-credentials-types.ts'

type Mode = 'paste' | 'fields' | 'sso'

interface AWSCredentialsModalProps {
  open: boolean
  onClose: () => void
  onSave: (creds: AWSCredentials) => void
  currentCreds: AWSCredentials | null
  ssoContent?: ReactNode
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
        region: json.Region ?? json.region,
      }
    }
  } catch { /* not JSON */ }

  // Try shell export format
  const envMatch = {
    accessKeyId: trimmed.match(/AWS_ACCESS_KEY_ID[=\s]+["']?(\S+?)["']?\s*$/m)?.[1],
    secretAccessKey: trimmed.match(/AWS_SECRET_ACCESS_KEY[=\s]+["']?(\S+?)["']?\s*$/m)?.[1],
    sessionToken: trimmed.match(/AWS_SESSION_TOKEN[=\s]+["']?(\S+?)["']?\s*$/m)?.[1],
    region: trimmed.match(/AWS_(?:DEFAULT_)?REGION[=\s]+["']?(\S+?)["']?\s*$/m)?.[1],
  }
  if (envMatch.accessKeyId && envMatch.secretAccessKey) {
    return {
      accessKeyId: envMatch.accessKeyId,
      secretAccessKey: envMatch.secretAccessKey,
      sessionToken: envMatch.sessionToken,
      region: envMatch.region,
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

const inputStyle = {
  width: '100%',
  padding: '6px 8px',
  background: '#2a2a3e',
  border: '1px solid #444',
  borderRadius: 4,
  color: '#eee',
  fontSize: 12,
  fontFamily: 'monospace' as const,
  outline: 'none',
}

const labelStyle = {
  display: 'block' as const,
  fontSize: 11,
  color: '#888',
  marginBottom: 3,
  marginTop: 10,
}

export function AWSCredentialsModal({ open, onClose, onSave, currentCreds, ssoContent }: AWSCredentialsModalProps) {
  const availableModes: Mode[] = ssoContent ? ['sso', 'paste', 'fields'] : ['paste', 'fields']
  const [mode, setMode] = useState<Mode>(availableModes[0])

  // Paste mode state
  const [text, setText] = useState('')
  const [parsed, setParsed] = useState<AWSCredentials | null>(null)
  const [pasteError, setPasteError] = useState<string | null>(null)

  // Fields mode state
  const [accessKeyId, setAccessKeyId] = useState('')
  const [secretAccessKey, setSecretAccessKey] = useState('')
  const [sessionToken, setSessionToken] = useState('')
  const [region, setRegion] = useState('us-east-1')
  const [expiration, setExpiration] = useState('')

  const handlePaste = useCallback((value: string) => {
    setText(value)
    setPasteError(null)
    const creds = parseCredentials(value)
    if (creds) {
      setParsed(creds)
    } else {
      setParsed(null)
      if (value.trim()) setPasteError('Could not parse credentials')
    }
  }, [])

  const handleSavePaste = useCallback(() => {
    if (parsed) {
      onSave(parsed)
      setText('')
      setParsed(null)
      onClose()
    }
  }, [parsed, onSave, onClose])

  const handleSaveFields = useCallback(() => {
    if (accessKeyId && secretAccessKey) {
      const creds: AWSCredentials = {
        accessKeyId,
        secretAccessKey,
        sessionToken: sessionToken || undefined,
        region: region || undefined,
        expiration: expiration || undefined,
      }
      onSave(creds)
      onClose()
    }
  }, [accessKeyId, secretAccessKey, sessionToken, region, expiration, onSave, onClose])

  const fieldsValid = accessKeyId.length > 0 && secretAccessKey.length > 0

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
        width: 460,
        maxWidth: '90vw',
        maxHeight: '80vh',
        overflowY: 'auto',
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
            {currentCreds.region && <div>Region: {currentCreds.region}</div>}
            {currentCreds.expiration && (
              <div>{formatExpiration(currentCreds.expiration)}</div>
            )}
          </div>
        )}

        {/* Mode tabs */}
        <div style={{ display: 'flex', gap: 4, marginBottom: 16 }}>
          {availableModes.map(m => (
            <button
              key={m}
              onClick={() => setMode(m)}
              style={{
                flex: 1,
                padding: '6px 0',
                background: mode === m ? '#3a3a5e' : 'transparent',
                border: `1px solid ${mode === m ? '#666' : '#444'}`,
                borderRadius: 4,
                color: mode === m ? '#eee' : '#888',
                fontSize: 12,
                fontWeight: mode === m ? 600 : 400,
                cursor: 'pointer',
              }}
            >
              {m === 'sso' ? 'SSO' : m === 'paste' ? 'Paste' : 'Fields'}
            </button>
          ))}
        </div>

        {/* Paste mode */}
        {mode === 'paste' && (
          <>
            <div style={{ fontSize: 12, color: '#888', marginBottom: 8 }}>
              Paste output of <code style={{ color: '#aaa' }}>aws configure export-credentials | pbcopy</code>
            </div>

            <textarea
              value={text}
              onChange={e => handlePaste(e.target.value)}
              placeholder={'{"AccessKeyId": "...", "SecretAccessKey": "...", ...}'}
              rows={5}
              style={{
                ...inputStyle,
                resize: 'vertical' as const,
              }}
            />

            {pasteError && (
              <div style={{ color: '#ff4444', fontSize: 12, marginTop: 6 }}>{pasteError}</div>
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
                {parsed.region && ` \u00b7 ${parsed.region}`}
                {parsed.expiration && ` \u00b7 ${formatExpiration(parsed.expiration)}`}
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 16 }}>
              <button onClick={onClose} style={cancelBtnStyle}>Cancel</button>
              <button
                onClick={handleSavePaste}
                disabled={!parsed}
                style={saveBtnStyle(!parsed)}
              >
                Save
              </button>
            </div>
          </>
        )}

        {/* Fields mode */}
        {mode === 'fields' && (
          <>
            <label style={labelStyle}>
              Access Key ID *
              <input
                value={accessKeyId}
                onChange={e => setAccessKeyId(e.target.value)}
                placeholder="AKIA..."
                style={inputStyle}
              />
            </label>

            <label style={labelStyle}>
              Secret Access Key *
              <input
                type="password"
                value={secretAccessKey}
                onChange={e => setSecretAccessKey(e.target.value)}
                placeholder="wJalr..."
                style={inputStyle}
              />
            </label>

            <label style={labelStyle}>
              Session Token
              <input
                value={sessionToken}
                onChange={e => setSessionToken(e.target.value)}
                placeholder="(optional, for temporary credentials)"
                style={inputStyle}
              />
            </label>

            <div style={{ display: 'flex', gap: 12 }}>
              <label style={{ ...labelStyle, flex: 1 }}>
                Region
                <input
                  value={region}
                  onChange={e => setRegion(e.target.value)}
                  placeholder="us-east-1"
                  style={inputStyle}
                />
              </label>

              <label style={{ ...labelStyle, flex: 1 }}>
                Expiration
                <input
                  value={expiration}
                  onChange={e => setExpiration(e.target.value)}
                  placeholder="(optional ISO date)"
                  style={inputStyle}
                />
              </label>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 16 }}>
              <button onClick={onClose} style={cancelBtnStyle}>Cancel</button>
              <button
                onClick={handleSaveFields}
                disabled={!fieldsValid}
                style={saveBtnStyle(!fieldsValid)}
              >
                Save
              </button>
            </div>
          </>
        )}

        {/* SSO mode */}
        {mode === 'sso' && ssoContent}
      </div>
    </div>
  )
}

const cancelBtnStyle = {
  padding: '6px 16px',
  background: 'transparent',
  border: '1px solid #555',
  borderRadius: 4,
  color: '#aaa',
  cursor: 'pointer',
  fontSize: 13,
}

const saveBtnStyle = (disabled: boolean) => ({
  padding: '6px 16px',
  background: disabled ? '#333' : '#4a9eff',
  border: 'none',
  borderRadius: 4,
  color: '#fff',
  cursor: disabled ? ('default' as const) : ('pointer' as const),
  fontSize: 13,
  opacity: disabled ? 0.5 : 1,
})

export type { AWSCredentials }
