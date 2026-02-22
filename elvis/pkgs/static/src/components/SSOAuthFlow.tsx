import { useState, useCallback, useEffect, useRef } from 'react'
import { SSOAuth } from 'aws-static-sso'
import type { DeviceAuth, SSOAccount, SSORole, SSOCredentials } from 'aws-static-sso'
import type { AWSCredentials } from '@elvis/core'

const PROXY_URL = 'https://aws-static-sso.ryan-0dc.workers.dev'
const SSO_SETTINGS_KEY = 'elvis-sso-settings'

type FlowState =
  | { step: 'idle' }
  | { step: 'authorizing'; deviceAuth: DeviceAuth }
  | { step: 'authenticated'; accounts: SSOAccount[] }
  | { step: 'roles'; accountId: string; accountName: string; roles: SSORole[] }
  | { step: 'ready'; creds: SSOCredentials }
  | { step: 'error'; message: string }

interface SSOSettings {
  startUrl: string
  region: string
}

function loadSSOSettings(): SSOSettings {
  try {
    const raw = localStorage.getItem(SSO_SETTINGS_KEY)
    if (raw) return JSON.parse(raw)
  } catch { /* ignore */ }
  return { startUrl: '', region: 'us-east-1' }
}

function saveSSOSettings(settings: SSOSettings) {
  localStorage.setItem(SSO_SETTINGS_KEY, JSON.stringify(settings))
}

interface SSOAuthFlowProps {
  onSave: (creds: AWSCredentials) => void
  onClose: () => void
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

const btnStyle = (primary: boolean, disabled = false) => ({
  padding: '6px 16px',
  background: disabled ? '#333' : primary ? '#4a9eff' : 'transparent',
  border: primary ? 'none' : '1px solid #555',
  borderRadius: 4,
  color: primary ? '#fff' : '#aaa',
  cursor: disabled ? ('default' as const) : ('pointer' as const),
  fontSize: 13,
  opacity: disabled ? 0.5 : 1,
})

export function SSOAuthFlow({ onSave, onClose }: SSOAuthFlowProps) {
  const [startUrl, setStartUrl] = useState(() => loadSSOSettings().startUrl)
  const [region, setRegion] = useState(() => loadSSOSettings().region)
  const [flowState, setFlowState] = useState<FlowState>({ step: 'idle' })
  const [loading, setLoading] = useState(false)
  const authRef = useRef<SSOAuth | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  // Cleanup on unmount
  useEffect(() => {
    return () => { abortRef.current?.abort() }
  }, [])

  const handleStart = useCallback(async () => {
    if (!startUrl.trim()) return
    saveSSOSettings({ startUrl, region })
    setLoading(true)
    setFlowState({ step: 'idle' })

    try {
      const auth = new SSOAuth({ startUrl, region, proxyUrl: PROXY_URL })
      authRef.current = auth

      const deviceAuth = await auth.startAuth()
      setFlowState({ step: 'authorizing', deviceAuth })

      // Open verification page
      window.open(deviceAuth.verificationUriComplete, '_blank')

      // Poll for token
      const abort = new AbortController()
      abortRef.current = abort
      await auth.pollForToken(deviceAuth, abort.signal)

      // Token acquired — list accounts
      const session = auth.getSession()
      if (!session) throw new Error('No session after token acquisition')
      const accounts = await session.listAccounts()
      setFlowState({ step: 'authenticated', accounts })
    } catch (e) {
      if ((e as Error).name !== 'AbortError') {
        setFlowState({ step: 'error', message: (e as Error).message })
      }
    } finally {
      setLoading(false)
    }
  }, [startUrl, region])

  const handleSelectAccount = useCallback(async (accountId: string, accountName: string) => {
    setLoading(true)
    try {
      const session = authRef.current?.getSession()
      if (!session) throw new Error('No active session')
      const roles = await session.listAccountRoles(accountId)
      setFlowState({ step: 'roles', accountId, accountName, roles })
    } catch (e) {
      setFlowState({ step: 'error', message: (e as Error).message })
    } finally {
      setLoading(false)
    }
  }, [])

  const handleSelectRole = useCallback(async (accountId: string, roleName: string) => {
    setLoading(true)
    try {
      const session = authRef.current?.getSession()
      if (!session) throw new Error('No active session')
      const creds = await session.getCredentials({ accountId, roleName })
      setFlowState({ step: 'ready', creds })

      // Convert to Elvis AWSCredentials format and save
      onSave({
        accessKeyId: creds.accessKeyId,
        secretAccessKey: creds.secretAccessKey,
        sessionToken: creds.sessionToken,
        expiration: new Date(creds.expiration).toISOString(),
        region,
      })
      onClose()
    } catch (e) {
      setFlowState({ step: 'error', message: (e as Error).message })
    } finally {
      setLoading(false)
    }
  }, [region, onSave, onClose])

  return (
    <div>
      {/* Start URL + Region (always shown) */}
      <label style={labelStyle}>
        SSO Start URL
        <input
          value={startUrl}
          onChange={e => setStartUrl(e.target.value)}
          placeholder="https://myorg.awsapps.com/start"
          style={inputStyle}
          disabled={loading}
        />
      </label>

      <label style={{ ...labelStyle, marginBottom: 12 }}>
        SSO Region
        <input
          value={region}
          onChange={e => setRegion(e.target.value)}
          placeholder="us-east-1"
          style={inputStyle}
          disabled={loading}
        />
      </label>

      {/* Idle state */}
      {flowState.step === 'idle' && (
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 16 }}>
          <button onClick={onClose} style={btnStyle(false)}>Cancel</button>
          <button
            onClick={handleStart}
            disabled={!startUrl.trim() || loading}
            style={btnStyle(true, !startUrl.trim())}
          >
            {loading ? 'Starting...' : 'Sign in with SSO'}
          </button>
        </div>
      )}

      {/* Authorizing: show device code */}
      {flowState.step === 'authorizing' && (
        <div style={{
          padding: '12px 16px',
          background: '#2a2a3e',
          borderRadius: 6,
          textAlign: 'center',
        }}>
          <div style={{ fontSize: 12, color: '#888', marginBottom: 8 }}>
            Enter this code in the browser window that just opened:
          </div>
          <div style={{
            fontSize: 28,
            fontWeight: 700,
            fontFamily: 'monospace',
            color: '#4a9eff',
            letterSpacing: 4,
            marginBottom: 8,
          }}>
            {flowState.deviceAuth.userCode}
          </div>
          <div style={{ fontSize: 11, color: '#666', marginBottom: 12 }}>
            Waiting for authorization...
          </div>
          <button
            onClick={() => window.open(flowState.deviceAuth.verificationUriComplete, '_blank')}
            style={{ ...btnStyle(false), fontSize: 11 }}
          >
            Re-open verification page
          </button>
        </div>
      )}

      {/* Authenticated: account selection */}
      {flowState.step === 'authenticated' && (
        <div>
          <div style={{ fontSize: 12, color: '#888', marginBottom: 8 }}>
            Select an AWS account:
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {flowState.accounts.map(account => (
              <button
                key={account.accountId}
                onClick={() => handleSelectAccount(account.accountId, account.accountName)}
                disabled={loading}
                style={{
                  padding: '8px 12px',
                  background: '#2a2a3e',
                  border: '1px solid #444',
                  borderRadius: 4,
                  color: '#eee',
                  fontSize: 12,
                  cursor: 'pointer',
                  textAlign: 'left',
                }}
              >
                <div style={{ fontWeight: 600 }}>{account.accountName}</div>
                <div style={{ color: '#888', fontSize: 11 }}>{account.accountId}</div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Roles: role selection */}
      {flowState.step === 'roles' && (
        <div>
          <div style={{ fontSize: 12, color: '#888', marginBottom: 8 }}>
            {flowState.accountName} \u2014 select a role:
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {flowState.roles.map(role => (
              <button
                key={role.roleName}
                onClick={() => handleSelectRole(flowState.accountId, role.roleName)}
                disabled={loading}
                style={{
                  padding: '8px 12px',
                  background: '#2a2a3e',
                  border: '1px solid #444',
                  borderRadius: 4,
                  color: '#eee',
                  fontSize: 12,
                  cursor: 'pointer',
                  textAlign: 'left',
                }}
              >
                {role.roleName}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Error */}
      {flowState.step === 'error' && (
        <div>
          <div style={{
            padding: '8px 12px',
            background: 'rgba(255, 68, 68, 0.1)',
            borderRadius: 4,
            fontSize: 12,
            color: '#ff4444',
            marginBottom: 12,
          }}>
            {flowState.message}
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
            <button onClick={onClose} style={btnStyle(false)}>Cancel</button>
            <button onClick={() => setFlowState({ step: 'idle' })} style={btnStyle(true)}>
              Try again
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
