import { Component } from 'react'
import type { ReactNode, ErrorInfo } from 'react'

interface ErrorBoundaryProps {
  children: ReactNode
  /** When this key changes, a caught error is automatically cleared. */
  resetKey?: string | number
  /** Label shown in the fallback header (e.g. "Viewer", "Controls"). */
  label?: string
}

interface ErrorBoundaryState {
  error: Error | null
  prevResetKey?: string | number
}

/**
 * Catches render/lifecycle errors in child tree and shows a recoverable fallback.
 * Automatically resets when `resetKey` changes (e.g. new material or iso level).
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { error }
  }

  static getDerivedStateFromProps(
    props: ErrorBoundaryProps,
    state: ErrorBoundaryState,
  ): Partial<ErrorBoundaryState> | null {
    if (state.error && props.resetKey !== state.prevResetKey) {
      return { error: null, prevResetKey: props.resetKey }
    }
    if (props.resetKey !== state.prevResetKey) {
      return { prevResetKey: props.resetKey }
    }
    return null
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`[ErrorBoundary${this.props.label ? `: ${this.props.label}` : ''}]`, error, info.componentStack)
  }

  render() {
    if (this.state.error) {
      const { label } = this.props
      const { error } = this.state
      return (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          padding: 24,
          gap: 12,
          background: '#1a1a2e',
          color: '#ccc',
        }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#ff6b6b' }}>
            {label ? `${label} crashed` : 'Something went wrong'}
          </div>
          <div style={{
            fontSize: 12,
            color: '#999',
            maxWidth: 400,
            textAlign: 'center',
            fontFamily: 'monospace',
            wordBreak: 'break-word',
          }}>
            {error.message}
          </div>
          <button
            onClick={() => this.setState({ error: null })}
            style={{
              marginTop: 4,
              padding: '6px 16px',
              background: '#4a9eff',
              border: 'none',
              borderRadius: 4,
              color: '#fff',
              fontSize: 13,
              cursor: 'pointer',
            }}
          >
            Retry
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
