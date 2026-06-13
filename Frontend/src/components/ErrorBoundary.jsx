// ── ErrorBoundary ─────────────────────────────────────────────────────────────
//  Stops a render-time exception in one subtree from blanking the whole app.
//  Used to isolate the decorative WebGL scene (a 3D/GPU failure must never take
//  down the marketing page) and to guard the page as a whole.
//
//  Pass `fallback={null}` to silently drop a failed subtree (e.g. the 3D scene),
//  or omit it to show a minimal recoverable message.
// ─────────────────────────────────────────────────────────────────────────────

import { Component } from 'react'

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    // surface it in the console for diagnosis without crashing the UI
    console.error(`[ErrorBoundary${this.props.label ? ' · ' + this.props.label : ''}]`, error, info)
  }

  render() {
    if (this.state.error) {
      if ('fallback' in this.props) return this.props.fallback
      const dev = import.meta.env && import.meta.env.DEV
      return (
        <div style={{
          padding: '40px 24px', textAlign: 'center',
          fontFamily: 'Inter, system-ui, sans-serif', color: '#8da8c4',
        }}>
          <p style={{ fontSize: '14px' }}>Something went wrong rendering this section.</p>
          {dev && (
            <pre style={{
              margin: '16px auto 0', maxWidth: '760px', textAlign: 'left',
              whiteSpace: 'pre-wrap', wordBreak: 'break-word',
              fontFamily: 'monospace', fontSize: '12px', color: '#ff8095',
              background: 'rgba(255,68,102,0.06)', border: '1px solid rgba(255,68,102,0.25)',
              borderRadius: '8px', padding: '14px 16px',
            }}>
              {String(this.state.error && (this.state.error.stack || this.state.error.message || this.state.error))}
            </pre>
          )}
          <button
            type="button"
            onClick={() => this.setState({ error: null })}
            style={{
              marginTop: '14px', padding: '8px 18px', borderRadius: '8px',
              border: '1px solid rgba(255,255,255,0.15)', background: 'rgba(255,255,255,0.04)',
              color: '#dde8f5', cursor: 'pointer', fontSize: '13px',
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
