// LoginPage.jsx
import { useEffect, useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  getGoogleAuthUrl,
  getGoogleRedirectUri,
  parseGoogleCallback,
  saveSession,
  isLoggedIn,
} from '../lib/auth'

export default function LoginPage() {
  const navigate      = useNavigate()
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState('')
  const [status,  setStatus]  = useState('')
  const authAttempted = useRef(false)

  useEffect(() => {
    const hasCode = new URLSearchParams(window.location.search).has('code')
    if (!hasCode && isLoggedIn()) {
      navigate('/dashboard', { replace: true })
    }
  }, [navigate])

  useEffect(() => {
    const code = parseGoogleCallback()
    if (!code) return
    if (authAttempted.current) return
    authAttempted.current = true

    const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'
    const redirectUri = getGoogleRedirectUri()

    setLoading(true)
    setStatus('Authenticating with Google...')

    fetch(`${BASE}/auth/google-callback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        code,
        redirect_uri: redirectUri,
      }),
    })
      .then(async (res) => {
        const payload = await res.json().catch(() => null)

        if (!res.ok) {
          const detail = payload?.detail ? `: ${payload.detail}` : ''
          throw new Error(`Login failed (${res.status})${detail}`)
        }

        return payload
      })
      .then(data => {
        saveSession(data)
        const saved = localStorage.getItem('fie_session')
        if (!saved) throw new Error('Session could not be saved.')
        window.history.replaceState(null, '', window.location.pathname)
        navigate('/dashboard', { replace: true })
      })
      .catch(err => {
        setError(err.message || 'Authentication failed. Please try again.')
        setLoading(false)
      })
  }, [navigate])

  return (
    <>
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(12px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to   { transform: rotate(360deg); }
        }
        @keyframes pulse-dot {
          0%, 100% { opacity: 1; transform: scale(1); }
          50%      { opacity: 0.4; transform: scale(0.75); }
        }
        .fade-in   { animation: fadeIn 0.5s cubic-bezier(0.16,1,0.3,1) both; }
        .d1 { animation-delay: 0.08s; }
        .d2 { animation-delay: 0.16s; }
        .d3 { animation-delay: 0.24s; }
        .d4 { animation-delay: 0.32s; }
        .google-btn {
          transition: all 0.18s ease;
          box-shadow: 0 1px 4px rgba(0,0,0,0.3);
        }
        .google-btn:hover {
          transform: translateY(-1px);
          box-shadow: 0 4px 16px rgba(0,0,0,0.35);
        }
        .google-btn:active {
          transform: scale(0.99);
        }
        .stat-row:not(:last-child) {
          border-bottom: 1px solid var(--border);
        }
      `}</style>

      <div style={{
        minHeight: '100vh',
        display: 'flex',
        background: 'var(--bg-primary)',
        position: 'relative',
        overflow: 'hidden',
      }}>

        {/* Subtle ambient glow */}
        <div style={{
          position: 'fixed', inset: 0, pointerEvents: 'none',
          background: `
            radial-gradient(ellipse at 15% 50%, rgba(0,212,255,0.05) 0%, transparent 60%),
            radial-gradient(ellipse at 85% 80%, rgba(0,255,136,0.03) 0%, transparent 50%)
          `,
        }}/>

        {/* Dot grid */}
        <div style={{
          position: 'fixed', inset: 0, pointerEvents: 'none',
          backgroundImage: `radial-gradient(rgba(0,212,255,0.08) 1px, transparent 1px)`,
          backgroundSize: '32px 32px',
          opacity: 0.5,
        }}/>

        {/* ══ LEFT PANEL ══════════════════════════════════ */}
        <div style={{
          width: '50%',
          minHeight: '100vh',
          borderRight: '1px solid var(--border)',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          padding: '40px 56px',
          position: 'relative',
        }} className="hidden-mobile">

          {/* Top — Logo */}
          <div className="fade-in d1" style={{
            display: 'flex', alignItems: 'center', gap: '10px',
          }}>
            <div style={{
              width: '30px', height: '30px',
              borderRadius: '8px',
              background: 'rgba(0,212,255,0.08)',
              border: '1px solid rgba(0,212,255,0.2)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '9px', fontWeight: 700,
              color: 'var(--accent-cyan)',
              letterSpacing: '0.05em',
            }}>FIE</div>
            <span style={{
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '12px', fontWeight: 500,
              color: 'var(--text-secondary)',
            }}>Failure Intelligence Engine</span>
          </div>

          {/* Middle — Headline */}
          <div style={{ maxWidth: '400px' }}>
            <div className="fade-in d1" style={{
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '10px', fontWeight: 600,
              letterSpacing: '0.2em',
              color: 'var(--accent-cyan)',
              marginBottom: '20px',
            }}>LLM OBSERVABILITY</div>

            <h1 className="fade-in d2" style={{
              fontSize: '36px',
              fontWeight: 700,
              lineHeight: 1.15,
              letterSpacing: '-0.02em',
              color: 'var(--text-primary)',
              marginBottom: '16px',
            }}>
              Monitor your LLMs.<br/>
              <span style={{
                background: 'linear-gradient(90deg, var(--accent-cyan), var(--accent-green))',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
              }}>Catch failures early.</span>
            </h1>

            <p className="fade-in d3" style={{
              fontSize: '14px',
              lineHeight: 1.7,
              color: 'var(--text-muted)',
              marginBottom: '36px',
              maxWidth: '340px',
            }}>
              Real-time detection, diagnosis, and auto-correction
              for production LLM failures.
            </p>

            {/* Stats row */}
            <div className="fade-in d4" style={{
              display: 'flex', gap: '0px',
              borderRadius: '12px',
              border: '1px solid var(--border)',
              overflow: 'hidden',
              background: 'var(--bg-card)',
            }}>
              {[
                { value: '5',    label: 'Fix strategies' },
                { value: '< 1s', label: 'Detection time' },
                { value: '99%',  label: 'Uptime SLA' },
              ].map(({ value, label }, i) => (
                <div key={label} className="stat-row" style={{
                  flex: 1,
                  padding: '16px',
                  borderRight: i < 2 ? '1px solid var(--border)' : 'none',
                  textAlign: 'center',
                }}>
                  <div style={{
                    fontFamily: 'JetBrains Mono, monospace',
                    fontSize: '18px', fontWeight: 700,
                    color: 'var(--accent-cyan)',
                    marginBottom: '4px',
                  }}>{value}</div>
                  <div style={{
                    fontSize: '11px',
                    color: 'var(--text-muted)',
                    fontFamily: 'JetBrains Mono, monospace',
                  }}>{label}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Bottom — version badge */}
          <div className="fade-in d4" style={{
            display: 'flex', alignItems: 'center', gap: '8px',
          }}>
            <div style={{
              width: '6px', height: '6px', borderRadius: '50%',
              background: 'var(--accent-green)',
              animation: 'pulse-dot 2.5s ease-in-out infinite',
            }}/>
            <span style={{
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '11px', color: 'var(--text-muted)',
            }}>v0.1.0 · All systems operational</span>
          </div>
        </div>

        {/* ══ RIGHT PANEL ══════════════════════════════════ */}
        <div style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '40px 32px',
        }}>
          <div style={{ width: '100%', maxWidth: '340px' }}>

            {/* Header */}
            <div className="fade-in d1" style={{ marginBottom: '32px' }}>
              <div style={{
                fontFamily: 'JetBrains Mono, monospace',
                fontSize: '10px', fontWeight: 600,
                letterSpacing: '0.15em',
                color: 'var(--text-muted)',
                marginBottom: '10px',
              }}>SIGN IN</div>
              <h2 style={{
                fontSize: '24px', fontWeight: 700,
                color: 'var(--text-primary)',
                letterSpacing: '-0.01em',
                marginBottom: '6px',
              }}>Welcome back</h2>
              <p style={{
                fontSize: '13px',
                color: 'var(--text-muted)',
                lineHeight: 1.6,
              }}>
                Sign in to access your monitoring dashboard.
              </p>
            </div>

            {/* Loading state */}
            {loading && (
              <div className="fade-in" style={{
                marginBottom: '20px',
                padding: '14px 16px',
                borderRadius: '10px',
                background: 'rgba(0,212,255,0.04)',
                border: '1px solid rgba(0,212,255,0.15)',
                display: 'flex', alignItems: 'center', gap: '10px',
              }}>
                <svg style={{ animation: 'spin 1s linear infinite', flexShrink: 0 }}
                  width="14" height="14" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="10" stroke="rgba(0,212,255,0.2)" strokeWidth="2"/>
                  <path d="M12 2a10 10 0 0 1 10 10"
                    stroke="var(--accent-cyan)" strokeWidth="2" strokeLinecap="round"/>
                </svg>
                <span style={{
                  fontFamily: 'JetBrains Mono, monospace',
                  fontSize: '12px', color: 'var(--accent-cyan)',
                }}>{status}</span>
              </div>
            )}

            {/* Error state */}
            {error && (
              <div className="fade-in" style={{
                marginBottom: '20px',
                padding: '14px 16px',
                borderRadius: '10px',
                background: 'rgba(255,68,102,0.05)',
                border: '1px solid rgba(255,68,102,0.2)',
              }}>
                <div style={{
                  fontFamily: 'JetBrains Mono, monospace',
                  fontSize: '11px', fontWeight: 600,
                  color: 'var(--accent-red)',
                  marginBottom: '2px',
                }}>AUTH ERROR</div>
                <div style={{
                  fontSize: '12px',
                  color: 'var(--text-secondary)',
                }}>{error}</div>
              </div>
            )}

            {/* Main card */}
            {!loading && (
              <div className="fade-in d2">

                {/* Google button */}
                <button
                  className="google-btn"
                  onClick={() => { window.location.href = getGoogleAuthUrl() }}
                  style={{
                    width: '100%',
                    display: 'flex', alignItems: 'center',
                    justifyContent: 'center', gap: '10px',
                    padding: '13px 20px',
                    borderRadius: '10px',
                    background: '#ffffff',
                    border: 'none',
                    cursor: 'pointer',
                    fontFamily: 'Syne, sans-serif',
                    fontSize: '14px', fontWeight: 600,
                    color: '#111827',
                    marginBottom: '16px',
                  }}
                >
                  <svg width="16" height="16" viewBox="0 0 18 18" fill="none">
                    <path fill="#4285F4" d="M16.51 8H8.98v3h4.3c-.18 1-.74 1.48-1.6 2.04v2.01h2.6a7.8 7.8 0 002.38-5.88c0-.57-.05-.66-.15-1.18z"/>
                    <path fill="#34A853" d="M8.98 17c2.16 0 3.97-.72 5.3-1.94l-2.6-2a4.8 4.8 0 01-7.18-2.54H1.83v2.07A8 8 0 008.98 17z"/>
                    <path fill="#FBBC05" d="M4.5 10.52a4.8 4.8 0 010-3.04V5.41H1.83a8 8 0 000 7.18l2.67-2.07z"/>
                    <path fill="#EA4335" d="M8.98 4.18c1.17 0 2.23.4 3.06 1.2l2.3-2.3A8 8 0 001.83 5.4L4.5 7.49a4.77 4.77 0 014.48-3.3z"/>
                  </svg>
                  Continue with Google
                </button>

                {/* Divider */}
                <div style={{
                  display: 'flex', alignItems: 'center', gap: '12px',
                  marginBottom: '16px',
                }}>
                  <div style={{ flex: 1, height: '1px', background: 'var(--border)' }}/>
                  <span style={{
                    fontFamily: 'JetBrains Mono, monospace',
                    fontSize: '10px', color: 'var(--text-muted)',
                  }}>WHAT YOU GET</span>
                  <div style={{ flex: 1, height: '1px', background: 'var(--border)' }}/>
                </div>

                {/* Features list */}
                <div style={{
                  borderRadius: '10px',
                  border: '1px solid var(--border)',
                  overflow: 'hidden',
                  marginBottom: '20px',
                  background: 'var(--bg-card)',
                }}>
                  {[
                    { icon: '⚡', text: 'API key generated automatically' },
                    { icon: '◎', text: 'Private dashboard with live feed' },
                    { icon: '◈', text: 'DiagnosticJury verdicts per failure' },
                    { icon: '◉', text: 'Slack alerts on model degradation' },
                  ].map((item, i, arr) => (
                    <div key={item.text} style={{
                      display: 'flex', alignItems: 'center',
                      gap: '12px', padding: '11px 14px',
                      borderBottom: i < arr.length - 1
                        ? '1px solid var(--border)' : 'none',
                    }}>
                      <span style={{
                        fontFamily: 'JetBrains Mono, monospace',
                        fontSize: '11px',
                        color: 'var(--accent-cyan)',
                        flexShrink: 0,
                      }}>{item.icon}</span>
                      <span style={{
                        fontSize: '12px',
                        color: 'var(--text-secondary)',
                        lineHeight: 1.4,
                      }}>{item.text}</span>
                    </div>
                  ))}
                </div>

                {/* Footer */}
                <p style={{
                  textAlign: 'center',
                  fontFamily: 'JetBrains Mono, monospace',
                  fontSize: '10px',
                  color: 'var(--text-muted)',
                  letterSpacing: '0.08em',
                }}>
                  SECURE · PRIVATE · NO PASSWORD
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  )
}
