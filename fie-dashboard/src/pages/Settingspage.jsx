import { useState } from 'react'
import { getSession, clearSession } from '../lib/auth'
import { useNavigate } from 'react-router-dom'

const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false)
  const handle = () => {
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  return (
    <button onClick={handle} style={{
      padding: '6px 12px', borderRadius: '6px', border: '1px solid var(--border)',
      background: copied ? 'rgba(0,255,136,0.1)' : 'var(--bg-hover)',
      color: copied ? 'var(--accent-green)' : 'var(--text-muted)',
      fontFamily: 'JetBrains Mono, monospace', fontSize: '11px',
      cursor: 'pointer', transition: 'all 0.15s ease', flexShrink: 0,
    }}>
      {copied ? '✓ Copied' : 'Copy'}
    </button>
  )
}

function Section({ title, children }) {
  return (
    <div style={{ marginBottom: '32px' }}>
      <div style={{
        fontFamily: 'JetBrains Mono, monospace', fontSize: '10px',
        fontWeight: 600, letterSpacing: '0.15em', color: 'var(--text-muted)',
        marginBottom: '12px', textTransform: 'uppercase',
      }}>{title}</div>
      <div style={{
        borderRadius: '12px', border: '1px solid var(--border)',
        background: 'var(--bg-card)', overflow: 'hidden',
      }}>{children}</div>
    </div>
  )
}

function Row({ label, value, action, mono = false, last = false }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '14px 18px',
      borderBottom: last ? 'none' : '1px solid var(--border)',
      gap: '16px',
    }}>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '3px' }}>{label}</div>
        <div style={{
          fontSize: '13px', color: 'var(--text-primary)',
          fontFamily: mono ? 'JetBrains Mono, monospace' : 'inherit',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}>{value}</div>
      </div>
      {action}
    </div>
  )
}

export default function SettingsPage() {
  const navigate  = useNavigate()
  const session   = getSession()
  const [apiKey, setApiKey]   = useState(session?.api_key || '')
  const [regen, setRegen]     = useState(false)
  const [regenDone, setRegenDone] = useState(false)
  const [error, setError]     = useState('')

  const name      = session?.name || 'User'
  const email     = session?.email || ''
  const plan      = session?.plan || 'free'
  const isAdmin   = session?.is_admin || false
  const tenantId  = session?.tenant_id || ''
  const initials  = name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase()

  const handleRegenerate = async () => {
    setRegen(true)
    setError('')
    try {
      const res = await fetch(`${BASE}/auth/regenerate-key`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session?.token}`,
        },
      })
      if (!res.ok) throw new Error('Failed to regenerate')
      const data = await res.json()
      setApiKey(data.api_key)
      const saved = JSON.parse(localStorage.getItem('fie_session') || '{}')
      saved.api_key = data.api_key
      localStorage.setItem('fie_session', JSON.stringify(saved))
      setRegenDone(true)
      setTimeout(() => setRegenDone(false), 3000)
    } catch (e) {
      setError(e.message)
    } finally {
      setRegen(false)
    }
  }

  const handleLogout = () => {
    clearSession()
    navigate('/login', { replace: true })
  }

  return (
    <>
      <style>{`
        @keyframes kpiIn {
          from { opacity: 0; transform: translateY(14px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to   { transform: rotate(360deg); }
        }
      `}</style>

      <div style={{ flex: 1, padding: '28px 32px', overflowY: 'auto', maxWidth: '800px' }}>

        {/* Header */}
        <div style={{ marginBottom: '28px', animation: 'kpiIn 0.5s ease both' }}>
          <h1 style={{ fontSize: '22px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px' }}>
            Settings
          </h1>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
            Manage your account, API key, and integration.
          </p>
        </div>

        {/* Profile */}
        <div style={{ animation: 'kpiIn 0.5s ease 0.05s both', opacity: 0 }}>
          <Section title="Profile">
            <div style={{ padding: '18px', display: 'flex', alignItems: 'center', gap: '14px', borderBottom: '1px solid var(--border)' }}>
              <div style={{
                width: '44px', height: '44px', borderRadius: '50%', flexShrink: 0,
                background: 'linear-gradient(135deg, var(--accent-cyan), var(--accent-green))',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontFamily: 'JetBrains Mono, monospace', fontSize: '14px', fontWeight: 700, color: '#0d1117',
              }}>{initials}</div>
              <div>
                <div style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-primary)' }}>{name}</div>
                <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' }}>{email}</div>
              </div>
              <div style={{ marginLeft: 'auto' }}>
                <span style={{
                  padding: '3px 10px', borderRadius: '20px',
                  fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', fontWeight: 700,
                  background: isAdmin ? 'rgba(248,81,73,0.12)' : 'rgba(88,166,255,0.1)',
                  color: isAdmin ? 'var(--accent-red)' : 'var(--accent-cyan)',
                  border: `1px solid ${isAdmin ? 'rgba(248,81,73,0.25)' : 'rgba(88,166,255,0.2)'}`,
                }}>{isAdmin ? 'ADMIN' : plan.toUpperCase()}</span>
              </div>
            </div>
            <Row label="Tenant ID" value={tenantId} mono last />
          </Section>
        </div>

        {/* API Key */}
        <div style={{ animation: 'kpiIn 0.5s ease 0.1s both', opacity: 0 }}>
          <Section title="API Key">
            <div style={{ padding: '16px 18px', borderBottom: '1px solid var(--border)' }}>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px' }}>Your API Key</div>
              <div style={{
                display: 'flex', alignItems: 'center', gap: '10px',
                padding: '10px 14px', borderRadius: '8px',
                background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border)',
              }}>
                <div style={{
                  flex: 1, fontFamily: 'JetBrains Mono, monospace',
                  fontSize: '13px', color: 'var(--accent-green)',
                  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                }}>{apiKey}</div>
                <CopyButton text={apiKey} />
              </div>
              {regenDone && (
                <div style={{ marginTop: '8px', fontSize: '12px', color: 'var(--accent-green)', fontFamily: 'JetBrains Mono, monospace' }}>
                  ✓ New key generated — update your @monitor decorator
                </div>
              )}
              {error && (
                <div style={{ marginTop: '8px', fontSize: '12px', color: 'var(--accent-red)', fontFamily: 'JetBrains Mono, monospace' }}>
                  {error}
                </div>
              )}
            </div>
            <div style={{ padding: '14px 18px' }}>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '10px' }}>
                Generate a new key — old key stops working immediately.
              </div>
              <button
                onClick={handleRegenerate}
                disabled={regen}
                style={{
                  display: 'flex', alignItems: 'center', gap: '8px',
                  padding: '8px 16px', borderRadius: '8px',
                  border: '1px solid var(--border)',
                  background: 'var(--bg-hover)', color: 'var(--text-secondary)',
                  fontFamily: 'JetBrains Mono, monospace', fontSize: '12px',
                  cursor: regen ? 'not-allowed' : 'pointer',
                  opacity: regen ? 0.6 : 1, transition: 'all 0.15s ease',
                }}
              >
                {regen && (
                  <svg style={{ animation: 'spin 1s linear infinite' }} width="12" height="12" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="10" stroke="rgba(255,255,255,0.2)" strokeWidth="2"/>
                    <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                )}
                {regen ? 'Regenerating...' : 'Regenerate API Key'}
              </button>
            </div>
          </Section>
        </div>

        {/* SDK Quick Start */}
        <div style={{ animation: 'kpiIn 0.5s ease 0.15s both', opacity: 0 }}>
          <Section title="Quick Start — Connect Your LLM">
            <div style={{ padding: '16px 18px', borderBottom: '1px solid var(--border)' }}>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px' }}>Step 1 — Install SDK</div>
              <div style={{
                display: 'flex', alignItems: 'center', gap: '10px',
                padding: '10px 14px', borderRadius: '8px',
                background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border)',
              }}>
                <code style={{ flex: 1, fontFamily: 'JetBrains Mono, monospace', fontSize: '13px', color: 'var(--accent-cyan)' }}>
                  pip install fie-sdk
                </code>
                <CopyButton text="pip install fie-sdk" />
              </div>
            </div>
            <div style={{ padding: '16px 18px' }}>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px' }}>Step 2 — Wrap your LLM</div>
              <div style={{ position: 'relative' }}>
                <pre style={{
                  margin: 0, padding: '14px', borderRadius: '8px',
                  background: 'rgba(0,0,0,0.35)', border: '1px solid var(--border)',
                  fontFamily: 'JetBrains Mono, monospace', fontSize: '12px',
                  lineHeight: 1.7, overflowX: 'auto', color: 'var(--text-secondary)',
                }}>
{`from fie import monitor

@monitor(
    api_key = "${apiKey}",
    fie_url = "${import.meta.env.VITE_API_URL?.replace('/api/v1','') || 'http://localhost:8000'}",
    mode    = "correct",
)
def call_your_llm(prompt: str) -> str:
    return your_llm(prompt)`}
                </pre>
                <div style={{ position: 'absolute', top: '10px', right: '10px' }}>
                  <CopyButton text={`from fie import monitor\n\n@monitor(\n    api_key = "${apiKey}",\n    fie_url = "${import.meta.env.VITE_API_URL?.replace('/api/v1','') || 'http://localhost:8000'}",\n    mode    = "correct",\n)\ndef call_your_llm(prompt: str) -> str:\n    return your_llm(prompt)`} />
                </div>
              </div>
            </div>
          </Section>
        </div>

        {/* Modes */}
        <div style={{ animation: 'kpiIn 0.5s ease 0.2s both', opacity: 0 }}>
          <Section title="Monitor Modes">
            {[
              { mode: 'monitor', color: 'var(--accent-cyan)',  title: 'Background monitoring', desc: 'User gets answer instantly. FIE checks in background. Best for speed.' },
              { mode: 'correct', color: 'var(--accent-green)', title: 'Real-time correction',   desc: 'FIE fixes answer before returning. Best for accuracy. ~2-3s overhead.', last: true },
            ].map(({ mode, color, title, desc, last }) => (
              <div key={mode} style={{
                padding: '14px 18px',
                borderBottom: last ? 'none' : '1px solid var(--border)',
                borderLeft: `3px solid ${color}`,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                  <code style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', color }}>{`mode="${mode}"`}</code>
                  <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>— {title}</span>
                </div>
                <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{desc}</div>
              </div>
            ))}
          </Section>
        </div>

        {/* Danger zone */}
        <div style={{ animation: 'kpiIn 0.5s ease 0.25s both', opacity: 0 }}>
          <Section title="Account">
            <div style={{ padding: '14px 18px' }}>
              <button
                onClick={handleLogout}
                style={{
                  padding: '8px 16px', borderRadius: '8px',
                  border: '1px solid rgba(255,68,102,0.3)',
                  background: 'rgba(255,68,102,0.06)', color: 'var(--accent-red)',
                  fontFamily: 'JetBrains Mono, monospace', fontSize: '12px',
                  cursor: 'pointer', transition: 'all 0.15s ease',
                }}
                onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,68,102,0.12)'}
                onMouseLeave={e => e.currentTarget.style.background = 'rgba(255,68,102,0.06)'}
              >
                Sign out
              </button>
            </div>
          </Section>
        </div>

      </div>
    </>
  )
}