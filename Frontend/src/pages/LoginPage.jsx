import { useEffect, useState, useRef } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import {
  getGoogleAuthUrl,
  getGoogleRedirectUri,
  parseGoogleCallback,
  saveSession,
  isLoggedIn,
} from '../lib/auth'

const PERKS = [
  { icon: 'M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z', label: 'Shadow jury detection' },
  { icon: 'M13 2L3 14h9l-1 8 10-12h-9l1-8z',             label: 'Auto-correction engine' },
  { icon: 'M18 20V10M12 20V4M6 20v-6',                   label: 'Live analytics dashboard' },
]

export default function LoginPage() {
  const navigate       = useNavigate()
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState('')
  const [status,  setStatus]  = useState('')
  const authAttempted = useRef(false)

  // redirect if already logged in
  useEffect(() => {
    const hasCode = new URLSearchParams(window.location.search).has('code')
    if (!hasCode && isLoggedIn()) navigate('/dashboard', { replace: true })
  }, [navigate])

  // handle Google OAuth callback
  useEffect(() => {
    const code = parseGoogleCallback()
    if (!code || authAttempted.current) return
    authAttempted.current = true
    const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'
    const redirectUri = getGoogleRedirectUri()
    setLoading(true)
    setStatus('Authenticating...')
    fetch(`${BASE}/auth/google-callback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, redirect_uri: redirectUri }),
    })
      .then(async res => {
        const payload = await res.json().catch(() => null)
        if (!res.ok) throw new Error(`Login failed (${res.status})${payload?.detail ? ': ' + payload.detail : ''}`)
        return payload
      })
      .then(data => {
        saveSession(data)
        if (!localStorage.getItem('fie_session')) throw new Error('Session could not be saved.')
        window.history.replaceState(null, '', window.location.pathname)
        navigate('/dashboard', { replace: true })
      })
      .catch(err => { setError(err.message || 'Authentication failed.'); setLoading(false) })
  }, [navigate])

  return (
    <>
      <style>{`
        @keyframes login-up {
          from { opacity: 0; transform: translateY(28px); }
          to   { opacity: 1; transform: translateY(0);    }
        }
        @keyframes login-fade {
          from { opacity: 0; }
          to   { opacity: 1; }
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        @keyframes pulse-ring {
          0%,100% { transform: translate(-50%,-50%) scale(1);    opacity: 0.35; }
          50%      { transform: translate(-50%,-50%) scale(1.06); opacity: 0.55; }
        }
        @keyframes float-orb {
          0%,100% { transform: translate(-50%,-50%) translate(0,0); }
          33%      { transform: translate(-50%,-50%) translate(18px,-14px); }
          66%      { transform: translate(-50%,-50%) translate(-12px,10px); }
        }
        @keyframes perk-in {
          from { opacity: 0; transform: translateX(-10px); }
          to   { opacity: 1; transform: translateX(0);     }
        }
        @keyframes glow-pulse {
          0%,100% { box-shadow: 0 0 20px rgba(0,212,255,0.08); }
          50%      { box-shadow: 0 0 36px rgba(0,212,255,0.18); }
        }
        @keyframes ring-spin-slow {
          from { transform: translate(-50%,-50%) rotate(0deg); }
          to   { transform: translate(-50%,-50%) rotate(360deg); }
        }
        @keyframes ring-pulse {
          0%,100% { opacity: 0.5; transform: translate(-50%,-50%) scale(1); }
          50%      { opacity: 1;   transform: translate(-50%,-50%) scale(1.04); }
        }
        @keyframes bg-breathe {
          0%,100% { opacity: 0.7; transform: translate(-50%,-50%) scale(1); }
          50%      { opacity: 1;   transform: translate(-50%,-50%) scale(1.08); }
        }

        .login-card {
          animation: login-up 0.7s cubic-bezier(0.16,1,0.3,1) both;
        }
        .login-label  { animation: login-fade 0.5s ease 0.15s both; }
        .login-title  { animation: login-up   0.6s cubic-bezier(0.16,1,0.3,1) 0.2s both; }
        .login-sub    { animation: login-up   0.6s cubic-bezier(0.16,1,0.3,1) 0.3s both; }
        .login-btn    { animation: login-up   0.6s cubic-bezier(0.16,1,0.3,1) 0.42s both; }
        .login-perks  { animation: login-up   0.6s cubic-bezier(0.16,1,0.3,1) 0.54s both; }
        .login-footer { animation: login-fade 0.5s ease 0.72s both; }

        .perk-row { animation: perk-in 0.4s cubic-bezier(0.16,1,0.3,1) both; }

        .google-btn {
          width: 100%;
          display: flex; align-items: center; justify-content: center; gap: 10px;
          padding: 14px 20px;
          border-radius: 12px;
          background: #fff;
          border: none;
          cursor: pointer;
          font-family: 'Syne', sans-serif;
          font-size: 14.5px; font-weight: 600;
          color: #111827;
          transition: transform 0.2s cubic-bezier(0.16,1,0.3,1), box-shadow 0.2s ease;
          box-shadow: 0 2px 8px rgba(0,0,0,0.25);
          position: relative;
          overflow: hidden;
        }
        .google-btn::after {
          content: '';
          position: absolute; inset: 0;
          background: linear-gradient(135deg, rgba(255,255,255,0) 40%, rgba(255,255,255,0.15) 100%);
          opacity: 0;
          transition: opacity 0.2s;
        }
        .google-btn:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 28px rgba(0,0,0,0.35);
        }
        .google-btn:hover::after { opacity: 1; }
        .google-btn:active { transform: translateY(0) scale(0.99); }

        .back-link {
          font-family: 'JetBrains Mono', monospace;
          font-size: 11px; color: #374f65;
          text-decoration: none;
          transition: color 0.2s;
          display: inline-flex; align-items: center; gap: 6px;
        }
        .back-link:hover { color: #00d4ff; }
      `}</style>

      <div style={{
        minHeight: '100vh',
        background: 'radial-gradient(ellipse at bottom, #1e0a2e 0%, #090a0f 55%, #070b12 100%)',
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        position: 'relative', overflow: 'hidden',
        fontFamily: 'Inter, sans-serif',
      }}>

        {/* ── Background layers ── */}

        {/* Dot grid */}
        <div style={{
          position: 'fixed', inset: 0, pointerEvents: 'none',
          backgroundImage: 'radial-gradient(rgba(0,212,255,0.055) 1px, transparent 1px)',
          backgroundSize: '28px 28px',
        }}/>

        {/* Deep center glow — breathes slowly */}
        <div style={{
          position: 'fixed', width: 800, height: 800, borderRadius: '50%',
          top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
          background: 'radial-gradient(circle, rgba(0,212,255,0.09) 0%, rgba(139,92,246,0.06) 40%, transparent 68%)',
          filter: 'blur(60px)',
          animation: 'bg-breathe 7s ease-in-out infinite',
          pointerEvents: 'none',
        }}/>

        {/* Purple drift orb — bottom left */}
        <div style={{
          position: 'fixed', width: 560, height: 560, borderRadius: '50%',
          bottom: '-8%', left: '-8%',
          background: 'radial-gradient(circle, rgba(139,92,246,0.15) 0%, transparent 60%)',
          filter: 'blur(90px)', animation: 'float-orb 26s ease-in-out infinite reverse',
          pointerEvents: 'none',
        }}/>

        {/* Cyan drift orb — top right */}
        <div style={{
          position: 'fixed', width: 480, height: 480, borderRadius: '50%',
          top: '-5%', right: '-5%',
          background: 'radial-gradient(circle, rgba(0,212,255,0.1) 0%, transparent 60%)',
          filter: 'blur(80px)', animation: 'float-orb 20s ease-in-out infinite',
          pointerEvents: 'none',
        }}/>

        {/* ── Concentric glow rings ── */}
        {/* Ring 1 — outermost, slow rotate */}
        <div style={{
          position: 'fixed', width: 780, height: 780, borderRadius: '50%',
          top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
          border: '1px solid rgba(0,212,255,0.06)',
          boxShadow: '0 0 40px rgba(0,212,255,0.03), inset 0 0 40px rgba(0,212,255,0.03)',
          animation: 'ring-spin-slow 40s linear infinite',
          pointerEvents: 'none',
        }}/>

        {/* Ring 2 — dashed, counter-rotate */}
        <div style={{
          position: 'fixed', width: 620, height: 620, borderRadius: '50%',
          top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
          border: '1px dashed rgba(167,139,250,0.1)',
          animation: 'ring-spin-slow 30s linear infinite reverse',
          pointerEvents: 'none',
        }}/>

        {/* Ring 3 — pulse, solid cyan */}
        <div style={{
          position: 'fixed', width: 480, height: 480, borderRadius: '50%',
          top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
          border: '1px solid rgba(0,212,255,0.09)',
          boxShadow: '0 0 24px rgba(0,212,255,0.06), inset 0 0 24px rgba(0,212,255,0.04)',
          animation: 'ring-pulse 5s ease-in-out infinite',
          pointerEvents: 'none',
        }}/>

        {/* Ring 4 — innermost, fast pulse */}
        <div style={{
          position: 'fixed', width: 340, height: 340, borderRadius: '50%',
          top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
          border: '1px solid rgba(167,139,250,0.08)',
          animation: 'ring-pulse 4s ease-in-out 0.8s infinite',
          pointerEvents: 'none',
        }}/>

        {/* ── Top bar ── */}
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, zIndex: 10,
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '20px 32px',
          borderBottom: '1px solid rgba(255,255,255,0.05)',
          backdropFilter: 'blur(20px) saturate(1.4)',
          background: 'rgba(7,11,18,0.85)',
          animation: 'login-fade 0.4s ease both',
        }}>
          <Link to="/" className="back-link">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2"><path d="M19 12H5M12 5l-7 7 7 7"/></svg>
            Back
          </Link>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{
              width: '28px', height: '28px', borderRadius: '7px',
              background: 'rgba(0,212,255,0.08)', border: '1px solid rgba(0,212,255,0.2)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontFamily: 'JetBrains Mono, monospace', fontSize: '8px', fontWeight: 800,
              color: '#00d4ff', letterSpacing: '0.04em',
            }}>FIE</div>
            <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', color: '#6e90b0', fontWeight: 500 }}>
              Failure Intelligence Engine
            </span>
          </div>
        </div>

        {/* ── Main card ── */}
        <div className="login-card" style={{
          width: '100%', maxWidth: '400px',
          margin: '0 20px',
          padding: '40px 36px',
          borderRadius: '20px',
          background: 'rgba(11,16,25,0.92)',
          border: '1px solid rgba(0,212,255,0.12)',
          boxShadow: '0 40px 100px rgba(0,0,0,0.7), 0 0 0 1px rgba(0,212,255,0.06), 0 0 80px rgba(0,212,255,0.04) inset',
          backdropFilter: 'blur(24px)',
          overflow: 'visible',
          position: 'relative', zIndex: 2,
          animation: 'login-up 0.65s cubic-bezier(0.16,1,0.3,1) both, glow-pulse 5s ease-in-out 1s infinite',
        }}>

          {/* Top accent line */}
          <div style={{
            position: 'absolute', top: 0, left: '20%', right: '20%', height: '1px',
            background: 'linear-gradient(90deg, transparent, rgba(0,212,255,0.7), rgba(167,139,250,0.4), transparent)',
            borderRadius: '0 0 4px 4px',
          }}/>

          {/* FIE badge */}
          <div className="login-label" style={{ textAlign: 'center', marginBottom: '28px' }}>
            <div style={{
              display: 'inline-flex', alignItems: 'center', gap: '7px',
              padding: '5px 14px', borderRadius: '20px',
              background: 'rgba(0,212,255,0.06)', border: '1px solid rgba(0,212,255,0.18)',
              fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', fontWeight: 700,
              color: '#00d4ff', letterSpacing: '0.18em',
            }}>
              <div style={{
                width: 5, height: 5, borderRadius: '50%',
                background: '#00d4ff',
                boxShadow: '0 0 8px #00d4ff',
                animation: 'glow-pulse 2s ease-in-out infinite',
              }}/>
              RUNTIME SECURITY
            </div>
          </div>

          {/* Headline */}
          <h1 className="login-title" style={{
            fontFamily: 'Inter, sans-serif',
            fontSize: '27px', fontWeight: 800,
            letterSpacing: '-0.03em', lineHeight: 'normal',
            color: '#f0f6ff', marginBottom: '10px', textAlign: 'center',
            overflow: 'visible',
          }}>
            Sign in to FIE
          </h1>

          <p className="login-sub" style={{
            fontSize: '14px', color: '#6e90b0',
            lineHeight: 1.6, textAlign: 'center',
            margin: '0 auto 28px', maxWidth: '280px',
          }}>
            Access your dashboard, SDK key, and live detection feed.
          </p>

          {/* Loading state */}
          {loading && (
            <div style={{
              marginBottom: '20px', padding: '14px 16px', borderRadius: '10px',
              background: 'rgba(0,212,255,0.04)', border: '1px solid rgba(0,212,255,0.15)',
              display: 'flex', alignItems: 'center', gap: '10px',
              animation: 'login-fade 0.3s ease both',
            }}>
              <svg style={{ animation: 'spin 0.9s linear infinite', flexShrink: 0 }} width="14" height="14" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="rgba(0,212,255,0.2)" strokeWidth="2"/>
                <path d="M12 2a10 10 0 0 1 10 10" stroke="#00d4ff" strokeWidth="2" strokeLinecap="round"/>
              </svg>
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', color: '#00d4ff' }}>{status}</span>
            </div>
          )}

          {/* Error state */}
          {error && (
            <div style={{
              marginBottom: '20px', padding: '12px 16px', borderRadius: '10px',
              background: 'rgba(255,68,102,0.05)', border: '1px solid rgba(255,68,102,0.2)',
              animation: 'login-fade 0.3s ease both',
            }}>
              <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', fontWeight: 700, color: '#ff4466', marginBottom: '3px', letterSpacing: '0.1em' }}>AUTH ERROR</div>
              <div style={{ fontSize: '12px', color: '#8da8c4' }}>{error}</div>
            </div>
          )}

          {/* Google button */}
          {!loading && (
            <div className="login-btn">
              <button className="google-btn" onClick={() => { window.location.href = getGoogleAuthUrl() }}>
                <svg width="17" height="17" viewBox="0 0 18 18" fill="none">
                  <path fill="#4285F4" d="M16.51 8H8.98v3h4.3c-.18 1-.74 1.48-1.6 2.04v2.01h2.6a7.8 7.8 0 002.38-5.88c0-.57-.05-.66-.15-1.18z"/>
                  <path fill="#34A853" d="M8.98 17c2.16 0 3.97-.72 5.3-1.94l-2.6-2a4.8 4.8 0 01-7.18-2.54H1.83v2.07A8 8 0 008.98 17z"/>
                  <path fill="#FBBC05" d="M4.5 10.52a4.8 4.8 0 010-3.04V5.41H1.83a8 8 0 000 7.18l2.67-2.07z"/>
                  <path fill="#EA4335" d="M8.98 4.18c1.17 0 2.23.4 3.06 1.2l2.3-2.3A8 8 0 001.83 5.4L4.5 7.49a4.77 4.77 0 014.48-3.3z"/>
                </svg>
                Continue with Google
              </button>
            </div>
          )}

          {/* Perks row */}
          {!loading && (
            <div className="login-perks" style={{ marginTop: '28px' }}>
              {/* Divider */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '18px' }}>
                <div style={{ flex: 1, height: '1px', background: 'rgba(255,255,255,0.06)' }}/>
                <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '9px', color: '#253545', letterSpacing: '0.14em' }}>UNLOCKS</span>
                <div style={{ flex: 1, height: '1px', background: 'rgba(255,255,255,0.06)' }}/>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {PERKS.map((p, i) => (
                  <div key={p.label} className="perk-row" style={{
                    display: 'flex', alignItems: 'center', gap: '11px',
                    animationDelay: `${0.62 + i * 0.08}s`,
                  }}>
                    <div style={{
                      width: 28, height: 28, borderRadius: '8px', flexShrink: 0,
                      background: 'rgba(0,212,255,0.05)', border: '1px solid rgba(0,212,255,0.1)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      color: '#00d4ff',
                    }}>
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d={p.icon}/>
                      </svg>
                    </div>
                    <span style={{ fontSize: '12.5px', color: '#6e90b0' }}>{p.label}</span>
                    <svg style={{ marginLeft: 'auto', opacity: 0.3 }} width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" strokeWidth="2.5"><path d="M20 6L9 17l-5-5"/></svg>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Footer */}
          <div className="login-footer" style={{
            marginTop: '28px', paddingTop: '20px',
            borderTop: '1px solid rgba(255,255,255,0.05)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '16px',
          }}>
            {['Secure', 'Private', 'No password'].map((t, i) => (
              <span key={t} style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                {i > 0 && <span style={{ color: '#1a2535' }}>·</span>}
                <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', color: '#374f65', letterSpacing: '0.06em' }}>{t}</span>
              </span>
            ))}
          </div>
        </div>

        {/* ── Bottom status ── */}
        <div className="login-footer" style={{
          position: 'fixed', bottom: '24px',
          display: 'flex', alignItems: 'center', gap: '7px',
        }}>
          <div style={{
            width: 5, height: 5, borderRadius: '50%',
            background: '#00ff88', boxShadow: '0 0 8px #00ff88',
            animation: 'glow-pulse 2.5s ease-in-out infinite',
          }}/>
          <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', color: '#374f65' }}>
            v3.0.0 · All systems operational
          </span>
        </div>

      </div>
    </>
  )
}
