import { useState, useEffect, useRef } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { isLoggedIn } from '../lib/auth'

// ── Data ──────────────────────────────────────────────────────────────────────

const STATS = [
  { value: 547,  suffix: '+', label: 'PyPI installs' },
  { value: 0,    suffix: '%', label: 'False positive rate' },
  { value: 96,   suffix: '%', label: 'GCG attack recall' },
  { value: 9,    suffix: '',  label: 'Detection layers' },
]

const FEATURES = [
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
      </svg>
    ),
    title: 'Hallucination Detection',
    desc: 'A shadow jury of 3 models cross-checks every answer. Failure Signal Vector captures agreement, entropy, and confidence — fed into an XGBoost classifier.',
    color: '#00d4ff',
  },
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
        <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
      </svg>
    ),
    title: 'Adversarial Attack Protection',
    desc: 'Nine detection layers catch prompt injection, jailbreaks, token smuggling, many-shot attacks, model extraction, and Crescendo multi-turn attacks.',
    color: '#ff4466',
  },
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
        <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>
      </svg>
    ),
    title: 'Auto-Correction',
    desc: 'Detects failure and automatically applies the right fix — jury consensus replacement, ground truth cache, prompt sanitization, or human escalation.',
    color: '#00ff88',
  },
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
        <rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/>
      </svg>
    ),
    title: 'Works Offline',
    desc: 'Local mode runs entirely on your machine with zero network calls. No API key, no server. Add one decorator and protection is immediate.',
    color: '#a78bfa',
  },
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
      </svg>
    ),
    title: 'LangGraph Pipeline',
    desc: 'Detection runs as a stateful LangGraph StateGraph — guard → signal extraction → DiagnosticJury — each stage a typed node with conditional routing.',
    color: '#fb923c',
  },
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
      </svg>
    ),
    title: 'Model Drift Monitoring',
    desc: 'EMA-based trend tracking alerts you when failure rate rises — before your users notice. Catch degradation from stale training data or bad updates.',
    color: '#00d4ff',
  },
]

const STEPS = [
  { n: '01', title: 'Install the SDK',        desc: 'pip install fie-sdk — one command, zero extra dependencies.' },
  { n: '02', title: 'Add the decorator',      desc: 'Wrap your LLM call with @monitor(mode="local") for instant offline detection.' },
  { n: '03', title: 'Connect for full power', desc: 'Sign in to unlock the shadow jury, XGBoost classifier, auto-correction, and live analytics.' },
]

const BENCHMARKS = [
  { method: 'POET rule-based (baseline)', recall: '56.4%', fpr: '38.7%', auc: '—',     highlight: false },
  { method: 'FIE XGBoost v3',            recall: '63.6%', fpr: '38.6%', auc: '0.677', highlight: true  },
]

const ATTACK_BENCHMARKS = [
  { method: 'GCG suffix attacks',     detection: '96.0%', fpr: '2.0%' },
  { method: 'JBC persona jailbreaks', detection: '52.0%', fpr: '2.0%' },
  { method: 'Direct injection',       detection: '95.0%', fpr: '2.0%' },
]

// syntax-coloured code lines
const CODE_LINES = [
  { tokens: [{ t: 'from ', c: '#79c0ff' }, { t: 'fie ', c: '#79c0ff' }, { t: 'import ', c: '#ff7b72' }, { t: 'monitor', c: '#e6edf3' }] },
  { tokens: [] },
  { tokens: [{ t: '@monitor', c: '#d2a8ff' }, { t: '(mode=', c: '#e6edf3' }, { t: '"local"', c: '#a5d6ff' }, { t: ')   ', c: '#e6edf3' }, { t: '# zero setup', c: '#8b949e' }] },
  { tokens: [{ t: 'def ', c: '#ff7b72' }, { t: 'ask_ai', c: '#d2a8ff' }, { t: '(prompt: ', c: '#e6edf3' }, { t: 'str', c: '#79c0ff' }, { t: ') -> ', c: '#e6edf3' }, { t: 'str', c: '#79c0ff' }, { t: ':', c: '#e6edf3' }] },
  { tokens: [{ t: '    return ', c: '#ff7b72' }, { t: 'your_llm(prompt)', c: '#e6edf3' }] },
  { tokens: [] },
  { tokens: [{ t: '# Hallucinations + attacks caught automatically', c: '#8b949e' }] },
  { tokens: [{ t: 'response ', c: '#e6edf3' }, { t: '= ', c: '#ff7b72' }, { t: 'ask_ai(', c: '#e6edf3' }, { t: '"Who won the 2022 World Cup?"', c: '#a5d6ff' }, { t: ')', c: '#e6edf3' }] },
]

// ── Hooks ─────────────────────────────────────────────────────────────────────

function useScrollReveal() {
  const ref = useRef(null)
  const [visible, setVisible] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const obs = new IntersectionObserver(([e]) => { if (e.isIntersecting) { setVisible(true); obs.disconnect() } }, { threshold: 0.12 })
    obs.observe(el)
    return () => obs.disconnect()
  }, [])
  return [ref, visible]
}

function useCounter(target, visible, duration = 1400) {
  const [val, setVal] = useState(0)
  const raf = useRef(null)
  useEffect(() => {
    if (!visible || target === 0) { setVal(target); return }
    let start = null
    const step = ts => {
      if (!start) start = ts
      const p = Math.min((ts - start) / duration, 1)
      const eased = 1 - Math.pow(1 - p, 3)
      setVal(Math.round(eased * target))
      if (p < 1) raf.current = requestAnimationFrame(step)
    }
    raf.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf.current)
  }, [target, visible, duration])
  return val
}

// ── Sub-components ────────────────────────────────────────────────────────────

function StatCard({ value, suffix, label, visible }) {
  const count = useCounter(value, visible)
  return (
    <div style={{ padding: '32px 24px', textAlign: 'center', position: 'relative' }}>
      <div style={{
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: 'clamp(28px, 3.5vw, 38px)', fontWeight: 800,
        color: 'var(--text-primary)', letterSpacing: '-0.03em',
        marginBottom: '6px',
        background: 'linear-gradient(135deg, #fff 30%, rgba(0,212,255,0.7) 100%)',
        WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
      }}>
        {count}{suffix}
      </div>
      <div style={{ fontSize: '12px', color: 'var(--text-muted)', letterSpacing: '0.02em' }}>{label}</div>
    </div>
  )
}

function FeatureCard({ icon, title, desc, color, delay }) {
  const [hov, setHov] = useState(false)
  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        padding: '26px 28px', borderRadius: '14px',
        background: hov ? `rgba(${hexToRgb(color)},0.04)` : 'var(--bg-card)',
        border: `1px solid ${hov ? color + '40' : 'var(--border)'}`,
        transition: 'all 0.25s ease',
        transform: hov ? 'translateY(-3px)' : 'none',
        boxShadow: hov ? `0 12px 40px rgba(${hexToRgb(color)},0.1)` : 'none',
        cursor: 'default',
      }}
    >
      <div style={{
        width: '40px', height: '40px', borderRadius: '10px',
        background: `rgba(${hexToRgb(color)},0.1)`,
        border: `1px solid rgba(${hexToRgb(color)},0.2)`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        color, marginBottom: '16px',
        transition: 'all 0.25s ease',
        boxShadow: hov ? `0 0 16px rgba(${hexToRgb(color)},0.25)` : 'none',
      }}>{icon}</div>
      <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '10px', letterSpacing: '-0.01em' }}>{title}</div>
      <div style={{ fontSize: '13px', lineHeight: 1.7, color: 'var(--text-muted)' }}>{desc}</div>
    </div>
  )
}

function hexToRgb(hex) {
  const h = hex.replace('#', '')
  const n = parseInt(h.length === 3 ? h.split('').map(x => x + x).join('') : h, 16)
  return `${(n >> 16) & 255},${(n >> 8) & 255},${n & 255}`
}

function RevealSection({ children, style }) {
  const [ref, visible] = useScrollReveal()
  return (
    <div ref={ref} style={{
      opacity: visible ? 1 : 0,
      transform: visible ? 'none' : 'translateY(24px)',
      transition: 'opacity 0.65s cubic-bezier(0.16,1,0.3,1), transform 0.65s cubic-bezier(0.16,1,0.3,1)',
      ...style,
    }}>
      {children}
    </div>
  )
}

// ── Main ──────────────────────────────────────────────────────────────────────

export default function LandingPage() {
  const navigate = useNavigate()
  const loggedIn = isLoggedIn()
  const [copied, setCopied] = useState(false)
  const [statsRef, statsVisible] = useScrollReveal()

  useEffect(() => {
    // If Google OAuth redirected here instead of /login, forward everything to /login
    const params = new URLSearchParams(window.location.search)
    if (params.has('code')) {
      navigate('/login' + window.location.search, { replace: true })
      return
    }
    if (loggedIn) navigate('/dashboard', { replace: true })
  }, [loggedIn, navigate])

  const copy = () => {
    navigator.clipboard.writeText('pip install fie-sdk')
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <>
      <style>{`
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(18px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn {
          from { opacity: 0; } to { opacity: 1; }
        }
        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50%       { transform: translateY(-8px); }
        }
        @keyframes glow-ring {
          0%, 100% { box-shadow: 0 0 0 0 rgba(0,212,255,0); }
          50%       { box-shadow: 0 0 0 6px rgba(0,212,255,0.12); }
        }
        @keyframes shimmer-line {
          0%   { background-position: -200% center; }
          100% { background-position: 200% center; }
        }
        @keyframes pulse-slow {
          0%, 100% { opacity: 0.6; transform: scale(1); }
          50%       { opacity: 1;   transform: scale(1.04); }
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to   { transform: rotate(360deg); }
        }

        .fu  { animation: fadeUp 0.6s cubic-bezier(0.16,1,0.3,1) both; }
        .fi  { animation: fadeIn 0.5s ease both; }
        .d1  { animation-delay: 0.06s; }
        .d2  { animation-delay: 0.14s; }
        .d3  { animation-delay: 0.22s; }
        .d4  { animation-delay: 0.30s; }
        .d5  { animation-delay: 0.40s; }
        .d6  { animation-delay: 0.50s; }

        .nav-link {
          font-family: 'Inter', sans-serif; font-size: 13px;
          color: var(--text-muted); text-decoration: none;
          transition: color 0.15s;
        }
        .nav-link:hover { color: var(--text-primary); }

        .cta-primary {
          display: inline-flex; align-items: center; gap: 7px;
          padding: 11px 22px; border-radius: 9px;
          background: var(--accent-cyan); color: #000;
          font-size: 13px; font-weight: 700;
          font-family: 'Inter', sans-serif;
          border: none; cursor: pointer; text-decoration: none;
          transition: opacity 0.15s, transform 0.2s, box-shadow 0.2s;
          box-shadow: 0 0 0 0 rgba(0,212,255,0);
          letter-spacing: -0.01em;
        }
        .cta-primary:hover {
          opacity: 0.9;
          transform: translateY(-2px);
          box-shadow: 0 8px 28px rgba(0,212,255,0.28);
        }

        .cta-secondary {
          display: inline-flex; align-items: center; gap: 7px;
          padding: 11px 22px; border-radius: 9px;
          background: transparent; color: var(--text-secondary);
          font-size: 13px; font-weight: 500;
          font-family: 'Inter', sans-serif;
          border: 1px solid var(--border); cursor: pointer; text-decoration: none;
          transition: border-color 0.2s, color 0.2s, transform 0.2s;
        }
        .cta-secondary:hover {
          border-color: rgba(255,255,255,0.3);
          color: var(--text-primary);
          transform: translateY(-2px);
        }

        .section-label {
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px; font-weight: 700;
          letter-spacing: 0.2em; color: var(--accent-cyan);
          text-transform: uppercase; margin-bottom: 16px;
          display: flex; align-items: center; gap: 8px;
        }
        .section-label::before {
          content: ''; display: block;
          width: 20px; height: 1px;
          background: var(--accent-cyan);
        }

        .table-row:not(:last-child) { border-bottom: 1px solid var(--border); }
        .table-row { transition: background 0.15s; }
        .table-row:hover { background: rgba(255,255,255,0.018) !important; }

        .pill {
          display: inline-flex; align-items: center; gap: 6px;
          padding: 4px 12px; border-radius: 20px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 11px; font-weight: 600;
          background: rgba(0,212,255,0.07);
          color: var(--accent-cyan);
          border: 1px solid rgba(0,212,255,0.18);
          letter-spacing: 0.04em;
        }
        .pill-dot {
          width: 5px; height: 5px; border-radius: 50%;
          background: var(--accent-green);
          animation: pulse-slow 2.4s ease-in-out infinite;
        }

        .code-block {
          background: #0d1117;
          border-radius: 14px;
          border: 1px solid rgba(255,255,255,0.08);
          overflow: hidden;
          box-shadow: 0 24px 64px rgba(0,0,0,0.4), 0 0 0 1px rgba(0,212,255,0.06);
          animation: float 5s ease-in-out infinite;
        }

        .step-line {
          position: absolute;
          top: 24px; left: calc(50% + 28px);
          width: calc(100% - 56px); height: 1px;
          background: linear-gradient(90deg, var(--border), transparent);
        }

        @media (max-width: 768px) {
          .hide-mobile { display: none !important; }
          .grid-3 { grid-template-columns: 1fr !important; }
          .grid-2 { grid-template-columns: 1fr !important; }
          .grid-4 { grid-template-columns: repeat(2,1fr) !important; }
          .code-split { grid-template-columns: 1fr !important; }
        }
      `}</style>

      <div style={{ minHeight: '100vh', background: 'var(--bg-primary)', color: 'var(--text-primary)', fontFamily: 'Inter, sans-serif', overflowX: 'hidden' }}>

        {/* ── Ambient background ───────────────────────────────────── */}
        <div style={{
          position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 0,
          background: `
            radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,212,255,0.07) 0%, transparent 70%),
            radial-gradient(ellipse 40% 30% at 80% 60%, rgba(0,255,136,0.04) 0%, transparent 60%)
          `,
        }}/>
        <div style={{
          position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 0,
          backgroundImage: 'radial-gradient(rgba(0,212,255,0.055) 1px, transparent 1px)',
          backgroundSize: '40px 40px',
        }}/>

        {/* ── Nav ──────────────────────────────────────────────────── */}
        <nav style={{
          position: 'sticky', top: 0, zIndex: 100,
          borderBottom: '1px solid var(--border)',
          background: 'rgba(7,11,20,0.82)',
          backdropFilter: 'blur(16px)',
        }}>
          <div style={{
            maxWidth: '1100px', margin: '0 auto', padding: '0 28px',
            height: '58px', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          }}>
            {/* Logo */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div style={{
                width: '30px', height: '30px', borderRadius: '8px',
                background: 'rgba(0,212,255,0.1)', border: '1px solid rgba(0,212,255,0.25)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontFamily: 'JetBrains Mono, monospace', fontSize: '9px', fontWeight: 800,
                color: 'var(--accent-cyan)', letterSpacing: '0.05em',
                animation: 'glow-ring 3s ease-in-out infinite',
              }}>FIE</div>
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                Failure Intelligence Engine
              </span>
            </div>
            {/* Links */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '28px' }}>
              <a href="https://github.com/AyushSingh110/Failure_Intelligence_System" target="_blank" rel="noopener noreferrer" className="nav-link hide-mobile">GitHub</a>
              <a href="https://pypi.org/project/fie-sdk/" target="_blank" rel="noopener noreferrer" className="nav-link hide-mobile">PyPI</a>
              {loggedIn
                ? <Link to="/dashboard" className="cta-primary" style={{ padding: '7px 16px', fontSize: '12px' }}>Dashboard →</Link>
                : <Link to="/login"     className="cta-primary" style={{ padding: '7px 16px', fontSize: '12px' }}>Sign in</Link>
              }
            </div>
          </div>
        </nav>

        {/* ── Hero ─────────────────────────────────────────────────── */}
        <section style={{ maxWidth: '1100px', margin: '0 auto', padding: '100px 28px 80px', position: 'relative', zIndex: 1 }}>

          {/* Badge */}
          <div className="fu d1" style={{ marginBottom: '24px' }}>
            <span className="pill">
              <span className="pill-dot"/>
              Open Source · Apache 2.0
            </span>
          </div>

          {/* Headline */}
          <h1 className="fu d2" style={{
            fontSize: 'clamp(36px, 5.5vw, 60px)', fontWeight: 800,
            lineHeight: 1.1, letterSpacing: '-0.035em',
            color: 'var(--text-primary)', maxWidth: '780px', marginBottom: '22px',
          }}>
            Catch LLM failures<br />
            <span style={{
              background: 'linear-gradient(90deg, #00d4ff 0%, #00ff88 55%, #a78bfa 100%)',
              backgroundSize: '200% auto',
              WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
              animation: 'shimmer-line 5s linear infinite',
            }}>before your users do.</span>
          </h1>

          {/* Sub */}
          <p className="fu d3" style={{
            fontSize: '16px', lineHeight: 1.75, color: 'var(--text-muted)',
            maxWidth: '530px', marginBottom: '40px',
          }}>
            Real-time hallucination detection, adversarial attack protection,
            and automatic correction — as a single Python decorator.
          </p>

          {/* CTAs */}
          <div className="fu d4" style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', marginBottom: '44px', alignItems: 'center' }}>
            {loggedIn
              ? <Link to="/dashboard" className="cta-primary">Go to Dashboard →</Link>
              : <Link to="/login"     className="cta-primary">Get started free</Link>
            }
            <a href="https://github.com/AyushSingh110/Failure_Intelligence_System" target="_blank" rel="noopener noreferrer" className="cta-secondary">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
              View on GitHub
            </a>
          </div>

          {/* Install pill */}
          <div className="fu d5" style={{
            display: 'inline-flex', alignItems: 'center', gap: '16px',
            padding: '11px 18px', borderRadius: '10px',
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid var(--border)',
            backdropFilter: 'blur(6px)',
          }}>
            <code style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '13px', color: 'var(--text-secondary)' }}>
              <span style={{ color: 'var(--accent-cyan)', userSelect: 'none' }}>$ </span>pip install fie-sdk
            </code>
            <button onClick={copy} style={{
              background: copied ? 'rgba(0,255,136,0.1)' : 'rgba(255,255,255,0.05)',
              border: `1px solid ${copied ? 'rgba(0,255,136,0.3)' : 'var(--border)'}`,
              borderRadius: '6px', cursor: 'pointer',
              padding: '3px 10px',
              color: copied ? 'var(--accent-green)' : 'var(--text-muted)',
              fontFamily: 'JetBrains Mono, monospace', fontSize: '11px',
              transition: 'all 0.2s ease',
              display: 'flex', alignItems: 'center', gap: '4px',
            }}>
              {copied
                ? <><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="20 6 9 17 4 12"/></svg>copied</>
                : <><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>copy</>
              }
            </button>
          </div>
        </section>

        {/* ── Stats bar ────────────────────────────────────────────── */}
        <div style={{ borderTop: '1px solid var(--border)', borderBottom: '1px solid var(--border)', background: 'rgba(255,255,255,0.012)', position: 'relative', zIndex: 1 }}>
          <div ref={statsRef} style={{ maxWidth: '1100px', margin: '0 auto', padding: '0 28px', display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)' }} className="grid-4">
            {STATS.map(({ value, suffix, label }, i) => (
              <div key={label} style={{ borderRight: i < 3 ? '1px solid var(--border)' : 'none' }}>
                <StatCard value={value} suffix={suffix} label={label} visible={statsVisible} />
              </div>
            ))}
          </div>
        </div>

        {/* ── Features ─────────────────────────────────────────────── */}
        <section style={{ maxWidth: '1100px', margin: '0 auto', padding: '96px 28px', position: 'relative', zIndex: 1 }}>
          <RevealSection>
            <div className="section-label">Capabilities</div>
            <h2 style={{
              fontSize: 'clamp(22px, 3vw, 32px)', fontWeight: 700, letterSpacing: '-0.025em',
              color: 'var(--text-primary)', maxWidth: '500px', lineHeight: 1.2, marginBottom: '52px',
            }}>Everything you need to<br/>trust your LLM in production.</h2>
          </RevealSection>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }} className="grid-3">
            {FEATURES.map((f, i) => (
              <RevealSection key={f.title} style={{ transitionDelay: `${i * 60}ms` }}>
                <FeatureCard {...f} delay={i} />
              </RevealSection>
            ))}
          </div>
        </section>

        {/* ── How it works ─────────────────────────────────────────── */}
        <section style={{ borderTop: '1px solid var(--border)', borderBottom: '1px solid var(--border)', background: 'rgba(255,255,255,0.012)', position: 'relative', zIndex: 1 }}>
          <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '96px 28px' }}>
            <RevealSection>
              <div className="section-label">How it works</div>
              <h2 style={{ fontSize: 'clamp(22px,3vw,32px)', fontWeight: 700, letterSpacing: '-0.025em', color: 'var(--text-primary)', marginBottom: '56px' }}>
                Up and running in three steps.
              </h2>
            </RevealSection>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '48px' }} className="grid-3">
              {STEPS.map((s, i) => (
                <RevealSection key={s.n} style={{ transitionDelay: `${i * 80}ms`, position: 'relative' }}>
                  {i < 2 && <div className="step-line hide-mobile"/>}
                  <div style={{
                    width: '44px', height: '44px', borderRadius: '12px',
                    background: 'rgba(0,212,255,0.08)', border: '1px solid rgba(0,212,255,0.2)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', fontWeight: 800,
                    color: 'var(--accent-cyan)', marginBottom: '18px',
                  }}>{s.n}</div>
                  <div style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '10px', letterSpacing: '-0.01em' }}>{s.title}</div>
                  <div style={{ fontSize: '13px', lineHeight: 1.7, color: 'var(--text-muted)' }}>{s.desc}</div>
                </RevealSection>
              ))}
            </div>
          </div>
        </section>

        {/* ── Code block ───────────────────────────────────────────── */}
        <section style={{ maxWidth: '1100px', margin: '0 auto', padding: '96px 28px', position: 'relative', zIndex: 1 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '72px', alignItems: 'center' }} className="code-split">
            <RevealSection>
              <div className="section-label">Integration</div>
              <h2 style={{ fontSize: 'clamp(22px,3vw,30px)', fontWeight: 700, letterSpacing: '-0.025em', color: 'var(--text-primary)', marginBottom: '18px', lineHeight: 1.25 }}>
                One decorator.<br />Full protection.
              </h2>
              <p style={{ fontSize: '14px', lineHeight: 1.75, color: 'var(--text-muted)', marginBottom: '28px' }}>
                Add{' '}
                <code style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', background: 'rgba(0,212,255,0.08)', padding: '2px 7px', borderRadius: '5px', color: 'var(--accent-cyan)' }}>
                  @monitor(mode="local")
                </code>
                {' '}to any LLM function. Works with OpenAI, Anthropic, Groq, Ollama — anything that returns a string.
              </p>
              <Link to={loggedIn ? '/dashboard' : '/login'} className="cta-primary" style={{ alignSelf: 'flex-start' }}>
                {loggedIn ? 'Open dashboard →' : 'Get started free'}
              </Link>
            </RevealSection>

            <RevealSection style={{ transitionDelay: '120ms' }}>
              <div className="code-block">
                {/* Title bar */}
                <div style={{
                  padding: '12px 18px', borderBottom: '1px solid rgba(255,255,255,0.07)',
                  display: 'flex', gap: '7px', alignItems: 'center',
                  background: 'rgba(255,255,255,0.02)',
                }}>
                  {['#ff5f57','#febc2e','#28c840'].map(c => (
                    <div key={c} style={{ width: '11px', height: '11px', borderRadius: '50%', background: c, opacity: 0.8 }}/>
                  ))}
                  <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color: 'rgba(255,255,255,0.3)', marginLeft: '10px' }}>example.py</span>
                </div>
                {/* Code */}
                <pre style={{ margin: 0, padding: '22px 24px', fontFamily: 'JetBrains Mono, monospace', fontSize: '12.5px', lineHeight: 1.8, overflowX: 'auto', whiteSpace: 'pre' }}>
                  {CODE_LINES.map((line, li) => (
                    <div key={li}>
                      {line.tokens.length === 0
                        ? <span>&nbsp;</span>
                        : line.tokens.map((tok, ti) => (
                            <span key={ti} style={{ color: tok.c }}>{tok.t}</span>
                          ))
                      }
                    </div>
                  ))}
                </pre>
              </div>
            </RevealSection>
          </div>
        </section>

        {/* ── Benchmarks ───────────────────────────────────────────── */}
        <section style={{ borderTop: '1px solid var(--border)', borderBottom: '1px solid var(--border)', background: 'rgba(255,255,255,0.012)', position: 'relative', zIndex: 1 }}>
          <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '96px 28px' }}>
            <RevealSection>
              <div className="section-label">Benchmarks</div>
              <h2 style={{ fontSize: 'clamp(22px,3vw,32px)', fontWeight: 700, letterSpacing: '-0.025em', color: 'var(--text-primary)', marginBottom: '10px' }}>
                Numbers that matter.
              </h2>
              <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '48px' }}>
                Evaluated on 1,757 labeled examples and JailbreakBench (Chao et al., 2024).
              </p>
            </RevealSection>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }} className="grid-2">
              {/* Hallucination */}
              <RevealSection>
                <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '12px', letterSpacing: '0.02em' }}>Hallucination Detection</div>
                <div style={{ borderRadius: '12px', border: '1px solid var(--border)', overflow: 'hidden', background: 'var(--bg-card)' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr 1fr', padding: '11px 18px', borderBottom: '1px solid var(--border)', background: 'rgba(255,255,255,0.025)' }}>
                    {['Method','Recall','FPR','AUC'].map(h => (
                      <span key={h} style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', fontWeight: 700, letterSpacing: '0.1em', color: 'var(--text-muted)' }}>{h}</span>
                    ))}
                  </div>
                  {BENCHMARKS.map((r, i) => (
                    <div key={i} className="table-row" style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr 1fr', padding: '14px 18px', background: r.highlight ? 'rgba(0,212,255,0.03)' : 'transparent' }}>
                      <span style={{ fontSize: '12px', color: r.highlight ? 'var(--text-primary)' : 'var(--text-muted)', fontWeight: r.highlight ? 600 : 400 }}>{r.method}</span>
                      <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', color: r.highlight ? 'var(--accent-cyan)' : 'var(--text-muted)', fontWeight: r.highlight ? 700 : 400 }}>{r.recall}</span>
                      <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', color: 'var(--text-muted)' }}>{r.fpr}</span>
                      <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', color: r.highlight ? 'var(--accent-green)' : 'var(--text-muted)' }}>{r.auc}</span>
                    </div>
                  ))}
                </div>
              </RevealSection>

              {/* Attack */}
              <RevealSection style={{ transitionDelay: '100ms' }}>
                <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '12px', letterSpacing: '0.02em' }}>Adversarial Attack Detection · JailbreakBench</div>
                <div style={{ borderRadius: '12px', border: '1px solid var(--border)', overflow: 'hidden', background: 'var(--bg-card)' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr', padding: '11px 18px', borderBottom: '1px solid var(--border)', background: 'rgba(255,255,255,0.025)' }}>
                    {['Attack Type','Detection','FPR'].map(h => (
                      <span key={h} style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', fontWeight: 700, letterSpacing: '0.1em', color: 'var(--text-muted)' }}>{h}</span>
                    ))}
                  </div>
                  {ATTACK_BENCHMARKS.map((r, i) => (
                    <div key={i} className="table-row" style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr', padding: '14px 18px' }}>
                      <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{r.method}</span>
                      <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', color: parseFloat(r.detection) > 80 ? 'var(--accent-green)' : 'var(--accent-cyan)', fontWeight: 700 }}>{r.detection}</span>
                      <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', color: 'var(--accent-green)' }}>{r.fpr}</span>
                    </div>
                  ))}
                </div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '10px', lineHeight: 1.6 }}>
                  Offline tier · 282 attacks + 100 benign (Stanford Alpaca)
                </div>
              </RevealSection>
            </div>
          </div>
        </section>

        {/* ── CTA ──────────────────────────────────────────────────── */}
        <section style={{ position: 'relative', zIndex: 1, overflow: 'hidden' }}>
          <div style={{
            position: 'absolute', inset: 0, pointerEvents: 'none',
            background: 'radial-gradient(ellipse 60% 80% at 50% 50%, rgba(0,212,255,0.06) 0%, transparent 70%)',
          }}/>
          <div style={{ maxWidth: '640px', margin: '0 auto', padding: '108px 28px', textAlign: 'center', position: 'relative' }}>
            <RevealSection>
              <div className="section-label" style={{ justifyContent: 'center' }}>Get started</div>
              <h2 style={{
                fontSize: 'clamp(26px, 4vw, 40px)', fontWeight: 800,
                letterSpacing: '-0.03em', color: 'var(--text-primary)',
                margin: '12px 0 18px', lineHeight: 1.15,
              }}>
                Your LLM is already failing.<br />
                <span style={{ color: 'var(--accent-cyan)' }}>Start catching it.</span>
              </h2>
              <p style={{ fontSize: '15px', color: 'var(--text-muted)', lineHeight: 1.7, marginBottom: '40px' }}>
                Free to use. Open source. Works in three lines of code.
              </p>
              <div style={{ display: 'flex', gap: '12px', justifyContent: 'center', flexWrap: 'wrap' }}>
                {loggedIn
                  ? <Link to="/dashboard" className="cta-primary">Open dashboard →</Link>
                  : <Link to="/login"     className="cta-primary">Get started free</Link>
                }
                <a href="https://github.com/AyushSingh110/Failure_Intelligence_System" target="_blank" rel="noopener noreferrer" className="cta-secondary">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
                  Star on GitHub
                </a>
              </div>
            </RevealSection>
          </div>
        </section>

        {/* ── Footer ───────────────────────────────────────────────── */}
        <footer style={{ borderTop: '1px solid var(--border)', padding: '32px 28px', position: 'relative', zIndex: 1 }}>
          <div style={{
            maxWidth: '1100px', margin: '0 auto',
            display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '14px',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{
                width: '20px', height: '20px', borderRadius: '5px',
                background: 'rgba(0,212,255,0.08)', border: '1px solid rgba(0,212,255,0.2)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontFamily: 'JetBrains Mono, monospace', fontSize: '7px', fontWeight: 800,
                color: 'var(--accent-cyan)',
              }}>FIE</div>
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color: 'var(--text-muted)' }}>
                © 2026 Ayush Singh · Apache 2.0
              </span>
            </div>
            <div style={{ display: 'flex', gap: '24px' }}>
              {[
                { label: 'GitHub', href: 'https://github.com/AyushSingh110/Failure_Intelligence_System' },
                { label: 'PyPI',   href: 'https://pypi.org/project/fie-sdk/' },
                { label: 'Issues', href: 'https://github.com/AyushSingh110/Failure_Intelligence_System/issues' },
              ].map(l => (
                <a key={l.label} href={l.href} target="_blank" rel="noopener noreferrer" className="nav-link" style={{ fontSize: '12px' }}>{l.label}</a>
              ))}
            </div>
          </div>
        </footer>

      </div>
    </>
  )
}
