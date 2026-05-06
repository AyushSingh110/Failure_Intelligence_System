import { useState } from 'react'
import { Link } from 'react-router-dom'
import { isLoggedIn } from '../lib/auth'

const CODE = `from fie import monitor

@monitor(mode="local")   # zero setup, works offline
def ask_ai(prompt: str) -> str:
    return your_llm(prompt)

# Hallucinations + prompt attacks caught automatically
response = ask_ai("Who won the 2022 World Cup?")`

const BENCHMARKS = [
  { method: 'POET rule-based (baseline)', recall: '56.4%', fpr: '38.7%', auc: '—' },
  { method: 'FIE XGBoost v3',            recall: '63.6%', fpr: '38.6%', auc: '0.677' },
]

const ATTACK_BENCHMARKS = [
  { method: 'GCG suffix attacks',     detection: '96.0%', fpr: '2.0%' },
  { method: 'JBC persona jailbreaks', detection: '52.0%', fpr: '2.0%' },
  { method: 'Direct injection',       detection: '95.0%', fpr: '2.0%' },
]

const FEATURES = [
  {
    title: 'Hallucination Detection',
    desc: 'A shadow jury of 3 independent models cross-checks every answer. The Failure Signal Vector captures agreement, entropy, and confidence — fed into a trained XGBoost classifier.',
  },
  {
    title: 'Adversarial Attack Protection',
    desc: 'Seven detection layers catch prompt injection, jailbreaks, token smuggling, indirect injection, GCG suffix attacks, and multi-turn Crescendo attacks — including obfuscated variants.',
  },
  {
    title: 'Auto-Correction',
    desc: 'When a failure is detected, FIE automatically applies the right fix — jury consensus replacement, ground truth cache override, prompt sanitization, or escalation to a human reviewer.',
  },
  {
    title: 'Works Offline',
    desc: 'Local mode runs entirely on your machine with zero network calls. No API key, no server, no setup. Add one decorator and you have protection immediately.',
  },
  {
    title: 'Self-Improving',
    desc: 'Submit feedback on any flagged inference and FIE stores the correct answer. Future similar queries use the cache directly. Labeled data periodically retrains the classifier.',
  },
  {
    title: 'Model Drift Monitoring',
    desc: "EMA-based trend tracking alerts you when your model's failure rate is rising — before your users notice. Catch degradation from stale training data or a bad model update.",
  },
]

const STEPS = [
  { n: '01', title: 'Install the SDK',       desc: 'pip install fie-sdk — one command, no extra dependencies.' },
  { n: '02', title: 'Add the decorator',     desc: 'Wrap your LLM function with @monitor(mode="local") for instant offline detection.' },
  { n: '03', title: 'Connect for full power',desc: 'Sign in to unlock the shadow jury, XGBoost classifier, auto-correction, and your analytics dashboard.' },
]

export default function LandingPage() {
  const loggedIn      = isLoggedIn()
  const [copied, setCopied] = useState(false)

  const copy = () => {
    navigator.clipboard.writeText('pip install fie-sdk')
    setCopied(true)
    setTimeout(() => setCopied(false), 1800)
  }

  return (
    <>
      <style>{`
        *, *::before, *::after { box-sizing: border-box; }
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(14px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .fu  { animation: fadeUp 0.55s cubic-bezier(0.16,1,0.3,1) both; }
        .d1  { animation-delay: 0.05s; }
        .d2  { animation-delay: 0.12s; }
        .d3  { animation-delay: 0.20s; }
        .d4  { animation-delay: 0.28s; }
        .d5  { animation-delay: 0.36s; }
        .nav-link {
          font-family: 'Inter', sans-serif;
          font-size: 13px;
          color: var(--text-muted);
          text-decoration: none;
          transition: color 0.15s;
        }
        .nav-link:hover { color: var(--text-primary); }
        .feature-card {
          padding: 24px 26px;
          border-radius: 12px;
          background: var(--bg-card);
          border: 1px solid var(--border);
          transition: border-color 0.2s;
        }
        .feature-card:hover { border-color: rgba(0,212,255,0.22); }
        .cta-primary {
          display: inline-flex; align-items: center; gap: 6px;
          padding: 10px 20px; border-radius: 8px;
          background: var(--accent-cyan); color: #000;
          font-size: 13px; font-weight: 600;
          font-family: 'Inter', sans-serif;
          border: none; cursor: pointer; text-decoration: none;
          transition: opacity 0.15s, transform 0.15s;
        }
        .cta-primary:hover { opacity: 0.87; transform: translateY(-1px); }
        .cta-secondary {
          display: inline-flex; align-items: center; gap: 6px;
          padding: 10px 20px; border-radius: 8px;
          background: transparent; color: var(--text-secondary);
          font-size: 13px; font-weight: 500;
          font-family: 'Inter', sans-serif;
          border: 1px solid var(--border); cursor: pointer; text-decoration: none;
          transition: border-color 0.15s, color 0.15s, transform 0.15s;
        }
        .cta-secondary:hover {
          border-color: rgba(255,255,255,0.22);
          color: var(--text-primary);
          transform: translateY(-1px);
        }
        .section-label {
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px; font-weight: 600;
          letter-spacing: 0.18em; color: var(--text-muted);
          text-transform: uppercase; margin-bottom: 14px;
        }
        .table-row:not(:last-child) { border-bottom: 1px solid var(--border); }
        .table-row:hover { background: rgba(255,255,255,0.015); }
        .pill {
          display: inline-block;
          padding: 3px 10px; border-radius: 20px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 11px; font-weight: 600;
          background: rgba(0,212,255,0.08);
          color: var(--accent-cyan);
          border: 1px solid rgba(0,212,255,0.2);
          letter-spacing: 0.05em;
        }
      `}</style>

      <div style={{ minHeight: '100vh', background: 'var(--bg-primary)', color: 'var(--text-primary)', fontFamily: 'Inter, sans-serif' }}>

        {/* ── Nav ─────────────────────────────────────────────── */}
        <nav style={{
          position: 'sticky', top: 0, zIndex: 50,
          borderBottom: '1px solid var(--border)',
          background: 'rgba(7,11,20,0.85)',
          backdropFilter: 'blur(12px)',
        }}>
          <div style={{
            maxWidth: '1080px', margin: '0 auto', padding: '0 24px',
            height: '56px', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div style={{
                width: '28px', height: '28px', borderRadius: '7px',
                background: 'rgba(0,212,255,0.1)', border: '1px solid rgba(0,212,255,0.25)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontFamily: 'JetBrains Mono, monospace', fontSize: '9px', fontWeight: 700,
                color: 'var(--accent-cyan)', letterSpacing: '0.05em',
              }}>FIE</div>
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', fontWeight: 500, color: 'var(--text-secondary)' }}>
                Failure Intelligence Engine
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '28px' }}>
              <a href="https://github.com/AyushSingh110/Failure_Intelligence_System" target="_blank" rel="noopener noreferrer" className="nav-link">GitHub</a>
              <a href="https://pypi.org/project/fie-sdk/" target="_blank" rel="noopener noreferrer" className="nav-link">PyPI</a>
              {loggedIn
                ? <Link to="/dashboard" className="cta-primary" style={{ padding: '7px 16px', fontSize: '12px' }}>Dashboard →</Link>
                : <Link to="/login"     className="cta-primary" style={{ padding: '7px 16px', fontSize: '12px' }}>Sign in</Link>
              }
            </div>
          </div>
        </nav>

        {/* ── Hero ─────────────────────────────────────────────── */}
        <section style={{ maxWidth: '1080px', margin: '0 auto', padding: '96px 24px 80px' }}>
          <div className="fu d1" style={{ marginBottom: '20px' }}>
            <span className="pill">Open Source · Apache 2.0</span>
          </div>
          <h1 className="fu d2" style={{
            fontSize: 'clamp(32px, 5vw, 52px)', fontWeight: 700,
            lineHeight: 1.12, letterSpacing: '-0.03em',
            color: 'var(--text-primary)', maxWidth: '720px', marginBottom: '20px',
          }}>
            Catch LLM failures<br />
            <span style={{
              background: 'linear-gradient(90deg, var(--accent-cyan) 0%, #4ade80 100%)',
              WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
            }}>before your users do.</span>
          </h1>
          <p className="fu d3" style={{
            fontSize: '16px', lineHeight: 1.7, color: 'var(--text-muted)',
            maxWidth: '520px', marginBottom: '36px',
          }}>
            Real-time hallucination detection, adversarial attack protection,
            and automatic correction — as a single Python decorator.
          </p>
          <div className="fu d4" style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', marginBottom: '40px' }}>
            {loggedIn
              ? <Link to="/dashboard" className="cta-primary">Go to Dashboard →</Link>
              : <Link to="/login"     className="cta-primary">Get started free</Link>
            }
            <a href="https://github.com/AyushSingh110/Failure_Intelligence_System" target="_blank" rel="noopener noreferrer" className="cta-secondary">
              View on GitHub
            </a>
          </div>
          <div className="fu d5" style={{
            display: 'inline-flex', alignItems: 'center', gap: '16px',
            padding: '10px 16px', borderRadius: '8px',
            background: 'var(--bg-card)', border: '1px solid var(--border)',
          }}>
            <code style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '13px', color: 'var(--text-secondary)' }}>
              <span style={{ color: 'var(--text-muted)', userSelect: 'none' }}>$ </span>pip install fie-sdk
            </code>
            <button onClick={copy} style={{
              background: 'none', border: 'none', cursor: 'pointer',
              padding: '2px 4px', borderRadius: '4px',
              color: copied ? 'var(--accent-green)' : 'var(--text-muted)',
              fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', transition: 'color 0.15s',
            }}>{copied ? 'copied' : 'copy'}</button>
          </div>
        </section>

        {/* ── Stats bar ─────────────────────────────────────────── */}
        <div style={{ borderTop: '1px solid var(--border)', borderBottom: '1px solid var(--border)', background: 'rgba(255,255,255,0.015)' }}>
          <div style={{ maxWidth: '1080px', margin: '0 auto', padding: '0 24px', display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)' }}>
            {[
              { value: '547+',  label: 'PyPI installs' },
              { value: '0%',    label: 'False positive rate' },
              { value: '96%',   label: 'GCG attack recall' },
              { value: '7',     label: 'Detection layers' },
            ].map(({ value, label }, i) => (
              <div key={label} style={{
                padding: '28px 24px', textAlign: 'center',
                borderRight: i < 3 ? '1px solid var(--border)' : 'none',
              }}>
                <div style={{
                  fontFamily: 'JetBrains Mono, monospace', fontSize: '26px', fontWeight: 700,
                  color: 'var(--text-primary)', letterSpacing: '-0.02em', marginBottom: '4px',
                }}>{value}</div>
                <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* ── Features ──────────────────────────────────────────── */}
        <section style={{ maxWidth: '1080px', margin: '0 auto', padding: '88px 24px' }}>
          <div style={{ marginBottom: '48px' }}>
            <div className="section-label">Capabilities</div>
            <h2 style={{
              fontSize: '28px', fontWeight: 700, letterSpacing: '-0.02em',
              color: 'var(--text-primary)', maxWidth: '480px', lineHeight: 1.25,
            }}>Everything you need to trust your LLM in production.</h2>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
            {FEATURES.map(f => (
              <div key={f.title} className="feature-card">
                <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '10px' }}>{f.title}</div>
                <div style={{ fontSize: '13px', lineHeight: 1.65, color: 'var(--text-muted)' }}>{f.desc}</div>
              </div>
            ))}
          </div>
        </section>

        {/* ── How it works ──────────────────────────────────────── */}
        <section style={{ borderTop: '1px solid var(--border)', borderBottom: '1px solid var(--border)', background: 'rgba(255,255,255,0.012)' }}>
          <div style={{ maxWidth: '1080px', margin: '0 auto', padding: '88px 24px' }}>
            <div style={{ marginBottom: '48px' }}>
              <div className="section-label">How it works</div>
              <h2 style={{ fontSize: '28px', fontWeight: 700, letterSpacing: '-0.02em', color: 'var(--text-primary)' }}>
                Up and running in three steps.
              </h2>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '40px' }}>
              {STEPS.map(s => (
                <div key={s.n}>
                  <div style={{
                    fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', fontWeight: 700,
                    color: 'var(--accent-cyan)', letterSpacing: '0.1em', marginBottom: '10px',
                  }}>{s.n}</div>
                  <div style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '8px' }}>{s.title}</div>
                  <div style={{ fontSize: '13px', lineHeight: 1.65, color: 'var(--text-muted)' }}>{s.desc}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── Code example ──────────────────────────────────────── */}
        <section style={{
          maxWidth: '1080px', margin: '0 auto', padding: '88px 24px',
          display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '64px', alignItems: 'center',
        }}>
          <div>
            <div className="section-label">Integration</div>
            <h2 style={{ fontSize: '26px', fontWeight: 700, letterSpacing: '-0.02em', color: 'var(--text-primary)', marginBottom: '16px', lineHeight: 1.25 }}>
              One decorator.<br />Full protection.
            </h2>
            <p style={{ fontSize: '13px', lineHeight: 1.7, color: 'var(--text-muted)', marginBottom: '24px' }}>
              Add{' '}
              <code style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', background: 'rgba(0,212,255,0.08)', padding: '1px 6px', borderRadius: '4px', color: 'var(--accent-cyan)' }}>
                @monitor(mode="local")
              </code>
              {' '}to any LLM function. Works with OpenAI, Anthropic, Groq, Ollama — anything that returns a string.
            </p>
            <Link to={loggedIn ? '/dashboard' : '/login'} className="cta-primary">
              {loggedIn ? 'Open dashboard →' : 'Get started free'}
            </Link>
          </div>
          <div style={{ background: '#0d1117', borderRadius: '12px', border: '1px solid var(--border)', overflow: 'hidden' }}>
            <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)', display: 'flex', gap: '6px', alignItems: 'center' }}>
              {['#ff5f57','#febc2e','#28c840'].map(c => (
                <div key={c} style={{ width: '10px', height: '10px', borderRadius: '50%', background: c, opacity: 0.7 }} />
              ))}
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', color: 'var(--text-muted)', marginLeft: '8px' }}>example.py</span>
            </div>
            <pre style={{
              margin: 0, padding: '20px',
              fontFamily: 'JetBrains Mono, monospace', fontSize: '12px',
              lineHeight: 1.75, color: '#c9d1d9', overflowX: 'auto', whiteSpace: 'pre',
            }}>{CODE}</pre>
          </div>
        </section>

        {/* ── Benchmarks ────────────────────────────────────────── */}
        <section style={{ borderTop: '1px solid var(--border)', borderBottom: '1px solid var(--border)', background: 'rgba(255,255,255,0.012)' }}>
          <div style={{ maxWidth: '1080px', margin: '0 auto', padding: '88px 24px' }}>
            <div style={{ marginBottom: '48px' }}>
              <div className="section-label">Benchmarks</div>
              <h2 style={{ fontSize: '28px', fontWeight: 700, letterSpacing: '-0.02em', color: 'var(--text-primary)', marginBottom: '8px' }}>
                Numbers that matter.
              </h2>
              <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
                Evaluated on 1,757 labeled examples and JailbreakBench (Chao et al., 2024).
              </p>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
              {/* Hallucination */}
              <div>
                <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '12px' }}>Hallucination Detection</div>
                <div style={{ borderRadius: '10px', border: '1px solid var(--border)', overflow: 'hidden', background: 'var(--bg-card)' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr 1fr', padding: '10px 16px', borderBottom: '1px solid var(--border)', background: 'rgba(255,255,255,0.02)' }}>
                    {['Method','Recall','FPR','AUC'].map(h => (
                      <span key={h} style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', fontWeight: 600, letterSpacing: '0.1em', color: 'var(--text-muted)' }}>{h}</span>
                    ))}
                  </div>
                  {BENCHMARKS.map((r, i) => (
                    <div key={i} className="table-row" style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr 1fr', padding: '12px 16px', background: i === 1 ? 'rgba(0,212,255,0.03)' : 'transparent' }}>
                      <span style={{ fontSize: '12px', color: i === 1 ? 'var(--text-primary)' : 'var(--text-muted)', fontWeight: i === 1 ? 500 : 400 }}>{r.method}</span>
                      <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', color: i === 1 ? 'var(--accent-cyan)' : 'var(--text-muted)', fontWeight: i === 1 ? 700 : 400 }}>{r.recall}</span>
                      <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', color: 'var(--text-muted)' }}>{r.fpr}</span>
                      <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', color: i === 1 ? 'var(--accent-green)' : 'var(--text-muted)' }}>{r.auc}</span>
                    </div>
                  ))}
                </div>
              </div>
              {/* Attack */}
              <div>
                <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '12px' }}>Adversarial Attack Detection · JailbreakBench</div>
                <div style={{ borderRadius: '10px', border: '1px solid var(--border)', overflow: 'hidden', background: 'var(--bg-card)' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr', padding: '10px 16px', borderBottom: '1px solid var(--border)', background: 'rgba(255,255,255,0.02)' }}>
                    {['Attack Type','Detection','FPR'].map(h => (
                      <span key={h} style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', fontWeight: 600, letterSpacing: '0.1em', color: 'var(--text-muted)' }}>{h}</span>
                    ))}
                  </div>
                  {ATTACK_BENCHMARKS.map((r, i) => (
                    <div key={i} className="table-row" style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr', padding: '12px 16px' }}>
                      <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{r.method}</span>
                      <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', color: parseFloat(r.detection) > 80 ? 'var(--accent-green)' : 'var(--accent-cyan)', fontWeight: 600 }}>{r.detection}</span>
                      <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', color: 'var(--accent-green)' }}>{r.fpr}</span>
                    </div>
                  ))}
                </div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '8px', lineHeight: 1.5 }}>
                  Offline package tier · 282 attacks + 100 benign (Stanford Alpaca)
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ── CTA ───────────────────────────────────────────────── */}
        <section style={{ maxWidth: '1080px', margin: '0 auto', padding: '88px 24px', textAlign: 'center' }}>
          <div className="section-label" style={{ display: 'flex', justifyContent: 'center' }}>Get started</div>
          <h2 style={{ fontSize: '32px', fontWeight: 700, letterSpacing: '-0.025em', color: 'var(--text-primary)', margin: '10px 0 16px' }}>
            Your LLM is already failing.<br />Start catching it.
          </h2>
          <p style={{ fontSize: '14px', color: 'var(--text-muted)', lineHeight: 1.7, maxWidth: '380px', margin: '0 auto 36px' }}>
            Free to use. Open source. Works in three lines.
          </p>
          <div style={{ display: 'flex', gap: '12px', justifyContent: 'center', flexWrap: 'wrap' }}>
            {loggedIn
              ? <Link to="/dashboard" className="cta-primary">Open dashboard →</Link>
              : <Link to="/login"     className="cta-primary">Get started free</Link>
            }
            <a href="https://github.com/AyushSingh110/Failure_Intelligence_System" target="_blank" rel="noopener noreferrer" className="cta-secondary">
              Star on GitHub
            </a>
          </div>
        </section>

        {/* ── Footer ────────────────────────────────────────────── */}
        <footer style={{ borderTop: '1px solid var(--border)', padding: '28px 24px' }}>
          <div style={{
            maxWidth: '1080px', margin: '0 auto',
            display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px',
          }}>
            <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color: 'var(--text-muted)' }}>
              © 2026 Ayush Singh · Apache 2.0
            </span>
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
