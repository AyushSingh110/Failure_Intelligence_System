import { Link } from 'react-router-dom'

/**
 * Shared shell for the Privacy Policy and Terms pages.
 *
 * These exist because Google requires a published OAuth app to link a privacy
 * policy and terms of service from its consent screen. They are public and
 * unauthenticated by design — Google's crawler and any prospective user must be
 * able to read them without signing in.
 */
export default function LegalPage({ title, updated, children }) {
  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-primary)', padding: '48px 20px' }}>
      <div style={{ maxWidth: 780, margin: '0 auto' }}>
        <Link
          to="/"
          style={{
            color: 'var(--accent-cyan)',
            textDecoration: 'none',
            fontSize: 14,
            display: 'inline-block',
            marginBottom: 28,
          }}
        >
          ← Back to Failure Intelligence Engine
        </Link>

        <h1 style={{ fontSize: 32, fontWeight: 700, margin: '0 0 8px', color: 'var(--text-primary, #e8eef5)' }}>
          {title}
        </h1>
        <p style={{ fontSize: 13, color: '#7a9bb8', margin: '0 0 36px' }}>
          Last updated: {updated}
        </p>

        <div
          style={{
            background: 'var(--bg-card)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-lg, 12px)',
            padding: '32px 34px',
            color: '#c5d5e4',
            fontSize: 15,
            lineHeight: 1.75,
          }}
          className="legal-body"
        >
          {children}
        </div>

        <p style={{ fontSize: 13, color: '#7a9bb8', marginTop: 28 }}>
          Questions? Email{' '}
          <a href="mailto:ayushsingh355vns@gmail.com" style={{ color: 'var(--accent-cyan)' }}>
            ayushsingh355vns@gmail.com
          </a>{' '}
          or open an issue on{' '}
          <a
            href="https://github.com/AyushSingh110/Failure_Intelligence_System"
            target="_blank"
            rel="noreferrer"
            style={{ color: 'var(--accent-cyan)' }}
          >
            GitHub
          </a>
          .
        </p>
      </div>

      <style>{`
        .legal-body h2 {
          font-size: 18px;
          font-weight: 650;
          color: #e8eef5;
          margin: 30px 0 10px;
        }
        .legal-body h2:first-child { margin-top: 0; }
        .legal-body p  { margin: 0 0 14px; }
        .legal-body ul { margin: 0 0 16px; padding-left: 22px; }
        .legal-body li { margin-bottom: 7px; }
        .legal-body code {
          background: var(--bg-elevated, #16202b);
          padding: 2px 6px;
          border-radius: 4px;
          font-size: 13.5px;
        }
        .legal-body a { color: var(--accent-cyan); }
        .legal-body strong { color: #e8eef5; }
      `}</style>
    </div>
  )
}
