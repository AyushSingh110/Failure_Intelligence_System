import { NavLink, Link, useNavigate, useLocation } from 'react-router-dom'
import { clearSession, getSession } from '../lib/auth'

const NAV_GROUPS = [
  {
    label: 'Monitor',
    items: [
      {
        path: '/dashboard',
        label: 'Overview',
        shortcut: 'G D',
        icon: (
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <rect x="3" y="3" width="7" height="7" rx="1.5"/>
            <rect x="14" y="3" width="7" height="7" rx="1.5"/>
            <rect x="3" y="14" width="7" height="7" rx="1.5"/>
            <rect x="14" y="14" width="7" height="7" rx="1.5"/>
          </svg>
        ),
      },
      {
        path: '/analytics',
        label: 'Analytics',
        shortcut: 'G A',
        icon: (
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
          </svg>
        ),
      },
      {
        path: '/alerts',
        label: 'Alerts',
        shortcut: 'G L',
        icon: (
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/>
            <path d="M13.73 21a2 2 0 0 1-3.46 0"/>
          </svg>
        ),
      },
    ],
  },
  {
    label: 'Tools',
    items: [
      {
        path: '/playground',
        label: 'Playground',
        shortcut: 'G P',
        icon: (
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <rect x="2" y="3" width="9" height="18" rx="2"/>
            <rect x="13" y="3" width="9" height="18" rx="2"/>
            <line x1="2" y1="12" x2="11" y2="12"/>
            <line x1="13" y1="12" x2="22" y2="12"/>
          </svg>
        ),
      },
      {
        path: '/analyze',
        label: 'Signal Inspector',
        shortcut: 'G S',
        icon: (
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <circle cx="11" cy="11" r="8"/>
            <path d="m21 21-4.35-4.35"/>
          </svg>
        ),
      },
      {
        path: '/diagnose',
        label: 'Diagnose',
        shortcut: 'G X',
        icon: (
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M9 3H5a2 2 0 0 0-2 2v4"/>
            <path d="M9 3h6M15 3h4a2 2 0 0 1 2 2v4"/>
            <path d="M3 9v6M21 9v6"/>
            <path d="M3 15v2a2 2 0 0 0 2 2h4M21 15v2a2 2 0 0 1-2 2h-4"/>
            <path d="M9 21h6"/><circle cx="12" cy="12" r="3"/>
          </svg>
        ),
      },
      {
        path: '/vault',
        label: 'Vault',
        shortcut: 'G V',
        icon: (
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <ellipse cx="12" cy="5" rx="9" ry="3"/>
            <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>
            <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
          </svg>
        ),
      },
    ],
  },
  {
    label: 'System',
    items: [
      {
        path: '/settings',
        label: 'Settings',
        shortcut: 'G ,',
        icon: (
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <circle cx="12" cy="12" r="3"/>
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
          </svg>
        ),
      },
    ],
  },
]

function NavItem({ path, label, icon, shortcut }) {
  return (
    <NavLink
      to={path}
      title={shortcut ? `${label} (${shortcut})` : label}
      style={({ isActive }) => ({
        display: 'flex',
        alignItems: 'center',
        gap: '9px',
        padding: '7px 10px',
        borderRadius: '7px',
        marginBottom: '1px',
        textDecoration: 'none',
        fontSize: '13px',
        fontWeight: isActive ? 600 : 400,
        color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
        background: isActive ? 'rgba(255,255,255,0.05)' : 'transparent',
        borderLeft: `2px solid ${isActive ? 'var(--accent-cyan)' : 'transparent'}`,
        transition: 'all 0.12s ease',
        position: 'relative',
      })}
      onMouseEnter={e => {
        const active = e.currentTarget.style.borderLeftColor !== 'transparent'
        if (!active) {
          e.currentTarget.style.background = 'rgba(255,255,255,0.03)'
          e.currentTarget.style.color = 'var(--text-primary)'
        }
      }}
      onMouseLeave={e => {
        const active = e.currentTarget.style.background.includes('0.05)')
        if (!active) {
          e.currentTarget.style.background = 'transparent'
          e.currentTarget.style.color = 'var(--text-secondary)'
        }
      }}
    >
      <span style={{ opacity: 0.7, flexShrink: 0 }}>{icon}</span>
      <span style={{ flex: 1 }}>{label}</span>
    </NavLink>
  )
}

export default function Sidebar() {
  const navigate  = useNavigate()
  const session   = getSession()
  const name      = session?.name || 'User'
  const email     = session?.email || ''
  const initials  = name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase()

  const handleLogout = () => {
    clearSession()
    navigate('/')
  }

  return (
    <>
      <style>{`
        .sidebar-nav-group-label {
          font-family: 'JetBrains Mono', monospace;
          font-size: 9px;
          font-weight: 700;
          letter-spacing: 0.18em;
          color: var(--text-muted);
          text-transform: uppercase;
          padding: 0 12px;
          margin: 14px 0 4px;
        }
        .sidebar-nav-group-label:first-child { margin-top: 0; }
      `}</style>

      <aside className="layout-sidebar" style={{
        width: '220px',
        minWidth: '220px',
        height: '100vh',
        position: 'sticky',
        top: 0,
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--bg-secondary)',
        borderRight: '1px solid var(--border)',
      }}>

        {/* ── Logo ───────────────────────────────────────────────── */}
        <div style={{ padding: '18px 16px 14px', borderBottom: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
            <div style={{
              width: '30px', height: '30px', borderRadius: '7px', flexShrink: 0,
              background: 'linear-gradient(135deg, rgba(0,212,255,0.15), rgba(0,212,255,0.05))',
              border: '1px solid rgba(0,212,255,0.2)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '9px', fontWeight: 800,
              color: 'var(--accent-cyan)', letterSpacing: '0.04em',
            }}>FIE</div>
            <div>
              <div style={{
                fontFamily: 'JetBrains Mono, monospace',
                fontSize: '12px', fontWeight: 600,
                color: 'var(--text-primary)', lineHeight: 1.2,
              }}>FIE Platform</div>
              <div style={{
                fontFamily: 'JetBrains Mono, monospace',
                fontSize: '9px', color: 'var(--text-muted)', marginTop: '2px',
              }}>v3.0.0</div>
            </div>
          </div>

          {/* System status row */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: '6px',
            padding: '6px 9px', borderRadius: '6px',
            background: 'rgba(0,255,136,0.04)',
            border: '1px solid rgba(0,255,136,0.12)',
          }}>
            <div style={{
              width: '5px', height: '5px', borderRadius: '50%',
              background: 'var(--accent-green)',
              animation: 'pulse-dot 2.5s ease-in-out infinite',
              flexShrink: 0,
            }}/>
            <span style={{
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '10px', color: 'var(--accent-green)', fontWeight: 600,
            }}>All systems operational</span>
          </div>
        </div>

        {/* ── Home link ──────────────────────────────────────────── */}
        <div style={{ padding: '10px 16px 6px' }}>
          <Link to="/" style={{
            display: 'inline-flex', alignItems: 'center', gap: '5px',
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '10px', color: 'var(--text-muted)',
            textDecoration: 'none', transition: 'color .15s',
          }}
          onMouseEnter={e => e.currentTarget.style.color = 'var(--accent-cyan)'}
          onMouseLeave={e => e.currentTarget.style.color = 'var(--text-muted)'}
          >
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <polyline points="15 18 9 12 15 6"/>
            </svg>
            Back to home
          </Link>
        </div>

        {/* ── Navigation ─────────────────────────────────────────── */}
        <nav style={{ flex: 1, padding: '4px 8px', overflowY: 'auto' }}>
          {NAV_GROUPS.map(group => (
            <div key={group.label}>
              <div className="sidebar-nav-group-label">{group.label}</div>
              {group.items.map(item => (
                <NavItem key={item.path} {...item} />
              ))}
            </div>
          ))}
        </nav>

        {/* ── User section ───────────────────────────────────────── */}
        <div style={{ borderTop: '1px solid var(--border)', padding: '12px 8px 10px' }}>
          {/* User card */}
          <div style={{
            display: 'flex', alignItems: 'center',
            gap: '9px', padding: '8px 10px',
            borderRadius: '8px', marginBottom: '2px',
            background: 'rgba(255,255,255,0.02)',
            border: '1px solid var(--border)',
          }}>
            <div style={{
              width: '28px', height: '28px', borderRadius: '50%', flexShrink: 0,
              background: 'linear-gradient(135deg, #00d4ff 0%, #00ff88 100%)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '10px', fontWeight: 800, color: '#07111c',
            }}>{initials}</div>
            <div style={{ overflow: 'hidden', flex: 1, minWidth: 0 }}>
              <div style={{
                fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)',
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}>{name}</div>
              <div style={{
                fontFamily: 'JetBrains Mono, monospace',
                fontSize: '10px', color: 'var(--text-muted)',
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}>{session?.plan || 'free'}</div>
            </div>
          </div>

          {/* Logout */}
          <button
            onClick={handleLogout}
            style={{
              width: '100%', display: 'flex', alignItems: 'center',
              gap: '7px', padding: '7px 10px',
              borderRadius: '7px', border: 'none',
              background: 'transparent', cursor: 'pointer',
              fontSize: '12px', color: 'var(--text-muted)',
              transition: 'all 0.12s ease', textAlign: 'left',
            }}
            onMouseEnter={e => {
              e.currentTarget.style.background = 'rgba(255,68,102,0.07)'
              e.currentTarget.style.color = 'var(--accent-red)'
            }}
            onMouseLeave={e => {
              e.currentTarget.style.background = 'transparent'
              e.currentTarget.style.color = 'var(--text-muted)'
            }}
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
              <polyline points="16 17 21 12 16 7"/>
              <line x1="21" y1="12" x2="9" y2="12"/>
            </svg>
            Sign out
          </button>
        </div>
      </aside>
    </>
  )
}
