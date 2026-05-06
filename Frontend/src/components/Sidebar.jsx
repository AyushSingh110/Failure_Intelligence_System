//sidebar.jsx
import { NavLink, Link, useNavigate } from 'react-router-dom'
import { clearSession, getSession } from '../lib/auth'

const NAV = [
  {
    path: '/dashboard',
    label: 'Dashboard',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
        stroke="currentColor" strokeWidth="1.8">
        <rect x="3" y="3" width="7" height="7" rx="1"/>
        <rect x="14" y="3" width="7" height="7" rx="1"/>
        <rect x="3" y="14" width="7" height="7" rx="1"/>
        <rect x="14" y="14" width="7" height="7" rx="1"/>
      </svg>
    ),
  },
  {
    path: '/analyze',
    label: 'Analyze',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
        stroke="currentColor" strokeWidth="1.8">
        <circle cx="11" cy="11" r="8"/>
        <path d="m21 21-4.35-4.35"/>
      </svg>
    ),
  },
  {
    path: '/diagnose',
    label: 'Diagnose',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
        stroke="currentColor" strokeWidth="1.8">
        <path d="M9 3H5a2 2 0 0 0-2 2v4"/>
        <path d="M9 3h6"/>
        <path d="M15 3h4a2 2 0 0 1 2 2v4"/>
        <path d="M3 9v6"/>
        <path d="M21 9v6"/>
        <path d="M3 15v2a2 2 0 0 0 2 2h4"/>
        <path d="M21 15v2a2 2 0 0 1-2 2h-4"/>
        <path d="M9 21h6"/>
        <circle cx="12" cy="12" r="3"/>
      </svg>
    ),
  },
  {
    path: '/alerts',
    label: 'Alerts',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
        stroke="currentColor" strokeWidth="1.8">
        <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/>
        <path d="M13.73 21a2 2 0 0 1-3.46 0"/>
      </svg>
    ),
  },
  {
    path: '/vault',
    label: 'Vault',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
        stroke="currentColor" strokeWidth="1.8">
        <ellipse cx="12" cy="5" rx="9" ry="3"/>
        <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>
        <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
      </svg>
    ),
  },
  {
    path: '/settings',
    label: 'Settings',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
        stroke="currentColor" strokeWidth="1.8">
        <circle cx="12" cy="12" r="3"/>
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
      </svg>
    ),
  },
]

export default function Sidebar() {
  const navigate = useNavigate()
  const session  = getSession()
  const name     = session?.name || 'User'
  const initials = name.split(' ').map(w => w[0]).join('').slice(0,2).toUpperCase()

  const handleLogout = () => {
    clearSession()
    navigate('/')
  }

  return (
    <aside style={{
      width: '220px',
      minWidth: '220px',
      height: '100vh',
      position: 'sticky',
      top: 0,
      display: 'flex',
      flexDirection: 'column',
      background: 'var(--bg-secondary)',
      borderRight: '1px solid var(--border)',
      padding: '24px 0',
    }}>

      {/* Logo */}
      <div style={{ padding: '0 20px 28px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '32px', height: '32px',
            borderRadius: '8px',
            background: 'rgba(0,212,255,0.1)',
            border: '1px solid rgba(0,212,255,0.25)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '10px', fontWeight: 700,
            color: 'var(--accent-cyan)',
          }}>FIE</div>
          <div>
            <div style={{
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '12px', fontWeight: 600,
              color: 'var(--text-primary)',
            }}>FIE Platform</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '5px', marginTop: '2px' }}>
              <div style={{
                width: '5px', height: '5px', borderRadius: '50%',
                background: 'var(--accent-green)',
                animation: 'pulse-dot 2.5s ease-in-out infinite',
              }}/>
              <span style={{
                fontFamily: 'JetBrains Mono, monospace',
                fontSize: '10px',
                color: 'var(--text-muted)',
              }}>live</span>
            </div>
          </div>
        </div>
      </div>

      {/* Back to landing */}
      <Link to="/" style={{
        display: 'flex', alignItems: 'center', gap: '6px',
        padding: '0 20px 16px',
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: '10px', color: 'var(--text-muted)',
        textDecoration: 'none', transition: 'color .2s',
      }}
      onMouseEnter={e => e.currentTarget.style.color = 'var(--accent-cyan)'}
      onMouseLeave={e => e.currentTarget.style.color = 'var(--text-muted)'}
      >
        ← Home
      </Link>

      {/* Section label */}
      <div style={{
        padding: '0 20px 8px',
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: '9px', fontWeight: 600,
        letterSpacing: '0.15em',
        color: 'var(--text-muted)',
      }}>NAVIGATION</div>

      {/* Nav items */}
      <nav style={{ flex: 1, padding: '0 10px' }}>
        {NAV.map(({ path, label, icon }) => (
          <NavLink
            key={path}
            to={path}
            style={({ isActive }) => ({
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              padding: '9px 12px',
              borderRadius: '8px',
              marginBottom: '2px',
              textDecoration: 'none',
              fontSize: '13px',
              fontWeight: isActive ? 600 : 400,
              color: isActive ? 'var(--accent-cyan)' : 'var(--text-secondary)',
              background: isActive ? 'rgba(0,212,255,0.08)' : 'transparent',
              border: isActive ? '1px solid rgba(0,212,255,0.12)' : '1px solid transparent',
              transition: 'all 0.15s ease',
            })}
            onMouseEnter={e => {
              if (!e.currentTarget.style.background.includes('0.08')) {
                e.currentTarget.style.background = 'rgba(255,255,255,0.03)'
                e.currentTarget.style.color = 'var(--text-primary)'
              }
            }}
            onMouseLeave={e => {
              if (!e.currentTarget.style.background.includes('0.08')) {
                e.currentTarget.style.background = 'transparent'
                e.currentTarget.style.color = 'var(--text-secondary)'
              }
            }}
          >
            {icon}
            {label}
          </NavLink>
        ))}
      </nav>

      {/* User section */}
      <div style={{
        padding: '16px 10px 0',
        borderTop: '1px solid var(--border)',
        marginTop: '8px',
      }}>
        {/* User row */}
        <div style={{
          display: 'flex', alignItems: 'center',
          gap: '10px', padding: '8px 12px',
          borderRadius: '8px',
          marginBottom: '4px',
        }}>
          <div style={{
            width: '28px', height: '28px',
            borderRadius: '50%',
            background: 'linear-gradient(135deg, var(--accent-cyan), var(--accent-green))',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '10px', fontWeight: 700,
            color: '#0d1117',
            flexShrink: 0,
          }}>{initials}</div>
          <div style={{ overflow: 'hidden' }}>
            <div style={{
              fontSize: '12px', fontWeight: 600,
              color: 'var(--text-primary)',
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            }}>{name}</div>
            <div style={{
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '10px',
              color: 'var(--text-muted)',
            }}>{session?.plan || 'free'}</div>
          </div>
        </div>

        {/* Logout */}
        <button
          onClick={handleLogout}
          style={{
            width: '100%', display: 'flex', alignItems: 'center',
            gap: '8px', padding: '8px 12px',
            borderRadius: '8px', border: 'none',
            background: 'transparent', cursor: 'pointer',
            fontSize: '13px', color: 'var(--text-muted)',
            transition: 'all 0.15s ease',
            fontFamily: 'Syne, sans-serif',
          }}
          onMouseEnter={e => {
            e.currentTarget.style.background = 'rgba(255,68,102,0.08)'
            e.currentTarget.style.color = 'var(--accent-red)'
          }}
          onMouseLeave={e => {
            e.currentTarget.style.background = 'transparent'
            e.currentTarget.style.color = 'var(--text-muted)'
          }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="1.8">
            <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
            <polyline points="16 17 21 12 16 7"/>
            <line x1="21" y1="12" x2="9" y2="12"/>
          </svg>
          Sign out
        </button>
      </div>
    </aside>
  )
}