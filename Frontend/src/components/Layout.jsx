import Sidebar from './Sidebar'

export default function Layout({ children }) {
  return (
    <div className="layout-root" style={{
      display: 'flex',
      minHeight: '100vh',
      background: 'var(--bg-primary)',
    }}>
      <div className="layout-sidebar"><Sidebar /></div>
      <main style={{
        flex: 1, display: 'flex', flexDirection: 'column',
        minWidth: 0, overflowX: 'hidden',
        backgroundImage: 'radial-gradient(ellipse 60% 40% at 70% 0%, rgba(0,212,255,0.03) 0%, transparent 60%)',
      }}>
        {children}
      </main>
    </div>
  )
}