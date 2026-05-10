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
      }}>
        {children}
      </main>
    </div>
  )
}