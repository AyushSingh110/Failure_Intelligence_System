import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { isLoggedIn } from './lib/auth'
import LoginPage     from './pages/LoginPage'
import DashboardPage from './pages/DashboardPage'
import AnalyzePage   from './pages/AnalyzePage'
import DiagnosePage  from './pages/DiagnosePage'
import AlertsPage    from './pages/AlertsPage'
import VaultPage     from './pages/VaultPage'
import SettingsPage  from './pages/SettingsPage'
import Layout        from './components/Layout'

function Protected({ children }) {
  const [authed, setAuthed] = useState(() => isLoggedIn())
  useEffect(() => { setAuthed(isLoggedIn()) }, [])
  if (authed === null) return null
  return authed
    ? <Layout>{children}</Layout>
    : <Navigate to="/login" replace />
}

function HomeRoute() {
  const location = useLocation()
  if (location.search.includes('code=')) return <Navigate to={`/login${location.search}`} replace />
  return <Navigate to={isLoggedIn() ? '/dashboard' : '/login'} replace />
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login"     element={<LoginPage />} />
        <Route path="/"          element={<HomeRoute />} />
        <Route path="/dashboard" element={<Protected><DashboardPage /></Protected>} />
        <Route path="/analyze"   element={<Protected><AnalyzePage /></Protected>} />
        <Route path="/diagnose"  element={<Protected><DiagnosePage /></Protected>} />
        <Route path="/alerts"    element={<Protected><AlertsPage /></Protected>} />
        <Route path="/vault"     element={<Protected><VaultPage /></Protected>} />
        <Route path="/settings"  element={<Protected><SettingsPage /></Protected>} />
      </Routes>
    </BrowserRouter>
  )
}