import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { isLoggedIn } from './lib/auth'
import ErrorBoundary from './components/ErrorBoundary'
import LandingPage   from './pages/LandingPage'
import LoginPage     from './pages/LoginPage'
import DashboardPage from './pages/DashboardPage'
import AnalyzePage   from './pages/Analyzepage'
import DiagnosePage    from './pages/Diagnosepage'
import PlaygroundPage  from './pages/Playgroundpage'
import AlertsPage      from './pages/AlertsPage'
import VaultPage     from './pages/Vaultpage'
import SettingsPage   from './pages/Settingspage'
import AnalyticsPage  from './pages/AnalyticsPage'
import PrivacyPage    from './pages/PrivacyPage'
import TermsPage      from './pages/TermsPage'
import Layout         from './components/Layout'

function Protected({ children }) {
  const [authed, setAuthed] = useState(() => isLoggedIn())
  useEffect(() => { setAuthed(isLoggedIn()) }, [])
  if (authed === null) return null
  return authed
    ? <Layout>{children}</Layout>
    : <Navigate to="/login" replace />
}

export default function App() {
  return (
    <BrowserRouter>
      <ErrorBoundary>
      <Routes>
        {/* Public */}
        <Route path="/"      element={<LandingPage />} />
        <Route path="/login" element={<LoginPage />} />

        {/* Public legal pages. Google requires a published OAuth app to link a
            privacy policy and terms of service from its consent screen, and its
            crawler must reach them without signing in — so these must stay
            OUTSIDE <Protected>. */}
        <Route path="/privacy" element={<PrivacyPage />} />
        <Route path="/terms"   element={<TermsPage />} />

        {/* Protected dashboard routes */}
        <Route path="/dashboard" element={<Protected><DashboardPage /></Protected>} />
        <Route path="/analyze"   element={<Protected><AnalyzePage /></Protected>} />
        <Route path="/diagnose"    element={<Protected><DiagnosePage /></Protected>} />
        <Route path="/playground"  element={<Protected><PlaygroundPage /></Protected>} />
        <Route path="/analytics" element={<Protected><AnalyticsPage /></Protected>} />
        <Route path="/alerts"    element={<Protected><AlertsPage /></Protected>} />
        <Route path="/vault"     element={<Protected><VaultPage /></Protected>} />
        <Route path="/settings"  element={<Protected><SettingsPage /></Protected>} />

        {/* Catch-all → landing */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      </ErrorBoundary>
    </BrowserRouter>
  )
}
