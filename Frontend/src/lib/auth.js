const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID || ''
const TOKEN_KEY = 'fie_session'
const GOOGLE_REDIRECT_KEY = 'fie_google_redirect_uri'

function normalizeSession(data) {
  if (!data || typeof data !== 'object') {
    return null
  }

  const token = data.token || data.access_token || ''

  return {
    ...data,
    token,
    access_token: token,
    // Preserved so an API-key session survives a normalize() round-trip.
    api_key: data.api_key || '',
  }
}

function getCurrentRedirectUri() {
  // VITE_REDIRECT_URI wins whenever it is configured.
  //
  // This used to derive the redirect_uri from window.location.origin, which
  // fails the moment the app is reached from any host other than the single
  // canonical one registered in Google Cloud Console — and there are several:
  //
  //   https://<deploy-hash>.failure-intelligence-system.pages.dev  (every
  //       Cloudflare Pages deployment gets its own preview subdomain)
  //   https://failure-intelligence-system-6h53.vercel.app
  //   a custom domain, later
  //
  // Google then rejects the request with `Error 400: redirect_uri_mismatch`,
  // and the error surfaces on Google's page rather than ours, so nothing in
  // the app logs explains it. Pinning the value makes the redirect_uri
  // deterministic: it is whatever is registered, regardless of which URL the
  // user happened to open.
  const configured = import.meta.env.VITE_REDIRECT_URI
  if (configured) {
    return configured
  }

  if (typeof window === 'undefined') {
    return ''
  }
  // Fallback for local dev, where VITE_REDIRECT_URI is usually unset.
  // Must include /login so Google redirects back to the page that actually
  // processes the ?code= callback — not the landing page.
  return window.location.origin + '/login'
}

export function getGoogleRedirectUri() {
  const savedRedirectUri = sessionStorage.getItem(GOOGLE_REDIRECT_KEY)
  return savedRedirectUri || getCurrentRedirectUri()
}

export function getGoogleAuthUrl() {
  const redirectUri = getCurrentRedirectUri()
  sessionStorage.setItem(GOOGLE_REDIRECT_KEY, redirectUri)

  const params = new URLSearchParams({
    client_id: GOOGLE_CLIENT_ID,
    redirect_uri: redirectUri,
    response_type: 'code',
    scope: 'openid email profile',
    access_type: 'online',
    prompt: 'select_account',
  })

  return `https://accounts.google.com/o/oauth2/v2/auth?${params}`
}

export function parseGoogleCallback() {
  const params = new URLSearchParams(window.location.search)
  return params.get('code')
}

export function saveSession(data) {
  try {
    const session = normalizeSession(data)

    if (!session?.token) {
      throw new Error('Missing session token in login response')
    }

    localStorage.setItem(TOKEN_KEY, JSON.stringify(session))
  } catch (err) {
    console.error('Error saving session:', err)
  }
}

export function getSession() {
  try {
    const data = localStorage.getItem(TOKEN_KEY)
    return data ? normalizeSession(JSON.parse(data)) : null
  } catch {
    return null
  }
}

export function clearSession() {
  localStorage.removeItem(TOKEN_KEY)
  sessionStorage.removeItem(GOOGLE_REDIRECT_KEY)
}

export function isLoggedIn() {
  const session = getSession()
  return Boolean(session?.token || session?.api_key)
}

// ── API-key sign-in ──────────────────────────────────────────────────────────
//
// A second way into the dashboard that does not depend on Google.
//
// The backend has always accepted `X-API-Key` (app/auth_guard.py::resolve_user
// takes either a bearer token or a key), but the frontend only ever offered
// Google. That made an outage in Google's consent-screen configuration a total
// lockout from your own operator console — including the screens you need to
// diagnose the outage.
//
// Keys are per-tenant and issued by the backend; this only stores one the user
// already has. It is not a bypass of authentication, just a different
// credential for the same check.

export function saveApiKeySession(apiKey, profile = {}) {
  const session = {
    ...profile,
    api_key: apiKey,
    token: '',            // no JWT — requests authenticate with the key itself
    auth_method: 'api_key',
  }
  localStorage.setItem(TOKEN_KEY, JSON.stringify(session))
  return session
}

/**
 * Auth headers for the current session, whichever credential it holds.
 * Returns {} when signed out, so callers can spread it unconditionally.
 */
export function getAuthHeaders() {
  const s = getSession()
  if (!s) return {}
  if (s.token) return { Authorization: `Bearer ${s.token}` }
  if (s.api_key) return { 'X-API-Key': s.api_key }
  return {}
}
