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
  }
}

function getCurrentRedirectUri() {
  if (typeof window === 'undefined') {
    return import.meta.env.VITE_REDIRECT_URI || ''
  }

  return `${window.location.origin}${window.location.pathname}`
}

export function getGoogleRedirectUri() {
  const savedRedirectUri = sessionStorage.getItem(GOOGLE_REDIRECT_KEY)
  return savedRedirectUri || import.meta.env.VITE_REDIRECT_URI || getCurrentRedirectUri()
}

export function getGoogleAuthUrl() {
  const redirectUri = import.meta.env.VITE_REDIRECT_URI || getCurrentRedirectUri()
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
  return Boolean(session?.token)
}
