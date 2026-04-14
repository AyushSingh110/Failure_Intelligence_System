/**
 * api.js - All FastAPI backend calls
 * Base URL reads from env variable
 */

const BASE = (import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1').replace(/\/$/, '')

async function request(method, path, body = null, token = null) {
  const headers = { 'Content-Type': 'application/json' }
  if (token) headers['Authorization'] = `Bearer ${token}`

  const res = await fetch(`${BASE}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : null,
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Request failed')
  }
  return res.json()
}

export const api = {
  // Auth
  loginGoogle: (email, name, picture) =>
    request('POST', '/auth/google', { email, name, picture }),

  getMe: (token) =>
    request('GET', '/auth/me', null, token),

  regenerateKey: (token) =>
    request('POST', '/auth/regenerate-key', null, token),

  // Inferences
  getInferences: (token) =>
    request('GET', '/inferences', null, token),

  getTrend: (token) =>
    request('GET', '/trend', null, token),

  // Monitor
  postMonitor: (body, token) =>
    request('POST', '/monitor', body, token),

  // Analyze
  analyzeOutputs: (model_outputs, token) =>
    request('POST', '/analyze', { model_outputs }, token),

  // Diagnose
  runDiagnose: (body, token) =>
    request('POST', '/diagnose', body, token),
}
