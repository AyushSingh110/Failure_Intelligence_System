/**
 * api.js - All FastAPI backend calls
 * Base URL reads from env variable
 */

import { demoResponseFor } from './demoData'

const BASE = (import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1').replace(/\/$/, '')

/**
 * DEMO-MODE FALLBACK
 * ------------------
 * The dashboard is hosted separately from the API. When the API is unreachable
 * — billing lapsed, instance asleep, cold start, network blip — every panel
 * used to throw and the visitor saw a broken application.
 *
 * That is worse than having no demo at all: it is the first thing a prospective
 * user clicks, and it makes a working project look abandoned.
 *
 * Read-only GETs now fall back to a bundled fixture and set `isBackendOffline`,
 * which the UI surfaces as a persistent "demo data" banner. Two rules keep this
 * honest:
 *
 *   1. Only GETs fall back. A POST that appears to succeed while the backend is
 *      down would be a lie about a write that never happened — scans, feedback
 *      and diagnoses still fail loudly.
 *   2. Real HTTP errors (401, 403, 422, 500) are NOT masked. Only transport
 *      failures and 5xx/429 fall back, so a genuine auth or validation bug is
 *      still visible instead of being hidden behind sample data.
 */

let backendOffline = false
const listeners = new Set()

export function isBackendOffline() {
  return backendOffline
}

/** Subscribe to offline-state changes. Returns an unsubscribe function. */
export function onBackendStatusChange(fn) {
  listeners.add(fn)
  return () => listeners.delete(fn)
}

function setOffline(value) {
  if (backendOffline === value) return
  backendOffline = value
  listeners.forEach((fn) => {
    try {
      fn(value)
    } catch {
      /* a broken listener must not break the data path */
    }
  })
}

async function request(method, path, body = null, token = null) {
  const headers = { 'Content-Type': 'application/json' }
  if (token) headers['Authorization'] = `Bearer ${token}`

  let res
  try {
    res = await fetch(`${BASE}${path}`, {
      method,
      headers,
      body: body ? JSON.stringify(body) : null,
    })
  } catch (networkErr) {
    // Transport-level failure: DNS, CORS, connection refused, offline.
    const demo = method === 'GET' ? demoResponseFor(path) : undefined
    if (demo !== undefined) {
      setOffline(true)
      return demo
    }
    throw new Error(`Backend unreachable: ${networkErr.message}`)
  }

  if (!res.ok) {
    // 5xx and 429 mean the server is unhealthy, not that the request was wrong.
    // 4xx (except 429) is a genuine client error and must stay visible.
    const serverUnhealthy = res.status >= 500 || res.status === 429
    const demo = method === 'GET' && serverUnhealthy ? demoResponseFor(path) : undefined
    if (demo !== undefined) {
      setOffline(true)
      return demo
    }
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Request failed')
  }

  setOffline(false)
  return res.json()
}

export const api = {
  // Auth — login goes through the Google OAuth code flow only
  // (see auth.js getGoogleAuthUrl + POST /auth/google-callback).
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

  // Export
  exportCsv: async (token) => {
    const headers = { 'Content-Type': 'application/json' }
    if (token) headers['Authorization'] = `Bearer ${token}`
    const res = await fetch(`${BASE}/inferences/export/csv`, { headers })
    if (!res.ok) throw new Error('Export failed')
    return res.blob()
  },
}
