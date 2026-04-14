/**
 * useData.js - Data fetching hooks
 * Polls FastAPI backend every 10 seconds
 */
import { useState, useEffect, useCallback } from 'react'
import { api } from '../lib/api'
import { getSession } from '../lib/auth'

export function useInferences() {
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetch = useCallback(async () => {
    try {
      const session = getSession()
      const res = await api.getInferences(session?.token)
      setData(res || [])
      setError(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetch()
    const interval = setInterval(fetch, 10000)
    return () => clearInterval(interval)
  }, [fetch])

  return { data, loading, error, refetch: fetch }
}

export function useTrend() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  const fetch = useCallback(async () => {
    try {
      const session = getSession()
      const res = await api.getTrend(session?.token)
      setData(res)
    } catch {
      setData(null)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetch()
    const interval = setInterval(fetch, 10000)
    return () => clearInterval(interval)
  }, [fetch])

  return { data, loading }
}

export function computeKPIs(inferences) {
  if (!inferences.length) {
    return {
      total: 0,
      highRisk: 0,
      riskPct: 0,
      avgEntropy: 0,
      avgAgreement: 0,
    }
  }

  const metrics = inferences.map((record) => record.metrics || {})
  const entropies = metrics.map((metric) => metric.entropy || 0).filter(Boolean)
  const agreements = metrics.map((metric) => metric.agreement_score || 0).filter(Boolean)

  const highRisk = inferences.filter((record) => (record.metrics?.entropy || 0) > 0.75).length

  return {
    total: inferences.length,
    highRisk,
    riskPct: Math.round((highRisk / inferences.length) * 100),
    avgEntropy: entropies.length
      ? +(entropies.reduce((sum, value) => sum + value, 0) / entropies.length).toFixed(3)
      : 0,
    avgAgreement: agreements.length
      ? +(agreements.reduce((sum, value) => sum + value, 0) / agreements.length).toFixed(3)
      : 0,
  }
}

export function buildTimeSeries(inferences) {
  const sorted = [...inferences]
    .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))
    .slice(-30)

  return sorted.map((record, index) => ({
    index: index + 1,
    entropy: +(record.metrics?.entropy || 0).toFixed(3),
    agreement: +(record.metrics?.agreement_score || 0).toFixed(3),
    time: record.timestamp
      ? new Date(record.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      : '',
  }))
}
