import { useState, useEffect } from 'react'
import {
  AreaChart, Area, BarChart, Bar, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Legend,
} from 'recharts'
import { useInferences, useTrend, computeKPIs, buildTimeSeries } from '../hooks/useData.js'
import { getSession } from '../lib/auth.js'

const BASE = (import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1').replace(/\/$/, '')

const ARCHETYPE_COLOR = {
  STABLE:             '#00ff88',
  HALLUCINATION_RISK: '#ff4466',
  MODEL_BLIND_SPOT:   '#ff4466',
  UNSTABLE_OUTPUT:    '#ffaa00',
  LOW_CONFIDENCE:     '#ffaa00',
}

function deriveArchetype(record) {
  if (record.archetype) return record.archetype
  const entropy   = record.metrics?.entropy || 0
  const agreement = record.metrics?.agreement_score || 0
  const highRisk  = record.metrics?.high_failure_risk === true
  const isAttack  = record.is_adversarial === true || record.adversarial?.is_attack === true
  if (isAttack)                          return 'MODEL_BLIND_SPOT'
  if (entropy >= 0.75 || highRisk)       return 'HALLUCINATION_RISK'
  if (entropy >= 0.4 && agreement < 0.5) return 'UNSTABLE_OUTPUT'
  if (agreement < 0.5)                   return 'LOW_CONFIDENCE'
  return 'STABLE'
}

function StatCard({ label, value, sub, color, delay = 0 }) {
  return (
    <div style={{
      padding: '18px 20px', borderRadius: '12px',
      background: 'var(--bg-card)', border: '1px solid var(--border)',
      animation: `kpiIn 0.5s ease ${delay}ms both`,
      position: 'relative', overflow: 'hidden',
      transition: 'border-color 0.2s, box-shadow 0.2s',
    }}
    onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--border-bright)'; e.currentTarget.style.boxShadow = '0 4px 20px rgba(0,0,0,0.3)' }}
    onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.boxShadow = 'none' }}
    >
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: '2px',
        background: color, opacity: 0.8,
      }}/>
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: '56px',
        background: `linear-gradient(180deg, ${color}0d 0%, transparent 100%)`,
        pointerEvents: 'none',
      }}/>
      <div style={{
        fontFamily: 'JetBrains Mono, monospace', fontSize: '9px',
        fontWeight: 700, letterSpacing: '0.14em',
        color: 'var(--text-muted)', marginBottom: '12px',
      }}>{label}</div>
      <div style={{
        fontFamily: 'JetBrains Mono, monospace', fontSize: '28px',
        fontWeight: 800, color: 'var(--text-primary)', lineHeight: 1, marginBottom: '7px',
      }}>{value}</div>
      <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color: 'var(--text-muted)' }}>{sub}</div>
    </div>
  )
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--bg-elevated)', border: '1px solid var(--border-bright)',
      borderRadius: '8px', padding: '10px 14px',
      fontFamily: 'JetBrains Mono, monospace', fontSize: '11px',
      boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
    }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: '7px', fontSize: '10px' }}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ display: 'flex', justifyContent: 'space-between', gap: '14px', marginBottom: '3px' }}>
          <span style={{ color: 'var(--text-muted)' }}>{p.dataKey}</span>
          <span style={{ color: p.color || p.stroke }}>{typeof p.value === 'number' ? p.value.toFixed(3) : p.value}</span>
        </div>
      ))}
    </div>
  )
}

function SectionLabel({ children }) {
  return (
    <div style={{
      fontFamily: 'JetBrains Mono, monospace', fontSize: '10px',
      fontWeight: 600, letterSpacing: '0.12em',
      color: 'var(--text-muted)', marginBottom: '16px',
    }}>{children}</div>
  )
}

// ── Percentile helper ────────────────────────────────────────────────────────
function pct(sorted, p) {
  if (!sorted.length) return 0
  return sorted[Math.min(Math.floor(sorted.length * p), sorted.length - 1)]
}

export default function AnalyticsPage() {
  const { data: inferences, loading } = useInferences()
  const { data: trend }               = useTrend()
  const [usage, setUsage]             = useState(null)
  const kpis                          = computeKPIs(inferences)
  const timeSeries                    = buildTimeSeries(inferences)

  useEffect(() => {
    const session = getSession()
    if (!session?.token) return
    fetch(`${BASE}/analytics/usage`, {
      headers: { Authorization: `Bearer ${session.token}` },
    })
      .then(r => r.ok ? r.json() : null)
      .then(data => setUsage(data))
      .catch(() => {})
  }, [])

  // Archetype distribution
  const archetypeCounts = inferences.reduce((acc, r) => {
    const a = deriveArchetype(r)
    acc[a] = (acc[a] || 0) + 1
    return acc
  }, {})
  const archetypeData = Object.entries(archetypeCounts)
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => ({
      name: name.replace(/_/g, ' '),
      count,
      color: ARCHETYPE_COLOR[name] || '#3d5166',
    }))

  // Model breakdown
  const modelCounts = inferences.reduce((acc, r) => {
    const m = r.model_name || 'unknown'
    acc[m] = (acc[m] || 0) + 1
    return acc
  }, {})
  const modelData = Object.entries(modelCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([name, count]) => ({ name: name.slice(0, 20), count }))

  // ── Confidence histogram ──────────────────────────────────────────────────
  const confBuckets = Array.from({ length: 10 }, (_, i) => ({
    range: `${i * 10}–${(i + 1) * 10}%`,
    count: 0,
    color: i < 3 ? '#ff4466' : i < 5 ? '#ffaa00' : i < 7 ? '#00d4ff' : '#00ff88',
  }))
  inferences.forEach(r => {
    const c = r.metrics?.confidence ?? r.metrics?.classifier_confidence ?? null
    if (c === null) return
    const idx = Math.min(Math.floor(c * 10), 9)
    confBuckets[idx].count++
  })

  // ── Attack trend by day ───────────────────────────────────────────────────
  const byDay = {}
  inferences.forEach(r => {
    if (!r.timestamp) return
    const day = new Date(r.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
    if (!byDay[day]) byDay[day] = { day, attacks: 0, safe: 0, risk: 0 }
    if (r.is_adversarial === true || r.adversarial?.is_attack === true) byDay[day].attacks++
    else if ((r.metrics?.entropy || 0) > 0.75) byDay[day].risk++
    else byDay[day].safe++
  })
  const trendData = Object.values(byDay).slice(-14)

  // ── Latency percentiles ───────────────────────────────────────────────────
  const latencies = inferences
    .map(r => r.metrics?.latency_ms ?? r.latency_ms ?? r.total_latency_ms ?? null)
    .filter(v => v !== null && v > 0)
    .sort((a, b) => a - b)
  const latP50 = pct(latencies, 0.50)
  const latP95 = pct(latencies, 0.95)
  const latP99 = pct(latencies, 0.99)

  // ── Low-confidence detections ─────────────────────────────────────────────
  const suspectDetections = inferences
    .filter(r => (r.is_adversarial === true || r.adversarial?.is_attack === true))
    .map(r => ({
      ...r,
      _conf: r.metrics?.confidence ?? r.metrics?.classifier_confidence ?? 1,
    }))
    .filter(r => r._conf < 0.65)
    .sort((a, b) => a._conf - b._conf)
    .slice(0, 15)

  // ── Entropy distribution buckets
  const entropyBuckets = [
    { range: '0.0–0.2', count: 0, color: '#00ff88' },
    { range: '0.2–0.4', count: 0, color: '#00d4ff' },
    { range: '0.4–0.6', count: 0, color: '#ffaa00' },
    { range: '0.6–0.8', count: 0, color: '#ff6600' },
    { range: '0.8–1.0', count: 0, color: '#ff4466' },
  ]
  inferences.forEach(r => {
    const e = r.metrics?.entropy || 0
    if (e < 0.2)      entropyBuckets[0].count++
    else if (e < 0.4) entropyBuckets[1].count++
    else if (e < 0.6) entropyBuckets[2].count++
    else if (e < 0.8) entropyBuckets[3].count++
    else              entropyBuckets[4].count++
  })

  if (loading) return (
    <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: 'JetBrains Mono, monospace', fontSize: '13px', color: 'var(--text-muted)' }}>
      Loading analytics...
    </div>
  )

  return (
    <>
      <style>{`
        @keyframes kpiIn { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:translateY(0); } }
      `}</style>

      <div style={{ flex: 1, padding: '28px 32px', overflowY: 'auto' }}>

        {/* Header */}
        <div style={{ marginBottom: '24px', animation: 'kpiIn 0.5s ease both' }}>
          <h1 style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px', letterSpacing: '-0.02em' }}>
            Analytics
          </h1>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace' }}>
            Failure patterns, model health, and usage breakdown
          </p>
        </div>

        {/* KPI row */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginBottom: '24px' }}>
          <StatCard label="TOTAL INFERENCES"  value={kpis.total}                       sub="all time"              color="var(--accent-cyan)"  delay={0}   />
          <StatCard label="HIGH RISK"          value={kpis.highRisk}                    sub={`${kpis.riskPct}% failure rate`} color={kpis.riskPct > 30 ? 'var(--accent-red)' : 'var(--accent-amber)'} delay={60}  />
          <StatCard label="ATTACKS BLOCKED"    value={kpis.attacks}                     sub="adversarial prompts"   color={kpis.attacks > 0 ? 'var(--accent-red)' : 'var(--accent-green)'} delay={120} />
          <StatCard label="AUTO-FIXES APPLIED" value={kpis.fixApplied}                  sub="corrections issued"    color="var(--accent-cyan)"  delay={180} />
        </div>

        {/* Usage quota (from /analytics/usage) */}
        {usage && (
          <div style={{
            padding: '16px 20px', borderRadius: '12px',
            background: 'var(--bg-card)', border: '1px solid var(--border)',
            marginBottom: '24px',
            animation: 'kpiIn 0.5s ease 0.2s both',
          }}>
            <SectionLabel>API USAGE THIS MONTH</SectionLabel>
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
              <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                  <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', color: 'var(--text-muted)' }}>
                    {usage.calls_used?.toLocaleString() || 0} / {usage.calls_limit?.toLocaleString() || 1000} calls
                  </span>
                  <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '12px',
                    color: (usage.calls_used / usage.calls_limit) > 0.8 ? 'var(--accent-red)' : 'var(--accent-green)' }}>
                    {Math.round((usage.calls_used / usage.calls_limit) * 100) || 0}%
                  </span>
                </div>
                <div style={{ height: '6px', borderRadius: '3px', background: 'rgba(255,255,255,0.06)' }}>
                  <div style={{
                    height: '100%', borderRadius: '3px',
                    width: `${Math.min((usage.calls_used / usage.calls_limit) * 100, 100) || 0}%`,
                    background: (usage.calls_used / usage.calls_limit) > 0.8
                      ? 'var(--accent-red)' : 'var(--accent-cyan)',
                    transition: 'width 1s ease',
                  }}/>
                </div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', color: 'var(--text-muted)', marginBottom: '2px' }}>PLAN</div>
                <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '13px', fontWeight: 700,
                  color: 'var(--accent-cyan)', textTransform: 'uppercase' }}>{usage.plan || 'free'}</div>
              </div>
            </div>
          </div>
        )}

        {/* Charts row 1 */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px', marginBottom: '14px' }}>

          {/* Signal time series */}
          <div style={{
            padding: '20px 22px', borderRadius: '12px',
            background: 'var(--bg-card)', border: '1px solid var(--border)',
            animation: 'kpiIn 0.5s ease 0.25s both',
          }}>
            <SectionLabel>ENTROPY & AGREEMENT OVER TIME</SectionLabel>
            {timeSeries.length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={timeSeries} margin={{ top: 4, right: 4, left: -22, bottom: 0 }}>
                  <defs>
                    <linearGradient id="gE" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ff4466" stopOpacity={0.15}/>
                      <stop offset="95%" stopColor="#ff4466" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="gA" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#00ff88" stopOpacity={0.12}/>
                      <stop offset="95%" stopColor="#00ff88" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="rgba(255,255,255,0.03)" strokeDasharray="4 4"/>
                  <XAxis dataKey="time" tick={{ fontFamily: 'JetBrains Mono', fontSize: 9, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false} interval="preserveStartEnd"/>
                  <YAxis domain={[0,1]} tick={{ fontFamily: 'JetBrains Mono', fontSize: 9, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false}/>
                  <Tooltip content={<ChartTooltip />}/>
                  <Area type="monotone" dataKey="entropy"   stroke="#ff4466" strokeWidth={1.5} fill="url(#gE)" dot={false}/>
                  <Area type="monotone" dataKey="agreement" stroke="#00ff88" strokeWidth={1.5} fill="url(#gA)" dot={false}/>
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace', fontSize: '12px' }}>
                No data yet
              </div>
            )}
            <div style={{ display: 'flex', gap: 16, marginTop: 10 }}>
              {[['#ff4466','Entropy'],['#00ff88','Agreement']].map(([c,l]) => (
                <div key={l} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <div style={{ width: 14, height: 2, background: c, borderRadius: 1 }}/>
                  <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: 'var(--text-muted)' }}>{l}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Entropy distribution */}
          <div style={{
            padding: '20px 22px', borderRadius: '12px',
            background: 'var(--bg-card)', border: '1px solid var(--border)',
            animation: 'kpiIn 0.5s ease 0.3s both',
          }}>
            <SectionLabel>ENTROPY DISTRIBUTION</SectionLabel>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={entropyBuckets} margin={{ top: 4, right: 4, left: -22, bottom: 0 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.03)" strokeDasharray="4 4" vertical={false}/>
                <XAxis dataKey="range" tick={{ fontFamily: 'JetBrains Mono', fontSize: 9, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false}/>
                <YAxis tick={{ fontFamily: 'JetBrains Mono', fontSize: 9, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false}/>
                <Tooltip content={<ChartTooltip />}/>
                <Bar dataKey="count" radius={[4,4,0,0]} animationDuration={1200}>
                  {entropyBuckets.map((e, i) => <Cell key={i} fill={e.color} opacity={0.85}/>)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Charts row 2 */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px', marginBottom: '14px' }}>

          {/* Failure archetypes */}
          <div style={{
            padding: '20px 22px', borderRadius: '12px',
            background: 'var(--bg-card)', border: '1px solid var(--border)',
            animation: 'kpiIn 0.5s ease 0.35s both',
          }}>
            <SectionLabel>FAILURE ARCHETYPES</SectionLabel>
            {archetypeData.length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={archetypeData} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 0 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.03)" strokeDasharray="4 4" horizontal={false}/>
                  <XAxis type="number" tick={{ fontFamily: 'JetBrains Mono', fontSize: 9, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false}/>
                  <YAxis type="category" dataKey="name" tick={{ fontFamily: 'JetBrains Mono', fontSize: 9, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false} width={110}/>
                  <Tooltip content={<ChartTooltip />}/>
                  <Bar dataKey="count" radius={[0,4,4,0]} animationDuration={1200}>
                    {archetypeData.map((e,i) => <Cell key={i} fill={e.color} opacity={0.8}/>)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace', fontSize: '12px' }}>
                No data yet
              </div>
            )}
          </div>

          {/* Model breakdown */}
          <div style={{
            padding: '20px 22px', borderRadius: '12px',
            background: 'var(--bg-card)', border: '1px solid var(--border)',
            animation: 'kpiIn 0.5s ease 0.4s both',
          }}>
            <SectionLabel>INFERENCES BY MODEL</SectionLabel>
            {modelData.length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={modelData} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 0 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.03)" strokeDasharray="4 4" horizontal={false}/>
                  <XAxis type="number" tick={{ fontFamily: 'JetBrains Mono', fontSize: 9, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false}/>
                  <YAxis type="category" dataKey="name" tick={{ fontFamily: 'JetBrains Mono', fontSize: 9, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false} width={120}/>
                  <Tooltip content={<ChartTooltip />}/>
                  <Bar dataKey="count" fill="var(--accent-cyan)" opacity={0.75} radius={[0,4,4,0]} animationDuration={1200}/>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace', fontSize: '12px' }}>
                No data yet
              </div>
            )}
          </div>
        </div>

        {/* EMA health */}
        {trend && (
          <div style={{
            padding: '20px 22px', borderRadius: '12px',
            background: 'var(--bg-card)', border: '1px solid var(--border)',
            animation: 'kpiIn 0.5s ease 0.45s both',
          }}>
            <SectionLabel>EMA HEALTH METRICS</SectionLabel>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
              {[
                { label: 'EMA ENTROPY',    val: trend.ema_entropy || 0,        threshold: 0.75, bad: true },
                { label: 'EMA AGREEMENT',  val: trend.ema_agreement || 0,      threshold: 0.5,  bad: false },
                { label: 'HIGH RISK RATE', val: trend.ema_high_risk_rate || 0, threshold: 0.4,  bad: true },
                { label: 'DEG. VELOCITY',  val: trend.degradation_velocity || 0, threshold: 0.05, bad: true },
              ].map(({ label, val, threshold, bad }) => {
                const warn  = bad ? val > threshold : val < threshold
                const color = warn ? 'var(--accent-red)' : 'var(--accent-green)'
                return (
                  <div key={label} style={{
                    padding: '14px 16px', borderRadius: '10px',
                    background: warn ? 'rgba(255,68,102,0.05)' : 'rgba(0,255,136,0.03)',
                    border: `1px solid ${warn ? 'rgba(255,68,102,0.2)' : 'var(--border)'}`,
                  }}>
                    <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '9px', letterSpacing: '0.12em', color: 'var(--text-muted)', marginBottom: '8px' }}>{label}</div>
                    <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '20px', fontWeight: 800, color, marginBottom: '4px' }}>{val.toFixed(3)}</div>
                    <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', color: warn ? color : 'var(--text-muted)' }}>
                      {warn ? '⚠ ABOVE THRESHOLD' : '✓ WITHIN RANGE'}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* ── NEW: Confidence histogram + Attack trend ─────────────── */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px', marginTop: '14px' }}>

          {/* Confidence histogram */}
          <div style={{ padding: '20px 22px', borderRadius: '12px', background: 'var(--bg-card)', border: '1px solid var(--border)', animation: 'kpiIn 0.5s ease 0.5s both' }}>
            <SectionLabel>DETECTION CONFIDENCE DISTRIBUTION</SectionLabel>
            <p style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', color: 'var(--text-muted)', marginBottom: '14px', lineHeight: 1.6 }}>
              Histogram of classifier confidence across all flagged inferences. Low bars at high confidence may indicate under-triggering.
            </p>
            {confBuckets.some(b => b.count > 0) ? (
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={confBuckets} margin={{ top: 4, right: 4, left: -22, bottom: 0 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.03)" strokeDasharray="4 4" vertical={false}/>
                  <XAxis dataKey="range" tick={{ fontFamily: 'JetBrains Mono', fontSize: 8, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false} angle={-20} textAnchor="end"/>
                  <YAxis tick={{ fontFamily: 'JetBrains Mono', fontSize: 9, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false}/>
                  <Tooltip content={<ChartTooltip />}/>
                  <Bar dataKey="count" radius={[4,4,0,0]} animationDuration={900}>
                    {confBuckets.map((b, i) => <Cell key={i} fill={b.color} opacity={0.85}/>)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ height: 180, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace', fontSize: '12px' }}>
                No confidence data yet
              </div>
            )}
          </div>

          {/* Attack trend by day */}
          <div style={{ padding: '20px 22px', borderRadius: '12px', background: 'var(--bg-card)', border: '1px solid var(--border)', animation: 'kpiIn 0.5s ease 0.55s both' }}>
            <SectionLabel>DAILY THREAT TREND — LAST 14 DAYS</SectionLabel>
            <p style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', color: 'var(--text-muted)', marginBottom: '14px', lineHeight: 1.6 }}>
              Breakdown of adversarial attacks, high-risk outputs, and safe inferences per day.
            </p>
            {trendData.length > 0 ? (
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={trendData} margin={{ top: 4, right: 4, left: -22, bottom: 0 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.03)" strokeDasharray="4 4" vertical={false}/>
                  <XAxis dataKey="day" tick={{ fontFamily: 'JetBrains Mono', fontSize: 8, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false}/>
                  <YAxis tick={{ fontFamily: 'JetBrains Mono', fontSize: 9, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false}/>
                  <Tooltip content={<ChartTooltip />}/>
                  <Bar dataKey="attacks" stackId="a" fill="#ff4466" opacity={0.85} radius={[0,0,0,0]}/>
                  <Bar dataKey="risk"    stackId="a" fill="#ffaa00" opacity={0.85} radius={[0,0,0,0]}/>
                  <Bar dataKey="safe"   stackId="a" fill="#00ff88" opacity={0.6}  radius={[4,4,0,0]}/>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ height: 180, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace', fontSize: '12px' }}>
                No data yet
              </div>
            )}
            <div style={{ display: 'flex', gap: 14, marginTop: 10 }}>
              {[['#ff4466','Attacks'],['#ffaa00','High risk'],['#00ff88','Safe']].map(([c,l]) => (
                <div key={l} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                  <div style={{ width: 10, height: 10, borderRadius: 2, background: c, opacity: 0.85 }}/>
                  <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, color: 'var(--text-muted)' }}>{l}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ── NEW: Latency percentiles ──────────────────────────────── */}
        {latencies.length > 0 && (
          <div style={{ padding: '20px 22px', borderRadius: '12px', background: 'var(--bg-card)', border: '1px solid var(--border)', marginTop: '14px', animation: 'kpiIn 0.5s ease 0.6s both' }}>
            <SectionLabel>PIPELINE LATENCY PERCENTILES</SectionLabel>
            <p style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', color: 'var(--text-muted)', marginBottom: '18px', lineHeight: 1.6 }}>
              Computed from {latencies.length} inferences with latency data. Stage-level breakdown (preflight / shadow / jury) requires backend instrumentation.
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
              {[
                { label: 'P50 (MEDIAN)', val: latP50, color: '#00ff88' },
                { label: 'P95',          val: latP95, color: '#ffaa00' },
                { label: 'P99',          val: latP99, color: '#ff4466' },
              ].map(({ label, val, color }) => (
                <div key={label} style={{ padding: '16px 18px', borderRadius: '10px', background: 'rgba(0,0,0,0.2)', border: '1px solid var(--border)', textAlign: 'center' }}>
                  <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '9px', letterSpacing: '0.14em', color: 'var(--text-muted)', marginBottom: '10px' }}>{label}</div>
                  <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '28px', fontWeight: 800, color, letterSpacing: '-0.03em', lineHeight: 1 }}>
                    {val ? `${val.toFixed(0)}` : '—'}
                  </div>
                  <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', color: 'var(--text-muted)', marginTop: '6px' }}>ms</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── NEW: Low-confidence detections ───────────────────────── */}
        {suspectDetections.length > 0 && (
          <div style={{ padding: '20px 22px', borderRadius: '12px', background: 'var(--bg-card)', border: '1px solid rgba(255,170,0,0.2)', marginTop: '14px', animation: 'kpiIn 0.5s ease 0.65s both' }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '8px' }}>
              <SectionLabel>LOW-CONFIDENCE DETECTIONS</SectionLabel>
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '9px', padding: '2px 8px', borderRadius: '4px', background: 'rgba(255,170,0,0.1)', color: '#ffaa00', border: '1px solid rgba(255,170,0,0.2)' }}>
                confidence &lt; 65%
              </span>
            </div>
            <p style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', color: 'var(--text-muted)', marginBottom: '16px', lineHeight: 1.6 }}>
              Flagged prompts where the classifier was uncertain. These are the most likely false positives — review them to tune your pipeline.
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: '10px 1fr auto auto auto', gap: '10px', padding: '0 8px 8px', borderBottom: '1px solid var(--border)', marginBottom: '6px' }}>
              {['', 'PROMPT', 'CONF', 'ENTROPY', 'TIME'].map((h, i) => (
                <div key={i} style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '9px', color: 'var(--text-muted)', letterSpacing: '0.1em', textAlign: i > 1 ? 'right' : 'left' }}>{h}</div>
              ))}
            </div>
            {suspectDetections.map((r, i) => {
              const conf = r._conf
              const confColor = conf < 0.4 ? '#ff4466' : conf < 0.55 ? '#ffaa00' : '#6e90b0'
              const entropy = r.metrics?.entropy || 0
              const time = r.timestamp ? new Date(r.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''
              return (
                <div key={r.request_id || i} style={{
                  display: 'grid', gridTemplateColumns: '10px 1fr auto auto auto', gap: '10px',
                  padding: '8px 8px', borderRadius: '6px', marginBottom: '2px',
                  background: i % 2 === 0 ? 'rgba(255,255,255,0.015)' : 'transparent',
                  transition: 'background 0.1s',
                }}
                onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,170,0,0.04)'}
                onMouseLeave={e => e.currentTarget.style.background = i % 2 === 0 ? 'rgba(255,255,255,0.015)' : 'transparent'}
                >
                  <div style={{ width: 6, height: 6, borderRadius: '50%', background: confColor, marginTop: 5, boxShadow: `0 0 4px ${confColor}` }}/>
                  <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {(r.input_text || '').slice(0, 90)}
                  </div>
                  <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', fontWeight: 700, color: confColor, textAlign: 'right' }}>
                    {(conf * 100).toFixed(0)}%
                  </div>
                  <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color: entropy > 0.75 ? 'var(--accent-red)' : 'var(--text-muted)', textAlign: 'right' }}>
                    {entropy.toFixed(3)}
                  </div>
                  <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', color: 'var(--text-muted)', textAlign: 'right' }}>{time}</div>
                </div>
              )
            })}
          </div>
        )}

      </div>
    </>
  )
}
