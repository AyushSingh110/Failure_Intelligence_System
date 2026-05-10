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
      borderTop: `2px solid ${color}`,
      animation: `kpiIn 0.5s ease ${delay}ms both`,
      position: 'relative', overflow: 'hidden',
    }}>
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: '48px',
        background: `linear-gradient(180deg, ${color}10 0%, transparent 100%)`,
        pointerEvents: 'none',
      }}/>
      <div style={{
        fontFamily: 'JetBrains Mono, monospace', fontSize: '9px',
        fontWeight: 600, letterSpacing: '0.14em',
        color: 'var(--text-muted)', marginBottom: '10px',
      }}>{label}</div>
      <div style={{
        fontFamily: 'JetBrains Mono, monospace', fontSize: '28px',
        fontWeight: 800, color, lineHeight: 1, marginBottom: '6px',
      }}>{value}</div>
      <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{sub}</div>
    </div>
  )
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--bg-card)', border: '1px solid var(--border)',
      borderRadius: '8px', padding: '10px 14px',
      fontFamily: 'JetBrains Mono, monospace', fontSize: '12px',
    }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: '6px' }}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color || p.stroke, marginBottom: '2px' }}>
          {p.dataKey}: {typeof p.value === 'number' ? p.value.toFixed(3) : p.value}
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

  // Entropy distribution buckets
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
          <h1 style={{ fontSize: '22px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px' }}>
            Analytics
          </h1>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
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
      </div>
    </>
  )
}
