// DashboardPage.jsx
import { useState, useEffect, useRef } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, BarChart, Bar, Cell,
} from 'recharts'
import { useInferences, useTrend, computeKPIs, buildTimeSeries } from '../hooks/useData.js'

// ── Animated counter ──────────────────────────────────────────────────────
function Counter({ to, suffix = '', decimals = 0, duration = 1200 }) {
  const [val, setVal] = useState(0)
  const ref = useRef(null)

  useEffect(() => {
    if (!to) return
    let start = null
    const step = (ts) => {
      if (!start) start = ts
      const progress = Math.min((ts - start) / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3)
      setVal(parseFloat((eased * to).toFixed(decimals)))
      if (progress < 1) ref.current = requestAnimationFrame(step)
    }
    ref.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(ref.current)
  }, [to, duration, decimals])

  return <>{decimals ? val.toFixed(decimals) : Math.round(val)}{suffix}</>
}

// ── KPI Card ──────────────────────────────────────────────────────────────
function KPICard({ label, value, suffix, decimals, sub, color, icon, delay }) {
  return (
    <div
      style={{
        padding: '20px 22px',
        borderRadius: '14px',
        background: 'var(--bg-card)',
        border: `1px solid var(--border)`,
        borderTop: `2px solid ${color}`,
        animation: `kpiIn 0.55s cubic-bezier(0.16,1,0.3,1) ${delay}ms both`,
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Background glow */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: '60px',
        background: `linear-gradient(180deg, ${color}12 0%, transparent 100%)`,
        pointerEvents: 'none',
      }}/>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
        <span style={{
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: '10px', fontWeight: 600,
          letterSpacing: '0.12em',
          color: 'var(--text-muted)',
        }}>{label}</span>
        <span style={{ color, opacity: 0.7 }}>{icon}</span>
      </div>

      <div style={{
        fontSize: '28px', fontWeight: 800,
        fontFamily: 'JetBrains Mono, monospace',
        color: 'var(--text-primary)',
        letterSpacing: '-0.02em',
        lineHeight: 1,
        marginBottom: '6px',
      }}>
        <Counter to={value} suffix={suffix} decimals={decimals} />
      </div>

      <div style={{
        fontSize: '12px',
        color: 'var(--text-muted)',
      }}>{sub}</div>
    </div>
  )
}

// ── Archetype badge ───────────────────────────────────────────────────────
const ARCHETYPE_COLOR = {
  STABLE:                '#00ff88',
  HALLUCINATION_RISK:    '#ff4466',
  MODEL_BLIND_SPOT:      '#ff4466',
  OVERCONFIDENT_FAILURE: '#ff4466',
  UNSTABLE_OUTPUT:       '#ffaa00',
  LOW_CONFIDENCE:        '#ffaa00',
  TEMPORAL_KNOWLEDGE_CUTOFF: '#ffaa00',
}

function ArchetypeBadge({ type }) {
  const color = ARCHETYPE_COLOR[type] || '#3d5166'
  const short = type?.replace(/_/g, ' ') || 'UNKNOWN'
  return (
    <span style={{
      padding: '2px 8px',
      borderRadius: '4px',
      fontFamily: 'JetBrains Mono, monospace',
      fontSize: '10px', fontWeight: 600,
      background: `${color}18`,
      color,
      border: `1px solid ${color}30`,
      whiteSpace: 'nowrap',
    }}>{short}</span>
  )
}

// ── Inference Row ─────────────────────────────────────────────────────────
function InferenceRow({ record, delay }) {
  const entropy   = record.metrics?.entropy || 0
  const isRisk    = entropy > 0.75
  const prompt    = record.input_text || ''
  const model     = record.model_name || 'unknown'
  const archetype = record.archetype || 'STABLE'
  const time      = record.timestamp
    ? new Date(record.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : ''

  return (
    <div
      style={{
        display: 'flex', alignItems: 'center',
        gap: '14px', padding: '12px 16px',
        borderRadius: '10px',
        background: isRisk ? 'rgba(255,68,102,0.04)' : 'transparent',
        border: `1px solid ${isRisk ? 'rgba(255,68,102,0.15)' : 'var(--border)'}`,
        marginBottom: '6px',
        animation: `slideRow 0.4s ease ${delay}ms both`,
        transition: 'background 0.15s ease',
        cursor: 'pointer',
      }}
      onMouseEnter={e => e.currentTarget.style.background = isRisk
        ? 'rgba(255,68,102,0.07)' : 'rgba(255,255,255,0.02)'}
      onMouseLeave={e => e.currentTarget.style.background = isRisk
        ? 'rgba(255,68,102,0.04)' : 'transparent'}
    >
      {/* Status dot */}
      <div style={{
        width: '7px', height: '7px',
        borderRadius: '50%', flexShrink: 0,
        background: isRisk ? 'var(--accent-red)' : 'var(--accent-green)',
        boxShadow: isRisk ? '0 0 6px var(--accent-red)' : '0 0 6px var(--accent-green)',
      }}/>

      {/* Prompt */}
      <div style={{ flex: 1, overflow: 'hidden' }}>
        <div style={{
          fontSize: '13px', color: 'var(--text-primary)',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}>
          {prompt.slice(0, 70) || 'No prompt'}
        </div>
        <div style={{
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px',
        }}>{model}</div>
      </div>

      {/* Archetype */}
      <ArchetypeBadge type={archetype} />

      {/* Entropy */}
      <div style={{
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: '12px',
        color: entropy > 0.75 ? 'var(--accent-red)' : entropy > 0.4 ? 'var(--accent-amber)' : 'var(--accent-green)',
        minWidth: '44px', textAlign: 'right',
      }}>
        {entropy.toFixed(3)}
      </div>

      {/* Time */}
      <div style={{
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: '11px', color: 'var(--text-muted)',
        minWidth: '48px', textAlign: 'right',
      }}>{time}</div>
    </div>
  )
}

// ── Custom tooltip ────────────────────────────────────────────────────────
function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--bg-card)',
      border: '1px solid var(--border)',
      borderRadius: '8px',
      padding: '10px 14px',
      fontFamily: 'JetBrains Mono, monospace',
      fontSize: '12px',
    }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: '6px' }}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color, marginBottom: '2px' }}>
          {p.dataKey}: {p.value}
        </div>
      ))}
    </div>
  )
}

// ── Main Dashboard ────────────────────────────────────────────────────────
export default function DashboardPage() {
  const { data: inferences, loading } = useInferences()
  const { data: trend } = useTrend()
  const kpis      = computeKPIs(inferences)
  const timeSeries = buildTimeSeries(inferences)
  const recent    = [...inferences].reverse().slice(0, 12)

  // Archetype distribution for bar chart
  const archetypeCounts = inferences.reduce((acc, r) => {
    const a = r.archetype || 'STABLE'
    acc[a] = (acc[a] || 0) + 1
    return acc
  }, {})
  const archetypeData = Object.entries(archetypeCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([name, count]) => ({
      name: name.replace(/_/g, ' ').slice(0, 18),
      count,
      color: ARCHETYPE_COLOR[name] || '#3d5166',
    }))

  if (loading) return (
    <div style={{
      flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: 'JetBrains Mono, monospace', fontSize: '13px',
      color: 'var(--text-muted)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <svg style={{ animation: 'spin 1s linear infinite' }} width="16" height="16"
          viewBox="0 0 24 24" fill="none">
          <circle cx="12" cy="12" r="10" stroke="rgba(0,212,255,0.2)" strokeWidth="2"/>
          <path d="M12 2a10 10 0 0 1 10 10"
            stroke="var(--accent-cyan)" strokeWidth="2" strokeLinecap="round"/>
        </svg>
        Loading inference data...
      </div>
    </div>
  )

  return (
    <div style={{ flex: 1, padding: '28px 32px', overflowY: 'auto' }}>

      {/* Header */}
      <div style={{ marginBottom: '28px', animation: 'kpiIn 0.5s ease both' }}>
        <h1 style={{
          fontSize: '22px', fontWeight: 700,
          color: 'var(--text-primary)', marginBottom: '4px',
        }}>Overview</h1>
        <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
          Real-time LLM monitoring · Auto-refreshes every 10s
        </p>
      </div>

      {/* Degradation banner */}
      {trend?.is_degrading && (
        <div style={{
          padding: '12px 16px', borderRadius: '10px',
          background: 'rgba(255,68,102,0.07)',
          border: '1px solid rgba(255,68,102,0.3)',
          borderLeft: '3px solid var(--accent-red)',
          marginBottom: '20px',
          display: 'flex', alignItems: 'center', gap: '10px',
          animation: 'kpiIn 0.4s ease both',
        }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
            stroke="var(--accent-red)" strokeWidth="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
            <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
          <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', color: 'var(--accent-red)', fontWeight: 600 }}>
            DEGRADATION DETECTED
          </span>
          <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
            Model performance declining — velocity: {trend.degradation_velocity?.toFixed(4)}
          </span>
        </div>
      )}

      {/* KPI Cards */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: '14px',
        marginBottom: '24px',
      }}>
        <KPICard
          label="TOTAL INFERENCES" value={kpis.total} suffix="" decimals={0}
          sub={`${kpis.highRisk} high-risk`}
          color="var(--accent-cyan)" delay={0}
          icon={<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>}
        />
        <KPICard
          label="HIGH RISK RATE" value={kpis.riskPct} suffix="%" decimals={0}
          sub={`${kpis.highRisk} flagged`}
          color={kpis.riskPct > 30 ? 'var(--accent-red)' : 'var(--accent-green)'} delay={80}
          icon={<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>}
        />
        <KPICard
          label="AVG ENTROPY" value={kpis.avgEntropy} suffix="" decimals={3}
          sub={kpis.avgEntropy > 0.75 ? '↑ above threshold' : '↓ within range'}
          color={kpis.avgEntropy > 0.75 ? 'var(--accent-amber)' : 'var(--accent-cyan)'} delay={160}
          icon={<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>}
        />
        <KPICard
          label="AVG AGREEMENT" value={kpis.avgAgreement} suffix="" decimals={3}
          sub={kpis.avgAgreement < 0.5 ? '↓ low agreement' : '↑ stable'}
          color={kpis.avgAgreement < 0.5 ? 'var(--accent-amber)' : 'var(--accent-green)'} delay={240}
          icon={<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>}
        />
      </div>

      {/* Charts row */}
      <div style={{
        display: 'grid', gridTemplateColumns: '1fr 1fr',
        gap: '16px', marginBottom: '24px',
      }}>

        {/* Signal time series */}
        <div style={{
          padding: '20px 22px', borderRadius: '14px',
          background: 'var(--bg-card)', border: '1px solid var(--border)',
          animation: 'kpiIn 0.6s ease 0.3s both',
        }}>
          <div style={{
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '10px', fontWeight: 600,
            letterSpacing: '0.12em', color: 'var(--text-muted)',
            marginBottom: '16px',
          }}>SIGNAL TIME SERIES</div>
          {timeSeries.length > 0 ? (
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={timeSeries} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.03)" strokeDasharray="4 4"/>
                <XAxis dataKey="time" tick={{ fontFamily: 'JetBrains Mono', fontSize: 10, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false} interval="preserveStartEnd"/>
                <YAxis domain={[0, 1]} tick={{ fontFamily: 'JetBrains Mono', fontSize: 10, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false}/>
                <Tooltip content={<ChartTooltip />}/>
                <Line type="monotone" dataKey="entropy" stroke="var(--accent-red)" strokeWidth={1.5} dot={false} animationDuration={1500}/>
                <Line type="monotone" dataKey="agreement" stroke="var(--accent-green)" strokeWidth={1.5} dot={false} animationDuration={1800}/>
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ height: '180px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace', fontSize: '12px' }}>
              No data yet
            </div>
          )}
          <div style={{ display: 'flex', gap: '16px', marginTop: '12px' }}>
            {[['var(--accent-red)', 'Entropy'], ['var(--accent-green)', 'Agreement']].map(([c, l]) => (
              <div key={l} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <div style={{ width: '16px', height: '2px', background: c, borderRadius: '1px' }}/>
                <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color: 'var(--text-muted)' }}>{l}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Archetype distribution */}
        <div style={{
          padding: '20px 22px', borderRadius: '14px',
          background: 'var(--bg-card)', border: '1px solid var(--border)',
          animation: 'kpiIn 0.6s ease 0.4s both',
        }}>
          <div style={{
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '10px', fontWeight: 600,
            letterSpacing: '0.12em', color: 'var(--text-muted)',
            marginBottom: '16px',
          }}>FAILURE ARCHETYPES</div>
          {archetypeData.length > 0 ? (
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={archetypeData} margin={{ top: 4, right: 4, left: -20, bottom: 20 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.03)" strokeDasharray="4 4" vertical={false}/>
                <XAxis dataKey="name" tick={{ fontFamily: 'JetBrains Mono', fontSize: 9, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false} angle={-30} textAnchor="end"/>
                <YAxis tick={{ fontFamily: 'JetBrains Mono', fontSize: 10, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false}/>
                <Tooltip content={<ChartTooltip />}/>
                <Bar dataKey="count" radius={[4,4,0,0]} animationDuration={1200}>
                  {archetypeData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} opacity={0.8}/>
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ height: '180px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace', fontSize: '12px' }}>
              No data yet
            </div>
          )}
        </div>
      </div>

      {/* Live Failure Feed */}
      <div style={{
        padding: '20px 22px', borderRadius: '14px',
        background: 'var(--bg-card)', border: '1px solid var(--border)',
        animation: 'kpiIn 0.6s ease 0.5s both',
      }}>
        <div style={{
          display: 'flex', justifyContent: 'space-between',
          alignItems: 'center', marginBottom: '16px',
        }}>
          <div>
            <div style={{
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '10px', fontWeight: 600,
              letterSpacing: '0.12em', color: 'var(--text-muted)',
            }}>LIVE INFERENCE FEED</div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <div style={{
              width: '6px', height: '6px', borderRadius: '50%',
              background: 'var(--accent-green)',
              animation: 'pulse-dot 2s ease-in-out infinite',
            }}/>
            <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color: 'var(--text-muted)' }}>
              {inferences.length} total
            </span>
          </div>
        </div>

        {recent.length > 0 ? (
          recent.map((r, i) => (
            <InferenceRow key={r.request_id || i} record={r} delay={i * 40} />
          ))
        ) : (
          <div style={{
            padding: '40px', textAlign: 'center',
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '13px', color: 'var(--text-muted)',
          }}>
            No inferences yet. Connect your LLM with the SDK to start monitoring.
          </div>
        )}
      </div>
    </div>
  )
}
