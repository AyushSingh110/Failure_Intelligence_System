// DashboardPage.jsx
import { useState, useEffect, useRef } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, BarChart, Bar, Cell, AreaChart, Area,
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
function KPICard({ label, value, suffix = '', decimals = 0, sub, color, delay = 0, trend }) {
  return (
    <div style={{
      padding: '18px 20px',
      borderRadius: '12px',
      background: 'var(--bg-card)',
      border: '1px solid var(--border)',
      borderTop: `2px solid ${color}`,
      animation: `kpiIn 0.55s cubic-bezier(0.16,1,0.3,1) ${delay}ms both`,
      position: 'relative',
      overflow: 'hidden',
    }}>
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: '50px',
        background: `linear-gradient(180deg, ${color}10 0%, transparent 100%)`,
        pointerEvents: 'none',
      }}/>

      <div style={{
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: '10px', fontWeight: 600,
        letterSpacing: '0.12em',
        color: 'var(--text-muted)',
        marginBottom: '10px',
      }}>{label}</div>

      <div style={{
        fontSize: '26px', fontWeight: 800,
        fontFamily: 'JetBrains Mono, monospace',
        color: 'var(--text-primary)',
        letterSpacing: '-0.02em',
        lineHeight: 1,
        marginBottom: '6px',
      }}>
        <Counter to={value} suffix={suffix} decimals={decimals} />
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{sub}</div>
        {trend !== undefined && (
          <div style={{
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '11px',
            color: trend > 0 ? 'var(--accent-red)' : trend < 0 ? 'var(--accent-green)' : 'var(--text-muted)',
          }}>
            {trend > 0 ? '↑' : trend < 0 ? '↓' : '–'}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Archetype badge ───────────────────────────────────────────────────────
const ARCHETYPE_COLOR = {
  STABLE:                    '#00ff88',
  HALLUCINATION_RISK:        '#ff4466',
  MODEL_BLIND_SPOT:          '#ff4466',
  OVERCONFIDENT_FAILURE:     '#ff4466',
  UNSTABLE_OUTPUT:           '#ffaa00',
  LOW_CONFIDENCE:            '#ffaa00',
  TEMPORAL_KNOWLEDGE_CUTOFF: '#ffaa00',
}

function ArchetypeBadge({ type }) {
  const color = ARCHETYPE_COLOR[type] || '#3d5166'
  const label = type?.replace(/_/g, ' ') || 'UNKNOWN'
  return (
    <span style={{
      padding: '2px 8px', borderRadius: '4px',
      fontFamily: 'JetBrains Mono, monospace',
      fontSize: '10px', fontWeight: 600,
      background: `${color}18`, color,
      border: `1px solid ${color}30`,
      whiteSpace: 'nowrap',
    }}>{label}</span>
  )
}

function AttackBadge() {
  return (
    <span style={{
      padding: '2px 8px', borderRadius: '4px',
      fontFamily: 'JetBrains Mono, monospace',
      fontSize: '10px', fontWeight: 600,
      background: 'rgba(255,68,102,0.15)',
      color: 'var(--accent-red)',
      border: '1px solid rgba(255,68,102,0.35)',
      whiteSpace: 'nowrap',
    }}>ATTACK</span>
  )
}

// ── Filter tabs ───────────────────────────────────────────────────────────
function FilterTab({ label, active, count, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '5px 12px',
        borderRadius: '6px',
        border: active ? '1px solid var(--accent-cyan)' : '1px solid transparent',
        background: active ? 'rgba(0,212,255,0.08)' : 'transparent',
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: '11px', fontWeight: 600,
        color: active ? 'var(--accent-cyan)' : 'var(--text-muted)',
        cursor: 'pointer',
        display: 'flex', alignItems: 'center', gap: '6px',
        transition: 'all 0.15s ease',
      }}
    >
      {label}
      {count !== undefined && (
        <span style={{
          background: active ? 'rgba(0,212,255,0.15)' : 'rgba(255,255,255,0.05)',
          color: active ? 'var(--accent-cyan)' : 'var(--text-muted)',
          borderRadius: '4px', padding: '1px 5px',
          fontSize: '10px',
        }}>{count}</span>
      )}
    </button>
  )
}

// ── Inference Row ─────────────────────────────────────────────────────────
function InferenceRow({ record, delay }) {
  const entropy    = record.metrics?.entropy || 0
  const isRisk     = entropy > 0.75
  const isAttack   = record.is_adversarial === true || record.adversarial?.is_attack === true
  const prompt     = record.input_text || ''
  const model      = record.model_name || 'unknown'
  const archetype  = record.archetype || 'STABLE'
  const confidence = record.metrics?.confidence || record.metrics?.classifier_confidence
  const time       = record.timestamp
    ? new Date(record.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : ''

  const borderColor = isAttack
    ? 'rgba(255,68,102,0.25)'
    : isRisk ? 'rgba(255,170,0,0.2)' : 'var(--border)'
  const bgColor = isAttack
    ? 'rgba(255,68,102,0.05)'
    : isRisk ? 'rgba(255,170,0,0.03)' : 'transparent'

  return (
    <div style={{
      display: 'flex', alignItems: 'center',
      gap: '12px', padding: '11px 14px',
      borderRadius: '9px',
      background: bgColor,
      border: `1px solid ${borderColor}`,
      marginBottom: '5px',
      animation: `slideRow 0.35s ease ${delay}ms both`,
      transition: 'background 0.15s ease',
      cursor: 'default',
    }}
      onMouseEnter={e => e.currentTarget.style.background = isAttack
        ? 'rgba(255,68,102,0.08)' : isRisk ? 'rgba(255,170,0,0.06)' : 'rgba(255,255,255,0.02)'}
      onMouseLeave={e => e.currentTarget.style.background = bgColor}
    >
      {/* Status dot */}
      <div style={{
        width: '6px', height: '6px',
        borderRadius: '50%', flexShrink: 0,
        background: isAttack ? 'var(--accent-red)' : isRisk ? 'var(--accent-amber)' : 'var(--accent-green)',
        boxShadow: `0 0 5px ${isAttack ? 'var(--accent-red)' : isRisk ? 'var(--accent-amber)' : 'var(--accent-green)'}`,
      }}/>

      {/* Prompt + model */}
      <div style={{ flex: 1, overflow: 'hidden' }}>
        <div style={{
          fontSize: '13px', color: 'var(--text-primary)',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}>
          {prompt.slice(0, 72) || '(no prompt)'}
        </div>
        <div style={{
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px',
        }}>{model}</div>
      </div>

      {/* Badges */}
      <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
        {isAttack && <AttackBadge />}
        <ArchetypeBadge type={archetype} />
      </div>

      {/* Entropy */}
      <div style={{
        fontFamily: 'JetBrains Mono, monospace', fontSize: '12px',
        color: entropy > 0.75 ? 'var(--accent-red)' : entropy > 0.4 ? 'var(--accent-amber)' : 'var(--accent-green)',
        minWidth: '44px', textAlign: 'right',
      }}>{entropy.toFixed(3)}</div>

      {/* Confidence */}
      {confidence !== undefined && (
        <div style={{
          fontFamily: 'JetBrains Mono, monospace', fontSize: '11px',
          color: 'var(--text-muted)', minWidth: '36px', textAlign: 'right',
        }}>{(confidence * 100).toFixed(0)}%</div>
      )}

      {/* Time */}
      <div style={{
        fontFamily: 'JetBrains Mono, monospace', fontSize: '11px',
        color: 'var(--text-muted)', minWidth: '46px', textAlign: 'right',
      }}>{time}</div>
    </div>
  )
}

// ── Custom chart tooltip ──────────────────────────────────────────────────
function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--bg-card)',
      border: '1px solid var(--border)',
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

// ── Section header ────────────────────────────────────────────────────────
function SectionLabel({ children }) {
  return (
    <div style={{
      fontFamily: 'JetBrains Mono, monospace',
      fontSize: '10px', fontWeight: 600,
      letterSpacing: '0.12em',
      color: 'var(--text-muted)',
      marginBottom: '16px',
    }}>{children}</div>
  )
}

// ── Model Health Panel ────────────────────────────────────────────────────
function ModelHealthPanel({ trend, kpis }) {
  const score  = trend?.current_score ?? null
  const drifting = trend?.is_degrading === true
  const velocity = trend?.degradation_velocity ?? 0

  const statusColor  = drifting ? 'var(--accent-red)' : 'var(--accent-green)'
  const statusLabel  = drifting ? 'DEGRADING' : 'HEALTHY'

  const bars = [
    { label: 'Risk Rate',   pct: kpis.riskPct,                   color: kpis.riskPct > 30 ? 'var(--accent-red)' : 'var(--accent-green)' },
    { label: 'Avg Entropy', pct: Math.round(kpis.avgEntropy * 100), color: kpis.avgEntropy > 0.75 ? 'var(--accent-red)' : 'var(--accent-cyan)' },
    { label: 'Agreement',   pct: Math.round(kpis.avgAgreement * 100), color: kpis.avgAgreement < 0.5 ? 'var(--accent-amber)' : 'var(--accent-green)' },
  ]

  return (
    <div style={{
      padding: '20px 22px', borderRadius: '12px',
      background: 'var(--bg-card)', border: '1px solid var(--border)',
      animation: 'kpiIn 0.6s ease 0.45s both',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '18px' }}>
        <SectionLabel>MODEL HEALTH</SectionLabel>
        <span style={{
          fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', fontWeight: 700,
          color: statusColor, letterSpacing: '0.08em',
          background: `${statusColor}15`,
          padding: '3px 8px', borderRadius: '4px',
          border: `1px solid ${statusColor}30`,
        }}>{statusLabel}</span>
      </div>

      {bars.map(({ label, pct, color }) => (
        <div key={label} style={{ marginBottom: '12px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
            <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color: 'var(--text-muted)' }}>{label}</span>
            <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color }}>{pct}%</span>
          </div>
          <div style={{ height: '4px', borderRadius: '2px', background: 'rgba(255,255,255,0.05)' }}>
            <div style={{
              height: '100%', borderRadius: '2px',
              width: `${Math.min(pct, 100)}%`,
              background: color,
              transition: 'width 1s ease',
            }}/>
          </div>
        </div>
      ))}

      {drifting && (
        <div style={{
          marginTop: '14px', padding: '10px 12px',
          borderRadius: '8px',
          background: 'rgba(255,68,102,0.07)',
          border: '1px solid rgba(255,68,102,0.2)',
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: '11px', color: 'var(--accent-red)',
        }}>
          Velocity: {velocity.toFixed(4)} · EMA degrading
        </div>
      )}

      {score !== null && (
        <div style={{
          marginTop: '14px', display: 'flex', justifyContent: 'space-between',
          fontFamily: 'JetBrains Mono, monospace', fontSize: '11px',
          color: 'var(--text-muted)',
          paddingTop: '14px',
          borderTop: '1px solid var(--border)',
        }}>
          <span>EMA score</span>
          <span style={{ color: 'var(--text-primary)' }}>{score.toFixed(4)}</span>
        </div>
      )}
    </div>
  )
}

// ── Quick stats strip ─────────────────────────────────────────────────────
function StatPill({ label, value, color }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: '8px',
      padding: '7px 14px', borderRadius: '8px',
      background: 'var(--bg-card)', border: '1px solid var(--border)',
      flex: 1,
    }}>
      <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', color: 'var(--text-muted)', letterSpacing: '0.08em' }}>{label}</span>
      <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '13px', fontWeight: 700, color: color || 'var(--text-primary)', marginLeft: 'auto' }}>{value}</span>
    </div>
  )
}

// ── Main Dashboard ────────────────────────────────────────────────────────
export default function DashboardPage() {
  const { data: inferences, loading } = useInferences()
  const { data: trend }               = useTrend()
  const [filter, setFilter]           = useState('all')

  const kpis       = computeKPIs(inferences)
  const timeSeries = buildTimeSeries(inferences)
  const sorted     = [...inferences].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))

  const filtered = filter === 'all'     ? sorted
    : filter === 'risk'    ? sorted.filter(r => (r.metrics?.entropy || 0) > 0.75)
    : filter === 'attacks' ? sorted.filter(r => r.is_adversarial === true || r.adversarial?.is_attack === true)
    : sorted

  const recent = filtered.slice(0, 14)

  const archetypeCounts = inferences.reduce((acc, r) => {
    const a = r.archetype || 'STABLE'
    acc[a] = (acc[a] || 0) + 1
    return acc
  }, {})
  const archetypeData = Object.entries(archetypeCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([name, count]) => ({
      name: name.replace(/_/g, ' ').slice(0, 16),
      count,
      color: ARCHETYPE_COLOR[name] || '#3d5166',
    }))

  const lastSeenStr = kpis.lastSeen
    ? kpis.lastSeen.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : '—'

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
        Loading...
      </div>
    </div>
  )

  return (
    <div style={{ flex: 1, padding: '28px 32px', overflowY: 'auto' }}>

      {/* Header */}
      <div style={{
        marginBottom: '24px',
        display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
        animation: 'kpiIn 0.5s ease both',
      }}>
        <div>
          <h1 style={{ fontSize: '22px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px' }}>
            Overview
          </h1>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
            Real-time LLM monitoring · Auto-refreshes every 10s
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <div style={{
              width: '6px', height: '6px', borderRadius: '50%',
              background: 'var(--accent-green)',
              animation: 'pulse-dot 2s ease-in-out infinite',
            }}/>
            <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color: 'var(--text-muted)' }}>
              Live
            </span>
          </div>
          {kpis.lastSeen && (
            <span style={{
              fontFamily: 'JetBrains Mono, monospace', fontSize: '11px',
              color: 'var(--text-muted)',
            }}>
              Last: {lastSeenStr}
            </span>
          )}
        </div>
      </div>

      {/* Degradation banner */}
      {trend?.is_degrading && (
        <div style={{
          padding: '11px 16px', borderRadius: '9px',
          background: 'rgba(255,68,102,0.06)',
          border: '1px solid rgba(255,68,102,0.25)',
          borderLeft: '3px solid var(--accent-red)',
          marginBottom: '20px',
          display: 'flex', alignItems: 'center', gap: '10px',
          animation: 'kpiIn 0.4s ease both',
        }}>
          <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color: 'var(--accent-red)', fontWeight: 700 }}>
            DEGRADATION DETECTED
          </span>
          <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
            Model performance declining · velocity {trend.degradation_velocity?.toFixed(4)}
          </span>
        </div>
      )}

      {/* KPI row 1 */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', marginBottom: '12px' }}>
        <KPICard
          label="TOTAL INFERENCES"
          value={kpis.total} suffix="" decimals={0}
          sub="all time"
          color="var(--accent-cyan)" delay={0}
        />
        <KPICard
          label="HIGH RISK"
          value={kpis.highRisk} suffix="" decimals={0}
          sub={`${kpis.riskPct}% of total`}
          color={kpis.riskPct > 30 ? 'var(--accent-red)' : 'var(--accent-amber)'} delay={60}
        />
        <KPICard
          label="ATTACKS DETECTED"
          value={kpis.attacks} suffix="" decimals={0}
          sub={kpis.attacks > 0 ? 'adversarial blocked' : 'none detected'}
          color={kpis.attacks > 0 ? 'var(--accent-red)' : 'var(--accent-green)'} delay={120}
        />
      </div>

      {/* KPI row 2 */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', marginBottom: '24px' }}>
        <KPICard
          label="AVG ENTROPY"
          value={kpis.avgEntropy} suffix="" decimals={3}
          sub={kpis.avgEntropy > 0.75 ? 'above threshold' : 'within range'}
          color={kpis.avgEntropy > 0.75 ? 'var(--accent-red)' : 'var(--accent-cyan)'} delay={180}
        />
        <KPICard
          label="AVG AGREEMENT"
          value={kpis.avgAgreement} suffix="" decimals={3}
          sub={kpis.avgAgreement < 0.5 ? 'low jury agreement' : 'stable'}
          color={kpis.avgAgreement < 0.5 ? 'var(--accent-amber)' : 'var(--accent-green)'} delay={240}
        />
        <KPICard
          label="FIX APPLIED"
          value={kpis.fixApplied} suffix="" decimals={0}
          sub={kpis.fixApplied > 0 ? 'corrections issued' : 'no corrections yet'}
          color={kpis.fixApplied > 0 ? 'var(--accent-cyan)' : 'var(--text-muted)'} delay={300}
        />
      </div>

      {/* Charts + Health panel */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 260px', gap: '14px', marginBottom: '20px' }}>

        {/* Signal time series */}
        <div style={{
          padding: '20px 22px', borderRadius: '12px',
          background: 'var(--bg-card)', border: '1px solid var(--border)',
          animation: 'kpiIn 0.6s ease 0.3s both',
        }}>
          <SectionLabel>SIGNAL TIME SERIES</SectionLabel>
          {timeSeries.length > 0 ? (
            <>
              <ResponsiveContainer width="100%" height={160}>
                <AreaChart data={timeSeries} margin={{ top: 4, right: 4, left: -22, bottom: 0 }}>
                  <defs>
                    <linearGradient id="gradEntropy" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ff4466" stopOpacity={0.15}/>
                      <stop offset="95%" stopColor="#ff4466" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="gradAgreement" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#00ff88" stopOpacity={0.12}/>
                      <stop offset="95%" stopColor="#00ff88" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="rgba(255,255,255,0.03)" strokeDasharray="4 4"/>
                  <XAxis dataKey="time" tick={{ fontFamily: 'JetBrains Mono', fontSize: 9, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false} interval="preserveStartEnd"/>
                  <YAxis domain={[0, 1]} tick={{ fontFamily: 'JetBrains Mono', fontSize: 9, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false}/>
                  <Tooltip content={<ChartTooltip />}/>
                  <Area type="monotone" dataKey="entropy" stroke="#ff4466" strokeWidth={1.5} fill="url(#gradEntropy)" dot={false}/>
                  <Area type="monotone" dataKey="agreement" stroke="#00ff88" strokeWidth={1.5} fill="url(#gradAgreement)" dot={false}/>
                </AreaChart>
              </ResponsiveContainer>
              <div style={{ display: 'flex', gap: '16px', marginTop: '10px' }}>
                {[['#ff4466', 'Entropy'], ['#00ff88', 'Agreement']].map(([c, l]) => (
                  <div key={l} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <div style={{ width: '14px', height: '2px', background: c, borderRadius: '1px' }}/>
                    <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color: 'var(--text-muted)' }}>{l}</span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div style={{ height: '160px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace', fontSize: '12px' }}>
              No data yet
            </div>
          )}
        </div>

        {/* Archetype distribution */}
        <div style={{
          padding: '20px 22px', borderRadius: '12px',
          background: 'var(--bg-card)', border: '1px solid var(--border)',
          animation: 'kpiIn 0.6s ease 0.38s both',
        }}>
          <SectionLabel>FAILURE ARCHETYPES</SectionLabel>
          {archetypeData.length > 0 ? (
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={archetypeData} margin={{ top: 4, right: 4, left: -22, bottom: 22 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.03)" strokeDasharray="4 4" vertical={false}/>
                <XAxis dataKey="name" tick={{ fontFamily: 'JetBrains Mono', fontSize: 8, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false} angle={-28} textAnchor="end"/>
                <YAxis tick={{ fontFamily: 'JetBrains Mono', fontSize: 9, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false}/>
                <Tooltip content={<ChartTooltip />}/>
                <Bar dataKey="count" radius={[4, 4, 0, 0]} animationDuration={1200}>
                  {archetypeData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} opacity={0.8}/>
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ height: '160px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace', fontSize: '12px' }}>
              No data yet
            </div>
          )}
        </div>

        {/* Model health */}
        <ModelHealthPanel trend={trend} kpis={kpis} />
      </div>

      {/* Live Feed */}
      <div style={{
        padding: '20px 22px', borderRadius: '12px',
        background: 'var(--bg-card)', border: '1px solid var(--border)',
        animation: 'kpiIn 0.6s ease 0.5s both',
      }}>
        {/* Feed header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
          <div style={{ display: 'flex', gap: '6px' }}>
            <FilterTab label="All"     active={filter === 'all'}     count={inferences.length}       onClick={() => setFilter('all')}/>
            <FilterTab label="Risk"    active={filter === 'risk'}    count={kpis.highRisk}           onClick={() => setFilter('risk')}/>
            <FilterTab label="Attacks" active={filter === 'attacks'} count={kpis.attacks}            onClick={() => setFilter('attacks')}/>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <div style={{
              width: '6px', height: '6px', borderRadius: '50%',
              background: 'var(--accent-green)',
              animation: 'pulse-dot 2s ease-in-out infinite',
            }}/>
            <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color: 'var(--text-muted)' }}>
              {filtered.length} {filter === 'all' ? 'total' : 'matched'}
            </span>
          </div>
        </div>

        {/* Column headers */}
        {recent.length > 0 && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: '18px 1fr auto auto auto auto',
            gap: '12px',
            padding: '0 14px 8px',
            marginBottom: '4px',
            borderBottom: '1px solid var(--border)',
          }}>
            {['', 'PROMPT · MODEL', 'ARCHETYPE', 'ENTROPY', 'CONF', 'TIME'].map((h, i) => (
              <div key={i} style={{
                fontFamily: 'JetBrains Mono, monospace', fontSize: '9px',
                color: 'var(--text-muted)', letterSpacing: '0.1em',
                textAlign: i > 2 ? 'right' : 'left',
              }}>{h}</div>
            ))}
          </div>
        )}

        {recent.length > 0 ? (
          recent.map((r, i) => (
            <InferenceRow key={r.request_id || i} record={r} delay={i * 30} />
          ))
        ) : (
          <div style={{
            padding: '48px', textAlign: 'center',
            fontFamily: 'JetBrains Mono, monospace',
          }}>
            <div style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '10px' }}>
              {filter === 'all' ? 'No inferences yet' : `No ${filter === 'risk' ? 'high-risk' : 'attack'} inferences`}
            </div>
            {filter === 'all' && (
              <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.2)' }}>
                Connect your LLM with the fie-sdk to start monitoring
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
