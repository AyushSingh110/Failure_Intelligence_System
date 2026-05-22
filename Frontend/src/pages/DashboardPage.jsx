import { useState, useEffect, useRef } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, BarChart, Bar, Cell, AreaChart, Area,
} from 'recharts'
import { useInferences, useTrend, computeKPIs, buildTimeSeries } from '../hooks/useData.js'
import { api } from '../lib/api.js'
import { getSession } from '../lib/auth.js'

// ── Animated counter ──────────────────────────────────────────────────────────
function Counter({ to, suffix = '', decimals = 0, duration = 1100 }) {
  const [val, setVal] = useState(0)
  const ref = useRef(null)
  useEffect(() => {
    if (!to) return
    let start = null
    const step = (ts) => {
      if (!start) start = ts
      const p = Math.min((ts - start) / duration, 1)
      const eased = 1 - Math.pow(1 - p, 3)
      setVal(parseFloat((eased * to).toFixed(decimals)))
      if (p < 1) ref.current = requestAnimationFrame(step)
    }
    ref.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(ref.current)
  }, [to, duration, decimals])
  return <>{decimals ? val.toFixed(decimals) : Math.round(val)}{suffix}</>
}

// ── Archetype helpers ─────────────────────────────────────────────────────────
const ARCHETYPE_COLOR = {
  STABLE:                    '#00ff88',
  HALLUCINATION_RISK:        '#ff4466',
  MODEL_BLIND_SPOT:          '#ff4466',
  OVERCONFIDENT_FAILURE:     '#ff4466',
  UNSTABLE_OUTPUT:           '#ffaa00',
  LOW_CONFIDENCE:            '#ffaa00',
  TEMPORAL_KNOWLEDGE_CUTOFF: '#ffaa00',
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

// ── Badges ────────────────────────────────────────────────────────────────────
function Badge({ label, color, dim }) {
  return (
    <span style={{
      padding: '2px 7px', borderRadius: '4px',
      fontFamily: 'JetBrains Mono, monospace',
      fontSize: '10px', fontWeight: 700,
      background: `${color}${dim || '14'}`,
      color,
      border: `1px solid ${color}28`,
      whiteSpace: 'nowrap',
      letterSpacing: '0.04em',
    }}>{label}</span>
  )
}

function ArchetypeBadge({ type }) {
  const color = ARCHETYPE_COLOR[type] || '#3d5166'
  return <Badge label={type?.replace(/_/g, ' ') || 'UNKNOWN'} color={color} />
}

// ── KPI Card ──────────────────────────────────────────────────────────────────
function KPICard({ label, value, suffix = '', decimals = 0, sub, color, delay = 0, icon }) {
  return (
    <div style={{
      padding: '18px 20px',
      borderRadius: '12px',
      background: 'var(--bg-card)',
      border: '1px solid var(--border)',
      animation: `kpiIn 0.5s cubic-bezier(0.16,1,0.3,1) ${delay}ms both`,
      position: 'relative',
      overflow: 'hidden',
      transition: 'border-color 0.2s, box-shadow 0.2s',
    }}
    onMouseEnter={e => {
      e.currentTarget.style.borderColor = 'var(--border-bright)'
      e.currentTarget.style.boxShadow = `0 4px 24px rgba(0,0,0,0.3)`
    }}
    onMouseLeave={e => {
      e.currentTarget.style.borderColor = 'var(--border)'
      e.currentTarget.style.boxShadow = 'none'
    }}
    >
      {/* Color accent stripe */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: '2px',
        background: color,
        opacity: 0.8,
      }}/>
      {/* Subtle color wash */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: '64px',
        background: `linear-gradient(180deg, ${color}0c 0%, transparent 100%)`,
        pointerEvents: 'none',
      }}/>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
        <div style={{
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: '9px', fontWeight: 700,
          letterSpacing: '0.14em',
          color: 'var(--text-muted)',
        }}>{label}</div>
        {icon && (
          <div style={{ color: `${color}70`, flexShrink: 0 }}>{icon}</div>
        )}
      </div>

      <div style={{
        fontSize: '28px', fontWeight: 800,
        fontFamily: 'JetBrains Mono, monospace',
        color: 'var(--text-primary)',
        letterSpacing: '-0.03em',
        lineHeight: 1,
        marginBottom: '8px',
      }}>
        <Counter to={value} suffix={suffix} decimals={decimals} />
      </div>

      <div style={{
        fontSize: '11px', color: 'var(--text-muted)',
        fontFamily: 'JetBrains Mono, monospace',
      }}>{sub}</div>
    </div>
  )
}

// ── Chart tooltip ─────────────────────────────────────────────────────────────
function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--bg-elevated)',
      border: '1px solid var(--border-bright)',
      borderRadius: '8px', padding: '10px 14px',
      fontFamily: 'JetBrains Mono, monospace', fontSize: '11px',
      boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
    }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: '7px', fontSize: '10px' }}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color || p.stroke, marginBottom: '3px', display: 'flex', justifyContent: 'space-between', gap: '16px' }}>
          <span style={{ color: 'var(--text-muted)' }}>{p.dataKey}</span>
          <span>{typeof p.value === 'number' ? p.value.toFixed(3) : p.value}</span>
        </div>
      ))}
    </div>
  )
}

// ── Card wrapper ──────────────────────────────────────────────────────────────
function Panel({ children, style, animDelay = 0 }) {
  return (
    <div style={{
      padding: '18px 20px', borderRadius: '12px',
      background: 'var(--bg-card)', border: '1px solid var(--border)',
      animation: `kpiIn 0.55s ease ${animDelay}ms both`,
      transition: 'border-color 0.2s',
      ...style,
    }}
    onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--border-bright)'}
    onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}
    >
      {children}
    </div>
  )
}

// ── Panel header ──────────────────────────────────────────────────────────────
function PanelHeader({ label, right }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
      <div style={{
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: '10px', fontWeight: 700,
        letterSpacing: '0.12em', color: 'var(--text-muted)',
      }}>{label}</div>
      {right}
    </div>
  )
}

// ── Filter tab ────────────────────────────────────────────────────────────────
function FilterTab({ label, active, count, onClick }) {
  return (
    <button onClick={onClick} style={{
      padding: '4px 10px', borderRadius: '5px',
      border: active ? '1px solid rgba(0,212,255,0.25)' : '1px solid transparent',
      background: active ? 'rgba(0,212,255,0.07)' : 'transparent',
      fontFamily: 'JetBrains Mono, monospace',
      fontSize: '11px', fontWeight: active ? 700 : 400,
      color: active ? 'var(--accent-cyan)' : 'var(--text-muted)',
      cursor: 'pointer',
      display: 'flex', alignItems: 'center', gap: '5px',
      transition: 'all 0.12s ease',
    }}
    onMouseEnter={e => { if (!active) e.currentTarget.style.color = 'var(--text-secondary)' }}
    onMouseLeave={e => { if (!active) e.currentTarget.style.color = 'var(--text-muted)' }}
    >
      {label}
      {count !== undefined && (
        <span style={{
          background: active ? 'rgba(0,212,255,0.12)' : 'rgba(255,255,255,0.04)',
          color: active ? 'var(--accent-cyan)' : 'var(--text-muted)',
          borderRadius: '3px', padding: '0 4px', fontSize: '10px', lineHeight: '16px',
        }}>{count}</span>
      )}
    </button>
  )
}

// ── Inference row ─────────────────────────────────────────────────────────────
function InferenceRow({ record, delay }) {
  const entropy   = record.metrics?.entropy || 0
  const isRisk    = entropy > 0.75
  const isAttack  = record.is_adversarial === true || record.adversarial?.is_attack === true
  const prompt    = record.input_text || ''
  const model     = record.model_name || 'unknown'
  const archetype = deriveArchetype(record)
  const confidence = record.metrics?.confidence || record.metrics?.classifier_confidence
  const time      = record.timestamp
    ? new Date(record.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : ''

  const dotColor = isAttack ? 'var(--accent-red)' : isRisk ? 'var(--accent-amber)' : 'var(--accent-green)'

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: '10px 1fr auto auto auto auto',
      gap: '12px',
      alignItems: 'center',
      padding: '9px 12px',
      borderRadius: '7px',
      background: isAttack
        ? 'rgba(255,68,102,0.04)'
        : isRisk ? 'rgba(255,170,0,0.03)' : 'transparent',
      border: `1px solid ${isAttack
        ? 'rgba(255,68,102,0.15)'
        : isRisk ? 'rgba(255,170,0,0.12)' : 'transparent'}`,
      marginBottom: '3px',
      animation: `slideRow 0.3s ease ${delay}ms both`,
      transition: 'background 0.12s ease, border-color 0.12s ease',
      cursor: 'default',
    }}
    onMouseEnter={e => {
      e.currentTarget.style.background = isAttack
        ? 'rgba(255,68,102,0.07)'
        : isRisk ? 'rgba(255,170,0,0.05)' : 'rgba(255,255,255,0.02)'
    }}
    onMouseLeave={e => {
      e.currentTarget.style.background = isAttack
        ? 'rgba(255,68,102,0.04)'
        : isRisk ? 'rgba(255,170,0,0.03)' : 'transparent'
    }}
    >
      <div style={{
        width: '6px', height: '6px', borderRadius: '50%',
        background: dotColor,
        boxShadow: `0 0 4px ${dotColor}`,
      }}/>

      <div style={{ overflow: 'hidden' }}>
        <div style={{
          fontSize: '12px', color: 'var(--text-primary)',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          fontWeight: isAttack ? 500 : 400,
        }}>
          {prompt.slice(0, 80) || '(no prompt)'}
        </div>
        <div style={{
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px',
        }}>{model}</div>
      </div>

      <div style={{ display: 'flex', gap: '5px', alignItems: 'center' }}>
        {isAttack && <Badge label="ATTACK" color="var(--accent-red)" dim="18" />}
        <ArchetypeBadge type={archetype} />
      </div>

      <div style={{
        fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', fontWeight: 600,
        color: entropy > 0.75 ? 'var(--accent-red)'
          : entropy > 0.4 ? 'var(--accent-amber)'
          : 'var(--accent-green)',
        minWidth: '42px', textAlign: 'right',
      }}>{entropy.toFixed(3)}</div>

      {confidence !== undefined ? (
        <div style={{
          fontFamily: 'JetBrains Mono, monospace', fontSize: '10px',
          color: 'var(--text-muted)', minWidth: '32px', textAlign: 'right',
        }}>{(confidence * 100).toFixed(0)}%</div>
      ) : <div />}

      <div style={{
        fontFamily: 'JetBrains Mono, monospace', fontSize: '10px',
        color: 'var(--text-muted)', minWidth: '42px', textAlign: 'right',
      }}>{time}</div>
    </div>
  )
}

// ── Health bar ────────────────────────────────────────────────────────────────
function HealthBar({ label, pct, color }) {
  return (
    <div style={{ marginBottom: '11px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
        <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', color: 'var(--text-muted)' }}>{label}</span>
        <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', color, fontWeight: 700 }}>{pct}%</span>
      </div>
      <div style={{ height: '3px', borderRadius: '2px', background: 'rgba(255,255,255,0.04)' }}>
        <div style={{
          height: '100%', borderRadius: '2px',
          width: `${Math.min(pct, 100)}%`,
          background: color,
          boxShadow: `0 0 6px ${color}60`,
          transition: 'width 1s cubic-bezier(0.16,1,0.3,1)',
        }}/>
      </div>
    </div>
  )
}

// ── Loading skeleton ──────────────────────────────────────────────────────────
function Skeleton({ w = '100%', h = '80px', r = '10px', delay = '0s' }) {
  return (
    <div style={{
      width: w, height: h, borderRadius: r,
      background: 'linear-gradient(90deg, rgba(255,255,255,0.02) 25%, rgba(255,255,255,0.05) 50%, rgba(255,255,255,0.02) 75%)',
      backgroundSize: '600px 100%',
      animation: `shimmer 1.8s ease-in-out infinite ${delay}`,
    }}/>
  )
}

// ── Main ──────────────────────────────────────────────────────────────────────
export default function DashboardPage() {
  const { data: inferences, loading, refetch } = useInferences()
  const { data: trend }                        = useTrend()
  const [filter, setFilter]                    = useState('all')
  const [refreshing, setRefreshing]            = useState(false)
  const [exporting, setExporting]              = useState(false)

  const handleRefresh = async () => {
    setRefreshing(true)
    await refetch()
    setTimeout(() => setRefreshing(false), 600)
  }

  const handleExport = async () => {
    if (exporting) return
    setExporting(true)
    try {
      const session = getSession()
      const blob = await api.exportCsv(session?.token)
      const url  = URL.createObjectURL(blob)
      const a    = document.createElement('a')
      a.href     = url
      a.download = `fie_inferences_${new Date().toISOString().slice(0, 10)}.csv`
      a.click()
      URL.revokeObjectURL(url)
    } catch (e) { console.error('Export failed:', e) }
    finally { setExporting(false) }
  }

  const kpis        = computeKPIs(inferences)
  const timeSeries  = buildTimeSeries(inferences)
  const sorted      = [...inferences].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
  const filtered    = filter === 'risk'    ? sorted.filter(r => (r.metrics?.entropy || 0) > 0.75)
                    : filter === 'attacks' ? sorted.filter(r => r.is_adversarial === true || r.adversarial?.is_attack === true)
                    : sorted
  const recent      = filtered.slice(0, 15)

  const archetypeCounts = inferences.reduce((acc, r) => {
    const a = deriveArchetype(r); acc[a] = (acc[a] || 0) + 1; return acc
  }, {})
  const archetypeData = Object.entries(archetypeCounts)
    .sort((a, b) => b[1] - a[1]).slice(0, 5)
    .map(([name, count]) => ({
      name: name.replace(/_/g, ' ').slice(0, 14),
      count,
      color: ARCHETYPE_COLOR[name] || '#3d5166',
    }))

  const lastSeenStr = kpis.lastSeen
    ? kpis.lastSeen.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : null

  const drifting = trend?.is_degrading === true
  const statusColor = drifting ? 'var(--accent-red)' : 'var(--accent-green)'
  const statusLabel = drifting ? 'DEGRADING' : 'HEALTHY'

  if (loading) return (
    <div className="dash-page" style={{ flex: 1, padding: '28px 32px', overflowY: 'auto' }}>
      <div style={{ marginBottom: '24px' }}>
        <Skeleton w="160px" h="20px" r="6px" />
        <div style={{ marginTop: '8px' }}><Skeleton w="240px" h="13px" r="4px" delay="0.05s" /></div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '12px', marginBottom: '12px' }}>
        {[0,1,2].map(i => <Skeleton key={i} h="96px" r="12px" delay={`${i*0.07}s`} />)}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '12px', marginBottom: '20px' }}>
        {[0,1,2].map(i => <Skeleton key={i} h="96px" r="12px" delay={`${(i+3)*0.07}s`} />)}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 250px', gap: '14px', marginBottom: '16px' }}>
        {[0,1,2].map(i => <Skeleton key={i} h="230px" r="12px" delay={`${i*0.08}s`} />)}
      </div>
      <Skeleton h="300px" r="12px" delay="0.3s" />
    </div>
  )

  return (
    <div className="dash-page" style={{ flex: 1, padding: '28px 32px', overflowY: 'auto' }}>

      {/* ── Page header ──────────────────────────────────────────── */}
      <div style={{
        marginBottom: '20px',
        display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
        animation: 'kpiIn 0.45s ease both',
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '4px' }}>
            <h1 style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>
              Overview
            </h1>
            <span style={{
              display: 'inline-flex', alignItems: 'center', gap: '5px',
              padding: '2px 8px', borderRadius: '4px',
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '10px', fontWeight: 700,
              background: drifting ? 'rgba(255,68,102,0.1)' : 'rgba(0,255,136,0.08)',
              color: statusColor,
              border: `1px solid ${drifting ? 'rgba(255,68,102,0.2)' : 'rgba(0,255,136,0.16)'}`,
              letterSpacing: '0.06em',
            }}>
              <span style={{
                width: '5px', height: '5px', borderRadius: '50%',
                background: statusColor,
                animation: 'pulse-dot 2s ease-in-out infinite',
                display: 'inline-block',
              }}/>
              {statusLabel}
            </span>
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace' }}>
            Real-time LLM monitoring · auto-refreshes every 10s
            {lastSeenStr && ` · last event ${lastSeenStr}`}
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '7px' }}>
          {inferences.length > 0 && (
            <button onClick={handleExport} disabled={exporting} style={{
              display: 'flex', alignItems: 'center', gap: '5px',
              padding: '6px 11px', borderRadius: '7px',
              border: '1px solid var(--border)', background: 'var(--bg-card)',
              cursor: exporting ? 'not-allowed' : 'pointer',
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '11px', color: 'var(--text-muted)',
              transition: 'all 0.12s ease', opacity: exporting ? 0.5 : 1,
            }}
            onMouseEnter={e => { if (!exporting) { e.currentTarget.style.borderColor = 'rgba(0,255,136,0.3)'; e.currentTarget.style.color = 'var(--accent-green)' }}}
            onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--text-muted)' }}
            >
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/>
              </svg>
              {exporting ? 'Exporting…' : 'Export CSV'}
            </button>
          )}
          <button onClick={handleRefresh} style={{
            display: 'flex', alignItems: 'center', gap: '5px',
            padding: '6px 11px', borderRadius: '7px',
            border: '1px solid var(--border)', background: 'var(--bg-card)',
            cursor: 'pointer', fontFamily: 'JetBrains Mono, monospace',
            fontSize: '11px', color: 'var(--text-muted)',
            transition: 'all 0.12s ease',
          }}
          onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(0,212,255,0.3)'; e.currentTarget.style.color = 'var(--accent-cyan)' }}
          onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--text-muted)' }}
          >
            <svg style={{ animation: refreshing ? 'spin 0.6s linear' : 'none' }}
              width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
              <path d="M23 4v6h-6M1 20v-6h6"/>
              <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"/>
            </svg>
            Refresh
          </button>
        </div>
      </div>

      {/* ── System status strip ───────────────────────────────────── */}
      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '8px',
        marginBottom: '20px',
        animation: 'kpiIn 0.45s ease 0.06s both',
      }}>
        {[
          { label: 'Pipeline',    value: 'LangGraph',          color: 'var(--accent-cyan)',  icon: '⬡' },
          { label: 'Guard layers', value: '9 active',          color: 'var(--accent-green)', icon: '⬡' },
          { label: 'Jury agents', value: '3 online',           color: 'var(--accent-green)', icon: '⬡' },
          { label: 'Threshold',   value: 'entropy > 0.75',     color: 'var(--text-muted)',   icon: '⬡' },
          { label: 'Tracked',     value: kpis.total > 0 ? `${kpis.total} inferences` : 'awaiting data',
            color: kpis.total > 0 ? 'var(--accent-cyan)' : 'var(--text-muted)', icon: '⬡' },
        ].map(({ label, value, color }) => (
          <div key={label} style={{
            padding: '10px 12px', borderRadius: '8px',
            background: 'var(--bg-card)', border: '1px solid var(--border)',
          }}>
            <div style={{
              fontFamily: 'JetBrains Mono, monospace', fontSize: '9px',
              color: 'var(--text-muted)', letterSpacing: '0.1em', marginBottom: '4px',
            }}>{label.toUpperCase()}</div>
            <div style={{
              fontFamily: 'JetBrains Mono, monospace', fontSize: '11px',
              fontWeight: 700, color,
            }}>{value}</div>
          </div>
        ))}
      </div>

      {/* ── Degradation alert ─────────────────────────────────────── */}
      {drifting && (
        <div style={{
          padding: '10px 14px', borderRadius: '8px',
          background: 'rgba(255,68,102,0.05)',
          border: '1px solid rgba(255,68,102,0.2)',
          borderLeft: '3px solid var(--accent-red)',
          marginBottom: '18px',
          display: 'flex', alignItems: 'center', gap: '12px',
          animation: 'kpiIn 0.4s ease both',
        }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent-red)" strokeWidth="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
            <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
          <div>
            <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color: 'var(--accent-red)', fontWeight: 700 }}>
              DEGRADATION DETECTED
            </span>
            <span style={{ fontSize: '12px', color: 'var(--text-secondary)', marginLeft: '10px' }}>
              Model performance declining · velocity {trend.degradation_velocity?.toFixed(4)}
            </span>
          </div>
        </div>
      )}

      {/* ── KPI row 1 ─────────────────────────────────────────────── */}
      <div className="dash-kpi-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '10px', marginBottom: '10px' }}>
        <KPICard
          label="TOTAL INFERENCES" value={kpis.total} sub="all time"
          color="var(--accent-cyan)" delay={0}
          icon={<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>}
        />
        <KPICard
          label="HIGH RISK" value={kpis.highRisk}
          sub={`${kpis.riskPct}% of total`}
          color={kpis.riskPct > 30 ? 'var(--accent-red)' : 'var(--accent-amber)'} delay={60}
          icon={<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/></svg>}
        />
        <KPICard
          label="ATTACKS BLOCKED" value={kpis.attacks}
          sub={kpis.attacks > 0 ? 'adversarial blocked' : 'none detected'}
          color={kpis.attacks > 0 ? 'var(--accent-red)' : 'var(--accent-green)'} delay={120}
          icon={<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>}
        />
      </div>

      {/* ── KPI row 2 ─────────────────────────────────────────────── */}
      <div className="dash-kpi-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '10px', marginBottom: '22px' }}>
        <KPICard
          label="AVG ENTROPY" value={kpis.avgEntropy} decimals={3}
          sub={kpis.avgEntropy > 0.75 ? 'above threshold' : 'within range'}
          color={kpis.avgEntropy > 0.75 ? 'var(--accent-red)' : 'var(--accent-cyan)'} delay={180}
        />
        <KPICard
          label="AVG AGREEMENT" value={kpis.avgAgreement} decimals={3}
          sub={kpis.avgAgreement < 0.5 ? 'low jury agreement' : 'shadow jury stable'}
          color={kpis.avgAgreement < 0.5 ? 'var(--accent-amber)' : 'var(--accent-green)'} delay={240}
        />
        <KPICard
          label="CORRECTIONS ISSUED" value={kpis.fixApplied}
          sub={kpis.fixApplied > 0 ? 'auto-fix applied' : 'no corrections yet'}
          color={kpis.fixApplied > 0 ? 'var(--accent-cyan)' : 'var(--text-muted)'} delay={300}
        />
      </div>

      {/* ── Charts row ────────────────────────────────────────────── */}
      <div className="dash-chart-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 250px', gap: '12px', marginBottom: '18px' }}>

        {/* Signal time series */}
        <Panel animDelay={300}>
          <PanelHeader
            label="SIGNAL TIME SERIES"
            right={
              <div style={{ display: 'flex', gap: '12px' }}>
                {[['#ff4466', 'Entropy'], ['#00ff88', 'Agreement']].map(([c, l]) => (
                  <div key={l} style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                    <div style={{ width: '12px', height: '2px', background: c, borderRadius: '1px' }}/>
                    <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', color: 'var(--text-muted)' }}>{l}</span>
                  </div>
                ))}
              </div>
            }
          />
          {timeSeries.length > 0 ? (
            <ResponsiveContainer width="100%" height={160}>
              <AreaChart data={timeSeries} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
                <defs>
                  <linearGradient id="gE" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#ff4466" stopOpacity={0.18}/>
                    <stop offset="100%" stopColor="#ff4466" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="gA" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#00ff88" stopOpacity={0.14}/>
                    <stop offset="100%" stopColor="#00ff88" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="rgba(255,255,255,0.025)" strokeDasharray="3 6"/>
                <XAxis dataKey="time" tick={{ fontFamily: 'JetBrains Mono', fontSize: 9, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false} interval="preserveStartEnd"/>
                <YAxis domain={[0,1]} tick={{ fontFamily: 'JetBrains Mono', fontSize: 9, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false}/>
                <Tooltip content={<ChartTooltip />}/>
                <Area type="monotone" dataKey="entropy"   stroke="#ff4466" strokeWidth={1.5} fill="url(#gE)" dot={false}/>
                <Area type="monotone" dataKey="agreement" stroke="#00ff88" strokeWidth={1.5} fill="url(#gA)" dot={false}/>
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <EmptyChart />
          )}
        </Panel>

        {/* Failure archetypes */}
        <Panel animDelay={360}>
          <PanelHeader label="FAILURE ARCHETYPES" />
          {archetypeData.length > 0 ? (
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={archetypeData} margin={{ top: 4, right: 4, left: -24, bottom: 24 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.025)" strokeDasharray="3 6" vertical={false}/>
                <XAxis dataKey="name" tick={{ fontFamily: 'JetBrains Mono', fontSize: 8, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false} angle={-25} textAnchor="end"/>
                <YAxis tick={{ fontFamily: 'JetBrains Mono', fontSize: 9, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false}/>
                <Tooltip content={<ChartTooltip />}/>
                <Bar dataKey="count" radius={[3,3,0,0]} animationDuration={1000}>
                  {archetypeData.map((e, i) => (
                    <Cell key={i} fill={e.color} opacity={0.75}/>
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <EmptyChart />
          )}
        </Panel>

        {/* Model health */}
        <Panel animDelay={420}>
          <PanelHeader
            label="MODEL HEALTH"
            right={
              <span style={{
                fontFamily: 'JetBrains Mono, monospace', fontSize: '9px', fontWeight: 700,
                color: statusColor, letterSpacing: '0.08em',
                background: `${drifting ? 'rgba(255,68,102' : 'rgba(0,255,136'}, 0.1)`,
                padding: '2px 7px', borderRadius: '3px',
                border: `1px solid ${drifting ? 'rgba(255,68,102,0.2)' : 'rgba(0,255,136,0.2)'}`,
              }}>{statusLabel}</span>
            }
          />
          <HealthBar label="Risk Rate"   pct={kpis.riskPct} color={kpis.riskPct > 30 ? 'var(--accent-red)' : 'var(--accent-green)'} />
          <HealthBar label="Avg Entropy" pct={Math.round(kpis.avgEntropy * 100)} color={kpis.avgEntropy > 0.75 ? 'var(--accent-red)' : 'var(--accent-cyan)'} />
          <HealthBar label="Agreement"   pct={Math.round(kpis.avgAgreement * 100)} color={kpis.avgAgreement < 0.5 ? 'var(--accent-amber)' : 'var(--accent-green)'} />

          {drifting && (
            <div style={{
              marginTop: '12px', padding: '8px 10px', borderRadius: '6px',
              background: 'rgba(255,68,102,0.06)', border: '1px solid rgba(255,68,102,0.18)',
              fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', color: 'var(--accent-red)',
            }}>
              velocity {trend?.degradation_velocity?.toFixed(4)} · EMA degrading
            </div>
          )}
          {trend?.current_score != null && (
            <div style={{
              marginTop: '12px', display: 'flex', justifyContent: 'space-between',
              fontFamily: 'JetBrains Mono, monospace', fontSize: '10px',
              color: 'var(--text-muted)', paddingTop: '10px', borderTop: '1px solid var(--border)',
            }}>
              <span>EMA score</span>
              <span style={{ color: 'var(--text-primary)' }}>{trend.current_score.toFixed(4)}</span>
            </div>
          )}
        </Panel>
      </div>

      {/* ── Live feed ─────────────────────────────────────────────── */}
      <Panel animDelay={480} style={{ padding: '16px 18px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
          <div style={{ display: 'flex', gap: '4px' }}>
            <FilterTab label="All"     active={filter === 'all'}     count={inferences.length} onClick={() => setFilter('all')}/>
            <FilterTab label="Risk"    active={filter === 'risk'}    count={kpis.highRisk}     onClick={() => setFilter('risk')}/>
            <FilterTab label="Attacks" active={filter === 'attacks'} count={kpis.attacks}      onClick={() => setFilter('attacks')}/>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <div style={{
              width: '5px', height: '5px', borderRadius: '50%',
              background: 'var(--accent-green)',
              animation: 'pulse-dot 2s ease-in-out infinite',
            }}/>
            <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', color: 'var(--text-muted)' }}>
              {filtered.length} {filter === 'all' ? 'total' : 'matched'}
            </span>
          </div>
        </div>

        {recent.length > 0 && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: '10px 1fr auto auto auto auto',
            gap: '12px',
            padding: '0 12px 7px',
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
          recent.map((r, i) => <InferenceRow key={r.request_id || i} record={r} delay={i * 25} />)
        ) : (
          <EmptyFeed filter={filter} />
        )}
      </Panel>
    </div>
  )
}

function EmptyChart() {
  return (
    <div style={{
      height: '160px', display: 'flex', alignItems: 'center', justifyContent: 'center',
      color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace', fontSize: '11px',
    }}>No data yet</div>
  )
}

function EmptyFeed({ filter }) {
  if (filter !== 'all') return (
    <div style={{ padding: '32px 16px', textAlign: 'center', fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', color: 'var(--text-muted)' }}>
      No {filter === 'risk' ? 'high-risk' : 'attack'} inferences in the current window
    </div>
  )
  return (
    <div style={{ padding: '40px 16px', textAlign: 'center' }}>
      <div style={{
        width: '40px', height: '40px', borderRadius: '10px',
        background: 'rgba(0,212,255,0.05)', border: '1px solid rgba(0,212,255,0.12)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        margin: '0 auto 14px',
      }}>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--accent-cyan)" strokeWidth="1.6">
          <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
        </svg>
      </div>
      <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '13px', color: 'var(--text-primary)', fontWeight: 600, marginBottom: '6px' }}>
        Awaiting first inference
      </div>
      <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '14px' }}>
        Send a prompt through the SDK to start monitoring
      </div>
      <div style={{
        display: 'inline-block', padding: '9px 14px', borderRadius: '7px',
        background: 'rgba(0,212,255,0.04)', border: '1px solid rgba(0,212,255,0.12)',
        fontFamily: 'JetBrains Mono, monospace', fontSize: '11px',
        color: 'var(--accent-cyan)', textAlign: 'left', lineHeight: 1.9,
      }}>
        <span style={{ color: 'var(--text-muted)' }}>$</span> pip install fie-sdk<br/>
        <span style={{ color: 'var(--text-muted)' }}>from fie import</span> monitor
      </div>
    </div>
  )
}
