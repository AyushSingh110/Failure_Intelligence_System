import { useTrend, useInferences, computeKPIs } from '../hooks/useData.js'

function EMACard({ label, value, threshold, higherIsBad = true, delay }) {
  const isWarning = higherIsBad ? value > threshold : value < threshold
  const color = isWarning ? '#ff4466' : '#00ff88'

  return (
    <div style={{
      padding: '18px 20px', borderRadius: '12px',
      background: 'var(--bg-card)',
      border: `1px solid ${isWarning ? 'rgba(255,68,102,0.25)' : 'var(--border)'}`,
      borderTop: `2px solid ${color}`,
      animation: `kpiIn 0.5s ease ${delay}ms both`,
      position: 'relative', overflow: 'hidden',
    }}>
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: '50px',
        background: `linear-gradient(180deg, ${color}10 0%, transparent 100%)`,
        pointerEvents: 'none',
      }}/>
      <div style={{
        fontFamily: 'JetBrains Mono, monospace', fontSize: '9px',
        fontWeight: 600, letterSpacing: '0.14em', color: 'var(--text-muted)',
        marginBottom: '10px',
      }}>{label}</div>
      <div style={{
        fontFamily: 'JetBrains Mono, monospace', fontSize: '24px',
        fontWeight: 800, color, lineHeight: 1, marginBottom: '6px',
      }}>
        {typeof value === 'number' ? value.toFixed(3) : value}
      </div>
      <div style={{
        fontFamily: 'JetBrains Mono, monospace', fontSize: '10px',
        color: isWarning ? color : 'var(--text-muted)',
        display: 'flex', alignItems: 'center', gap: '4px',
      }}>
        <div style={{ width: '5px', height: '5px', borderRadius: '50%', background: color, flexShrink: 0 }}/>
        {isWarning ? 'ABOVE THRESHOLD' : 'WITHIN RANGE'}
      </div>
    </div>
  )
}

function AlertRow({ record, delay }) {
  const entropy  = record.metrics?.entropy || 0
  const time     = record.timestamp
    ? new Date(record.timestamp).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
    : ''
  const archetype = record.archetype || 'STABLE'
  const archColor = entropy > 0.75 ? '#ff4466' : '#ffaa00'

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: '14px',
      padding: '12px 16px', borderRadius: '10px',
      background: 'rgba(255,68,102,0.04)',
      border: '1px solid rgba(255,68,102,0.15)',
      marginBottom: '6px',
      animation: `kpiIn 0.35s ease ${delay}ms both`,
      transition: 'background 0.15s ease',
    }}
    onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,68,102,0.08)'}
    onMouseLeave={e => e.currentTarget.style.background = 'rgba(255,68,102,0.04)'}
    >
      <div style={{ width: '7px', height: '7px', borderRadius: '50%', flexShrink: 0, background: '#ff4466', boxShadow: '0 0 6px #ff4466' }}/>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: '13px', color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {(record.input_text || 'No prompt').slice(0, 70)}
        </div>
        <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px' }}>
          {record.model_name} · {record.request_id}
        </div>
      </div>
      <span style={{
        padding: '2px 8px', borderRadius: '4px',
        fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', fontWeight: 600,
        background: `${archColor}18`, color: archColor, border: `1px solid ${archColor}30`,
        whiteSpace: 'nowrap',
      }}>{archetype.replace(/_/g, ' ')}</span>
      <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', color: '#ff4466', minWidth: '40px', textAlign: 'right' }}>
        {entropy.toFixed(3)}
      </div>
      <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color: 'var(--text-muted)', minWidth: '90px', textAlign: 'right' }}>
        {time}
      </div>
    </div>
  )
}

export default function AlertsPage() {
  const { data: trend }       = useTrend()
  const { data: inferences, loading } = useInferences()
  const kpis = computeKPIs(inferences)

  const highRiskFeed = [...inferences]
    .filter(r => (r.metrics?.entropy || 0) > 0.75)
    .reverse()
    .slice(0, 20)

  if (loading) return (
    <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'JetBrains Mono, monospace', fontSize: '13px', color: 'var(--text-muted)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <svg style={{ animation: 'spin 1s linear infinite' }} width="14" height="14" viewBox="0 0 24 24" fill="none">
          <circle cx="12" cy="12" r="10" stroke="rgba(0,212,255,0.2)" strokeWidth="2"/>
          <path d="M12 2a10 10 0 0 1 10 10" stroke="var(--accent-cyan)" strokeWidth="2" strokeLinecap="round"/>
        </svg>
        Loading alerts...
      </div>
    </div>
  )

  const isDegrading = trend?.is_degrading || false

  return (
    <>
      <style>{`
        @keyframes kpiIn { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:translateY(0); } }
        @keyframes spin   { from { transform:rotate(0deg); } to { transform:rotate(360deg); } }
        @keyframes pulse  { 0%,100% { opacity:1; } 50% { opacity:0.4; } }
      `}</style>

      <div style={{ flex: 1, padding: '28px 32px', overflowY: 'auto' }}>

        {/* Header */}
        <div style={{ marginBottom: '24px', animation: 'kpiIn 0.5s ease both' }}>
          <h1 style={{ fontSize: '22px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px' }}>Alerts</h1>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
            EMA-based degradation monitoring · Auto-refreshes every 10s
          </p>
        </div>

        {/* Degradation banner */}
        <div style={{
          padding: '14px 18px', borderRadius: '12px', marginBottom: '24px',
          background: isDegrading ? 'rgba(255,68,102,0.07)' : 'rgba(0,255,136,0.05)',
          border: `1px solid ${isDegrading ? 'rgba(255,68,102,0.3)' : 'rgba(0,255,136,0.2)'}`,
          borderLeft: `4px solid ${isDegrading ? '#ff4466' : '#00ff88'}`,
          display: 'flex', alignItems: 'center', gap: '12px',
          animation: 'kpiIn 0.5s ease 0.05s both', opacity: 0,
        }}>
          <div style={{
            width: '8px', height: '8px', borderRadius: '50%', flexShrink: 0,
            background: isDegrading ? '#ff4466' : '#00ff88',
            boxShadow: `0 0 8px ${isDegrading ? '#ff4466' : '#00ff88'}`,
            animation: isDegrading ? 'pulse 1.5s ease-in-out infinite' : 'none',
          }}/>
          <div>
            <div style={{
              fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', fontWeight: 700,
              color: isDegrading ? '#ff4466' : '#00ff88', marginBottom: '2px',
            }}>
              {isDegrading ? '⚠ DEGRADATION DETECTED' : '✓ ALL SYSTEMS HEALTHY'}
            </div>
            <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
              {isDegrading
                ? `Model performance is declining. Velocity: ${trend?.degradation_velocity?.toFixed(4) || '—'} · High-risk rate: ${(trend?.ema_high_risk_rate * 100 || 0).toFixed(1)}%`
                : `High-risk rate: ${(kpis.riskPct || 0)}% · ${kpis.highRisk} flagged out of ${kpis.total} total inferences`
              }
            </div>
          </div>
        </div>

        {/* EMA Metric Cards */}
        <div style={{
          fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', fontWeight: 600,
          letterSpacing: '0.14em', color: 'var(--text-muted)', marginBottom: '12px',
          animation: 'kpiIn 0.5s ease 0.1s both', opacity: 0,
        }}>EMA HEALTH METRICS</div>

        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)',
          gap: '12px', marginBottom: '28px',
        }}>
          <EMACard
            label="EMA ENTROPY"
            value={trend?.ema_entropy || 0}
            threshold={0.75}
            higherIsBad={true}
            delay={100}
          />
          <EMACard
            label="EMA AGREEMENT"
            value={trend?.ema_agreement || 0}
            threshold={0.5}
            higherIsBad={false}
            delay={160}
          />
          <EMACard
            label="HIGH RISK RATE"
            value={trend?.ema_high_risk_rate || 0}
            threshold={0.4}
            higherIsBad={true}
            delay={220}
          />
          <EMACard
            label="DEG. VELOCITY"
            value={trend?.degradation_velocity || 0}
            threshold={0.05}
            higherIsBad={true}
            delay={280}
          />
        </div>

        {/* Summary stats */}
        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)',
          gap: '12px', marginBottom: '28px',
          animation: 'kpiIn 0.5s ease 0.35s both', opacity: 0,
        }}>
          {[
            { label: 'SIGNALS RECORDED', value: trend?.signals_count || kpis.total },
            { label: 'TOTAL HIGH RISK',  value: kpis.highRisk },
            { label: 'RISK PERCENTAGE',  value: `${kpis.riskPct}%` },
          ].map(({ label, value }) => (
            <div key={label} style={{
              padding: '14px 16px', borderRadius: '10px',
              background: 'var(--bg-card)', border: '1px solid var(--border)',
              display: 'flex', alignItems: 'center', gap: '12px',
            }}>
              <div>
                <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '9px', letterSpacing: '0.12em', color: 'var(--text-muted)', marginBottom: '4px' }}>{label}</div>
                <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '20px', fontWeight: 800, color: 'var(--text-primary)' }}>{value}</div>
              </div>
            </div>
          ))}
        </div>

        {/* High-risk feed */}
        <div style={{ animation: 'kpiIn 0.5s ease 0.4s both', opacity: 0 }}>
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            marginBottom: '12px',
          }}>
            <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', fontWeight: 600, letterSpacing: '0.14em', color: 'var(--text-muted)' }}>
              HIGH-RISK INFERENCE FEED
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#ff4466', animation: 'pulse 1.5s ease-in-out infinite' }}/>
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color: 'var(--text-muted)' }}>
                {highRiskFeed.length} high-risk
              </span>
            </div>
          </div>

          {highRiskFeed.length === 0 ? (
            <div style={{
              padding: '40px', textAlign: 'center', borderRadius: '12px',
              border: '1px dashed var(--border)',
              fontFamily: 'JetBrains Mono, monospace', fontSize: '13px', color: 'var(--text-muted)',
            }}>
              <div style={{ fontSize: '28px', marginBottom: '10px', opacity: 0.3 }}>✓</div>
              No high-risk inferences detected. System is healthy.
            </div>
          ) : (
            highRiskFeed.map((r, i) => (
              <AlertRow key={r.request_id || i} record={r} delay={i * 30} />
            ))
          )}
        </div>
      </div>
    </>
  )
}