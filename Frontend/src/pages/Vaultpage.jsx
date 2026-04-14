import { useState, useMemo } from 'react'
import { useInferences } from '../hooks/useData.js'

const ARCHETYPE_COLOR = {
  STABLE: '#00ff88', HALLUCINATION_RISK: '#ff4466',
  MODEL_BLIND_SPOT: '#ff4466', OVERCONFIDENT_FAILURE: '#ff4466',
  UNSTABLE_OUTPUT: '#ffaa00', LOW_CONFIDENCE: '#ffaa00',
}

function Badge({ label, color }) {
  const c = color || ARCHETYPE_COLOR[label] || '#3d5166'
  return (
    <span style={{
      padding: '2px 8px', borderRadius: '4px',
      fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', fontWeight: 600,
      background: `${c}18`, color: c, border: `1px solid ${c}30`, whiteSpace: 'nowrap',
    }}>{label?.replace(/_/g, ' ')}</span>
  )
}

function EntropyBar({ value }) {
  const color = value > 0.75 ? '#ff4466' : value > 0.4 ? '#ffaa00' : '#00ff88'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: '100px' }}>
      <div style={{ flex: 1, height: '3px', background: 'var(--border)', borderRadius: '2px' }}>
        <div style={{ width: `${Math.min(value * 100, 100)}%`, height: '100%', background: color, borderRadius: '2px', transition: 'width 0.4s ease' }}/>
      </div>
      <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color, minWidth: '36px' }}>
        {value.toFixed(3)}
      </span>
    </div>
  )
}

function DetailDrawer({ record, onClose }) {
  if (!record) return null
  const metrics = record.metrics || {}
  const isRisk  = (metrics.entropy || 0) > 0.75

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 50,
      display: 'flex', alignItems: 'flex-end', justifyContent: 'flex-end',
    }}>
      <div
        onClick={onClose}
        style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.5)' }}
      />
      <div style={{
        position: 'relative', width: '480px', height: '100vh',
        background: 'var(--bg-secondary)', borderLeft: '1px solid var(--border)',
        overflowY: 'auto', padding: '24px',
        animation: 'slideInRight 0.25s ease both',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color: 'var(--text-muted)', letterSpacing: '0.1em' }}>
            INFERENCE DETAIL
          </div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '18px' }}>×</button>
        </div>

        {/* Status */}
        <div style={{
          padding: '12px 14px', borderRadius: '8px', marginBottom: '16px',
          background: isRisk ? 'rgba(255,68,102,0.06)' : 'rgba(0,255,136,0.05)',
          border: `1px solid ${isRisk ? 'rgba(255,68,102,0.2)' : 'rgba(0,255,136,0.15)'}`,
          display: 'flex', alignItems: 'center', gap: '8px',
        }}>
          <div style={{ width: '7px', height: '7px', borderRadius: '50%', background: isRisk ? 'var(--accent-red)' : 'var(--accent-green)', boxShadow: `0 0 6px ${isRisk ? 'var(--accent-red)' : 'var(--accent-green)'}` }}/>
          <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', color: isRisk ? 'var(--accent-red)' : 'var(--accent-green)', fontWeight: 600 }}>
            {isRisk ? 'HIGH RISK' : 'STABLE'}
          </span>
          <Badge label={record.archetype || 'STABLE'} />
        </div>

        {/* Meta */}
        {[
          ['Request ID', record.request_id],
          ['Model', record.model_name],
          ['Timestamp', record.timestamp ? new Date(record.timestamp).toLocaleString() : '—'],
          ['Latency', record.metrics?.latency_ms ? `${record.metrics.latency_ms}ms` : '—'],
        ].map(([label, val]) => (
          <div key={label} style={{ display: 'flex', gap: '12px', marginBottom: '10px' }}>
            <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color: 'var(--text-muted)', minWidth: '90px' }}>{label}</div>
            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', wordBreak: 'break-all' }}>{val || '—'}</div>
          </div>
        ))}

        <div style={{ height: '1px', background: 'var(--border)', margin: '16px 0' }}/>

        {/* Metrics */}
        <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', letterSpacing: '0.12em', color: 'var(--text-muted)', marginBottom: '12px' }}>METRICS</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '16px' }}>
          {[
            ['Entropy', (metrics.entropy || 0).toFixed(3), metrics.entropy > 0.75 ? '#ff4466' : '#00ff88'],
            ['Agreement', (metrics.agreement_score || 0).toFixed(3), metrics.agreement_score < 0.5 ? '#ffaa00' : '#00ff88'],
            ['FSD Score', (metrics.fsd_score || 0).toFixed(3), 'var(--accent-cyan)'],
            ['Emb. Distance', (metrics.embedding_distance || 0).toFixed(3), 'var(--text-secondary)'],
          ].map(([label, val, color]) => (
            <div key={label} style={{ padding: '10px 12px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
              <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px' }}>{label}</div>
              <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '16px', fontWeight: 700, color }}>{val}</div>
            </div>
          ))}
        </div>

        <div style={{ height: '1px', background: 'var(--border)', margin: '16px 0' }}/>

        {/* Prompt */}
        <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', letterSpacing: '0.12em', color: 'var(--text-muted)', marginBottom: '8px' }}>INPUT PROMPT</div>
        <div style={{ padding: '12px', borderRadius: '8px', background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border)', fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: '16px', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
          {record.input_text || '—'}
        </div>

        {/* Output */}
        <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', letterSpacing: '0.12em', color: 'var(--text-muted)', marginBottom: '8px' }}>MODEL OUTPUT</div>
        <div style={{ padding: '12px', borderRadius: '8px', background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border)', fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.6, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
          {record.output_text || '—'}
        </div>
      </div>
    </div>
  )
}

export default function VaultPage() {
  const { data: inferences, loading } = useInferences()
  const [search,   setSearch]   = useState('')
  const [model,    setModel]    = useState('all')
  const [risk,     setRisk]     = useState('all')
  const [selected, setSelected] = useState(null)

  const models = useMemo(() => {
    const set = new Set(inferences.map(r => r.model_name).filter(Boolean))
    return ['all', ...set]
  }, [inferences])

  const filtered = useMemo(() => {
    return inferences.filter(r => {
      const matchSearch = !search || (r.input_text || '').toLowerCase().includes(search.toLowerCase()) || (r.request_id || '').includes(search)
      const matchModel  = model === 'all' || r.model_name === model
      const entropy     = r.metrics?.entropy || 0
      const matchRisk   = risk === 'all' || (risk === 'high' && entropy > 0.75) || (risk === 'stable' && entropy <= 0.75)
      return matchSearch && matchModel && matchRisk
    }).reverse()
  }, [inferences, search, model, risk])

  if (loading) return (
    <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'JetBrains Mono, monospace', fontSize: '13px', color: 'var(--text-muted)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <svg style={{ animation: 'spin 1s linear infinite' }} width="14" height="14" viewBox="0 0 24 24" fill="none">
          <circle cx="12" cy="12" r="10" stroke="rgba(0,212,255,0.2)" strokeWidth="2"/>
          <path d="M12 2a10 10 0 0 1 10 10" stroke="var(--accent-cyan)" strokeWidth="2" strokeLinecap="round"/>
        </svg>
        Loading vault...
      </div>
    </div>
  )

  return (
    <>
      <style>{`
        @keyframes kpiIn { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:translateY(0); } }
        @keyframes spin   { from { transform:rotate(0deg); } to { transform:rotate(360deg); } }
        @keyframes slideInRight { from { transform:translateX(40px); opacity:0; } to { transform:translateX(0); opacity:1; } }
        .vault-row:hover { background: rgba(255,255,255,0.02) !important; cursor: pointer; }
      `}</style>

      <div style={{ flex: 1, padding: '28px 32px', overflowY: 'auto' }}>

        {/* Header */}
        <div style={{ marginBottom: '24px', animation: 'kpiIn 0.5s ease both' }}>
          <h1 style={{ fontSize: '22px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px' }}>Vault</h1>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
            {inferences.length} total inferences · click any row for details
          </p>
        </div>

        {/* Filters */}
        <div style={{ display: 'flex', gap: '10px', marginBottom: '16px', flexWrap: 'wrap', animation: 'kpiIn 0.5s ease 0.05s both', opacity: 0 }}>
          <input
            value={search} onChange={e => setSearch(e.target.value)}
            placeholder="Search prompts or IDs..."
            style={{
              flex: '1', minWidth: '200px', padding: '8px 12px',
              borderRadius: '8px', border: '1px solid var(--border)',
              background: 'var(--bg-card)', color: 'var(--text-primary)',
              fontFamily: 'JetBrains Mono, monospace', fontSize: '12px',
              outline: 'none',
            }}
          />
          <select value={model} onChange={e => setModel(e.target.value)} style={{
            padding: '8px 12px', borderRadius: '8px', border: '1px solid var(--border)',
            background: 'var(--bg-card)', color: 'var(--text-secondary)',
            fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', cursor: 'pointer',
          }}>
            {models.map(m => <option key={m} value={m}>{m === 'all' ? 'All models' : m}</option>)}
          </select>
          <select value={risk} onChange={e => setRisk(e.target.value)} style={{
            padding: '8px 12px', borderRadius: '8px', border: '1px solid var(--border)',
            background: 'var(--bg-card)', color: 'var(--text-secondary)',
            fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', cursor: 'pointer',
          }}>
            <option value="all">All risk levels</option>
            <option value="high">High risk only</option>
            <option value="stable">Stable only</option>
          </select>
        </div>

        {/* Table */}
        <div style={{ borderRadius: '12px', border: '1px solid var(--border)', background: 'var(--bg-card)', overflow: 'hidden', animation: 'kpiIn 0.5s ease 0.1s both', opacity: 0 }}>
          {/* Header row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 120px 140px 110px 80px', gap: '0', padding: '10px 16px', borderBottom: '1px solid var(--border)', background: 'rgba(0,0,0,0.2)' }}>
            {['PROMPT', 'MODEL', 'ARCHETYPE', 'ENTROPY', 'TIME'].map(h => (
              <div key={h} style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '9px', fontWeight: 600, letterSpacing: '0.12em', color: 'var(--text-muted)' }}>{h}</div>
            ))}
          </div>

          {filtered.length === 0 ? (
            <div style={{ padding: '40px', textAlign: 'center', fontFamily: 'JetBrains Mono, monospace', fontSize: '13px', color: 'var(--text-muted)' }}>
              No inferences match your filters.
            </div>
          ) : (
            filtered.map((r, i) => {
              const entropy  = r.metrics?.entropy || 0
              const isRisk   = entropy > 0.75
              const time     = r.timestamp ? new Date(r.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''
              return (
                <div
                  key={r.request_id || i}
                  className="vault-row"
                  onClick={() => setSelected(r)}
                  style={{
                    display: 'grid', gridTemplateColumns: '1fr 120px 140px 110px 80px',
                    alignItems: 'center', gap: '0', padding: '11px 16px',
                    borderBottom: i < filtered.length - 1 ? '1px solid var(--border)' : 'none',
                    background: isRisk ? 'rgba(255,68,102,0.03)' : 'transparent',
                    transition: 'background 0.12s ease',
                    animation: `kpiIn 0.35s ease ${Math.min(i * 20, 300)}ms both`,
                  }}
                >
                  <div style={{ overflow: 'hidden', paddingRight: '12px' }}>
                    <div style={{ fontSize: '13px', color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {(r.input_text || 'No prompt').slice(0, 60)}
                    </div>
                    <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px' }}>
                      {r.request_id}
                    </div>
                  </div>
                  <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {r.model_name || '—'}
                  </div>
                  <div><Badge label={r.archetype || 'STABLE'} /></div>
                  <EntropyBar value={entropy} />
                  <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color: 'var(--text-muted)' }}>{time}</div>
                </div>
              )
            })
          )}
        </div>
      </div>

      {selected && <DetailDrawer record={selected} onClose={() => setSelected(null)} />}
    </>
  )
}