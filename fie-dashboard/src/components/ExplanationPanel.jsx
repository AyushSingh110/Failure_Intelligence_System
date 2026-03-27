import { useMemo, useState } from 'react'

function SectionLabel({ children }) {
  return (
    <div style={{
      fontFamily: 'JetBrains Mono, monospace',
      fontSize: '9px',
      fontWeight: 600,
      letterSpacing: '0.12em',
      color: 'var(--text-muted)',
      marginBottom: '8px',
    }}>
      {children}
    </div>
  )
}

function ConfidenceMeter({ value, color = 'var(--accent-cyan)' }) {
  const pct = Math.max(0, Math.min(100, Math.round((value || 0) * 100)))
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
      <div style={{ flex: 1, height: '6px', borderRadius: '999px', background: 'rgba(255,255,255,0.06)' }}>
        <div style={{
          width: `${pct}%`,
          height: '100%',
          borderRadius: '999px',
          background: color,
          transition: 'width 0.35s ease',
        }} />
      </div>
      <span style={{
        minWidth: '42px',
        textAlign: 'right',
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: '11px',
        color,
      }}>
        {pct}%
      </span>
    </div>
  )
}

function SignalCard({ signal }) {
  const directionColor = signal.direction?.includes('failure')
    ? 'var(--accent-red)'
    : signal.direction?.includes('fix')
      ? 'var(--accent-cyan)'
      : signal.direction?.includes('stability')
        ? 'var(--accent-green)'
        : 'var(--text-secondary)'

  return (
    <div style={{
      padding: '12px',
      borderRadius: '10px',
      background: 'rgba(255,255,255,0.02)',
      border: '1px solid var(--border)',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', marginBottom: '6px' }}>
        <div>
          <div style={{
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '11px',
            fontWeight: 600,
            color: 'var(--text-primary)',
          }}>
            {signal.name?.replace(/_/g, ' ')}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
            {signal.source}
          </div>
        </div>
        <div style={{
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: '11px',
          color: directionColor,
          whiteSpace: 'nowrap',
        }}>
          {(signal.normalized_score || 0).toFixed(2)}
        </div>
      </div>
      <div style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
        {signal.summary}
      </div>
    </div>
  )
}

function StepCard({ step, index }) {
  return (
    <div style={{
      position: 'relative',
      padding: '12px 12px 12px 36px',
      borderRadius: '10px',
      background: 'rgba(255,255,255,0.02)',
      border: '1px solid var(--border)',
    }}>
      <div style={{
        position: 'absolute',
        top: '14px',
        left: '12px',
        width: '14px',
        height: '14px',
        borderRadius: '50%',
        border: '1px solid rgba(0,212,255,0.35)',
        color: 'var(--accent-cyan)',
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: '9px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        {index + 1}
      </div>
      <div style={{
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: '10px',
        color: 'var(--text-muted)',
        marginBottom: '4px',
      }}>
        {step.stage?.replace(/_/g, ' ')}
      </div>
      <div style={{ fontSize: '13px', color: 'var(--text-primary)', fontWeight: 600, marginBottom: '6px' }}>
        {step.decision}
      </div>
      <div style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
        {step.reason}
      </div>
      {step.inputs_used?.length > 0 && (
        <div style={{
          marginTop: '8px',
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: '10px',
          color: 'var(--text-muted)',
        }}>
          inputs: {step.inputs_used.join(', ')}
        </div>
      )}
    </div>
  )
}

export default function ExplanationPanel({ internalExplanation, externalExplanation, isAdmin = false }) {
  const tabs = useMemo(() => {
    const base = []
    if (externalExplanation) base.push({ key: 'external', label: 'User-safe' })
    if (isAdmin && internalExplanation) base.push({ key: 'internal', label: 'Internal' })
    return base
  }, [externalExplanation, internalExplanation, isAdmin])

  const [activeTab, setActiveTab] = useState(tabs[0]?.key || 'external')
  const explanation = activeTab === 'internal' ? internalExplanation : externalExplanation

  if (!explanation) return null

  return (
    <div style={{
      marginTop: '16px',
      padding: '16px',
      borderRadius: '12px',
      background: 'var(--bg-card)',
      border: '1px solid var(--border)',
      animation: 'kpiIn 0.35s ease both',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '12px', marginBottom: '14px' }}>
        <div>
          <div style={{
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '10px',
            fontWeight: 600,
            letterSpacing: '0.12em',
            color: 'var(--text-muted)',
            marginBottom: '6px',
          }}>
            EXPLAINABILITY
          </div>
          <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
            {explanation.summary}
          </div>
        </div>
        {tabs.length > 1 && (
          <div style={{ display: 'flex', gap: '6px' }}>
            {tabs.map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                style={{
                  padding: '6px 10px',
                  borderRadius: '999px',
                  border: `1px solid ${activeTab === tab.key ? 'rgba(0,212,255,0.28)' : 'var(--border)'}`,
                  background: activeTab === tab.key ? 'rgba(0,212,255,0.12)' : 'transparent',
                  color: activeTab === tab.key ? 'var(--accent-cyan)' : 'var(--text-muted)',
                  fontFamily: 'JetBrains Mono, monospace',
                  fontSize: '11px',
                  cursor: 'pointer',
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>
        )}
      </div>

      <div style={{ marginBottom: '14px' }}>
        <SectionLabel>EXPLANATION CONFIDENCE</SectionLabel>
        <ConfidenceMeter value={explanation.explanation_confidence} />
      </div>

      {explanation.attributions?.length > 0 && (
        <div style={{ marginBottom: '14px' }}>
          <SectionLabel>TOP FACTORS</SectionLabel>
          <div style={{ display: 'grid', gap: '8px' }}>
            {explanation.attributions.map((item) => (
              <div key={`${item.factor}-${item.details}`} style={{
                padding: '10px 12px',
                borderRadius: '10px',
                border: '1px solid var(--border)',
                background: 'rgba(255,255,255,0.02)',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '10px', marginBottom: '4px' }}>
                  <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color: 'var(--text-primary)' }}>
                    {item.factor?.replace(/_/g, ' ')}
                  </span>
                  <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color: 'var(--accent-cyan)' }}>
                    {(item.impact_score || 0).toFixed(2)}
                  </span>
                </div>
                <div style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                  {item.details}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {explanation.decision_trace?.length > 0 && (
        <div style={{ marginBottom: '14px' }}>
          <SectionLabel>DECISION TRACE</SectionLabel>
          <div style={{ display: 'grid', gap: '8px' }}>
            {explanation.decision_trace.map((step, index) => (
              <StepCard key={`${step.stage}-${index}`} step={step} index={index} />
            ))}
          </div>
        </div>
      )}

      {explanation.signals?.length > 0 && (
        <div style={{ marginBottom: '14px' }}>
          <SectionLabel>SIGNALS</SectionLabel>
          <div style={{ display: 'grid', gap: '8px', gridTemplateColumns: '1fr 1fr' }}>
            {explanation.signals.map((signal) => (
              <SignalCard key={`${signal.name}-${signal.source}`} signal={signal} />
            ))}
          </div>
        </div>
      )}

      {explanation.evidence?.length > 0 && (
        <div style={{ marginBottom: '14px' }}>
          <SectionLabel>EVIDENCE</SectionLabel>
          <div style={{ display: 'grid', gap: '8px' }}>
            {explanation.evidence.map((item, index) => (
              <div key={`${item.type}-${index}`} style={{
                padding: '12px',
                borderRadius: '10px',
                border: '1px solid var(--border)',
                background: 'rgba(255,255,255,0.02)',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', marginBottom: '6px' }}>
                  <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color: 'var(--text-primary)' }}>
                    {item.type?.replace(/_/g, ' ')}
                  </span>
                  <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color: 'var(--text-muted)' }}>
                    {Math.round((item.confidence || 0) * 100)}%
                  </span>
                </div>
                <div style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                  {item.content_preview}
                </div>
                <div style={{ marginTop: '6px', fontSize: '11px', color: 'var(--text-muted)' }}>
                  source: {item.source}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {(explanation.alternatives_considered?.length > 0 || explanation.uncertainty_notes?.length > 0) && (
        <div style={{ display: 'grid', gap: '12px', gridTemplateColumns: '1fr 1fr' }}>
          <div>
            <SectionLabel>ALTERNATIVES</SectionLabel>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
              {(explanation.alternatives_considered || []).map((item) => (
                <span key={item} style={{
                  padding: '4px 8px',
                  borderRadius: '999px',
                  border: '1px solid var(--border)',
                  background: 'rgba(255,255,255,0.02)',
                  color: 'var(--text-muted)',
                  fontFamily: 'JetBrains Mono, monospace',
                  fontSize: '10px',
                }}>
                  {item.replace(/_/g, ' ')}
                </span>
              ))}
            </div>
          </div>
          <div>
            <SectionLabel>UNCERTAINTY NOTES</SectionLabel>
            <div style={{ display: 'grid', gap: '6px' }}>
              {(explanation.uncertainty_notes || []).map((note, index) => (
                <div key={`${note}-${index}`} style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                  {note}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
