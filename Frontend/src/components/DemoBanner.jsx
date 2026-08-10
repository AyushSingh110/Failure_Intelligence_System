import { useEffect, useState } from 'react'
import { AlertTriangle } from 'lucide-react'
import { isBackendOffline, onBackendStatusChange } from '../lib/api'

/**
 * Persistent banner shown whenever the dashboard is rendering bundled sample
 * data because the API is unreachable.
 *
 * This exists to keep the fallback honest. Silently substituting fixture data
 * would be worse than the original broken state — a visitor would read sample
 * numbers as real detections. The banner is deliberately non-dismissible and
 * high-contrast: if you are looking at demo data, you should always know it.
 */
export default function DemoBanner() {
  const [offline, setOffline] = useState(isBackendOffline())

  useEffect(() => onBackendStatusChange(setOffline), [])

  if (!offline) return null

  return (
    <div
      role="status"
      aria-live="polite"
      className="sticky top-0 z-50 flex items-center gap-3 border-b border-amber-500/40 bg-amber-500/10 px-4 py-2 text-sm text-amber-200 backdrop-blur"
    >
      <AlertTriangle className="h-4 w-4 shrink-0" aria-hidden="true" />
      <p className="leading-snug">
        <span className="font-semibold">Demo data.</span>{' '}
        The API backend is unreachable, so this dashboard is showing bundled
        sample traffic — not live detections. Published evaluation numbers are in{' '}
        <a
          href="https://github.com/AyushSingh110/Failure_Intelligence_System/blob/main/docs/RESEARCH_LOG.md"
          target="_blank"
          rel="noreferrer"
          className="underline underline-offset-2 hover:text-amber-100"
        >
          docs/RESEARCH_LOG.md
        </a>
        .
      </p>
    </div>
  )
}
