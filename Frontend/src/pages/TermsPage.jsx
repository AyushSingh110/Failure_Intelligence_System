import LegalPage from '../components/LegalPage'

/**
 * Terms of service.
 *
 * Deliberately blunt about the limitations, because FIE is a security tool and
 * the failure mode of overselling one is that somebody deploys it in front of
 * real users and believes they are protected. The over-refusal and recall
 * numbers quoted here are the project's own published measurements.
 */
export default function TermsPage() {
  return (
    <LegalPage title="Terms of Service" updated="11 August 2026">
      <h2>What this is</h2>
      <p>
        Failure Intelligence Engine (FIE) is a free, open-source research project released
        under the Apache 2.0 licence. It is maintained by one person as a research and
        portfolio project, not a commercial product.
      </p>

      <h2>Read this before you rely on it</h2>
      <p>
        FIE is a guardrail, and guardrails create a false sense of security when their limits
        are not understood. The project publishes its own weaknesses, and they are significant:
      </p>
      <ul>
        <li>
          <strong>It over-blocks safe prompts.</strong> On standardised over-refusal benchmarks
          FIE flags <strong>53.6% of safe XSTest prompts</strong> and{' '}
          <strong>90.4% of OR-Bench-hard prompts</strong> as attacks. This is a documented,
          unsolved problem — and a 20-billion-parameter guard model fails the same test at 80%.
        </li>
        <li>
          <strong>It misses attacks.</strong> Macro recall is about 85.8% on decontaminated
          benchmarks. Gradient-optimised suffix attacks (GCG) are caught roughly 73.7% of the
          time, and harm described in euphemism evades detection about half the time.
        </li>
        <li>
          <strong>It is not a content moderator.</strong> Hate speech, harassment and self-harm
          content phrased as ordinary conversation are outside its design scope.
        </li>
        <li>
          <strong>It is not a black box.</strong> The source is public, so anyone who reads it
          can craft prompts that stay below threshold.
        </li>
      </ul>
      <p>
        Use FIE as one layer of defence, never the only one. Do not use it where a missed
        attack or a wrongly blocked request would cause harm — including safety-critical,
        medical, legal or financial decision-making — without independent review.
      </p>

      <h2>The hosted service</h2>
      <p>
        The hosted API and demo run on free infrastructure. There is{' '}
        <strong>no uptime guarantee, no support guarantee, and no data durability
        guarantee.</strong> The service may be slow, asleep, rate-limited, or discontinued at
        any time without notice. For anything you depend on, self-host it.
      </p>
      <p>Please do not:</p>
      <ul>
        <li>send production user data, credentials or personal information to the hosted instance</li>
        <li>use it to attack, overload or probe the infrastructure it runs on</li>
        <li>use it to develop attacks against systems you are not authorised to test</li>
        <li>resell access to the hosted endpoint as a paid service</li>
      </ul>

      <h2>No warranty</h2>
      <p>
        FIE is provided <strong>"as is", without warranty of any kind</strong>, express or
        implied, including but not limited to warranties of merchantability, fitness for a
        particular purpose and non-infringement. To the maximum extent permitted by law, the
        author is not liable for any claim, damages or other liability arising from the use of
        this software or service — including any harm resulting from an attack it failed to
        detect or a legitimate request it wrongly blocked.
      </p>
      <p>
        This mirrors the{' '}
        <a
          href="https://github.com/AyushSingh110/Failure_Intelligence_System/blob/main/LICENSE"
          target="_blank"
          rel="noreferrer"
        >
          Apache 2.0 licence
        </a>
        , which governs the software itself.
      </p>

      <h2>Your account</h2>
      <p>
        You are responsible for keeping your API key secret. You may delete your data or
        request account deletion at any time. Accounts may be removed if used to abuse the
        service.
      </p>

      <h2>Research use</h2>
      <p>
        FIE is intended for security research, evaluation and defensive use. If you find a
        prompt that defeats it, or a harmless prompt it wrongly blocks, reporting it is
        genuinely welcome — over-refusal reports are currently the most useful contribution
        anyone can make.
      </p>

      <h2>Changes</h2>
      <p>
        These terms may change; the date at the top reflects the last revision. Continued use
        after a change constitutes acceptance.
      </p>
    </LegalPage>
  )
}
