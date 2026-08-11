import LegalPage from '../components/LegalPage'

/**
 * Privacy policy.
 *
 * Written from what the code actually does, not from a template:
 *  - app/auth_routes.py  -> the three Google profile fields stored
 *  - app/schemas.py      -> InferenceRequest fields persisted to MongoDB
 *  - fie/_telemetry.py   -> the exact anonymous ping payload
 *  - fie/adversarial.py  -> scan cache keys on SHA-256, never raw prompt text
 *  - app/main.py         -> Sentry runs with send_default_pii=False
 *
 * If any of those change, this page must change with them.
 */
export default function PrivacyPage() {
  return (
    <LegalPage title="Privacy Policy" updated="11 August 2026">
      <h2>Summary</h2>
      <p>
        Failure Intelligence Engine (FIE) is an open-source research project maintained by an
        individual, not a company. It is not funded by advertising and your data is never sold
        or shared with third parties for marketing. This page describes exactly what is
        collected, why, and how to avoid it.
      </p>

      <h2>The SDK collects almost nothing</h2>
      <p>
        If you install <code>fie-sdk</code> from PyPI and use it in <code>local</code> mode, the
        detector runs entirely on your machine. <strong>Your prompts never leave your
        computer.</strong> There is no API key, no account, and no network call in the
        detection path.
      </p>
      <p>
        The SDK sends one anonymous ping on import containing only:
      </p>
      <ul>
        <li>the FIE version (e.g. <code>1.18.0</code>)</li>
        <li>your operating system family (e.g. <code>Windows</code>)</li>
        <li>your Python version (e.g. <code>3.11.9</code>)</li>
      </ul>
      <p>
        No prompts, no IP-derived identity, no account, no unique device ID. It exists solely
        to tell the maintainer which versions are still in use. Disable it completely by
        setting <code>FIE_NO_TELEMETRY=1</code>.
      </p>

      <h2>If you sign in to the dashboard</h2>
      <p>Signing in with Google stores exactly three fields from your Google profile:</p>
      <ul>
        <li>your email address</li>
        <li>your display name</li>
        <li>your profile picture URL</li>
      </ul>
      <p>
        These identify your account and nothing else. FIE requests only the{' '}
        <code>openid</code>, <code>email</code> and <code>profile</code> scopes — it cannot read
        your Gmail, Drive, contacts or calendar, and never asks for that access.
      </p>
      <p>
        FIE does not receive or store your Google password. Authentication happens on Google's
        servers; FIE only receives a short-lived authorisation code.
      </p>

      <h2>If you send data to the hosted API</h2>
      <p>
        When you use the hosted API in <code>monitor</code> or <code>correct</code> mode, the
        prompts and model outputs you submit are stored so the dashboard can show your history:
        the prompt text, the model output, the model name, a timestamp, and detection results.
      </p>
      <p>
        <strong>Do not send production user data, secrets or personal information to the
        hosted instance.</strong> It is a research deployment on free infrastructure, run by
        one person, with no uptime or durability guarantee. If you need to monitor sensitive
        traffic, self-host it — the full source is on GitHub and the deployment guide is in the
        repository.
      </p>
      <p>
        The scan cache keys on a SHA-256 hash of the prompt rather than the text itself, so
        cached entries do not retain readable prompt content.
      </p>

      <h2>The public demo</h2>
      <p>
        Prompts entered in the demo are scanned in memory and are not written to the database.
        The Space host (Hugging Face) may keep standard server logs, which are outside FIE's
        control.
      </p>

      <h2>Third parties</h2>
      <ul>
        <li><strong>Google</strong> — sign-in only.</li>
        <li><strong>MongoDB Atlas</strong> — stores accounts and inference history.</li>
        <li><strong>Hugging Face</strong> — hosts the API and demo.</li>
        <li><strong>Cloudflare Pages</strong> — hosts this dashboard.</li>
        <li><strong>Groq</strong> — receives prompts only when hallucination monitoring is enabled, to run shadow models.</li>
        <li><strong>Sentry</strong> — optional error tracking, configured with <code>send_default_pii=False</code> so prompt text and user data are never transmitted.</li>
      </ul>

      <h2>Retention and deletion</h2>
      <p>
        You can delete individual inference records, or all of them, from the dashboard at any
        time. To delete your account entirely, email the address below and it will be removed
        along with its stored records.
      </p>

      <h2>Children</h2>
      <p>
        FIE is a developer tool and is not directed at anyone under 13.
      </p>

      <h2>Changes</h2>
      <p>
        Material changes will be reflected in the date at the top of this page. Because the
        project is fully open source, you can also audit exactly what is collected by reading{' '}
        <a
          href="https://github.com/AyushSingh110/Failure_Intelligence_System"
          target="_blank"
          rel="noreferrer"
        >
          the source code
        </a>
        .
      </p>
    </LegalPage>
  )
}
