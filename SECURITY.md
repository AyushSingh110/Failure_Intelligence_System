# Security Policy

## Supported Versions

| Version | Supported |
| --- | --- |
| fie-sdk v1.13.x (latest) | Yes |
| fie-sdk v1.12.x | Critical fixes only |
| fie-sdk v1.11.x | Critical fixes only |
| fie-sdk < v1.11.0 | No |

---

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability in FIE — whether in the SDK, the backend API, the pre-flight guard, or the dashboard — please report it privately so we can fix it before it is publicly disclosed.

### How to report

Email: **[ayushsingh355vns@gmail.com](mailto:ayushsingh355vns@gmail.com)**

Use the subject line: `[SECURITY] Brief description of the issue`

Include as much of the following as possible:

- A description of the vulnerability and its potential impact
- Steps to reproduce the issue
- Affected component (SDK / backend API / frontend / other)
- Any proof-of-concept code or examples (if safe to share)

### What to expect

- **Acknowledgement** within 48 hours of your report.
- **Status update** within 7 days — whether we confirmed the issue, need more information, or determined it is not a vulnerability.
- **Fix timeline** — critical vulnerabilities will be patched as fast as possible, typically within 14 days. You will be notified when a fix is released.
- **Credit** — if you would like to be credited in the release notes, let us know in your report.

---

## Scope

The following are in scope for security reports:

- Authentication bypass or privilege escalation in the dashboard or API
- Pre-flight guard bypass — a crafted prompt that evades adversarial detection
- API key exposure or leakage
- Injection vulnerabilities (SQL, command, prompt, indirect)
- Unauthorized access to another user's tenant data (MongoDB isolation)
- Denial-of-service vulnerabilities in the `/monitor` or `/playground` endpoints

The following are **out of scope**:

- Vulnerabilities in third-party dependencies (report those upstream)
- Issues that require physical access to the server
- Social engineering attacks
- Rate limiting bypass (we acknowledge current limits are basic)

---

## Disclosure Policy

We follow **coordinated disclosure**. We ask that you give us a reasonable amount of time to fix a confirmed vulnerability before publishing any details publicly. We will work with you to agree on an appropriate disclosure timeline.

---

## Security Design Notes

FIE is a security product. Here is how it protects itself:

- **API keys** are generated using Python's `secrets` module (cryptographically secure, 36^16 combinations).
- **JWT tokens** are signed with `HS256` and expire after 24 hours.
- **Tenant isolation** — all MongoDB queries are scoped to `tenant_id`. Users cannot query other tenants' data.
- **Pre-flight guard** — adversarial prompts are blocked server-side before reaching any LLM, regardless of SDK configuration.
- **Admin endpoints** require both valid authentication and `is_admin: true` on the user record.
- **Playground results** are not persisted to MongoDB — sandbox requests leave no trace in analytics.
