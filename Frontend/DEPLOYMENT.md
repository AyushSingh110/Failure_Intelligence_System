# FIE Frontend Deployment

## Platform: Cloudflare Pages

The frontend (React/Vite) now includes both the landing page and the dashboard in one project.
No separate static deployment needed.

---

## Architecture

```text
yourdomain.com          →  /               LandingPage.jsx  (public, no auth)
yourdomain.com/login    →  /login          LoginPage.jsx    (Google OAuth)
yourdomain.com/dashboard →  /dashboard     DashboardPage    (protected)
yourdomain.com/analyze   →  /analyze       AnalyzePage      (protected)
... etc
```

All routes are handled by React Router inside a single Cloudflare Pages deployment.

---

## Cloudflare Pages — Project Settings

| Setting        | Value                                     |
| -------------- | ----------------------------------------- |
| Repository     | AyushSingh110/Failure_Intelligence_System |
| Root directory | `Frontend`                                |
| Build command  | `npm run build`                           |
| Output dir     | `dist`                                    |
| Branch         | `main`                                    |

### SPA routing fix (required)

React Router needs all paths to serve `index.html`. Create this file:

**`Frontend/public/_redirects`**

```text
/*    /index.html   200
```

This is already created — Cloudflare Pages uses it to serve the React app for every URL.

---

## Environment variables (set in Cloudflare Pages dashboard)

```env
VITE_API_URL=https://failure-intelligence-system-800748790940.asia-south1.run.app/api/v1
VITE_GOOGLE_CLIENT_ID=800748790940-m6h2foakt6g2an2iss8974ekb00hjjo7.apps.googleusercontent.com
VITE_REDIRECT_URI=https://yourdomain.com
```

**Important:** `VITE_REDIRECT_URI` must exactly match what Google has in its authorized redirect URIs.

---

## Google OAuth setup

In Google Cloud Console → APIs & Services → Credentials → your OAuth 2.0 client:

**Authorized JavaScript origins:**

```text
https://yourdomain.com
```

**Authorized redirect URIs:**

```text
https://yourdomain.com
```

Google redirects back to the root `/`. The landing page detects `?code=` and forwards to `/login?code=...` automatically.

---

## Steps to deploy / update

1. Push to `main` → Cloudflare Pages auto-builds and deploys (~60–90 seconds)
2. No manual steps needed after initial setup

## Initial Cloudflare Pages setup

1. Cloudflare Dashboard → Pages → Create a project → Connect to Git
2. Select `AyushSingh110/Failure_Intelligence_System`
3. Set **root directory** = `Frontend`
4. Build command: `npm run build` · Output dir: `dist`
5. Add environment variables (from the table above)
6. Deploy
7. Add custom domain in Pages → Custom domains

---

## Backend

Running on Google Cloud Run (Asia South 1) — no changes needed:

```text
https://failure-intelligence-system-800748790940.asia-south1.run.app
```
