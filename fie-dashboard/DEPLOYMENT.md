# Frontend Deployment

## Vercel project settings

- Framework preset: `Vite`
- Root directory: `fie-dashboard`
- Build command: `npm run build`
- Output directory: `dist`

## Required environment variables

- `VITE_API_URL`
- `VITE_GOOGLE_CLIENT_ID`
- `VITE_REDIRECT_URI`

## Production values

```env
VITE_API_URL=https://failure-intelligence-system-800748790940.asia-south1.run.app/api/v1
VITE_GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
VITE_REDIRECT_URI=https://your-vercel-app.vercel.app
```

## After Vercel gives you the public URL

1. Update the backend `GOOGLE_REDIRECT_URI` to the same Vercel URL.
2. Add the Vercel URL to Google OAuth authorized origins.
3. Add the Vercel URL to Google OAuth authorized redirect URIs.
4. Redeploy backend if you changed backend environment variables.
