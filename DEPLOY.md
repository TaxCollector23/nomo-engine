# Deploying a live demo

Two pieces, two hosts:

| piece | what it is | where |
|---|---|---|
| `frontend/` | Next.js dashboard | Vercel (already deployed) |
| `nomo` backend | long-running Python process holding WebSockets | any container host with WebSocket support (Render, Railway, Fly.io, Cloud Run, a VM) |

Vercel's serverless functions cannot host the backend: searches run for seconds to minutes and
the telemetry stream is a persistent WebSocket.

## 1. Backend (example: Render, using `render.yaml`)
1. Push this repo to GitHub.
2. Render → New → Blueprint → pick the repo. It builds the `Dockerfile` and health-checks `/healthz`.
3. When it's live, open `https://<your-service>.onrender.com/catalog`; you should see JSON.

Any other host: build the `Dockerfile`, expose the port in `$PORT`, run **exactly one instance**
(runs live in memory; a second replica would not see the first one's runs).

## 2. Connect the dashboard
1. Vercel → project → Settings → Environment Variables:
   `NEXT_PUBLIC_NOMO_API = https://<your-service>.onrender.com` (https; the client derives `wss://`).
2. Redeploy. `NEXT_PUBLIC_*` values are baked in at build time, so a redeploy is required.
3. Lock the backend to your dashboard: set `NOMO_CORS_ORIGINS=https://<your-vercel-domain>`
   on the backend (comma-separate several origins).

## 3. Know the limits
- **Free tiers sleep.** The first request after idle can take ~30-60 s while the container wakes,
  and a restart discards in-memory runs. For a pitch, open the page a minute beforehand or use a paid always-on instance.
- **Public endpoint.** Anyone with the URL can start searches. `NOMO_MAX_ACTIVE_RUNS` (default 2)
  caps concurrency (extra requests get HTTP 429); `NOMO_MAX_RUNS_KEPT` (default 50) bounds memory.
  Add auth before sharing the URL widely.
- **Domain.** The auto-generated `*.vercel.app` name persists with the project; add a custom
  domain in Vercel → Settings → Domains for a name you control.
