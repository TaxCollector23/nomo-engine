# Deploying Nomo v3

Backend: `https://nomo-engine.onrender.com` (Render, free plan)
Dashboard: Vercel (Next.js, `frontend/`)

## 1. Update the Render service to v3

Replace the repository contents with this folder, commit, push. Render redeploys automatically
if auto-deploy is on (otherwise: service → Manual Deploy → Deploy latest commit).

Which runtime is your service using? (Render dashboard → service → Settings)

| runtime | settings |
|---|---|
| **Docker** (recommended) | Dockerfile path `./Dockerfile`. Nothing else to set. |
| **Python 3** | Build command `pip install -r requirements.txt` · Start command `uvicorn nomo.telemetry.server:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips=* --no-access-log` |

Health check path (Settings → Health Checks): `/healthz`

## 2. Environment variables (Render → service → Environment)

| variable | set to | why |
|---|---|---|
| `NOMO_ADMIN_TOKEN` | a long random string (password manager) | unlocks `/admin` and the dashboard's admin page. **Unset = admin disabled.** |
| `NOMO_CORS_ORIGINS` | `https://<your-app>.vercel.app` (comma-separate several) | only your dashboard can call the API from a browser |
| `NOMO_MAX_ACTIVE_RUNS` | `2` | concurrent searches (free plan has 0.1 CPU) |
| `NOMO_MAX_POP` / `NOMO_MAX_GENS` | `128` / `150` | per-search size caps (HTTP 422 above) |
| `NOMO_MAX_RUNS_PER_CLIENT_HOUR` | `30` | per-browser rate limit (HTTP 429) |
| `NOMO_LOG_LEVEL` | `INFO` | `DEBUG` for more |
| `NOMO_LOG_GEN_EVERY` | `5` | log search progress every N generations |
| `NOMO_LOG_RAW_IP` | `0` | `1` stores raw IPs instead of salted hashes (check your privacy obligations first) |
| `NOMO_LOG_SALT` | optional random string | keeps IP hashes stable across restarts |

Check: `https://nomo-engine.onrender.com/` returns `{"service":"nomo-backend","version":"0.3.0",...}`.

## 3. Dashboard on Vercel

`frontend/.env.production` already points at `https://nomo-engine.onrender.com`, so a plain
redeploy works. A `NEXT_PUBLIC_NOMO_API` variable set in Vercel overrides it (for another backend).
Vercel project settings: Root Directory = `frontend`, framework = Next.js.
After deploying, set `NOMO_CORS_ORIGINS` on Render to the Vercel URL.

Pages: `/` launcher · `/runs/<id>` live search · `/admin` logs, users, runs, stats (needs the token).

## 4. Free-plan behaviour
- **Sleeps after ~15 min idle.** The dashboard now shows "waking the server… Ns" and retries for up to
  2 minutes, so the first visit just waits instead of failing. Open it a minute before a demo.
- **Restarts wipe memory:** runs, the admin log rings and user registry reset on each deploy/wake.
  Render's own **Logs** tab keeps the stdout copy of every record (see OBSERVABILITY.md).
- **0.1 CPU:** a 64×60 search takes tens of seconds instead of ~2 s locally.

## Local development
    pip install -e ".[server,dev]"
    python -m nomo.cli serve --port 8765        # backend
    cd frontend && echo "NEXT_PUBLIC_NOMO_API=http://127.0.0.1:8765" > .env.local && npm install && npm run dev

Alternative free host without a card: `deploy/huggingface/` (Docker Space, port 7860).
