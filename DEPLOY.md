# Deploying Nomo v5

Backend: `https://nomo-engine.onrender.com` (Render, free plan)
Dashboard: Vercel (Next.js, `frontend/`)

## 1. Update the Render service to v5

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

New in v4 (all optional):

| variable | set to | why |
|---|---|---|
| `MALLOC_ARENA_MAX` | `2` (already set in the Dockerfile) | keeps memory use low with worker threads |
| `NOMO_ANTHROPIC_API_KEY` | an Anthropic API key | Copilot phrases answers with Claude (paid, per use). Unset = free built-in answers |
| `NOMO_COPILOT_MODEL` | default `claude-haiku-4-5-20251001` | model used when the key is set |
| `NOMO_MAX_UPLOADS` | `20` | uploaded models kept in memory (oldest dropped first) |
| `NOMO_MAX_UPLOAD_MB` | `50` | largest accepted upload |
| `NOMO_DISABLE_COREML` | `1` to switch off | frees memory if the server is ever short; the Core ML option then shows as unavailable |

Check: `https://nomo-engine.onrender.com/` returns `{"service":"nomo-backend","version":"0.5.0",...}` after Render has deployed the release commit.

**Python-runtime services** (not Docker) must also install the export packages; `requirements.txt`
lists `onnx`, `reportlab`, `pyyaml`, and optional Core ML tooling. Core ML `.mlpackage` generation
requires the native ML-storage extension; on Linux the API reports that format as unavailable when it is
not present.

## 2.5 v5 API and release path

- `POST /models/{id}/calibration?filename=calibration.npz` accepts 100–500 finite input tensors. Use
  `.npz` keys `inputs`/`X`/`x`/`data`, plus `aux`/`A`/`metadata` when a model has guard metadata.
- `GET /runs/{id}/workbench` returns the six-level `nomo.workbench/1` state. `POST /runs/{id}/workbench/targets`
  reweights the cached feasible candidates and records the new target weights.
- `POST /runs/{id}/export` accepts `{ "key": "recommended", "formats": ["enterprise", "pdf", "c11"],
  "archive": "zip" }` or `"tar.gz"`.
- `nomo-cli pipeline --config nomo.yaml --out release.tar.gz` runs search, optional calibration-driven PTQ,
  and the same structured release builder offline.

## 3. Dashboard on Vercel

`frontend/.env.production` already points at `https://nomo-engine.onrender.com`, so a plain
redeploy works. A `NEXT_PUBLIC_NOMO_API` variable set in Vercel overrides it (for another backend).
Vercel project settings: Root Directory = `frontend`, framework = Next.js.
After deploying, set `NOMO_CORS_ORIGINS` on Render to the Vercel URL.

Pages: `/` launcher · `/runs/<id>` live search, Workbench, and exports · `/admin` logs, users, runs, stats
(needs the token).

## 4. Free-plan behaviour
- **Sleeps after ~15 min idle.** The dashboard now shows "waking the server… Ns" and retries for up to
  2 minutes, so the first visit just waits instead of failing. Open it a minute before a demo.
- **Restarts wipe memory:** runs, the admin log rings and user registry reset on each deploy/wake.
  Render's own **Logs** tab keeps the stdout copy of every record (see OBSERVABILITY.md).
- **0.1 CPU:** a 64×60 search takes tens of seconds instead of ~2 s locally. Exporting the camera model in
  every format takes about 4 s locally, so expect up to a minute on the free plan.
- **512 MB memory:** measured worst case is 382 MB (camera search plus all export formats). Exports run one at
  a time; a second export waits up to 2 minutes, then gets a "server is busy" message.
- **Uploaded models live in memory,** so a restart forgets them: users re-upload after the server wakes.

## Local development
    pip install -e ".[server,dev]"
    python -m nomo.cli serve --port 8765        # backend
    cd frontend && echo "NEXT_PUBLIC_NOMO_API=http://127.0.0.1:8765" > .env.local && npm install && npm run dev

Alternative free host without a card: `deploy/huggingface/` (Docker Space, port 7860).
