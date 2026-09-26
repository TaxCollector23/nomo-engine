# Logs and telemetry (v3)

Every log record is one JSON object with these core fields:

```json
{"ts": 1790452403.42, "time": "2026-09-26T19:53:23.422Z", "level": "INFO",
 "stream": "access", "event": "http.request", "...": "event-specific fields"}
```

## Streams and events

| stream | events | key fields |
|---|---|---|
| `backend` | `backend.start`, `backend.stop`, `admin.disabled`, `cors.open`, `admin.denied`, `log.files_disabled` | version, pid, effective config, uptime, interrupted runs |
| `access` | `http.request` (INFO; WARNING for 4xx; ERROR for 5xx), `http.unhandled` | request_id, method, path, query, status, duration_ms, client_id, ip_hash, origin |
| `users` | `user.first_seen` | client_id, ip_hash, user_agent, origin |
| `runs` | `run.created`, `run.started`, `run.progress`, `run.completed`, `run.stopped`, `run.failed`, `run.rejected`, `run.stop_requested`, `run.evicted` | run_id, client_id, config, gen, hv, front, unique, wall_s, recommended design and its (energy, latency, accuracy), error + traceback |
| `telemetry` | `ws.open`, `ws.close`, `ws.overflow`, `ws.not_found`, `ws.error` | session, run_id, client_id, since, mode (full / replay / snapshot), backlog, sent, duration_s, close reason |
| `errors` | copy of every ERROR/CRITICAL record from any stream | includes `exc` (traceback) |

`/healthz` hits are not logged (Render polls it constantly); set `NOMO_LOG_HEALTH=1` to include them.

## Where to read them

1. **Dashboard → /admin** (token = `NOMO_ADMIN_TOKEN`): filter by stream, level and text; live tail;
   download NDJSON; users, runs and stats tabs.
2. **API** (`Authorization: Bearer <token>`):
   ```
   curl -H "Authorization: Bearer $TOKEN" "https://nomo-engine.onrender.com/admin/logs?stream=runs&limit=100"
   curl -H "Authorization: Bearer $TOKEN" "https://nomo-engine.onrender.com/admin/logs/download?stream=all" -o nomo.ndjson
   curl -H "Authorization: Bearer $TOKEN" https://nomo-engine.onrender.com/admin/users
   curl -H "Authorization: Bearer $TOKEN" https://nomo-engine.onrender.com/admin/runs
   curl -H "Authorization: Bearer $TOKEN" https://nomo-engine.onrender.com/admin/stats
   ```
   Filters: `stream=all|backend|access|users|runs|telemetry|errors`, `limit` (≤5000), `level=WARNING`,
   `since_ts=<unix seconds>`, `contains=<text>`.
3. **Render → service → Logs**: the stdout copy of every record. It survives restarts, subject to Render's own retention.
   Search it by event name, e.g. `run.failed` or a `run_id`.
4. **Files** in `NOMO_LOG_DIR` (`/tmp/nomo-logs` in the container): `<stream>.log`, 5 MB × 3 rotation.
   On Render's free plan the disk is ephemeral, so treat these as a local-development convenience.

## Users and privacy

There are no accounts. The dashboard creates a random id per browser (localStorage `nomo.client_id`) and
sends it as `X-Nomo-Client` / `?client=`. Clients without one (curl, scripts) are grouped by
`anon-<ip_hash>`. IPs are stored as a salted SHA-256 prefix unless `NOMO_LOG_RAW_IP=1`. User agents
and origins are kept (truncated to 200 chars). If you add real accounts later, log the account id in
the same `client_id` field and every view keeps working.

## Correlating a problem

Every HTTP response carries `X-Request-ID`. A user report → find the request in `access` →
its `client_id` → that user's `run.created` → the `run_id` → `run.*` and `ws.*` events for that run.
