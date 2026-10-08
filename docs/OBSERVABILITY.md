# Nomo Engine observability guide

The dashboard has two observable paths: REST control/read APIs and a sequenced WebSocket telemetry stream. The
browser client keeps the connection state visible and validates protocol envelopes before rendering them.

## Run telemetry

The dashboard uses these API paths:

- `GET /catalog`, `GET /presets`, and `GET /runs` for the launcher.
- `POST /runs` to start a search; `POST /runs/{run_id}/stop` to request a stop.
- `GET /runs/{run_id}` and `GET /runs/{run_id}/designs/{design_key}` for metadata and the selected design.
- `POST /runs/{run_id}/copilot` for grounded answers and `POST /runs/{run_id}/export` for selected artifacts.
- `GET /ws/runs/{run_id}?since=<sequence>` for resumable telemetry.

The client-side wire definitions and guard are in `frontend/src/lib/telemetry/protocol.ts`; connection and replay
behavior are in `frontend/src/lib/telemetry/client.ts` and `useRunTelemetry.ts`. Sequence numbers allow a reconnecting
client to request events after the last accepted envelope.

## Admin logs

The `/admin` dashboard reads authenticated `/admin/summary`, `/admin/runs`, `/admin/logs`, and
`/admin/logs/download` endpoints. The admin token is entered by the operator and held in session storage only. It is
never committed, placed in a URL, or included in a client build.

## What the numbers mean

- Search progress, evaluations, hypervolume, feasible fraction, and the final design are runtime telemetry.
- Energy, latency, accuracy, and hardware coefficients can be proxies or calibrated values; the design payload carries
  provenance and assumptions.
- Browser-Lab predictions are local calculations and do not become server measurements.
- A successful frontend build or HTTP 200 does not prove a hardware measurement or compiler validation.

## Incident checklist

When a run appears stuck or incomplete, record the API URL, run id, last sequence number, connection state, and whether
the API returned `run.failed`. Do not paste admin tokens or private model files into public issues. Reproduce with a
built-in model first, then attach only the smallest safe configuration needed to diagnose the parser or search.
