# Nomo Engine — v3 - fixed (v0.3)

**v3 - fixed** — hosted Python backend via `render.yaml` + `Dockerfile`, and v0.3 observability:
structured JSON logging, anonymous user tracking, and a token-protected admin API + `/admin` page.

Tri-domain (ANN / SNN / symbolic) hardware-aware NSGA-II search + compiler to NIR and bare-metal C11.
Full math and design: `docs/SPEC.md`.

## Setup
    pip install -e ".[server,dev]"
    python setup.py build_ext --inplace     # optional C++ kernel; skipped automatically if no compiler
    pytest -q                               # 73 tests

## Use
    nomo search  --model perception_cnn --hardware akd1500 --out run.json
    nomo search  --model attitude_policy --hardware akd1500 --out run.json
    nomo compile --model attitude_policy --hardware akd1500 --genome run.json --out build/
    nomo serve   --port 8765                # WebSocket telemetry, protocol in SPEC §9

## Dashboard
    cd frontend && npm install && npm run dev      # http://localhost:3000, talks to `nomo serve`
See `frontend/README.md`; screenshots in `docs/img/`.

## Live deployment
Backend: https://nomo-engine.onrender.com · setup and environment variables in `DEPLOY.md`.
Logs, users and telemetry: `OBSERVABILITY.md` · admin UI at `<dashboard>/admin`.

| | |
|---|---|
| Engine — repo | https://github.com/TaxCollector23/nomo-engine |
| Engine — live dashboard | https://frontend-gray-ten-c3tj1luab7.vercel.app |
| Engine — release | https://github.com/TaxCollector23/nomo-engine/releases/tag/v3-fixed |
| Prototype — repo | https://github.com/TaxCollector23/nomo-ai |

## What's new in v0.3
- Structured JSON logging in six streams (backend, access, users, runs, telemetry, errors) to stdout,
  in-memory rings and rotating files.
- Anonymous user tracking (per-browser id, hashed IPs), per-client rate limit, size caps.
- Token-protected admin API and `/admin` page: filterable live logs, NDJSON export, users, runs, stats.
- Dashboard waits through free-tier cold starts, defaults to the Render backend, sends client ids.
- Root `/` service endpoint; lifespan shutdown logging; `nomo/__init__.py` restored (was missing).

## Verification
- `pytest -q`: 73 tests (7 new observability tests) — invariants/operators (property tests), non-dominated sort vs brute force,
  exact 3-D hypervolume vs Monte Carlo, pipelined latency bounds, routing LP vs closed forms,
  LUT/surrogate, symbolic projections, golden == C++ == C11 bit-exactness (C11 also under UBSan),
  NIR strict (stock `nir.read`) and extended round-trips, WebSocket resume/snapshot.
- Frontend: strict `tsc`, `next build`, headless-browser E2E against a live server.

## Status
- Energy/latency coefficients are PLACEHOLDERS; replace via CostLUT with on-device measurements.
- No snn-mlir / microTVM output (see SPEC §0); direct C11 backend + MLIR-ready manifest instead.
- C11 lowering: dense chains, 8-bit activations, rate-coded (L)IF. TTFS/conv/16-bit raise UnsupportedLowering.
