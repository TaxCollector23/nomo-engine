# Nomo Engine — v3 - fixed

**v3 - fixed** — adds `render.yaml` + `Dockerfile` for the hosted Python backend, and makes the
optional C++ kernel build non-fatal (pure-Python fallback when no toolchain is present).

Tri-domain (ANN / SNN / symbolic) hardware-aware NSGA-II search + compiler to NIR and bare-metal C11.
Full math and design: `docs/SPEC.md`.

## Setup
    pip install -e ".[server,dev]"
    python setup.py build_ext --inplace     # optional C++ kernel; skipped automatically if no compiler
    pytest -q                               # 66 tests

## Use
    nomo search  --model perception_cnn --hardware akd1500 --out run.json
    nomo search  --model attitude_policy --hardware akd1500 --out run.json
    nomo compile --model attitude_policy --hardware akd1500 --genome run.json --out build/
    nomo serve   --port 8765                # WebSocket telemetry, protocol in SPEC §9

## Dashboard
    cd frontend && npm install && npm run dev      # http://localhost:3000, talks to `nomo serve`
See `frontend/README.md`; screenshots in `docs/img/`.

## Live demo deployment
See `DEPLOY.md` (Dockerfile + `render.yaml` included; the dashboard on Vercel needs this backend hosted separately).

Backend (hosted Python, Render blueprint): `render.yaml` → `nomo-backend`
Dashboard (Next.js, Vercel): https://frontend-gray-ten-c3tj1luab7.vercel.app
Prototype: https://github.com/TaxCollector23/nomo-ai

## Verification
- `pytest -q`: 66 tests — invariants/operators (property tests), non-dominated sort vs brute force,
  exact 3-D hypervolume vs Monte Carlo, pipelined latency bounds, routing LP vs closed forms,
  LUT/surrogate, symbolic projections, golden == C++ == C11 bit-exactness (C11 also under UBSan),
  NIR strict (stock `nir.read`) and extended round-trips, WebSocket resume/snapshot.
- Frontend: strict `tsc`, `next build`, headless-browser E2E against a live server.

## Status
- Energy/latency coefficients are PLACEHOLDERS; replace via CostLUT with on-device measurements.
- No snn-mlir / microTVM output (see SPEC §0); direct C11 backend + MLIR-ready manifest instead.
- C11 lowering: dense chains, 8-bit activations, rate-coded (L)IF. TTFS/conv/16-bit raise UnsupportedLowering.
