# Nomo Engine progress

Current phase: Gate 0 — per-layer training integrity

Status: Gate 0 implementation and parity checks complete. Engine and landing deployments are live.

## Completed

- Fair global comparison: the reference and browser planners select the best uniform-precision/recompute/offload/stage plan, and report precision gain separately from per-layer gain.
- Cost accounting: pipeline bubble and inter-stage communication are charged per adjacent stage; CPU activation offload charges the configured PCIe/host bandwidth and is called out as a customer-overridable assumption.
- Recommendation guardrails: ample-memory cases do not choose offload; stage-count selection prefers the minimum feasible count unless the charged time model shows a documented >5% benefit.
- Published presets: Meta Llama 3 8B, Meta Llama 3 70B, and Mixtral 8x7B. Llama 3 8B is the default; the tiny config is explicitly an example.
- Adaptive time/cost formatting and whole-run totals for the assumed 1,000-step run.
- Removed the former missing-interval answer-card label; uncalibrated packs are now described as point estimates with no calibrated interval claim.
- Added fixed-seed Python/TypeScript golden parity cases for all three presets.

## Verification evidence

- `python -m pytest -q` in `nomo-planner`: 13 passed.
- `npm run build`: passed.
- `npm run verify`: passed, 14,248 checks, worst relative difference `4.37e-16`.
- Llama 3 8B and 70B fixed-seed searches are reproducible in the reference and browser engines. Under the current model, precision gain is about 15.8%; per-layer gain is 0% because the current assumptions make uniform FP8 the fastest feasible precision choice. This is reported rather than inflated.

## Remaining / next

- Serve and neuromorphic per-layer recommendation packs remain Preview/contract surfaces; no recommendation is fabricated there.
- Auditor/log calibration, calibrated uncertainty for new packs, product packs, and the cost tracker are not yet implemented.
- Live smoke check: the engine URL resolves to the neuromorphic dashboard and the landing page's Open Engine links resolve to it. Dedicated 1440px/390px screenshot evidence remains a follow-up because the current browser harness does not expose viewport controls in this run.
