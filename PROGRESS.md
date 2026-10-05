# Nomo Engine progress

Current phase: Gate 0 closeout / Phase 1 serving slice complete

Status: Gate 0 implementation is complete, but its acceptance checklist is not fully closed: the expected positive per-layer gain is not demonstrated under the current assumptions, and dedicated 1440px/390px screenshot evidence is still pending. The multi-mode shell and neuromorphic dashboard are live.

## Completed

- Fair global comparison: the reference and browser planners select the best uniform-precision/recompute/offload/stage plan, and report precision gain separately from per-layer gain.
- Cost accounting: pipeline bubble and inter-stage communication are charged per adjacent stage; CPU activation offload charges the configured PCIe/host bandwidth and is called out as a customer-overridable assumption.
- Recommendation guardrails: ample-memory cases do not choose offload; stage-count selection prefers the minimum feasible count unless the charged time model shows a documented >5% benefit.
- Published presets: Meta Llama 3 8B, Meta Llama 3 70B, and Mixtral 8x7B. Llama 3 8B is the default; the tiny config is explicitly an example.
- Adaptive time/cost formatting and whole-run totals for the assumed 1,000-step run.
- Removed the former missing-interval answer-card label; uncalibrated packs are now described as point estimates with no calibrated interval claim.
- Added fixed-seed Python/TypeScript golden parity cases for all three presets.

## Verification evidence

- `python -m pytest -q` in `nomo-planner`: 17 passed.
- `npm run build`: passed.
- `npm run verify`: passed, 14,248 checks, worst relative difference `4.37e-16`.
- Llama 3 8B and 70B fixed-seed searches are reproducible in the reference and browser engines. Under the current model, precision gain is about 15.8%; per-layer gain is 0% because the current assumptions make uniform FP8 the fastest feasible precision choice. This is reported rather than inflated.

## Remaining / next

- Phase 1 serving slice is now real: the shared graph drives per-node weight precision and attention KV-cache precision, with memory, latency, cost, quality-budget constraints, locks, and bounded/exhaustive search reported in the UI. Neuromorphic remains a link to the existing validated dashboard rather than a shared-graph recommendation/export surface.
- Phase 2 auditor/log parsing and customer-log calibration are not implemented.
- Phase 3 has the earlier calibrated-training uncertainty artifact, but the full experiment designer and end-to-end trust layer are not implemented.
- Phase 4 products (chip design, RL post-training, reliability/goodput, serving fleets, fine-tuning, and TCO/procurement) are not implemented.
- Phase 5 public cost tracker is not implemented.
- Final zip packaging and full 1440px/390px browser evidence are not complete.
- Live smoke check: the public mode shell loads, its Serve tab shows a bounded per-layer recommendation, and its Open Engine link resolves to the neuromorphic dashboard. Dedicated 1440px/390px screenshot evidence remains a follow-up because the current browser harness does not expose viewport controls in this run.

## Prompt completion matrix

| Scope | Status | Evidence / limitation |
|---|---|---|
| Gate 0 reference + TypeScript cost/baseline work | Built | 17 pytest tests, `npm run build`, 14,248 parity checks |
| Gate 0 acceptance | Partial | 0% extra per-layer gain for Llama 3 8B/70B under current assumptions; viewport screenshots pending |
| Phase 1 shared graph | Partial | Train and Serve recommendations are real; Neuromorphic shared-graph recommendation/export remains pending |
| Phase 2 auditor/calibration | Not built | No parser, round-trip fixtures, or customer-log refit |
| Phase 3 uncertainty | Partial foundation | Existing calibrated-training bootstrap only; no experiment designer/full coverage |
| Phase 4 product packs | Not built | No new packs beyond existing planner domains |
| Phase 5 cost tracker | Not built | No cited public tracker |
| Final delivery zips | Not built | Reports and repositories are updated; zips are still outstanding |
