# Nomo Engine delivery report

## Delivered in this release

The per-layer training slice now has a fair comparison baseline and auditable accounting. A “global baseline” is no longer BF16-only: it searches the same uniform precision, recompute, CPU-offload, and stage choices as the per-layer search. The UI separates the gain attributable to precision from the gain attributable to per-layer choices.

The cost model charges pipeline bubble, inter-stage activation transfer, and CPU activation offload. The latter is labelled as a PCIe/host bandwidth assumption and can be overridden in the planner inputs. Adaptive formatting avoids zero-looking timings/costs and shows estimated totals for a 1,000-step run beside per-step values.

The browser planner includes published config presets for Llama 3 8B, Llama 3 70B, and Mixtral 8x7B, defaults to Llama 3 8B, and keeps the tiny transformer only as an example. The Python reference and TypeScript browser engine agree on fixed-seed golden cases for all three.

The shared graph now also drives a real serving search. It selects per-node weight precision and per-attention-node KV-cache precision, charges tensor-parallel weight memory and conservatively replicated KV memory, applies latency/cost/quality constraints, respects node locks, and reports whether the search was exhaustive or bounded. The Python reference is `nomo-planner/nomo_planner/serving.py`; the browser port is `src/planner/serving.ts`.

The Phase 2 auditor is now a real Lab module. It parses Megatron and vLLM command lines plus DeepSpeed JSON into one canonical run record, attaches observed step-time/throughput/memory log metrics, supports core Megatron round-trip export, and marks absent fields or uncalibrated assumptions instead of filling them with invented values. The Python reference is `nomo-planner/nomo_planner/auditor.py`; the browser port is `src/planner/auditor.ts`.

## Verification

- Python: 21 tests passed.
- TypeScript production build: passed.
- Python/TypeScript parity: `npm run verify` passed with 14,248 checks and worst relative difference `4.37e-16`.
- Regression calibration checks: the existing 22 published calibration predictions still pass unchanged.

## Honest boundaries

The current formulas are engineering estimates. FP8 quality, GPU memory bandwidth, framework overhead, batching behavior, and cluster utilisation need customer measurements. With the current assumptions, Llama 3 8B and 70B show about 15.8% training precision gain and 0% additional training per-layer gain; the engine reports that result instead of attributing the FP8 gain to layer decisions. The serving tab is now a real estimate/recommendation surface, but it remains calibration-dependent. Neuromorphic remains a contract surface until its shared-graph recommendation/export integration is implemented.

## Coverage of the pasted development prompt

| Requirement | Delivery status |
|---|---|
| Gate 0 Python-first layer accounting, fair baseline, cost model, presets, parity | Built and verified |
| Gate 0 expected positive per-layer gain and 1440px/390px screenshot evidence | Not fully satisfied; current assumptions produce 0% extra layer gain and viewport evidence remains pending |
| Phase 1 shared graph across Train, Serve, and Neuromorphic | Partial; Train and Serve are real, Neuromorphic links to the existing dashboard |
| Phase 2 config auditor and customer-log calibration | Partial; auditor and log attachment are built, customer-log calibration/refitting remains |
| Phase 3 full uncertainty trust layer and experiment designer | Partial foundation only |
| Phase 4 chip/RL/reliability/fleet/fine-tuning/TCO products | Not built |
| Phase 5 public cited cost tracker | Not built |
| Final browser matrix, zip bundles, and all phase gate reports | Not complete |

The repository and live sites therefore represent a verified Gate 0 implementation plus the restored multi-mode product shell, not completion of the entire pasted roadmap. No later feature is represented as implemented without its engine, tests, and evidence.

Current live deployment after this release: `dpl_39PKHAJUyv7uk53B15S8rhXnFpwK`. The public mode shell's Serve tab is smoke-tested and its Open Engine link still targets the neuromorphic dashboard.

## Deployment

The engine repository is deployed from `main`. The [frontend-gray-ten URL](https://frontend-gray-ten-c3tj1luab7.vercel.app/) is restored as the multi-mode shell and Lab, latest deployment `dpl_GoSQEvdBYdS6mWPCnqGkJg8h8NDV`; its Neuromorphic chips mode links to the separate [neuromorphic dashboard](https://nomo-engine-dashboard.vercel.app/), deployment `dpl_CyRWoKMFj8fjxx77VP5Y7BeXDgLS`. The [nomo-ai landing page](https://nomoailanding.vercel.app/) was refreshed successfully as deployment `dpl_9xqFV18tFwCVMRKQhV3AQqZ5E6C4`, and its Open Engine links point to the mode shell.
