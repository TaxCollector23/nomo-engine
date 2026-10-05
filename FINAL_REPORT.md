# Nomo Engine delivery report

## Delivered in this release

The per-layer training slice now has a fair comparison baseline and auditable accounting. A “global baseline” is no longer BF16-only: it searches the same uniform precision, recompute, CPU-offload, and stage choices as the per-layer search. The UI separates the gain attributable to precision from the gain attributable to per-layer choices.

The cost model charges pipeline bubble, inter-stage activation transfer, and CPU activation offload. The latter is labelled as a PCIe/host bandwidth assumption and can be overridden in the planner inputs. Adaptive formatting avoids zero-looking timings/costs and shows estimated totals for a 1,000-step run beside per-step values.

The browser planner includes published config presets for Llama 3 8B, Llama 3 70B, and Mixtral 8x7B, defaults to Llama 3 8B, and keeps the tiny transformer only as an example. The Python reference and TypeScript browser engine agree on fixed-seed golden cases for all three.

## Verification

- Python: 13 tests passed.
- TypeScript production build: passed.
- Python/TypeScript parity: `npm run verify` passed with 14,248 checks and worst relative difference `4.37e-16`.
- Regression calibration checks: the existing 22 published calibration predictions still pass unchanged.

## Honest boundaries

The current formulas are engineering estimates. FP8 quality, PCIe/host bandwidth, framework overhead, and cluster utilisation need customer measurements. With the current assumptions, Llama 3 8B and 70B show about 15.8% precision gain and 0% additional per-layer gain; the engine reports that result instead of attributing the FP8 gain to layer decisions. Serve and neuromorphic tabs remain Preview/contract surfaces until their real recommendation packs are implemented.

## Coverage of the pasted development prompt

| Requirement | Delivery status |
|---|---|
| Gate 0 Python-first layer accounting, fair baseline, cost model, presets, parity | Built and verified |
| Gate 0 expected positive per-layer gain and 1440px/390px screenshot evidence | Not fully satisfied; current assumptions produce 0% extra layer gain and viewport evidence remains pending |
| Phase 1 shared graph across Train, Serve, and Neuromorphic | Partial; Train is real, Serve is not, Neuromorphic links to the existing dashboard |
| Phase 2 config auditor and customer-log calibration | Not built |
| Phase 3 full uncertainty trust layer and experiment designer | Partial foundation only |
| Phase 4 chip/RL/reliability/fleet/fine-tuning/TCO products | Not built |
| Phase 5 public cited cost tracker | Not built |
| Final browser matrix, zip bundles, and all phase gate reports | Not complete |

The repository and live sites therefore represent a verified Gate 0 implementation plus the restored multi-mode product shell, not completion of the entire pasted roadmap. No later feature is represented as implemented without its engine, tests, and evidence.

## Deployment

The engine repository is deployed from `main`. The [frontend-gray-ten URL](https://frontend-gray-ten-c3tj1luab7.vercel.app/) is restored as the multi-mode shell and Lab, latest deployment `dpl_64cuBerLt4ApVuno9NrNya2Wcqhw`; its Neuromorphic chips mode links to the separate [neuromorphic dashboard](https://nomo-engine-dashboard.vercel.app/), deployment `dpl_CyRWoKMFj8fjxx77VP5Y7BeXDgLS`. The [nomo-ai landing page](https://nomoailanding.vercel.app/) was refreshed successfully as deployment `dpl_DyTeRQ6HGrbDf6qfBvQDt895LVJD`, and its Open Engine links point to the mode shell.
