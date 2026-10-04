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

## Deployment

The engine repository is deployed from `main`; the Vercel engine project uses `frontend/` as its root so the live `frontend-gray-ten` URL remains the neuromorphic engine dashboard. The public landing project remains `nomo-ai`; it is refreshed only when landing-source changes are present.
