# Nomo Engine

The public landing page for [Nomo AI](https://github.com/TaxCollector23/nomo-ai), an open research and compiler project for hardware-aware neural architecture search, physical AI, and neuromorphic hardware.

The current engine is [Nomo Engine v5](https://github.com/TaxCollector23/nomo-engine). It searches energy, latency, accuracy, memory, and hardware constraints with constrained NSGA-II, streams run telemetry to the dashboard, and turns a selected design into a calibration-aware software/RTL release package.

## Current platform surface

- Tri-domain search across continuous, spiking, and symbolic execution with hard policy and hardware invariants.
- Model upload for ONNX, safe PyTorch state dictionaries, and `nomo.graph/1` JSON graphs, with architecture-family contracts for MLP, CNN, FNO, ViT/attention, GNN/message-passing, and hybrid workloads.
- Calibration upload for 100–500 finite tensors, MSE/KL post-training quantisation, and per-edge activation ranges carried into export manifests.
- Six-level Workbench: system topology, ANN/SNN/SYM partitioning, hardware graph, cycle-emulation boundary, RTL, and silicon-floorplan proxy state.
- Native integer deployment driver with embedded golden vectors, C11 microkernel/CMake, SystemVerilog/Chisel boundaries, and generic Yosys/OpenROAD scripts.
- Operational modes for low-power neuromorphic, hard real-time, radiation-hardened, and on-chip-learning deployments, plus a narrow authenticated HITL benchmark protocol.
- Zip or `tar.gz` packages with model, runtime, SDK, EDA, and validation evidence separated into a canonical tree.

The product keeps evidence boundaries explicit: calibration measures runtime fidelity and quantisation ranges, while task accuracy remains a proxy until labeled/oracle evaluation. PPA, thermal, cycle, and power values remain proxies until synthesis, simulation, or trusted hardware-in-the-loop measurement. Built-in demo weights are marked as synthetic.

## Development

The deployable mode-selection shell and Nomo Lab are the Vite application in
the repository root. The historical Next.js neuromorphic dashboard remains in
`frontend/` and is deployed separately. The mode shell links to that dashboard
when the Neuromorphic chips mode is selected.

```bash
npm install
npm run dev
```

Build the production site with:

```bash
npm run build
```

## Public links

- Landing page repository: <https://github.com/TaxCollector23/nomo-ai>
- Landing page: <https://nomoailanding.vercel.app/>
- Nomo Engine mode selector and Lab: <https://frontend-gray-ten-c3tj1luab7.vercel.app/>
- Neuromorphic engine dashboard: <https://nomo-engine-dashboard.vercel.app/>
- Live Python backend: <https://nomo-engine.onrender.com/>
- Engine repository: <https://github.com/TaxCollector23/nomo-engine>
- Engine release line: `v0.5.0` (the repository `main` branch is the deployment source; verify the live `/` response after Render redeploys)
- Engine admin panel: <https://frontend-gray-ten-c3tj1luab7.vercel.app/admin>
- Engine plain-English guide: <https://github.com/TaxCollector23/nomo-engine/blob/main/GUIDE.md>
- Historical v3 release: <https://github.com/TaxCollector23/nomo-engine/releases/tag/v3-fixed>
- V1 prototype: <https://nomoaiprototype.vercel.app/>

The engine’s silicon coefficients are placeholders until calibrated with the measurement LUT or HITL protocol, and accuracy values remain a proxy until promoted by measured data. The mode shell intentionally keeps those boundaries visible.

## Landing page routes

- `/research` — research direction, preprint placeholders, and source material
- `/architecture` — current NIR exports, compiler boundaries, and future work
- `/benchmarks` — measurement status and benchmark limitations
- `/docs` — engine documentation, deployment links, quick start commands, and the v5 release boundary

These pages are linked with normal anchor elements and open in a new tab from the primary navigation.


## Nomo Lab (`/lab`)

An interactive planner that runs entirely in the browser (no server):

| Module | Question it answers |
|---|---|
| Train a model | How should this model be trained on these GPUs? (parallel layout, ZeRO, recomputation, precision) |
| Serve a model | Cheapest tokens within a speed limit and quality budget (precision, KV cache, GPUs, batch) |
| Design a model | Which model to build for a budget and lifetime usage (size, shape, attention type, training length) |
| Audit a run | What topology and precision did an existing Megatron, DeepSpeed, vLLM, or log describe? |
| Infrastructure products | What do chip, RL, reliability, fleet, fine-tuning, and TCO assumptions imply? |
| Cost tracker | Which public compute and provider-price rows have cited evidence, and what must be measured locally? |
| Evidence | Predictions vs 22 published measured runs (Narayanan et al. 2021), calibrated and not |
| Methods | Equations, verification, references |

Three views: **Guided** (plain answers), **Explore** (every control, what-ifs, full tables), **Rigor** (equations,
provenance of every number, editable assumptions). Panels can be switched on and off; preferences are remembered.

Exports: every file downloadable individually or as one zip:

```
nomo-plan_<question>_<date>/
  README.txt
  1-summary/          plan-summary.md, recommendation.json
  2-data/             best-tradeoffs.csv, all-evaluated-plans.csv, settings.json
  3-figures/          tradeoff-chart.svg
  4-launch-configs/   megatron-lm-args.sh + deepspeed-config.json | vllm-serve.sh | model-config.json
  5-paper-materials/  best-tradeoffs-table.tex, methods.md, references.bib
```

### Enterprise workspace and optional sync

The Lab's Enterprise workspace keeps named projects, saved planner or deterministic simulation runs, and a
module-linked review journal in the browser, with side-by-side comparison, exact configuration restore, and JSON
import/export. It is local-only until the user explicitly enters a compatible platform API URL and chooses to sync. The optional sync sends project/run/note records to the
dependency-free `nomo-planner` service; bearer tokens stay in the current browser session and are never included in
workspace exports. The platform API must be deployed with `NOMO_API_TOKEN` and `NOMO_CORS_ORIGINS` for shared use;
the Lab requires HTTPS for non-local platform URLs and accepts HTTP only for localhost development. Workspace imports
are capped at 5 MB in the browser. Authentication, tenant isolation, backups, and permissions are not claimed by the
static Lab itself.

### Engine and verification
`src/planner/` is a TypeScript port of the Python reference planner (nomo-planner). `npm run verify` compares both
on 1,050 plans across seven problems, every best-trade-off set, recommendation and counterfactual, and all 22
calibration predictions, layer goldens, product goldens, auditor fixtures, and evidence/cost boundaries (14,337 checks; worst relative difference 4.37e-16). `npm run bench` times the searches.
Limits: only A100 training is calibrated; serving throughput is an uncalibrated upper bound; precision and
attention-type quality effects, prices and training utilisation are labelled assumptions.

### A1–A3 uncertainty status
The A1–A3 uncertainty layer is implemented for calibrated A100 training. The dependency-free Python reference and
source rows live in `nomo-planner/`; `studies/export_uncertainty.py` produces the checked-in bootstrap artifact at
`src/planner/uncertainty.json`. The Lab shows medians, central 90% predictive intervals, uncertainty whiskers, and
posterior probability of being best for front plans. The leave-one-out study reports 18/22 covered runs (81.8%) against
a nominal 90% target, with a 95% Wilson interval of 61.5%–92.7%; this is evidence on 22 published rows, not a future
guarantee. Serving and co-design remain point-estimate packs until published calibration data exists; no calibrated interval is claimed for them.

### Phase 1 layer-aware planner
The Lab now has a shared model-graph surface at `/lab#layers`. Paste or upload a Hugging Face `config.json` to expand
embedding, attention, MLP, and output nodes with parameter, FLOP, activation, and KV-cache accounting. The Train tab
searches bounded per-node precision, recomputation, CPU-offload, and contiguous pipeline-stage candidates, applies locks,
and compares the selected result with the global-only baseline. Search is exhaustive only for small spaces; larger results
are labelled bounded. FP8 quality, offload bandwidth, framework overhead, and cluster utilization remain assumptions until
customer measurements are supplied. Serve now has a bounded per-layer decision pack using the same graph. Neuromorphic
can export the same graph as a contract JSON and still opens the validated historical dashboard; no placement recommendation
or new compiler integration is claimed.

The design note and cited formulas are in [`docs/PHASE_1_LAYER_PLANNER_DESIGN.md`](docs/PHASE_1_LAYER_PLANNER_DESIGN.md).

### Gate 0 fair per-layer training comparison
The layer planner now ships published Llama 3 8B, Llama 3 70B, and Mixtral 8x7B presets, with Llama 3 8B as the default and the tiny model only as an example. It compares a per-layer plan with the best global-only plan allowed the same options, reports precision gain separately from per-layer gain, and charges pipeline bubble, inter-stage communication, and CPU activation offload using customer-overridable bandwidth assumptions. Step and whole-run totals use adaptive units. Fixed-seed parity cases live in `scripts/layer-golden.json` and are checked by `npm run verify`.

### Phase 2–5 product slices

The `Audit a run` module parses Megatron commands, DeepSpeed JSON, vLLM commands, and a small set of log metrics into
one canonical record. It can re-serialize the fields it actually observed in the same source format and flags options it
did not understand. A local CSV fit applies a multiplicative customer scale, central 90% range, and leave-one-out error;
it remains a customer preview, not a published six-parameter refit. Recommendation-sensitive benchmark candidates are
ranked by uncertainty divided by benchmark hours, and known published model names connect to the shared graph planner.
DeepSpeed/vLLM export is intentionally loss-aware: unsupported framework fields are not fabricated.

`Infrastructure products` contains reference estimators for chip bottlenecks, RL rollout/training/reward scheduling,
checkpoint/restart goodput, hourly fleet sizing, fine-tuning modes, and buy/rent TCO. Values are user inputs or examples,
with cited formulas and physics-sanity tests; they are not measured product specifications. `Cost tracker` contains cited
model-card GPU-hour rows and provider-token price rows plus a local physical-cost calculator and provider/physical
comparison. Rows are hand-entered and not scraped at runtime, and provider prices must be rechecked before procurement.

The gate reports in [docs/](docs/) record what is built and what remains partial. The current surface specification,
deployment guide, and observability guide are [`docs/SPEC.md`](docs/SPEC.md), [`docs/DEPLOY.md`](docs/DEPLOY.md), and
[`docs/OBSERVABILITY.md`](docs/OBSERVABILITY.md). The strict line-by-line audit is in
[`docs/ROADMAP_AUDIT.md`](docs/ROADMAP_AUDIT.md). Core DeepSpeed/vLLM round-trip fixtures, editable product inputs and
JSON/CSV exports, graph-contract re-import, local completed-experiment CSV ingestion, and the conservative training “Safest plan” option are shipped. Binary
Generic checker-backed ONNX structural lowering and safe state-dict inspection are implemented when the optional ONNX
reader is available. Framework-specific ONNX graph-to-cost lowering, the neuromorphic server adapter,
customer/published serving calibration, and broader source coverage remain explicitly open.
