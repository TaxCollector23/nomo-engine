# Nomo AI landing page

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
- Main Nomo Engine dashboard: <https://frontend-gray-ten-c3tj1luab7.vercel.app/>
- Live Python backend: <https://nomo-engine.onrender.com/>
- Engine repository: <https://github.com/TaxCollector23/nomo-engine>
- Engine release line: `v0.5.0` (the repository `main` branch is the deployment source; verify the live `/` response after Render redeploys)
- Engine admin panel: <https://frontend-gray-ten-c3tj1luab7.vercel.app/admin>
- Engine plain-English guide: <https://github.com/TaxCollector23/nomo-engine/blob/main/GUIDE.md>
- Historical v3 release: <https://github.com/TaxCollector23/nomo-engine/releases/tag/v3-fixed>
- V1 prototype: <https://nomoaiprototype.vercel.app/>

The engine’s silicon coefficients are placeholders until calibrated with the measurement LUT or HITL protocol, and accuracy values remain a proxy until promoted by measured data. This landing page intentionally keeps those boundaries visible.

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

### Engine and verification
`src/planner/` is a TypeScript port of the Python reference planner (nomo-planner). `npm run verify` compares both
on 1,050 plans across seven problems, every best-trade-off set, recommendation and counterfactual, and all 22
calibration predictions (14,194 numbers; worst difference 2e-16). `npm run bench` times the searches.
Limits: only A100 training is calibrated; serving throughput is an uncalibrated upper bound; precision and
attention-type quality effects, prices and training utilisation are labelled assumptions.
