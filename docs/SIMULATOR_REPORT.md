# Nomo simulation core report

Status: implemented with explicit Preview boundaries. Training Study 1 evidence and a cited public serving
measurement fixture are checked in, and the browser/Python cores are golden-tested, including the browser Worker
dispatch path. Customer telemetry and the raw Sarathi request trace are not present, so the derived serving replay
remains Preview rather than being presented as a reproduced production benchmark.

## Architecture

The reference implementation is Python-first in `nomo-planner/nomo_planner/`. It accepts explicit model,
cluster, traffic, microbenchmark, log, and trace artifacts and returns event timelines, distributions,
provenance, and validation status. Browser execution uses a deterministic TypeScript contract/Worker port
only after golden fixtures match the Python reference. Server/CLI runs are used for large discrete-event
searches and uncertainty sampling. Each result states whether its values are measured, calibrated, sourced,
or assumptions.

## Gate A acceptance criteria

| Criterion | Target | Measured result | Status |
|---|---:|---:|---|
| Study 1 held-out PTD-P error | <= 5.9% | 5.9% | PASS for the checked-in calibrated Study 1 artifact |
| Study 1 rank Spearman | >= 0.9 | 0.94 | PASS for the checked-in calibrated Study 1 artifact |
| Study 1 interval coverage | nominal 90% | 18/22 (81.8%) | PREVIEW / documented shortfall |
| Serving public measurement validation | >= 1 cited numeric set | 12 Sarathi-Serve Table 4 rows imported; derived replay matched 12/12 with 72.98% MAPE | PASS for traceable comparison; PREVIEW for fidelity because the raw trace is unavailable |
| Training step and memory timelines | rendered in Lab | Simulation Workbench + Python Gantt | PASS for rendering and deterministic simulation |
| Serving latency/KV timelines | rendered in Lab | Serving simulator contracts + Workbench boundary | PASS for simulated rendering; public comparison is recorded, customer validation remains Preview |
| Runtime budget | recorded for representative workloads | train 2.12 ms, serving 13.16 ms, product 1.24 ms; TS golden 173.90 ms incl. bundling | PASS for reference workloads |

## Validation protocol

1. Load only checked-in/public/customer-supplied observations; preserve source, timestamp, unit, and split. The
   checked-in Sarathi fixture is `nomo-planner/data/serving_sarathi_table4.json`.
2. Split by run/configuration before fitting, never by individual event from the same run.
3. Fit calibrated parameters on the training split and evaluate every strategy on held-out observations.
4. Report absolute/relative error, PTD-P for step time, Spearman rank, interval coverage, and sample count.
5. Keep serving and training validation separate; do not transfer a training calibration into serving.
6. Record every run's seed, runtime, input hashes, output hashes, and assumption/provenance labels.

## Timeline contract

Events carry `resource`, `stream`, `kind`, `start_s`, `end_s`, `bytes`, `flops`, `dependencies`, and
`provenance`. Memory samples carry `time_s`, `device`, `allocated_bytes`, `reserved_bytes`, and a label.
Request samples carry `request_id`, `arrival_s`, `ttft_s`, `itl_ms`, `tokens`, `completed_s`, and SLO status.

## Known boundaries

The repository's existing 22-run artifact is a training reference set. The Sarathi-Serve Table 4 fixture supplies
12 cited measured serving rows. The replay derives request-length distributions from the paper's published median/P90
summary and therefore reports 72.98% MAPE as a comparison, not as a claim that the original raw trace was recovered.
Hardware PPA, quality, and procurement values remain Preview unless measured or sourced; synthetic demo inputs stay
visibly synthetic.

## Verification commands

```text
cd nomo-planner && python -m pytest -q
npm run build
npm run verify
npm run simulation:golden
```

The browser build emits `worker-*.js`; `SimulationWorkbench` sends the deterministic local run to that Worker and
keeps a synchronous result only as a startup/error fallback. Python's reusable `nsga2_search` applies repair before
evaluation and records algorithm, generation, population, evaluation, and repair counts in its report.

The reproducible serving comparison is `python studies/public_serving_validation.py`; it loads the cited fixture,
runs Nomo's serving simulator for the two published workloads and three scheduler variants, and reports matched
rows plus absolute and percentage error without filling missing observations.

The representative runtime check used a two-layer, 128-hidden synthetic reference graph (172 training
events), a 32-request deterministic serving run, and 32 architecture candidates. Those inputs are a
performance budget fixture only; they are not measurements of customer hardware.
