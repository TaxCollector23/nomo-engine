# Nomo simulation core report

Status: implemented with explicit Preview boundaries. Training Study 1 evidence is checked in and the
browser/Python cores are golden-tested, including the browser Worker dispatch path. This report is not marked as fully passed because the repository
does not contain a numeric public serving measurement fixture or customer telemetry.

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
| Serving public measurement validation | >= 1 cited numeric set | 0 imported | OPEN; vLLM is cited prior art, not a local measurement row |
| Training step and memory timelines | rendered in Lab | Simulation Workbench + Python Gantt | PASS for rendering and deterministic simulation |
| Serving latency/KV timelines | rendered in Lab | Serving simulator contracts + Workbench boundary | PASS for simulated rendering; measured validation is OPEN |
| Runtime budget | recorded for representative workloads | train 2.12 ms, serving 13.16 ms, product 1.24 ms; TS golden 173.90 ms incl. bundling | PASS for reference workloads |

## Validation protocol

1. Load only checked-in/public/customer-supplied observations; preserve source, timestamp, unit, and split.
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

The repository's existing 22-run artifact is a training reference set, not a serving measurement set. No
serving validation number may be promoted until a cited measurement trace is imported. Hardware PPA, quality,
and procurement values remain Preview unless measured or sourced; synthetic demo inputs stay visibly synthetic.

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

The representative runtime check used a two-layer, 128-hidden synthetic reference graph (172 training
events), a 32-request deterministic serving run, and 32 architecture candidates. Those inputs are a
performance budget fixture only; they are not measurements of customer hardware.
