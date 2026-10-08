# Nomo v6 enterprise co-design surface

This document is the implementation boundary for the v6 co-design workflow. Nomo is a local-first
search, compilation, and release tool: it can produce a reproducible design package and expose the
interfaces needed to connect real simulators, EDA tools, and hardware measurements. It does not claim
silicon results from a proxy model.

## Architecture outline

```text
project / enterprise profile / model upload / nomo.yaml
        |
        v
ingest -> ModelGraph + architecture contract + provenance
        |
        +--> tri-domain policy -> constrained NSGA-II -> Pareto archive
        |                              |
        |                              +--> target weights -> recommendation
        |
        +--> bounded accelerator co-search -> PE/SRAM/bandwidth/precision Pareto evidence
        |
        +--> calibration tensors -> MSE/KL PTQ -> activation edge ranges
        |
        v
RunContext / Workbench state
        |
        +--> native integer QGraph -> Python integer driver + C11 microkernel
        +--> float/reference exports -> PyTorch, ONNX, Core ML, NIR, PDF
        +--> silicon boundary -> SystemVerilog, Chisel, Yosys/OpenROAD, DEF
        +--> validation ledger -> proxy, simulated cycle/cache evidence, calibration fidelity, TTFS efficiency, HITL
        |
        v
zip or tar.gz structured release package
```

The central IR remains a schedulable chain so the existing search invariants remain hard constraints.
Architecture-family metadata prevents that chain from being presented as a generic flattened MLP:

| Family | Preserved contract | Current release boundary |
|---|---|---|
| MLP | vector shapes | PyTorch, ONNX, NIR, C11 |
| CNN | spatial tensor shapes | PyTorch, ONNX, NIR |
| FNO | spatial dynamics and spectral-mode metadata | PyTorch/reference path and NIR metadata |
| ViT / attention | token sequence and attention dimensions | PyTorch/reference path and ONNX-family metadata |
| GNN | graph/message-passing metadata | PyTorch/reference path and NIR metadata |
| Hybrid | per-block family contracts | backend capability is reported per export |

`nomo.architecture/1` is serialized in uploaded model metadata, design manifests, and validation reports.
Operators use `in_shape`, `out_shape`, `preserve_spatial`, and family-specific metadata instead of silently
flattening a spatial or graph workload.

## Operational modes

`GET /catalog` exposes four machine-readable mode contracts:

- `low_power_neuromorphic`: event-driven execution, power budget, clock/power gating tags.
- `hard_realtime`: explicit period budget, deterministic memory, fail-safe and HITL-required tags.
- `radiation_hardened`: TMR/ECC/thermal-profile tags for the target implementation.
- `on_chip_learning`: bounded plasticity, replay protection, and online-calibration tags.

Modes are recorded in `RunIn.mode`, `design.json`, the Workbench response, and the package manifest.
Warnings are explicit when a mode is selected without the budget or policy needed to enforce its contract.

## API contracts

### Projects and enterprise profiles

`POST /projects` creates an owner-scoped project workspace. Include its `project_id` in `POST /runs` to
associate searches with the project. Set `NOMO_STATE_DB` to a durable SQLite path for local or on-premise
history; the free hosted deployment intentionally remains ephemeral when it is unset.

`POST /enterprise/profile/validate` accepts a versioned YAML-derived JSON profile and returns its canonical
form, fingerprint, and evidence requirements. A profile can be attached to a run with `enterprise_profile`;
the server enforces its target hardware, allowed domains, and pinned precisions.

### Start a run

```http
POST /runs
Content-Type: application/json
```

```json
{
  "model": "attitude_policy",
  "hardware": "akd1500",
  "mode": "low_power_neuromorphic",
  "pop_size": 64,
  "generations": 60,
  "budgets": {"accuracy_drop_max": 4.0, "energy_j": 0.001},
  "search": {
    "allow_continuous": true,
    "allow_spiking": true,
    "allow_symbolic": true,
    "codings": ["rate", "ttfs"],
    "asf_weights": [2.0, 1.0, 1.0]
  }
}
```

The response is `{ "run_id": "...", "warnings": [] }`. The run keeps the selected mode, policy,
weights provenance, and calibration attachment in its export context.

### Attach calibration data

```http
POST /models/{model_id}/calibration?filename=calibration.npz
Content-Type: application/octet-stream
```

Accepted payloads are `.npz`, `.npy`, and JSON. The required batch has 100–500 finite samples matching
the model input shape. `.npz` accepts `inputs`/`X`/`x`/`data`; guarded physical-AI models also require
`aux`/`A`/`metadata` with the exact auxiliary width. Optional `labels` are retained for provenance.

The response is a `nomo.ptq/1` report. It includes `sample_count`, `edge_amax`, MSE/KL candidates,
selected method, and the source filename. PTQ changes activation ranges for compilation; it does not
convert a search accuracy proxy into a measured task score.

### Workbench

```http
GET  /runs/{run_id}/workbench
POST /runs/{run_id}/workbench/targets
Content-Type: application/json
```

```json
{
  "energy": 1.0,
  "latency": 1.0,
  "accuracy": 1.0
}
```

`nomo.workbench/1` returns six synchronized levels:

1. `system_topology` — model blocks and dataflow edges.
2. `partitioning` — ANN/SNN/SYM domains and crossings.
3. `hardware_graph` — PE array, SRAM, interconnect, and mapping.
4. `cycle_emulation` — cycle/waveform boundary and measured-vs-estimated metrics.
5. `rtl` — generated SystemVerilog datapath nodes.
6. `silicon_floorplan` — thermal-density proxy and future placement state.

All levels keep a stable layer/design selection and a revision number. Target weights re-score the cached
feasible archive; they do not mutate the original run or fabricate a new search result.

### Co-search and bounded emulation

`POST /runs/{run_id}/co-design` explores a finite accelerator space around cached deployment candidates.
The result labels PE, SRAM, bandwidth, precision, energy, latency, accuracy-loss, and area evidence as
analytic/model priors until a simulator or target measurement replaces them.

`POST /runs/{run_id}/emulation` compiles the selected supported integer graph into Nomo's deterministic
cycle/cache model. It returns per-layer cycles, memory traffic, cache hit/miss data, and spike/event counts
with `backend.capability = simulated` and `physical_measurement = false`. It is not SystemC, Verilator, Gem5,
or a physical-board measurement.

### Export

```http
POST /runs/{run_id}/export
Content-Type: application/json
```

```json
{
  "key": "recommended",
  "formats": ["enterprise", "pdf", "nir", "c11"],
  "archive": "tar.gz"
}
```

`archive` is `zip` or `tar.gz`. `enterprise` enables the structured package tree; the legacy format names
remain available for compatibility and each unavailable format includes a reason.

## Canonical release tree

```text
nomo_<model>_<design>/
├── manifest.json
├── model/
│   ├── weights.npz
│   ├── design.json
│   ├── graph.nir                       # or *.unavailable.txt
│   └── model_quantized.onnx             # or *.unavailable.txt
├── runtime/
│   ├── deploy_model.py                  # integer driver when QGraph lowering exists
│   ├── deploy_integer_weights.npz
│   ├── deploy_model_float.py            # explicit reference fallback
│   ├── c11_microkernel/{include,src}/
│   └── CMakeLists.txt
├── software_sdk/
│   ├── deploy_model.py
│   ├── c11_microkernel/{include,src}/
│   └── SystemC_tb/README.md
├── silicon_eda/
│   ├── rtl/{pe_array.sv,custom_npu_core.sv}
│   ├── chisel/CustomNpuCore.scala
│   ├── eda_scripts/{synth.tcl,floorplan.def}
│   └── ppa_report.json
├── validation/
│   ├── sensitivity_report.json
│   ├── closed_loop_telemetry.json
│   └── Nomo_Research_Brief.pdf
└── benchmarks/
    ├── sensitivity_report.json
    └── ppa_report.json
```

The root `README.md` in a package identifies legacy paths and points consumers to the canonical tree.
Integer deployment is only promoted when Nomo can embed a bit-exact golden-vector check. Operator-family
graphs without an integer lowering keep their reference path and a machine-readable limitation.

## Integer runtime and C11 boundary

The generated Python driver and C11 microkernel share the QGraph semantics:

- signed int8 activations and weights where the selected graph permits them;
- int64 accumulators with deterministic round-half-up requantisation;
- signed Q16.16 guard and symbolic arithmetic;
- saturating clamps and explicit LIF state transitions;
- embedded reference inputs, auxiliary values, and outputs for a bit-exact self-check.

The C11 API is intentionally small and freestanding-friendly. C11 emission is currently limited to dense
integer chains, supported rate-coded LIF segments, symbolic linear blocks, and the existing physical guards.
Unsupported TTFS/conv/operator-family cases are rejected or retained as explicit unavailable artifacts.

## RTL and EDA integration boundary

The release emits a synthesizable integer PE-array boundary, a top-level `custom_npu_core`, and a Chisel
interface stub with dimensions derived from the first compatible integer stage. `synth.tcl` is a generic
Yosys/OpenROAD entry point and `floorplan.def` is a starting integration artifact. `ppa_report.json` is
labelled `source: proxy`; it must be replaced or supplemented by synthesis, timing, thermal, and power
results before any hardware claim is made.

## HITL and evidence promotion

`nomo.hardware.hitl` supports two deliberately narrow transports:

- `HITLClient`: authenticated JSON `POST /v1/benchmark` to a trusted benchmark agent;
- `SocketBenchmarkClient`: length-prefixed JSON over an explicitly supplied socket.

The client exchanges artifact hashes and measurement fields such as `cycle_count`, `latency_ms`,
`energy_uj`, `cache_hit_ratio`, and `bus_contention`. It does not execute arbitrary commands on the
target. A returned measurement can be attached to `RunContext.hitl_measurement` and appears in
`validation/closed_loop_telemetry.json`.

Every validation report separates:

| Evidence | Meaning |
|---|---|
| `zero_shot_sensitivity_proxies` | search coefficients and accuracy proxy; not task measurement |
| `empirical_validation_results` | calibration-set reference/runtime fidelity |
| `neuromorphic_efficiency` | deterministic TTFS event-vs-dense operation estimate |
| `unified_cost_function` | energy/latency/area objective with area source marked proxy |
| `hitl` | externally measured values, when a trusted agent supplied them |

The bounded emulator's cycle/cache output is recorded separately as `simulated`; it must not be promoted to
physical latency, energy, or PPA without target evidence.

Synthetic built-in weights are marked `DEMO WEIGHTS` in PDF and package documentation. Calibration
fidelity is not task accuracy. Proxy PPA is not silicon PPA.

## Offline pipeline

Copy [`nomo.yaml.example`](../nomo.yaml.example) to `nomo.yaml`, add a calibration file if available,
and run:

```bash
nomo-cli pipeline --config nomo.yaml --out nomo_release.tar.gz
```

The config supports `model`, `hardware`, `mode`, `budgets`, `search.population`, `search.rounds`,
`search.seed`, optional `calibration`, and `export.formats`/`export.archive`. The offline pipeline and
the service use the same `RunContext` and structured release builder.
