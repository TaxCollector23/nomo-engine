# Nomo Engine surface specification

This document describes the surfaces that are actually present in the `nomo-engine` repository and its public
deployments. It deliberately separates the browser Lab, the historical neuromorphic dashboard, and the hosted API;
they are related products, not one monolithic binary.

## 1. Public surfaces

| Surface | Source in this repository | Role |
|---|---|---|
| Mode shell and Nomo Lab | repository-root Vite app (`src/`, `src/lab/`) | Browser-local LLM planning, shared graph accounting, auditing, product previews, evidence, and methods |
| Neuromorphic dashboard | `frontend/` Next.js app | Launcher, live search telemetry, layer inspection, Copilot, and export controls for the hosted engine API |
| Hosted engine API | `https://nomo-engine.onrender.com/` | Server-side catalog, uploads, searches, telemetry, Copilot, and export generation used by `frontend/` |

The hosted API implementation is not part of this checkout. The dashboard treats it as an explicit API boundary and
reports server failures rather than silently substituting a browser result.

## 2. Graph and model inputs

The browser Lab accepts Hugging Face-style transformer `config.json` data and Nomo graph-contract JSON. The graph
builder in `src/planner/layers.ts` computes resident parameters, FLOPs, activation memory, KV-cache accounting, and
decision nodes for the Train, Serve, and contract-export tabs.

The hosted dashboard upload surface accepts `.onnx`, `.pt` / `.pth`, and Nomo JSON graph files through the API. The
binary parsers live behind that server boundary; the browser-only shared Lab does not claim to parse binary ONNX or
state-dict files. This distinction is intentional because a JSON graph/config is the only input that can stay fully
local in the Lab.

## 3. Search and evidence boundary

The hosted dashboard runs the server-backed tri-domain search over continuous, spiking, and symbolic layers. The
browser Lab contains the deterministic reference planners used for its planning slices and parity checks. Every
quality, hardware, price, and silicon value is labelled as measured, calibrated, specification, customer input,
assumption, or placeholder where applicable.

The shared-graph Train search compares per-layer choices with a best global-only plan using the same precision,
recompute, offload, and stage options. It charges pipeline bubble, inter-stage transfer, and CPU activation offload
with customer-overridable bandwidth assumptions. The current browser search is bounded and deterministic; it is not
represented as a full NSGA-II implementation.

## 4. NIR export

The historical dashboard is the source of truth for server-backed NIR, C11, ONNX, PyTorch, Core ML, and design
exports. Availability is returned by the API capability manifest for the selected model and design; the UI does not
pretend every backend is available for every graph.

The browser Lab's **Download graph contract** button exports a JSON accounting contract only. It does not claim a
neuromorphic placement, chip mapping, NIR compilation, or silicon validation.

## 5. Telemetry protocol

The dashboard client mirrors telemetry protocol version 1 in `frontend/src/lib/telemetry/protocol.ts`. A run uses
REST for catalog/configuration/design/export actions and `/ws/runs/{run_id}` for sequenced `run.started`, `eval.batch`,
`gen.completed`, `run.completed`, `run.failed`, and `snapshot` envelopes. The client validates the version and
envelope type before updating the run store.

## 6. Related references

- [Phase 1 layer-planner design](PHASE_1_LAYER_PLANNER_DESIGN.md)
- [Roadmap audit](ROADMAP_AUDIT.md)
- [Deployment guide](DEPLOY.md)
- [Observability guide](OBSERVABILITY.md)
