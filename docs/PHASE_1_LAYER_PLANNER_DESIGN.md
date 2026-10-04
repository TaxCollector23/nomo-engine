# Phase 1.1–1.2: shared model graph and per-layer training planner

Status: implementation note for the first layer-aware planning slice.

## Objective

Make one uploaded transformer description usable by the training, serving, and
neuromorphic surfaces. This slice adds a conservative, deterministic graph
contract and a per-layer training search. The current global training planner
stays available as the baseline so the UI can show the incremental gain from
layer-aware decisions.

## Method

1. Parse a Hugging Face `config.json`-shaped object in the browser and in the
   Python reference. Required fields are `num_hidden_layers`,
   `hidden_size`, `num_attention_heads`, and `intermediate_size`; vocabulary,
   KV heads, MoE expert count/top-k, and tied embeddings are optional.
2. Expand the config into an ordered graph: embedding, attention and MLP
   nodes for every transformer block, then the output head. Each node carries
   parameter count, forward FLOPs, activation bytes, and KV-cache bytes.
3. Search the per-layer training decisions with a deterministic constrained
   enumerator for small spaces: contiguous non-empty pipeline stages,
   FP8/BF16 precision, recompute, and CPU activation offload. Larger spaces
   use the same objective/repair interface but are capped and reported as
   incomplete rather than pretending to be exhaustive.
4. Repair invalid candidates before evaluation. Stage IDs are contiguous and
   non-empty; locks are absolute; per-device memory must fit; and the first
   and last graph nodes default to BF16 unless the user explicitly overrides
   them.
5. Optimize step time, estimated cluster cost, and memory headroom. The
   global planner remains the comparison baseline, and the UI reports the
   percentage change only when both plans are feasible.

The recommended integration point is `src/planner/layers.ts`: the graph and
generic layer-search types are independent of the existing domain packs. The
Python implementation in `nomo-planner/nomo_planner/layers.py` is the
reference for parsing, graph accounting, repair, and objective semantics.

## Sources and formulas

- Transformer parameter and FLOP accounting follows PaLM's training
  accounting: Chowdhery et al. (2022),
  <https://arxiv.org/abs/2204.02311>.
- Activation-memory terms follow Korthikanti et al. (2022),
  <https://arxiv.org/abs/2205.05198>.
- Recomputation semantics follow Chen et al. (2016),
  <https://arxiv.org/abs/1604.06174>.
- FP8 is a selectable throughput assumption, not a quality claim; the UI
  labels its quality effect as customer-supplied until evaluation data exists.
- CPU offload is modelled as an explicit transfer penalty and memory relief.
  The default transfer bandwidth is a placeholder and is labelled as such.

No new benchmark numbers are introduced. Existing calibrated A100 values and
published observations remain unchanged.

## Validation plan

- Python unit tests cover config parsing, graph totals, lock preservation,
  stage contiguity, memory feasibility, endpoint precision defaults, and
  monotonicity of offload/recompute effects.
- TypeScript tests are represented by a deterministic golden export and the
  existing `npm run verify` parity checks. The port must match Python output to
  the existing `1e-10` relative tolerance.
- Run `python -m pytest -q`, `npm run build`, `npm run verify`, and the browser
  smoke check at 1440px and 390px. Confirm no console errors and keyboard
  access for upload, tab, lock, and export controls.

## UI changes

- Add a shared model input that accepts Hugging Face `config.json` text or a
  local JSON file; keep the existing preset models as fallback examples.
- Add three graph tabs over the same graph: Train, Serve, and Neuromorphic.
  This slice wires Train first and leaves existing neuromorphic export behavior
  intact.
- Show layer badges, stage boundaries, lock controls, and a baseline-versus-
  layer-aware comparison. Guided mode stays plain; Explore exposes decisions;
  Rigor exposes equations, source labels, and editable assumptions.

## What remains an assumption

- FP8 quality impact, CPU transfer bandwidth/latency, framework overhead, and
  cluster-specific utilization are assumptions until customer measurements are
  supplied.
- Per-layer serving and config auditing are intentionally not claimed as
  complete by this slice. The graph contract is shared so they can consume the
  same uploaded model next.
- Exhaustive search is guaranteed only for the bounded small-space mode. Any
  capped search is explicitly marked non-exhaustive in results and exports.
