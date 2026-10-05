# Gate 0 report — layer-aware training

Status: Partial, implementation shipped.

Built:

- Python-first layer graph and deterministic browser port.
- Published Llama 3 8B, Llama 3 70B, and Mixtral 8x7B presets.
- Fair global-only comparison using the same precision, recompute, offload, and stage choices.
- Separate precision gain and additional per-layer gain.
- Pipeline bubble, inter-stage communication, CPU activation-offload, memory, cost, and whole-run accounting.
- Fixed-seed Python/TypeScript golden cases in scripts/layer-golden.json.

Evidence:

- Python reference tests and browser parity pass.
- Under the current assumptions, Llama 3 8B/70B show approximately 15.8% precision gain and 0% additional per-layer gain.

Open acceptance items:

- A positive per-layer gain is not demonstrated by the current assumptions. This is reported, not manufactured.
- Dedicated 1440px and 390px screenshots are not part of the current evidence bundle.
- FP8 quality, framework overhead, utilization, and offload bandwidth remain customer-overridable assumptions.
