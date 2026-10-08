# Framework auditor boundary

The Python reference auditor in `nomo-planner/nomo_planner/auditor.py` now has
explicit, versioned validation fixtures. A fixture is a deliberately bounded
contract: it names the upstream release/documentation snapshot and the fields
whose type, enum, range, or cross-field relationship Nomo checks. It is not an
attempt to reimplement a framework's complete parser.

## Registered fixtures

| Fixture | Input | Bounded fields | Upstream reference |
|---|---|---|---|
| `megatron-core-v0.19.2` | Megatron CLI | model/name, sequence and micro-batch sizes, TP/PP/DP sizes, precision exclusivity, activation-recompute granularity/method/count, distributed-saved-activation prerequisites | [Megatron Core transformer configuration](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.19.2/megatron/core/transformer/transformer_config.py) |
| `deepspeed-v0.19.8` | DeepSpeed JSON | model path, micro/global batch and accumulation sizes, ZeRO stage 0–3, bf16/fp16 booleans and exclusivity, bounded TP/PP sizes | [DeepSpeed 0.19.8 configuration reference](https://deepspeed.readthedocs.io/en/v0.19.8/config-json.html) |
| `torchtitan-v0.2.2-toml` | TorchTitan TOML | model name/flavor, local batch, sequence length, dtype, DP/TP/PP/CP degrees, activation-checkpoint mode | [TorchTitan v0.2.2 release](https://github.com/pytorch/torchtitan/releases/tag/v0.2.2) |
| `vllm-v0.6.2` | vLLM CLI | TP/PP, model length, sequence and token scheduler limits, dtype, prefix-cache flag exclusivity, v0.6.2 quantization choices | [vLLM 0.6.2 documentation](https://docs.vllm.ai/_/downloads/en/v0.6.2/pdf/) |

The fixture registry is part of the Python module and is tested for stable IDs,
version labels, source links, and non-empty field lists. `FrameworkValidation`
reports `validated`, `partial`, `invalid`, or `unvalidated`. Unknown CLI options
and JSON/TOML fields remain in same-format exports and appear in the report;
they never receive guessed semantics.

## What this does not claim

This is semantic validation for the listed fields, not exhaustive framework
coverage or a target-framework dry run. Other Megatron flags, DeepSpeed blocks,
TorchTitan fields and newer typed-Python configuration, vLLM releases, launcher
arguments, model-specific divisibility constraints, installed-version behavior,
and log-to-performance calibration remain outside the fixtures. The browser
TypeScript auditor remains a separate parity surface and does not silently claim
these Python-only fixture checks. M3 is therefore still `IMPLEMENTED / PREVIEW`
until framework-version selection, full same-format coverage, target-framework
execution checks, and customer-log calibration are supplied.

## Verification

The focused tests cover valid and invalid cases, unknown-field preservation,
same-format round trips, TorchTitan TOML dispatch, and all four fixture IDs:

```powershell
cd nomo-planner
python -m pytest -q tests/test_auditor.py
```
