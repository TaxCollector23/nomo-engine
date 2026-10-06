"""Production-facing configuration exports for the simulation modules.

The exporters intentionally return plain JSON-compatible dictionaries.  They
are projections of a measured/configured run, not claims that the target
framework will accept every option for every version.  Callers get a warning
when a framework-specific field cannot be represented without guessing.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from .serving_sim import ServingAssumptions, SimulationResult as ServingResult
from .simcore import OperatorGraph, TrainingOptions, TrainingSimulation


def _framework_name(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    aliases = {"megatron-lm": "megatron", "deepspeed": "deepspeed", "torchtitan": "torchtitan", "megatron": "megatron"}
    if normalized not in aliases:
        raise ValueError("training framework must be megatron, deepspeed, or torchtitan")
    return aliases[normalized]


def export_training_config(
    graph: OperatorGraph,
    options: TrainingOptions,
    *,
    framework: str = "megatron",
    simulation: TrainingSimulation | None = None,
) -> dict[str, Any]:
    """Export a conservative M1 config for Megatron, DeepSpeed, or torchtitan."""

    target = _framework_name(framework)
    parallel = options.parallelism
    common = {
        "model": graph.name,
        "sequence_length": graph.sequence_length,
        "micro_batches": options.micro_batches,
        "precision": options.precision,
        "tensor_parallel": parallel.tensor,
        "pipeline_parallel": parallel.pipeline,
        "data_parallel": parallel.data,
        "schedule": options.schedule,
        "recompute": options.recompute,
        "offload": options.offload,
        "sequence_parallel": options.sequence_parallel,
        "context_parallel": options.context_parallel,
        "expert_parallel": options.expert_parallel,
        "zero_stage": options.zero_stage,
        "fsdp": options.fsdp,
    }
    warnings = [
        "This is a checked projection of the simulation inputs; verify flags against the installed framework version.",
        "Analytical timeline values are not a replacement for a target-framework dry run.",
    ]
    if target == "megatron":
        config: dict[str, Any] = {
            "command": "torchrun pretrain_gpt.py",
            "args": {
                "tensor-model-parallel-size": parallel.tensor,
                "pipeline-model-parallel-size": parallel.pipeline,
                "sequence-parallel": options.sequence_parallel > 1,
                "context-parallel-size": options.context_parallel,
                "expert-model-parallel-size": options.expert_parallel,
                "recompute-activations": options.recompute,
                "bf16": options.precision == "bf16",
                "fp16": options.precision == "fp16",
                "fp8": options.precision == "fp8",
                "num-layers": graph.name,
            },
        }
    elif target == "deepspeed":
        config = {
            "train_micro_batch_size_per_gpu": "auto",
            "zero_optimization": {"stage": options.zero_stage},
            "tensor_parallel": {"tp_size": parallel.tensor},
            "pipeline_parallel": {"stages": parallel.pipeline},
            "bf16": {"enabled": options.precision == "bf16"},
            "fp16": {"enabled": options.precision == "fp16"},
            "activation_checkpointing": {"partition_activations": options.recompute},
        }
    else:
        config = {
            "model": graph.name,
            "parallelism": {"data": parallel.data, "tensor": parallel.tensor, "pipeline": parallel.pipeline},
            "compile": {"dtype": options.precision, "activation_checkpointing": options.recompute},
            "fsdp": {"enabled": options.fsdp, "shard_degree": max(1, parallel.data)},
        }
        warnings.append("torchtitan field names are version-sensitive and should be checked against its TOML schema.")
    output = {"schema_version": 1, "framework": target, "common": common, "config": config, "warnings": warnings}
    if simulation is not None:
        output["simulation"] = {
            "step_time_s": simulation.step_time_s,
            "event_count": len(simulation.events),
            "peak_memory_by_gpu": list(simulation.peak_memory_by_gpu),
            "provenance": list(simulation.provenance),
        }
    return output


def _serving_name(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    if normalized not in {"vllm", "sglang", "tensorrt-llm", "tensorrt"}:
        raise ValueError("serving framework must be vllm, sglang, or tensorrt-llm")
    return "tensorrt-llm" if normalized == "tensorrt" else normalized


def export_serving_config(
    assumptions: ServingAssumptions,
    result: ServingResult | None = None,
    *,
    framework: str = "vllm",
) -> dict[str, Any]:
    """Export a serving configuration for vLLM, SGLang, or TensorRT-LLM."""

    target = _serving_name(framework)
    common = {
        "tensor_parallel": assumptions.tensor_parallel,
        "replicas": assumptions.replicas,
        "max_batch_size": assumptions.max_batch_size,
        "chunked_prefill_tokens": assumptions.chunked_prefill_tokens,
        "prefix_cache": assumptions.prefix_cache_enabled,
        "speculative_decoding": assumptions.speculative_enabled,
        "disaggregate_prefill_decode": bool(assumptions.disaggregate),
        "kv_block_tokens": assumptions.kv_block_tokens,
    }
    if target == "vllm":
        config: Mapping[str, Any] = {"tensor_parallel_size": assumptions.tensor_parallel, "enable_prefix_caching": assumptions.prefix_cache_enabled, "enable_chunked_prefill": assumptions.chunked_prefill_tokens > 0, "max_num_seqs": assumptions.max_batch_size}
    elif target == "sglang":
        config = {"tp_size": assumptions.tensor_parallel, "chunked_prefill_size": assumptions.chunked_prefill_tokens or None, "enable_prefix_caching": assumptions.prefix_cache_enabled, "max_running_requests": assumptions.max_batch_size}
    else:
        config = {"tp_size": assumptions.tensor_parallel, "kv_cache_block_size": assumptions.kv_block_tokens, "max_batch_size": assumptions.max_batch_size, "enable_chunked_context": assumptions.chunked_prefill_tokens > 0}
    output: dict[str, Any] = {
        "schema_version": 1,
        "framework": target,
        "common": common,
        "config": dict(config),
        "warnings": [
            "This export maps simulator controls to documented configuration concepts; verify the installed framework release.",
            "Serving rates and SLO results remain simulated until measured trace calibration is supplied.",
        ],
    }
    if result is not None:
        output["simulation"] = {"seed": result.seed, "metrics": dict(result.metrics), "metric_labels": dict(result.metric_labels)}
    return output


__all__ = ["export_serving_config", "export_training_config"]
