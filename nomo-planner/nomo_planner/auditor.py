"""Deterministic config and log auditor for common LLM run descriptions.

The auditor intentionally produces a canonical, inspectable record instead of
claiming that a command line is a complete performance calibration. It can
parse the high-value topology/precision fields from Megatron, DeepSpeed JSON,
and vLLM commands, plus a small set of observed log metrics.
"""

from __future__ import annotations

import json
import re
import shlex
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _option(tokens: list[str], *names: str) -> str | None:
    for index, token in enumerate(tokens):
        for name in names:
            if token == name and index + 1 < len(tokens):
                return tokens[index + 1]
            if token.startswith(f"{name}="):
                return token.split("=", 1)[1]
    return None


@dataclass(frozen=True)
class AuditRun:
    source_format: str
    model: str | None
    sequence_length: int | None
    micro_batch_size: int | None
    tensor_parallel: int
    pipeline_parallel: int
    data_parallel: int | None
    precision: str
    zero_stage: int | None
    command: str | None
    observed_step_time_s: float | None
    observed_tokens_per_s: float | None
    observed_memory_bytes: float | None
    warnings: tuple[str, ...] = ()
    unrecognized_options: tuple[str, ...] = ()
    passthrough_tokens: tuple[str, ...] = ()
    passthrough_fields: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"warnings": list(self.warnings)}

    def to_megatron_args(self) -> str:
        args = [
            "--tensor-model-parallel-size", str(self.tensor_parallel),
            "--pipeline-model-parallel-size", str(self.pipeline_parallel),
        ]
        if self.model:
            args += ["--model", self.model]
        if self.sequence_length:
            args += ["--seq-length", str(self.sequence_length)]
        if self.micro_batch_size:
            args += ["--micro-batch-size", str(self.micro_batch_size)]
        if self.data_parallel:
            args += ["--data-parallel-size", str(self.data_parallel)]
        if self.precision == "bf16":
            args.append("--bf16")
        elif self.precision == "fp8":
            args.append("--fp8")
        elif self.precision == "fp16":
            args.append("--fp16")
        args.extend(self.passthrough_tokens)
        return " ".join(shlex.quote(value) for value in args)

    def to_vllm_command(self) -> str:
        """Emit the fields vLLM can represent without inventing measurements."""
        args = ["vllm", "serve", self.model or "<model>"]
        if self.tensor_parallel > 1:
            args += ["--tensor-parallel-size", str(self.tensor_parallel)]
        if self.sequence_length:
            args += ["--max-model-len", str(self.sequence_length)]
        if self.micro_batch_size:
            args += ["--max-num-seqs", str(self.micro_batch_size)]
        if self.precision == "bf16":
            args += ["--dtype", "bfloat16"]
        elif self.precision == "fp16":
            args += ["--dtype", "float16"]
        elif self.precision != "unknown":
            args += ["--dtype", self.precision]
        args.extend(self.passthrough_tokens)
        return " ".join(shlex.quote(value) for value in args)

    def to_deepspeed_config(self) -> dict[str, Any]:
        """Emit a conservative DeepSpeed JSON projection of the audit record."""
        config: dict[str, Any] = {
            "train_micro_batch_size_per_gpu": self.micro_batch_size or "auto",
            "tensor_parallel": {"tp_size": self.tensor_parallel},
            "pipeline_parallel": {"stages": self.pipeline_parallel},
        }
        # Preserve fields that were not interpreted, including fields nested in
        # blocks that the auditor projects.  The canonical fields win on a key
        # collision; passthrough data is not a semantic validation mechanism.
        for key, value in self.passthrough_fields.items():
            current = config.get(key)
            if isinstance(current, Mapping) and isinstance(value, Mapping):
                merged = deepcopy(dict(value))
                merged.update(deepcopy(dict(current)))
                config[key] = merged
            elif key not in config:
                config[key] = deepcopy(value)
        if self.zero_stage is not None:
            zero = config.get("zero_optimization")
            if isinstance(zero, Mapping):
                config["zero_optimization"] = deepcopy(dict(zero)) | {"stage": self.zero_stage}
            else:
                config["zero_optimization"] = {"stage": self.zero_stage}
        if self.precision == "bf16":
            bf16 = config.get("bf16")
            config["bf16"] = (deepcopy(dict(bf16)) if isinstance(bf16, Mapping) else {}) | {"enabled": True}
        elif self.precision == "fp16":
            fp16 = config.get("fp16")
            config["fp16"] = (deepcopy(dict(fp16)) if isinstance(fp16, Mapping) else {}) | {"enabled": True}
        if self.model:
            config["model_name_or_path"] = self.model
        return config

    def export_same_format(self) -> str:
        """Return a corrected representation in the source format, not a recommendation."""
        if self.source_format == "vllm":
            return self.to_vllm_command()
        if self.source_format == "deepspeed-json":
            return json.dumps(self.to_deepspeed_config(), indent=2, sort_keys=True)
        return self.to_megatron_args()


def _base(source_format: str, *, command: str | None = None) -> AuditRun:
    return AuditRun(source_format, None, None, None, 1, 1, None, "unknown", None, command, None, None, None, ())


def _unknown_options(tokens: list[str], known: set[str]) -> tuple[str, ...]:
    names = {token.split("=", 1)[0] for token in tokens if token.startswith("--")}
    return tuple(sorted(names - known))


def _passthrough_tokens(tokens: list[str], unknown: tuple[str, ...]) -> tuple[str, ...]:
    """Keep unknown CLI options losslessly visible in same-format exports."""

    if not unknown:
        return ()
    names = set(unknown)
    result: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        name = token.split("=", 1)[0]
        if name in names:
            result.append(token)
            if (
                "=" not in token
                and index + 1 < len(tokens)
                # The auditor only treats long options as option boundaries.
                # Preserve a single-dash token as an opaque value because its
                # meaning is framework-specific and cannot be validated here.
                and not tokens[index + 1].startswith("--")
            ):
                result.append(tokens[index + 1])
                index += 1
        index += 1
    return tuple(result)


_DEEPSPEED_TOP_LEVEL_FIELDS = {
    "model_name_or_path",
    "train_micro_batch_size_per_gpu",
    "tensor_parallel",
    "pipeline_parallel",
    "zero_optimization",
    "bf16",
    "fp16",
}

_DEEPSPEED_INTERPRETED_NESTED_FIELDS = {
    "zero_optimization": {"stage"},
    "tensor_parallel": {"tp_size", "tp"},
    "pipeline_parallel": {"stages"},
    "bf16": {"enabled"},
    "fp16": {"enabled"},
}


def _deepspeed_passthrough_fields(config: Mapping[str, Any]) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Return unparsed DeepSpeed fields without asserting their semantics."""

    unknown: list[str] = []
    passthrough: dict[str, Any] = {}
    for key, value in config.items():
        if key not in _DEEPSPEED_TOP_LEVEL_FIELDS:
            unknown.append(str(key))
            passthrough[key] = deepcopy(value)

    for section, interpreted in _DEEPSPEED_INTERPRETED_NESTED_FIELDS.items():
        if section not in config:
            continue
        value = config[section]
        if isinstance(value, Mapping):
            extra = {key: deepcopy(item) for key, item in value.items() if key not in interpreted}
            if extra:
                passthrough[section] = extra
                unknown.extend(f"{section}.{key}" for key in extra)
            continue
        # A scalar pipeline_parallel value is a supported shorthand. Other
        # shapes are retained because this parser cannot interpret them safely.
        if section == "pipeline_parallel" and _number(value) is not None:
            continue
        unknown.append(section)
        passthrough[section] = deepcopy(value)

    return tuple(sorted(set(unknown))), passthrough


def parse_megatron_command(command: str) -> AuditRun:
    tokens = shlex.split(command)
    result = _base("megatron", command=command)
    warnings: list[str] = []
    unknown = _unknown_options(tokens, {"--model", "--model-name", "--model-type", "--seq-length", "--max-position-embeddings", "--micro-batch-size", "--micro-batch-size-per-gpu", "--tensor-model-parallel-size", "--tensor-parallel-size", "--pipeline-model-parallel-size", "--pipeline-parallel-size", "--data-parallel-size", "--fp8", "--bf16", "--fp16"})
    result = replace(
        result,
        model=_option(tokens, "--model", "--model-name", "--model-type"),
        sequence_length=int(_number(_option(tokens, "--seq-length", "--max-position-embeddings"), 0) or 0) or None,
        micro_batch_size=int(_number(_option(tokens, "--micro-batch-size", "--micro-batch-size-per-gpu"), 0) or 0) or None,
        tensor_parallel=int(_number(_option(tokens, "--tensor-model-parallel-size", "--tensor-parallel-size"), 1) or 1),
        pipeline_parallel=int(_number(_option(tokens, "--pipeline-model-parallel-size", "--pipeline-parallel-size"), 1) or 1),
        data_parallel=int(_number(_option(tokens, "--data-parallel-size"), 0) or 0) or None,
        precision="fp8" if "--fp8" in tokens else "bf16" if "--bf16" in tokens else "fp16" if "--fp16" in tokens else "unknown",
    )
    if result.model is None:
        warnings.append("model name was not present in the command")
    if result.precision == "unknown":
        warnings.append("precision flag was not present; quality impact is unknown")
    if unknown:
        warnings.append(f"options were not interpreted: {', '.join(unknown)}")
    return replace(result, warnings=tuple(warnings), unrecognized_options=unknown,
                   passthrough_tokens=_passthrough_tokens(tokens, unknown))


def parse_vllm_command(command: str) -> AuditRun:
    tokens = shlex.split(command)
    dtype = (_option(tokens, "--dtype") or "unknown").lower()
    if dtype in {"half", "float16", "bfloat16"}:
        dtype = "bf16" if dtype == "bfloat16" else "fp16"
    result = _base("vllm", command=command)
    warnings: list[str] = []
    unknown = _unknown_options(tokens, {"--max-model-len", "--max-seq-len", "--max-num-seqs", "--tensor-parallel-size", "--dtype"})
    result = replace(
        result,
        model=next((token for token in tokens[1:] if not token.startswith("-") and "/" in token), None),
        sequence_length=int(_number(_option(tokens, "--max-model-len", "--max-seq-len"), 0) or 0) or None,
        micro_batch_size=int(_number(_option(tokens, "--max-num-seqs"), 0) or 0) or None,
        tensor_parallel=int(_number(_option(tokens, "--tensor-parallel-size"), 1) or 1),
        precision=dtype,
    )
    if result.model is None:
        warnings.append("model path was not detected; pass a Hugging Face model path")
    if unknown:
        warnings.append(f"options were not interpreted: {', '.join(unknown)}")
    return replace(result, warnings=tuple(warnings), unrecognized_options=unknown,
                   passthrough_tokens=_passthrough_tokens(tokens, unknown))


def parse_deepspeed_config(config: Mapping[str, Any], *, command: str | None = None) -> AuditRun:
    zero = config.get("zero_optimization") if isinstance(config.get("zero_optimization"), Mapping) else {}
    tensor = config.get("tensor_parallel") if isinstance(config.get("tensor_parallel"), Mapping) else {}
    pipeline = config.get("pipeline_parallel")
    bf16 = config.get("bf16") if isinstance(config.get("bf16"), Mapping) else {}
    fp16 = config.get("fp16") if isinstance(config.get("fp16"), Mapping) else {}
    precision = "bf16" if bf16.get("enabled") else "fp16" if fp16.get("enabled") else "unknown"
    warnings: list[str] = []
    unknown, passthrough = _deepspeed_passthrough_fields(config)
    if precision == "unknown":
        warnings.append("DeepSpeed JSON did not enable bf16 or fp16")
    if unknown:
        warnings.append(f"DeepSpeed fields were not interpreted: {', '.join(unknown)}")
    return replace(
        _base("deepspeed-json", command=command),
        model=str(config.get("model_name_or_path")) if config.get("model_name_or_path") else None,
        micro_batch_size=int(_number(config.get("train_micro_batch_size_per_gpu"), 0) or 0) or None,
        tensor_parallel=int(_number(tensor.get("tp_size", tensor.get("tp", 1)), 1) or 1),
        pipeline_parallel=int(_number(pipeline.get("stages", pipeline) if isinstance(pipeline, Mapping) else pipeline, 1) or 1),
        precision=precision,
        zero_stage=int(_number(zero.get("stage"), 0) or 0) or None,
        warnings=tuple(warnings), unrecognized_options=unknown,
        passthrough_fields=passthrough,
    )


def parse_log_metrics(log: str, run: AuditRun | None = None) -> AuditRun:
    result = run or _base("log")
    step = re.search(r"(?:step time|iteration time|elapsed)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|s)", log, re.I)
    tokens = re.search(r"(?:tokens?\s*/\s*s|tokens per second)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", log, re.I)
    memory = re.search(r"(?:memory|allocated)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(GB|GiB|MB|MiB)", log, re.I)
    step_s = float(step.group(1)) / (1000 if step.group(2).lower() == "ms" else 1) if step else None
    factor = {"gb": 1e9, "gib": 2**30, "mb": 1e6, "mib": 2**20}
    memory_bytes = float(memory.group(1)) * factor[memory.group(2).lower()] if memory else None
    return replace(result, observed_step_time_s=step_s, observed_tokens_per_s=float(tokens.group(1)) if tokens else None, observed_memory_bytes=memory_bytes)


def audit_text(value: str, *, log: str = "") -> AuditRun:
    stripped = value.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, Mapping):
        result = parse_deepspeed_config(parsed)
    elif "vllm" in stripped.lower():
        result = parse_vllm_command(stripped)
    else:
        result = parse_megatron_command(stripped)
    return parse_log_metrics(log, result)
