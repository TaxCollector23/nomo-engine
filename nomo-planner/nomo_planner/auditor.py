"""Deterministic config and log auditor for common LLM run descriptions.

The auditor intentionally produces a canonical, inspectable record instead of
claiming that a framework config is a complete performance calibration. It can
parse bounded, versioned common fields from Megatron, DeepSpeed JSON,
TorchTitan TOML, and vLLM commands, plus a small set of observed log metrics.
Fields outside a registered fixture remain opaque and are retained by exports.
"""

from __future__ import annotations

import json
import re
import shlex
import tomllib
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
class FrameworkValidationFixture:
    """A pinned, intentionally small contract for one framework release.

    The auditor must never turn a moving framework CLI into an implied schema.
    Each fixture therefore names the upstream release/documentation snapshot and
    lists only fields whose type/domain is checked below.  Fields outside the
    list remain opaque and are retained by the same-format exporters.
    """

    fixture_id: str
    framework: str
    version: str
    source_url: str
    format: str
    supported_fields: tuple[str, ...]


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    code: str
    message: str
    severity: str = "error"

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class FrameworkValidation:
    """Result for the bounded fixture, not a claim of full framework validity."""

    framework: str
    fixture_id: str | None
    version: str | None
    source_url: str | None
    status: str
    checked_fields: tuple[str, ...] = ()
    unknown_fields: tuple[str, ...] = ()
    issues: tuple[ValidationIssue, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "fixture_id": self.fixture_id,
            "version": self.version,
            "source_url": self.source_url,
            "status": self.status,
            "checked_fields": list(self.checked_fields),
            "unknown_fields": list(self.unknown_fields),
            "issues": [issue.as_dict() for issue in self.issues],
        }


FRAMEWORK_VALIDATION_FIXTURES: dict[str, FrameworkValidationFixture] = {
    "megatron-core-v0.19.2": FrameworkValidationFixture(
        fixture_id="megatron-core-v0.19.2",
        framework="megatron",
        version="core_v0.19.2",
        source_url="https://github.com/NVIDIA/Megatron-LM/blob/core_v0.19.2/megatron/core/transformer/transformer_config.py",
        format="cli",
        supported_fields=(
            "--model",
            "--model-name",
            "--model-type",
            "--seq-length",
            "--max-position-embeddings",
            "--micro-batch-size",
            "--micro-batch-size-per-gpu",
            "--tensor-model-parallel-size",
            "--tensor-parallel-size",
            "--pipeline-model-parallel-size",
            "--pipeline-parallel-size",
            "--data-parallel-size",
            "--bf16",
            "--fp16",
            "--fp8",
            "--recompute-granularity",
            "--recompute-method",
            "--recompute-num-layers",
            "--distribute-saved-activations",
        ),
    ),
    "deepspeed-v0.19.8": FrameworkValidationFixture(
        fixture_id="deepspeed-v0.19.8",
        framework="deepspeed",
        version="0.19.8",
        source_url="https://deepspeed.readthedocs.io/en/v0.19.8/config-json.html",
        format="json",
        supported_fields=(
            "model_name_or_path",
            "train_micro_batch_size_per_gpu",
            "gradient_accumulation_steps",
            "train_batch_size",
            "zero_optimization.stage",
            "bf16.enabled",
            "fp16.enabled",
            "tensor_parallel.tp_size",
            "tensor_parallel.tp",
            "pipeline_parallel.stages",
        ),
    ),
    "torchtitan-v0.2.2-toml": FrameworkValidationFixture(
        fixture_id="torchtitan-v0.2.2-toml",
        framework="torchtitan",
        version="v0.2.2",
        source_url="https://github.com/pytorch/torchtitan/releases/tag/v0.2.2",
        format="toml",
        supported_fields=(
            "model.name",
            "model.flavor",
            "training.local_batch_size",
            "training.seq_len",
            "training.dtype",
            "parallelism.data_parallel_replicate_degree",
            "parallelism.data_parallel_shard_degree",
            "parallelism.tensor_parallel_degree",
            "parallelism.pipeline_parallel_degree",
            "parallelism.context_parallel_degree",
            "activation_checkpoint.mode",
        ),
    ),
    "vllm-v0.6.2": FrameworkValidationFixture(
        fixture_id="vllm-v0.6.2",
        framework="vllm",
        version="v0.6.2",
        source_url="https://docs.vllm.ai/_/downloads/en/v0.6.2/pdf/",
        format="cli",
        supported_fields=(
            "--tensor-parallel-size",
            "--pipeline-parallel-size",
            "--max-model-len",
            "--max-seq-len",
            "--max-num-seqs",
            "--max-num-batched-tokens",
            "--dtype",
            "--enable-prefix-caching",
            "--no-enable-prefix-caching",
            "--quantization",
            "-q",
        ),
    ),
}


def _fixture(fixture_id: str, framework: str) -> FrameworkValidationFixture | None:
    fixture = FRAMEWORK_VALIDATION_FIXTURES.get(fixture_id)
    if fixture is None or fixture.framework != framework:
        return None
    return fixture


def _validation_report(
    *,
    framework: str,
    fixture_id: str,
    checked_fields: list[str],
    unknown_fields: tuple[str, ...] = (),
    issues: list[ValidationIssue] | None = None,
) -> FrameworkValidation:
    fixture = _fixture(fixture_id, framework)
    actual_issues = tuple(issues or ())
    if fixture is None:
        return FrameworkValidation(
            framework=framework,
            fixture_id=None,
            version=None,
            source_url=None,
            status="unvalidated",
            checked_fields=tuple(sorted(set(checked_fields))),
            unknown_fields=tuple(sorted(set(unknown_fields))),
            issues=(ValidationIssue("fixture", "unknown_fixture", f"No versioned {framework} fixture is registered for {fixture_id}"),),
        )
    has_errors = any(issue.severity == "error" for issue in actual_issues)
    status = "invalid" if has_errors else "partial" if unknown_fields or actual_issues else "validated"
    return FrameworkValidation(
        framework=framework,
        fixture_id=fixture.fixture_id,
        version=fixture.version,
        source_url=fixture.source_url,
        status=status,
        checked_fields=tuple(sorted(set(checked_fields))),
        unknown_fields=tuple(sorted(set(unknown_fields))),
        issues=actual_issues,
    )


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value.strip())
    return None


def _issue(path: str, code: str, message: str, *, severity: str = "error") -> ValidationIssue:
    return ValidationIssue(path, code, message, severity)


def _config_value(config: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _mapping_paths(config: Mapping[str, Any], prefix: str = "") -> tuple[str, ...]:
    paths: list[str] = []
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping) and value:
            paths.extend(_mapping_paths(value, path))
        else:
            paths.append(path)
    return tuple(paths)


def _positive_int_issue(path: str, value: Any, *, allow_minus_one: bool = False) -> ValidationIssue | None:
    parsed = _int_value(value)
    if parsed is None or (parsed <= 0 and not (allow_minus_one and parsed == -1)):
        suffix = " or -1" if allow_minus_one else ""
        return _issue(path, "invalid_integer", f"{path} must be a positive integer{suffix}; got {value!r}")


def _nonempty_string_issue(path: str, value: Any) -> ValidationIssue | None:
    if not isinstance(value, str) or not value.strip():
        return _issue(path, "invalid_string", f"{path} must be a non-empty string; got {value!r}")
    return None
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
    validation: FrameworkValidation | None = None
    source_config: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self) | {"warnings": list(self.warnings)}
        # The raw TorchTitan config is retained internally for a loss-aware
        # TOML export, but is redundant in the canonical audit record.
        result.pop("source_config", None)
        if self.validation is not None:
            result["validation"] = self.validation.as_dict()
        return result

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
        if self.pipeline_parallel > 1:
            args += ["--pipeline-parallel-size", str(self.pipeline_parallel)]
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
        # blocks that the auditor projects. Valid nested maps are merged with
        # the canonical projection; an opaque scalar shape wins at the top
        # level so malformed/unknown input is not silently discarded.
        for key, value in self.passthrough_fields.items():
            current = config.get(key)
            if isinstance(current, Mapping) and isinstance(value, Mapping):
                merged = deepcopy(dict(value))
                merged.update(deepcopy(dict(current)))
                config[key] = merged
            elif key not in config or not isinstance(value, Mapping):
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

    def to_torchtitan_config(self) -> dict[str, Any]:
        """Return a bounded, loss-aware TorchTitan config projection."""

        config = deepcopy(dict(self.source_config or {}))

        def set_path(path: str, value: Any) -> None:
            section, _ = path.split(".", 1)
            target = config.get(section)
            if not isinstance(target, Mapping):
                target = {}
                config[section] = target
            target = dict(target)
            target[key] = value
            config[section] = target

        training = config.get("training") if isinstance(config.get("training"), Mapping) else {}
        parallelism = config.get("parallelism") if isinstance(config.get("parallelism"), Mapping) else {}
        if self.micro_batch_size:
            key = "local_batch_size" if "local_batch_size" in training else "batch_size"
            set_path(f"training.{key}", self.micro_batch_size)
        if self.sequence_length:
            key = "seq_len" if "seq_len" in training else "max_context_length"
            set_path(f"training.{key}", self.sequence_length)
        if self.tensor_parallel:
            set_path("parallelism.tensor_parallel_degree", self.tensor_parallel)
        if self.pipeline_parallel:
            set_path("parallelism.pipeline_parallel_degree", self.pipeline_parallel)
        if self.precision != "unknown":
            dtype = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(self.precision, self.precision)
            set_path("training.dtype", dtype)
        if self.model and isinstance(config.get("model"), Mapping) and "name" in config["model"]:
            set_path("model.name", self.model)
        return config

    def export_same_format(self) -> str:
        """Return a corrected representation in the source format, not a recommendation."""
        if self.source_format == "vllm":
            return self.to_vllm_command()
        if self.source_format == "deepspeed-json":
            return json.dumps(self.to_deepspeed_config(), indent=2, sort_keys=True)
        if self.source_format == "torchtitan-toml":
            return _toml_dumps(self.to_torchtitan_config())
        return self.to_megatron_args()


def _base(source_format: str, *, command: str | None = None) -> AuditRun:
    return AuditRun(source_format, None, None, None, 1, 1, None, "unknown", None, command, None, None, None, ())


def _unknown_options(tokens: list[str], known: set[str]) -> tuple[str, ...]:
    names = {token.split("=", 1)[0] for token in tokens if token.startswith("--")}
    return tuple(sorted(names - known))


def _passthrough_tokens(tokens: list[str], unknown: tuple[str, ...]) -> tuple[str, ...]:
    """Keep opaque and validated-but-not-projected CLI options visible."""

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


_MEGATRON_CORE_OPTIONS = {
    "--model",
    "--model-name",
    "--model-type",
    "--seq-length",
    "--max-position-embeddings",
    "--micro-batch-size",
    "--micro-batch-size-per-gpu",
    "--tensor-model-parallel-size",
    "--tensor-parallel-size",
    "--pipeline-model-parallel-size",
    "--pipeline-parallel-size",
    "--data-parallel-size",
    "--fp8",
    "--bf16",
    "--fp16",
}
_MEGATRON_VALIDATED_OPTIONS = {
    "--recompute-granularity",
    "--recompute-method",
    "--recompute-num-layers",
    "--distribute-saved-activations",
}

_VLLM_CORE_OPTIONS = {
    "--max-model-len",
    "--max-seq-len",
    "--max-num-seqs",
    "--tensor-parallel-size",
    "--pipeline-parallel-size",
    "--dtype",
}
_VLLM_VALIDATED_OPTIONS = {
    "--max-num-batched-tokens",
    "--enable-prefix-caching",
    "--no-enable-prefix-caching",
    "--quantization",
}


_DEEPSPEED_TOP_LEVEL_FIELDS = {
    "model_name_or_path",
    "train_micro_batch_size_per_gpu",
    "gradient_accumulation_steps",
    "train_batch_size",
    "tensor_parallel",
    "pipeline_parallel",
    "zero_optimization",
    "bf16",
    "fp16",
}

_DEEPSPEED_PROJECTED_TOP_LEVEL_FIELDS = {
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
    """Return non-projected fields while distinguishing unknown fields."""

    unknown: list[str] = []
    passthrough: dict[str, Any] = {}
    for key, value in config.items():
        if key not in _DEEPSPEED_TOP_LEVEL_FIELDS:
            unknown.append(str(key))
            passthrough[key] = deepcopy(value)
        elif key not in _DEEPSPEED_PROJECTED_TOP_LEVEL_FIELDS:
            # These fields have a bounded validator but are not part of the
            # canonical AuditRun, so retain them for same-format export.
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
        # A scalar pipeline_parallel value is a supported shorthand and is
        # retained in its source shape. Other shapes are also retained because
        # this parser cannot interpret them safely.
        if section == "pipeline_parallel" and _number(value) is not None:
            passthrough[section] = deepcopy(value)
            continue
        unknown.append(section)
        passthrough[section] = deepcopy(value)

    return tuple(sorted(set(unknown))), passthrough


def _option_values(tokens: list[str], name: str) -> tuple[Any, ...]:
    values: list[Any] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == name:
            if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                values.append(tokens[index + 1])
                index += 1
            else:
                values.append(True)
        elif token.startswith(f"{name}="):
            values.append(token.split("=", 1)[1])
        index += 1
    return tuple(values)


def _validate_megatron(tokens: list[str], unknown: tuple[str, ...], *, fixture_id: str) -> FrameworkValidation:
    checked: list[str] = []
    issues: list[ValidationIssue] = []
    model_values = _option_values(tokens, "--model") + _option_values(tokens, "--model-name") + _option_values(tokens, "--model-type")
    if model_values:
        checked.append("--model")
        for value in model_values:
            if value is True:
                issues.append(_issue("--model", "missing_value", "Megatron model option requires a value"))
            else:
                problem = _nonempty_string_issue("--model", value)
                if problem is not None:
                    issues.append(problem)
    numeric_options = (
        ("--seq-length", ("--seq-length", "--max-position-embeddings")),
        ("--micro-batch-size", ("--micro-batch-size", "--micro-batch-size-per-gpu")),
        ("--tensor-model-parallel-size", ("--tensor-model-parallel-size", "--tensor-parallel-size")),
        ("--pipeline-model-parallel-size", ("--pipeline-model-parallel-size", "--pipeline-parallel-size")),
        ("--data-parallel-size", ("--data-parallel-size",)),
        ("--recompute-num-layers", ("--recompute-num-layers",)),
    )
    for canonical, names in numeric_options:
        values = tuple(value for name in names for value in _option_values(tokens, name))
        if not values:
            continue
        checked.append(canonical)
        for value in values:
            problem = _positive_int_issue(canonical, value)
            if problem is not None:
                issues.append(problem)

    precision_flags = [name for name in ("--bf16", "--fp16", "--fp8") if _option_values(tokens, name)]
    checked.extend(precision_flags)
    if len(precision_flags) > 1:
        issues.append(_issue("precision", "mutually_exclusive", "Megatron precision flags are mutually exclusive"))

    for name, allowed in (
        ("--recompute-granularity", {"full", "selective"}),
        ("--recompute-method", {"uniform", "block"}),
    ):
        values = _option_values(tokens, name)
        if not values:
            continue
        checked.append(name)
        for value in values:
            if not isinstance(value, str) or value not in allowed:
                issues.append(_issue(name, "invalid_choice", f"{name} must be one of {sorted(allowed)}; got {value!r}"))

    distributed = bool(_option_values(tokens, "--distribute-saved-activations"))
    if distributed:
        checked.append("--distribute-saved-activations")
        tp = _int_value(_option(tokens, "--tensor-model-parallel-size", "--tensor-parallel-size"))
        granularity = _option(tokens, "--recompute-granularity")
        method = _option(tokens, "--recompute-method")
        if tp is None or tp <= 1:
            issues.append(_issue("--distribute-saved-activations", "requires_tensor_parallel", "distributed saved activations require tensor parallel size > 1"))
        if granularity != "full":
            issues.append(_issue("--distribute-saved-activations", "requires_full_recompute", "distributed saved activations require --recompute-granularity full"))
        if method is None:
            issues.append(_issue("--distribute-saved-activations", "requires_recompute_method", "distributed saved activations require --recompute-method"))

    return _validation_report(
        framework="megatron",
        fixture_id=fixture_id,
        checked_fields=checked,
        unknown_fields=unknown,
        issues=issues,
    )


_VLLM_QUANTIZATION_V062 = {
    "aqlm",
    "awq",
    "deepspeedfp",
    "tpu_int8",
    "fp8",
    "fbgemm_fp8",
    "marlin",
    "gguf",
    "gptq_marlin_24",
    "gptq_marlin",
    "awq_marlin",
    "gptq",
    "compressed-tensors",
    "bitsandbytes",
    "qqq",
    "experts_int8",
    "neuron_quant",
    "none",
}


def _validate_vllm(tokens: list[str], unknown: tuple[str, ...], *, fixture_id: str) -> FrameworkValidation:
    checked: list[str] = []
    issues: list[ValidationIssue] = []
    for name in (
        "--tensor-parallel-size",
        "--pipeline-parallel-size",
        "--max-model-len",
        "--max-seq-len",
        "--max-num-seqs",
        "--max-num-batched-tokens",
    ):
        values = _option_values(tokens, name)
        if not values:
            continue
        checked.append(name)
        for value in values:
            problem = _positive_int_issue(name, value)
            if problem is not None:
                issues.append(problem)

    dtype_values = _option_values(tokens, "--dtype")
    if dtype_values:
        checked.append("--dtype")
        allowed = {"auto", "half", "float16", "bfloat16", "float", "float32"}
        for value in dtype_values:
            if not isinstance(value, str) or value.lower() not in allowed:
                issues.append(_issue("--dtype", "invalid_choice", f"--dtype must be one of {sorted(allowed)}; got {value!r}"))

    cache_flags = [name for name in ("--enable-prefix-caching", "--no-enable-prefix-caching") if _option_values(tokens, name)]
    checked.extend(cache_flags)
    if len(cache_flags) > 1:
        issues.append(_issue("prefix_caching", "mutually_exclusive", "vLLM prefix-cache enable and disable flags are mutually exclusive"))

    quantization = _option(tokens, "--quantization", "-q")
    if quantization is not None:
        checked.append("--quantization")
        if quantization.lower() not in _VLLM_QUANTIZATION_V062:
            issues.append(_issue("--quantization", "invalid_choice", f"vLLM 0.6.2 does not document quantization method {quantization!r}"))

    return _validation_report(
        framework="vllm",
        fixture_id=fixture_id,
        checked_fields=checked,
        unknown_fields=unknown,
        issues=issues,
    )


def _validate_bool_field(path: str, value: Any) -> ValidationIssue | None:
    if not isinstance(value, bool):
        return _issue(path, "invalid_boolean", f"{path} must be a JSON boolean; got {value!r}")
    return None


def _validate_deepspeed(config: Mapping[str, Any], unknown: tuple[str, ...], *, fixture_id: str) -> FrameworkValidation:
    checked: list[str] = []
    issues: list[ValidationIssue] = []
    model_present, model_value = _config_value(config, "model_name_or_path")
    if model_present:
        checked.append("model_name_or_path")
        problem = _nonempty_string_issue("model_name_or_path", model_value)
        if problem is not None:
            issues.append(problem)
    for path in ("train_micro_batch_size_per_gpu", "gradient_accumulation_steps", "train_batch_size"):
        present, value = _config_value(config, path)
        if not present:
            continue
        checked.append(path)
        if value == "auto":
            issues.append(_issue(path, "deferred_value", f"{path}=auto is left to the launcher and was not semantically checked", severity="warning"))
        else:
            problem = _positive_int_issue(path, value)
            if problem is not None:
                issues.append(problem)

    zero_present, zero_value = _config_value(config, "zero_optimization.stage")
    if zero_present:
        checked.append("zero_optimization.stage")
        stage = _int_value(zero_value)
        if stage is None or stage < 0 or stage > 3:
            issues.append(_issue("zero_optimization.stage", "invalid_range", f"zero_optimization.stage must be an integer from 0 to 3; got {zero_value!r}"))

    enabled_precisions: list[str] = []
    for section in ("bf16", "fp16"):
        present, value = _config_value(config, f"{section}.enabled")
        if not present:
            continue
        checked.append(f"{section}.enabled")
        problem = _validate_bool_field(f"{section}.enabled", value)
        if problem is not None:
            issues.append(problem)
        elif value:
            enabled_precisions.append(section)
    if len(enabled_precisions) > 1:
        issues.append(_issue("precision", "mutually_exclusive", "DeepSpeed bf16.enabled and fp16.enabled cannot both be true"))

    for path in ("tensor_parallel.tp_size", "tensor_parallel.tp", "pipeline_parallel.stages"):
        present, value = _config_value(config, path)
        if not present:
            section, key = path.split(".", 1)
            shorthand_present, shorthand = _config_value(config, section)
            if shorthand_present and not isinstance(shorthand, Mapping) and section == "pipeline_parallel":
                checked.append(path)
                problem = _positive_int_issue(path, shorthand)
                if problem is not None:
                    issues.append(problem)
            continue
        checked.append(path)
        problem = _positive_int_issue(path, value)
        if problem is not None:
            issues.append(problem)

    return _validation_report(
        framework="deepspeed",
        fixture_id=fixture_id,
        checked_fields=checked,
        unknown_fields=unknown,
        issues=issues,
    )


def _validate_torchtitan(config: Mapping[str, Any], unknown: tuple[str, ...], *, fixture_id: str) -> FrameworkValidation:
    checked: list[str] = []
    issues: list[ValidationIssue] = []
    for path in ("model.name", "model.flavor"):
        present, value = _config_value(config, path)
        if not present:
            continue
        checked.append(path)
        problem = _nonempty_string_issue(path, value)
        if problem is not None:
            issues.append(problem)
    positive_paths = (
        "training.local_batch_size",
        "training.seq_len",
        "parallelism.data_parallel_replicate_degree",
        "parallelism.tensor_parallel_degree",
        "parallelism.pipeline_parallel_degree",
        "parallelism.context_parallel_degree",
    )
    for path in positive_paths:
        present, value = _config_value(config, path)
        if not present:
            continue
        checked.append(path)
        problem = _positive_int_issue(path, value)
        if problem is not None:
            issues.append(problem)

    present, value = _config_value(config, "parallelism.data_parallel_shard_degree")
    if present:
        checked.append("parallelism.data_parallel_shard_degree")
        problem = _positive_int_issue("parallelism.data_parallel_shard_degree", value, allow_minus_one=True)
        if problem is not None:
            issues.append(problem)

    present, value = _config_value(config, "training.dtype")
    if present:
        checked.append("training.dtype")
        if value not in {"bfloat16", "float32"}:
            issues.append(_issue("training.dtype", "invalid_choice", f"TorchTitan v0.2.2 fixture accepts bfloat16 or float32; got {value!r}"))

    present, value = _config_value(config, "activation_checkpoint.mode")
    if present:
        checked.append("activation_checkpoint.mode")
        if value not in {"none", "selective", "full"}:
            issues.append(_issue("activation_checkpoint.mode", "invalid_choice", f"activation_checkpoint.mode must be none, selective, or full; got {value!r}"))

    return _validation_report(
        framework="torchtitan",
        fixture_id=fixture_id,
        checked_fields=checked,
        unknown_fields=unknown,
        issues=issues,
    )


def _toml_key(value: str) -> str:
    return value if re.fullmatch(r"[A-Za-z0-9_-]+", value) else json.dumps(value)


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("TOML cannot represent NaN or infinity")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_scalar(item) for item in value) + "]"
    raise ValueError(f"unsupported TOML value {value!r}")


def _toml_dumps(config: Mapping[str, Any]) -> str:
    """Serialize the simple TOML shapes used by TorchTitan fixtures.

    Unsupported values raise instead of being dropped.  That is deliberately
    safer than pretending an arbitrary framework config was round-tripped.
    """

    lines: list[str] = []

    def emit_table(path: tuple[str, ...], values: Mapping[str, Any]) -> None:
        scalar_items = [(key, value) for key, value in values.items() if not isinstance(value, Mapping)]
        nested_items = [(key, value) for key, value in values.items() if isinstance(value, Mapping)]
        if path:
            if lines:
                lines.append("")
            lines.append("[" + ".".join(_toml_key(str(part)) for part in path) + "]")
        for key, value in sorted(scalar_items, key=lambda item: str(item[0])):
            lines.append(f"{_toml_key(str(key))} = {_toml_scalar(value)}")
        for key, value in sorted(nested_items, key=lambda item: str(item[0])):
            emit_table(path + (str(key),), value)

    emit_table((), config)
    return "\n".join(lines) + "\n"


def parse_megatron_command(command: str, *, fixture_id: str = "megatron-core-v0.19.2") -> AuditRun:
    tokens = shlex.split(command)
    result = _base("megatron", command=command)
    warnings: list[str] = []
    known = _MEGATRON_CORE_OPTIONS | _MEGATRON_VALIDATED_OPTIONS
    unknown = _unknown_options(tokens, known)
    validation = _validate_megatron(tokens, unknown, fixture_id=fixture_id)
    result = replace(
        result,
        model=_option(tokens, "--model", "--model-name", "--model-type"),
        sequence_length=int(_number(_option(tokens, "--seq-length", "--max-position-embeddings"), 0) or 0) or None,
        micro_batch_size=int(_number(_option(tokens, "--micro-batch-size", "--micro-batch-size-per-gpu"), 0) or 0) or None,
        tensor_parallel=int(_number(_option(tokens, "--tensor-model-parallel-size", "--tensor-parallel-size"), 1) or 1),
        pipeline_parallel=int(_number(_option(tokens, "--pipeline-model-parallel-size", "--pipeline-parallel-size"), 1) or 1),
        data_parallel=int(_number(_option(tokens, "--data-parallel-size"), 0) or 0) or None,
        precision="fp8" if "--fp8" in tokens else "bf16" if "--bf16" in tokens else "fp16" if "--fp16" in tokens else "unknown",
        validation=validation,
    )
    if result.model is None:
        warnings.append("model name was not present in the command")
    if result.precision == "unknown":
        warnings.append("precision flag was not present; quality impact is unknown")
    if unknown:
        warnings.append(f"options were not interpreted: {', '.join(unknown)}")
    for issue in validation.issues:
        warnings.append(f"semantic validation: {issue.message}")
    return replace(result, warnings=tuple(warnings), unrecognized_options=unknown,
                   passthrough_tokens=_passthrough_tokens(tokens, unknown + tuple(_MEGATRON_VALIDATED_OPTIONS)))


def parse_vllm_command(command: str, *, fixture_id: str = "vllm-v0.6.2") -> AuditRun:
    tokens = shlex.split(command)
    dtype = (_option(tokens, "--dtype") or "unknown").lower()
    if dtype in {"half", "float16", "bfloat16"}:
        dtype = "bf16" if dtype == "bfloat16" else "fp16"
    result = _base("vllm", command=command)
    warnings: list[str] = []
    known = _VLLM_CORE_OPTIONS | _VLLM_VALIDATED_OPTIONS
    unknown = _unknown_options(tokens, known)
    validation = _validate_vllm(tokens, unknown, fixture_id=fixture_id)
    result = replace(
        result,
        model=next((token for token in tokens[1:] if not token.startswith("-") and "/" in token), None),
        sequence_length=int(_number(_option(tokens, "--max-model-len", "--max-seq-len"), 0) or 0) or None,
        micro_batch_size=int(_number(_option(tokens, "--max-num-seqs"), 0) or 0) or None,
        tensor_parallel=int(_number(_option(tokens, "--tensor-parallel-size"), 1) or 1),
        pipeline_parallel=int(_number(_option(tokens, "--pipeline-parallel-size"), 1) or 1),
        precision=dtype,
        validation=validation,
    )
    if result.model is None:
        warnings.append("model path was not detected; pass a Hugging Face model path")
    if unknown:
        warnings.append(f"options were not interpreted: {', '.join(unknown)}")
    for issue in validation.issues:
        warnings.append(f"semantic validation: {issue.message}")
    return replace(result, warnings=tuple(warnings), unrecognized_options=unknown,
                   passthrough_tokens=_passthrough_tokens(tokens, unknown + tuple(_VLLM_VALIDATED_OPTIONS)))


def parse_deepspeed_config(config: Mapping[str, Any], *, command: str | None = None, fixture_id: str = "deepspeed-v0.19.8") -> AuditRun:
    zero = config.get("zero_optimization") if isinstance(config.get("zero_optimization"), Mapping) else {}
    tensor = config.get("tensor_parallel") if isinstance(config.get("tensor_parallel"), Mapping) else {}
    pipeline = config.get("pipeline_parallel")
    bf16 = config.get("bf16") if isinstance(config.get("bf16"), Mapping) else {}
    fp16 = config.get("fp16") if isinstance(config.get("fp16"), Mapping) else {}
    precision = "bf16" if bf16.get("enabled") else "fp16" if fp16.get("enabled") else "unknown"
    warnings: list[str] = []
    unknown, passthrough = _deepspeed_passthrough_fields(config)
    validation = _validate_deepspeed(config, unknown, fixture_id=fixture_id)
    stage_value = zero.get("stage")
    parsed_stage = _int_value(stage_value)
    if precision == "unknown":
        warnings.append("DeepSpeed JSON did not enable bf16 or fp16")
    if unknown:
        warnings.append(f"DeepSpeed fields were not interpreted: {', '.join(unknown)}")
    for issue in validation.issues:
        warnings.append(f"semantic validation: {issue.message}")
    return replace(
        _base("deepspeed-json", command=command),
        model=str(config.get("model_name_or_path")) if config.get("model_name_or_path") else None,
        micro_batch_size=int(_number(config.get("train_micro_batch_size_per_gpu"), 0) or 0) or None,
        tensor_parallel=int(_number(tensor.get("tp_size", tensor.get("tp", 1)), 1) or 1),
        pipeline_parallel=int(_number(pipeline.get("stages", pipeline) if isinstance(pipeline, Mapping) else pipeline, 1) or 1),
        precision=precision,
        zero_stage=parsed_stage if parsed_stage is not None and parsed_stage >= 0 else None,
        warnings=tuple(warnings), unrecognized_options=unknown,
        passthrough_fields=passthrough,
        validation=validation,
    )


def parse_torchtitan_config(
    config: Mapping[str, Any] | str,
    *,
    command: str | None = None,
    fixture_id: str = "torchtitan-v0.2.2-toml",
) -> AuditRun:
    """Parse the bounded TorchTitan 0.2.2 TOML fixture.

    TorchTitan has moved from TOML presets toward typed Python configuration.
    This function intentionally accepts only a parsed mapping or TOML text and
    validates the documented v0.2.2 training/parallelism subset.  Other keys
    stay in the source config and are reported as unknown rather than guessed.
    """

    if isinstance(config, str):
        try:
            parsed = tomllib.loads(config)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"invalid TorchTitan TOML: {exc}") from exc
        source_text = config
    elif isinstance(config, Mapping):
        parsed = deepcopy(dict(config))
        source_text = command
    else:
        raise TypeError("TorchTitan config must be TOML text or a mapping")

    fixture = _fixture(fixture_id, "torchtitan")
    supported = set(fixture.supported_fields) if fixture is not None else set()
    unknown = tuple(sorted(path for path in _mapping_paths(parsed) if path not in supported))
    validation = _validate_torchtitan(parsed, unknown, fixture_id=fixture_id)
    warnings: list[str] = []
    if unknown:
        warnings.append(f"TorchTitan fields were not interpreted: {', '.join(unknown)}")
    for issue in validation.issues:
        warnings.append(f"semantic validation: {issue.message}")

    model_section = parsed.get("model") if isinstance(parsed.get("model"), Mapping) else {}
    training = parsed.get("training") if isinstance(parsed.get("training"), Mapping) else {}
    parallelism = parsed.get("parallelism") if isinstance(parsed.get("parallelism"), Mapping) else {}
    model_value = model_section.get("name")
    sequence_value = training.get("seq_len", training.get("max_context_length"))
    micro_value = training.get("local_batch_size", training.get("batch_size"))
    tp_value = parallelism.get("tensor_parallel_degree", 1)
    pp_value = parallelism.get("pipeline_parallel_degree", 1)
    replicate_value = parallelism.get("data_parallel_replicate_degree", 1)
    shard_value = parallelism.get("data_parallel_shard_degree")
    replicate = _int_value(replicate_value)
    shard = _int_value(shard_value) if shard_value is not None else None
    if replicate is not None and replicate > 0 and shard is not None and shard > 0:
        data_parallel = replicate * shard
    elif replicate is not None and replicate > 1:
        data_parallel = replicate
    elif shard is not None and shard > 0:
        data_parallel = shard
    else:
        data_parallel = None
    dtype = str(training.get("dtype", "unknown")).lower()
    precision = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(dtype, dtype)

    return replace(
        _base("torchtitan-toml", command=source_text),
        model=str(model_value) if model_value else None,
        sequence_length=int(sequence_value) if _int_value(sequence_value) is not None and _int_value(sequence_value) > 0 else None,
        micro_batch_size=int(micro_value) if _int_value(micro_value) is not None and _int_value(micro_value) > 0 else None,
        tensor_parallel=int(tp_value) if _int_value(tp_value) is not None and _int_value(tp_value) > 0 else 1,
        pipeline_parallel=int(pp_value) if _int_value(pp_value) is not None and _int_value(pp_value) > 0 else 1,
        data_parallel=data_parallel,
        precision=precision,
        warnings=tuple(warnings),
        unrecognized_options=unknown,
        validation=validation,
        source_config=parsed,
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
    elif all(marker in stripped for marker in ("[training]", "[parallelism]")):
        result = parse_torchtitan_config(stripped)
    else:
        result = parse_megatron_command(stripped)
    return parse_log_metrics(log, result)
