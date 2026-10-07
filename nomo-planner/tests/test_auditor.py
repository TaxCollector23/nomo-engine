import json

from nomo_planner.auditor import (
    FRAMEWORK_VALIDATION_FIXTURES,
    audit_text,
    parse_deepspeed_config,
    parse_log_metrics,
    parse_megatron_command,
    parse_torchtitan_config,
    parse_vllm_command,
)


def test_megatron_command_extracts_topology_and_roundtrips_core_flags():
    run = parse_megatron_command("torchrun pretrain.py --model meta/llama --seq-length 4096 --micro-batch-size=2 --tensor-model-parallel-size 4 --pipeline-model-parallel-size=2 --bf16")
    assert run.source_format == "megatron"
    assert (run.model, run.sequence_length, run.micro_batch_size) == ("meta/llama", 4096, 2)
    assert (run.tensor_parallel, run.pipeline_parallel, run.precision) == (4, 2, "bf16")
    roundtrip = parse_megatron_command(run.to_megatron_args())
    assert (roundtrip.tensor_parallel, roundtrip.pipeline_parallel, roundtrip.precision) == (4, 2, "bf16")


def test_megatron_fp16_and_unknown_values_roundtrip_without_validation():
    run = parse_megatron_command(
        "torchrun pretrain.py --model meta/llama --fp16 --untested-token -opaque-value"
    )
    exported = run.to_megatron_args()
    assert "--fp16" in exported
    assert "--untested-token -opaque-value" in exported
    roundtrip = parse_megatron_command(exported)
    assert roundtrip.precision == "fp16"
    assert "--untested-token" in roundtrip.unrecognized_options


def test_deepspeed_json_and_vllm_command_are_canonicalized():
    deep = parse_deepspeed_config({"model_name_or_path": "meta/llama", "train_micro_batch_size_per_gpu": 4, "zero_optimization": {"stage": 2}, "tensor_parallel": {"tp_size": 2}, "bf16": {"enabled": True}})
    assert (deep.source_format, deep.micro_batch_size, deep.tensor_parallel, deep.zero_stage, deep.precision) == ("deepspeed-json", 4, 2, 2, "bf16")
    deep_export = json.loads(deep.export_same_format())
    assert deep_export["model_name_or_path"] == "meta/llama"
    assert deep_export["tensor_parallel"]["tp_size"] == 2
    vllm = parse_vllm_command("vllm serve meta/llama --tensor-parallel-size 8 --max-model-len 8192 --dtype bfloat16")
    assert (vllm.source_format, vllm.model, vllm.sequence_length, vllm.tensor_parallel, vllm.precision) == ("vllm", "meta/llama", 8192, 8, "bf16")
    assert "--tensor-parallel-size 8" in vllm.export_same_format()


def test_log_metrics_are_attached_without_claiming_calibration():
    run = parse_log_metrics("step time: 125 ms, tokens/s: 9876, allocated: 12 GiB")
    assert run.observed_step_time_s == 0.125
    assert run.observed_tokens_per_s == 9876
    assert run.observed_memory_bytes == 12 * 2**30
    assert "calibrated" not in json.dumps(run.as_dict()).lower()


def test_unsupported_options_are_flagged_instead_of_silently_dropped():
    run = parse_megatron_command("torchrun pretrain.py --model meta/llama --bf16 --untested-new-flag 1")
    assert "--untested-new-flag" in run.unrecognized_options
    assert "not interpreted" in " ".join(run.warnings)
    assert "--untested-new-flag 1" in run.export_same_format()

    deep = parse_deepspeed_config({"bf16": {"enabled": True}, "new_optimizer": {"foo": 1}})
    assert deep.unrecognized_options == ("new_optimizer",)
    assert json.loads(deep.export_same_format())["new_optimizer"] == {"foo": 1}

    vllm = parse_vllm_command("vllm serve meta/llama --dtype bfloat16 --new-scheduler-flag 7")
    assert "--new-scheduler-flag 7" in vllm.export_same_format()


def test_deepspeed_preserves_uninterpreted_nested_fields_in_known_blocks():
    deep = parse_deepspeed_config(
        {
            "model_name_or_path": "meta/llama",
            "train_micro_batch_size_per_gpu": 4,
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "overlap_comm": True,
            },
            "tensor_parallel": {"tp_size": 2, "tp_grain_size": 128},
            "pipeline_parallel": {"stages": 2, "activation_checkpoint_interval": 4},
            "bf16": {"enabled": True, "loss_scale": 0},
            "optimizer": {"type": "OneBitAdam", "params": {"lr": 1e-4}},
        }
    )
    exported = json.loads(deep.export_same_format())
    assert exported["zero_optimization"]["stage"] == 3
    assert exported["zero_optimization"]["offload_optimizer"] == {"device": "cpu", "pin_memory": True}
    assert exported["zero_optimization"]["overlap_comm"] is True
    assert exported["tensor_parallel"]["tp_grain_size"] == 128
    assert exported["pipeline_parallel"]["activation_checkpoint_interval"] == 4
    assert exported["bf16"]["loss_scale"] == 0
    assert exported["optimizer"] == {"type": "OneBitAdam", "params": {"lr": 1e-4}}
    assert "zero_optimization.offload_optimizer" in deep.unrecognized_options
    assert "pipeline_parallel.activation_checkpoint_interval" in deep.unrecognized_options
    assert "optimizer" in deep.unrecognized_options
    assert "not interpreted" in " ".join(deep.warnings)


def test_vllm_framework_options_are_preserved_and_bounded_semantics_are_checked():
    run = parse_vllm_command(
        "vllm serve meta/llama --dtype bfloat16 --enable-prefix-caching "
        "--max-num-batched-tokens 4096 --quantization awq"
    )
    exported = run.export_same_format()
    assert "--enable-prefix-caching" in exported
    assert "--max-num-batched-tokens 4096" in exported
    assert "--quantization awq" in exported
    assert run.unrecognized_options == ()
    assert run.validation is not None
    assert run.validation.status == "validated"


def test_audit_text_dispatches_json_and_logs():
    run = audit_text('{"fp16":{"enabled":true},"zero_optimization":{"stage":1}}', log="iteration time = 2 s")
    assert run.source_format == "deepspeed-json"
    assert run.precision == "fp16"
    assert run.observed_step_time_s == 2


def test_versioned_framework_fixtures_are_explicit_and_bounded():
    assert set(FRAMEWORK_VALIDATION_FIXTURES) == {
        "megatron-core-v0.19.2",
        "deepspeed-v0.19.8",
        "torchtitan-v0.2.2-toml",
        "vllm-v0.6.2",
    }
    assert all(fixture.version and fixture.source_url and fixture.supported_fields for fixture in FRAMEWORK_VALIDATION_FIXTURES.values())


def test_megatron_fixture_validates_documented_recompute_contract_and_preserves_extras():
    run = parse_megatron_command(
        "torchrun pretrain.py --model meta/llama --seq-length 4096 --micro-batch-size 2 "
        "--tensor-model-parallel-size 2 --pipeline-model-parallel-size 2 --bf16 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 2 "
        "--distribute-saved-activations --future-flag opaque"
    )
    assert run.validation is not None
    assert run.validation.status == "partial"
    assert run.validation.fixture_id == "megatron-core-v0.19.2"
    assert "--recompute-granularity" in run.validation.checked_fields
    assert "--future-flag" in run.validation.unknown_fields
    assert "--future-flag opaque" in run.export_same_format()


def test_megatron_fixture_rejects_conflicting_precision_and_bad_recompute_values():
    run = parse_megatron_command(
        "torchrun pretrain.py --bf16 --fp16 --tensor-model-parallel-size 1 "
        "--recompute-granularity selective --recompute-num-layers 0 --distribute-saved-activations"
    )
    assert run.validation is not None
    assert run.validation.status == "invalid"
    codes = {issue.code for issue in run.validation.issues}
    assert {"mutually_exclusive", "invalid_integer", "requires_tensor_parallel", "requires_full_recompute", "requires_recompute_method"} <= codes


def test_deepspeed_fixture_validates_common_fields_and_roundtrips_noncanonical_fields():
    run = parse_deepspeed_config(
        {
            "model_name_or_path": "meta/llama",
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 8,
            "train_batch_size": 16,
            "zero_optimization": {"stage": 3, "offload_optimizer": {"device": "cpu"}},
            "bf16": {"enabled": True},
        }
    )
    assert run.validation is not None
    assert run.validation.status == "partial"
    assert "gradient_accumulation_steps" in run.validation.checked_fields
    exported = json.loads(run.export_same_format())
    assert exported["gradient_accumulation_steps"] == 8
    assert exported["zero_optimization"]["offload_optimizer"] == {"device": "cpu"}


def test_deepspeed_fixture_rejects_bad_stage_and_precision_pair():
    run = parse_deepspeed_config(
        {
            "zero_optimization": {"stage": 4},
            "bf16": {"enabled": True},
            "fp16": {"enabled": True},
            "gradient_accumulation_steps": 0,
        }
    )
    assert run.validation is not None
    assert run.validation.status == "invalid"
    codes = {issue.code for issue in run.validation.issues}
    assert {"invalid_range", "mutually_exclusive", "invalid_integer"} <= codes


def test_deepspeed_opaque_scalar_shapes_are_not_dropped_by_export():
    run = parse_deepspeed_config({"tensor_parallel": 2, "pipeline_parallel": 3})
    exported = json.loads(run.export_same_format())
    assert exported["tensor_parallel"] == 2
    assert exported["pipeline_parallel"] == 3
    assert "tensor_parallel" in run.unrecognized_options
    assert run.validation is not None
    assert run.validation.status == "partial"


def test_torchtitan_toml_fixture_parses_validates_and_preserves_unknown_tables():
    source = """
[job]
description = "nightly audit"

[model]
name = "llama3"
flavor = "8B"

[training]
local_batch_size = 2
seq_len = 4096
dtype = "bfloat16"

[parallelism]
data_parallel_replicate_degree = 1
data_parallel_shard_degree = 8
tensor_parallel_degree = 2
pipeline_parallel_degree = 2
context_parallel_degree = 1

[activation_checkpoint]
mode = "selective"
"""
    run = parse_torchtitan_config(source)
    assert (run.source_format, run.model, run.sequence_length, run.micro_batch_size) == ("torchtitan-toml", "llama3", 4096, 2)
    assert (run.tensor_parallel, run.pipeline_parallel, run.data_parallel, run.precision) == (2, 2, 8, "bf16")
    assert run.validation is not None
    assert run.validation.status == "partial"
    assert "job.description" in run.validation.unknown_fields
    exported = run.export_same_format()
    reparsed = parse_torchtitan_config(exported)
    assert (reparsed.tensor_parallel, reparsed.pipeline_parallel, reparsed.data_parallel, reparsed.precision) == (2, 2, 8, "bf16")
    assert "description = \"nightly audit\"" in exported


def test_torchtitan_fixture_rejects_invalid_bounded_fields_and_audit_text_dispatches():
    source = """
[training]
local_batch_size = 0
seq_len = 4096
dtype = "float16"
[parallelism]
data_parallel_shard_degree = 0
tensor_parallel_degree = 2
[activation_checkpoint]
mode = "unknown"
"""
    run = audit_text(source)
    assert run.source_format == "torchtitan-toml"
    assert run.validation is not None
    assert run.validation.status == "invalid"
    assert {issue.code for issue in run.validation.issues} >= {"invalid_integer", "invalid_choice"}


def test_vllm_fixture_validates_documented_scheduler_controls():
    run = parse_vllm_command(
        "vllm serve meta/llama --tensor-parallel-size 2 --pipeline-parallel-size 2 "
        "--max-model-len 8192 --max-num-seqs 32 --max-num-batched-tokens 4096 "
        "--dtype bfloat16 --enable-prefix-caching --quantization awq"
    )
    assert run.validation is not None
    assert run.validation.status == "validated"
    assert run.pipeline_parallel == 2
    assert "--max-num-batched-tokens 4096" in run.export_same_format()


def test_vllm_fixture_rejects_conflicting_cache_flags_and_bad_quantization():
    run = parse_vllm_command(
        "vllm serve meta/llama --dtype nonsense --enable-prefix-caching "
        "--no-enable-prefix-caching --quantization invented"
    )
    assert run.validation is not None
    assert run.validation.status == "invalid"
    assert {issue.code for issue in run.validation.issues} == {"invalid_choice", "mutually_exclusive"}
