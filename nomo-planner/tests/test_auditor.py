import json

from nomo_planner.auditor import audit_text, parse_deepspeed_config, parse_log_metrics, parse_megatron_command, parse_vllm_command


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


def test_vllm_framework_options_are_preserved_without_semantic_interpretation():
    run = parse_vllm_command(
        "vllm serve meta/llama --dtype bfloat16 --enable-prefix-caching "
        "--max-num-batched-tokens 4096 --quantization awq"
    )
    exported = run.export_same_format()
    assert "--enable-prefix-caching" in exported
    assert "--max-num-batched-tokens 4096" in exported
    assert "--quantization awq" in exported
    assert "--enable-prefix-caching" in run.unrecognized_options
    assert "--max-num-batched-tokens" in run.unrecognized_options
    assert "--quantization" in run.unrecognized_options
    assert "not interpreted" in " ".join(run.warnings)


def test_audit_text_dispatches_json_and_logs():
    run = audit_text('{"fp16":{"enabled":true},"zero_optimization":{"stage":1}}', log="iteration time = 2 s")
    assert run.source_format == "deepspeed-json"
    assert run.precision == "fp16"
    assert run.observed_step_time_s == 2
