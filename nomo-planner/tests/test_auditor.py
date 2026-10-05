import json

from nomo_planner.auditor import audit_text, parse_deepspeed_config, parse_log_metrics, parse_megatron_command, parse_vllm_command


def test_megatron_command_extracts_topology_and_roundtrips_core_flags():
    run = parse_megatron_command("torchrun pretrain.py --model meta/llama --seq-length 4096 --micro-batch-size=2 --tensor-model-parallel-size 4 --pipeline-model-parallel-size=2 --bf16")
    assert run.source_format == "megatron"
    assert (run.model, run.sequence_length, run.micro_batch_size) == ("meta/llama", 4096, 2)
    assert (run.tensor_parallel, run.pipeline_parallel, run.precision) == (4, 2, "bf16")
    roundtrip = parse_megatron_command(run.to_megatron_args())
    assert (roundtrip.tensor_parallel, roundtrip.pipeline_parallel, roundtrip.precision) == (4, 2, "bf16")


def test_deepspeed_json_and_vllm_command_are_canonicalized():
    deep = parse_deepspeed_config({"train_micro_batch_size_per_gpu": 4, "zero_optimization": {"stage": 2}, "tensor_parallel": {"tp_size": 2}, "bf16": {"enabled": True}})
    assert (deep.source_format, deep.micro_batch_size, deep.tensor_parallel, deep.zero_stage, deep.precision) == ("deepspeed-json", 4, 2, 2, "bf16")
    vllm = parse_vllm_command("vllm serve meta/llama --tensor-parallel-size 8 --max-model-len 8192 --dtype bfloat16")
    assert (vllm.source_format, vllm.model, vllm.sequence_length, vllm.tensor_parallel, vllm.precision) == ("vllm", "meta/llama", 8192, 8, "bf16")


def test_log_metrics_are_attached_without_claiming_calibration():
    run = parse_log_metrics("step time: 125 ms, tokens/s: 9876, allocated: 12 GiB")
    assert run.observed_step_time_s == 0.125
    assert run.observed_tokens_per_s == 9876
    assert run.observed_memory_bytes == 12 * 2**30
    assert "calibrated" not in json.dumps(run.as_dict()).lower()


def test_audit_text_dispatches_json_and_logs():
    run = audit_text('{"fp16":{"enabled":true},"zero_optimization":{"stage":1}}', log="iteration time = 2 s")
    assert run.source_format == "deepspeed-json"
    assert run.precision == "fp16"
    assert run.observed_step_time_s == 2
