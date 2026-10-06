from nomo_planner.audit_reports import compare_audits, render_audit_html, render_audit_pdf
from nomo_planner.auditor import parse_megatron_command
from nomo_planner.serving_sim import ServingAssumptions, simulate_serving
from nomo_planner.simcore import Parallelism, RooflineHardware, TrainingOptions, Topology, build_operator_graph, simulate_training_step
from nomo_planner.simulation_exports import export_serving_config, export_training_config


def tiny_graph():
    return build_operator_graph({
        "_name_or_path": "tiny", "num_hidden_layers": 1, "hidden_size": 8,
        "intermediate_size": 16, "num_attention_heads": 2, "num_key_value_heads": 1,
        "vocab_size": 32,
    }, sequence_length=2)


def test_training_exports_include_framework_projection_and_timeline():
    graph = tiny_graph()
    options = TrainingOptions(parallelism=Parallelism(tensor=2), schedule="interleaved", sequence_parallel=2)
    simulation = simulate_training_step(graph, RooflineHardware(1e9, 1e9), Topology(2), options)
    for framework in ("megatron", "deepspeed", "torchtitan"):
        exported = export_training_config(graph, options, framework=framework, simulation=simulation)
        assert exported["framework"] == framework
        assert exported["simulation"]["event_count"] == len(simulation.events)


def test_serving_exports_expose_config_and_simulated_metric_labels():
    assumptions = ServingAssumptions(request_count=2, arrival_process="poisson", arrival_rate_per_s=2, prompt_tokens=8, answer_tokens=4)
    result = simulate_serving(assumptions=assumptions)
    for framework in ("vllm", "sglang", "tensorrt-llm"):
        exported = export_serving_config(assumptions, result, framework=framework)
        assert exported["framework"] == framework
        assert set(exported["simulation"]["metric_labels"].values()) == {"simulated"}


def test_audit_diff_and_html_pdf_reports_are_shareable():
    current = parse_megatron_command("torchrun pretrain.py --model meta/llama --bf16")
    candidate = parse_megatron_command("torchrun pretrain.py --model meta/llama --bf16 --tensor-model-parallel-size 2")
    report = compare_audits(current, candidate)
    assert any(item["field"] == "tensor_parallel" for item in report["changes"])
    html = render_audit_html({"title": "<unsafe>", **report})
    assert "<unsafe>" not in html and "&lt;unsafe&gt;" in html
    pdf = render_audit_pdf(report)
    assert pdf.startswith(b"%PDF-1.4") and b"%%EOF" in pdf
