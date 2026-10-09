import base64
import json
import sys
import threading
import types
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from nomo_planner.api import PlatformHTTPServer, PlatformService, handle_mcp_message
from nomo_planner.cli import build_parser
from nomo_planner.platform import NotFoundError, PlatformStore, render_html_report
from nomo_planner.sdk import PlatformClient


@pytest.fixture
def store(tmp_path):
    return PlatformStore(tmp_path / "platform.sqlite3")


def test_project_run_artifact_lifecycle_and_report(store):
    project = store.create_project("model study", description="local", metadata={"owner": "test"})
    run = store.create_run(project["id"], name="baseline", config={"batch": 8}, result={"loss": 0.2})
    artifact = store.create_artifact(project["id"], "metrics.json", {"loss": 0.2}, run_id=run["id"])

    assert store.get_project(project["id"])["metadata"] == {"owner": "test"}
    assert store.list_runs(project["id"])[0]["config"] == {"batch": 8}
    assert store.get_artifact(artifact["id"])["sha256"] == artifact["sha256"]
    report = store.render_report(run["id"])
    assert report["report"]["run"]["result"] == {"loss": 0.2}
    assert report["json_artifact"]["media_type"] == "application/json"
    html_artifact = store.get_artifact(report["html_artifact"]["id"])
    assert "Run report" in html_artifact["content"]
    assert f"{run['id']}" in html_artifact["content"]
    assert len(store.list_artifacts(project["id"], run_id=run["id"])) == 3


def test_store_validates_status_json_and_project_boundaries(store):
    project = store.create_project("one")
    other = store.create_project("two")
    run = store.create_run(project["id"])
    with pytest.raises(ValueError, match="status"):
        store.update_run(run["id"], status="unknown")
    with pytest.raises(ValueError, match="same project"):
        store.create_artifact(other["id"], "bad", {}, run_id=run["id"])
    with pytest.raises(ValueError, match="JSON serializable"):
        store.create_run(project["id"], config={"not_json": object()})
    with pytest.raises(NotFoundError):
        store.get_run("missing")


def test_html_report_escapes_user_supplied_values():
    output = render_html_report({"project": {"name": "<script>"}, "run": {"name": "<img>", "result": {}}})
    assert "&lt;script&gt;" in output
    assert "&lt;img&gt;" in output
    assert "<script>" not in output


def test_sdk_local_and_http_modes(store):
    local = PlatformClient(store=store)
    project = local.create_project("sdk")
    assert local.list_projects()[0]["id"] == project["id"]

    server = PlatformHTTPServer(("127.0.0.1", 0), store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        remote = PlatformClient(base_url=f"http://127.0.0.1:{server.server_port}")
        created = remote.create_project("http sdk", metadata={"from": "http"})
        assert remote.list_projects()[-1]["id"] == created["id"]
        run = remote.create_run(created["id"], result={"score": 3}, status="running")
        assert remote.get_run(run["id"])["result"] == {"score": 3}
        artifact = remote.create_artifact(created["id"], "remote.json", {"ok": True}, run_id=run["id"])
        assert remote.list_artifacts(created["id"], run_id=run["id"])[0]["id"] == artifact["id"]
        assert remote.list_runs(created["id"])[0]["id"] == run["id"]
        updated = remote.update_run(run["id"], status="completed", result={"score": 4}, metadata={"review": "ready"})
        assert updated["status"] == "completed"
        assert updated["result"] == {"score": 4}
        assert updated["metadata"] == {"review": "ready"}
        report = remote.render_report(run["id"])
        assert report["html_artifact"]["media_type"].startswith("text/html")
        request = Request(f"http://127.0.0.1:{server.server_port}/health")
        with urlopen(request) as response:
            assert json.loads(response.read())["status"] == "ok"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_healthz_capabilities_cors_and_bearer_auth(store):
    server = PlatformHTTPServer(("127.0.0.1", 0), store, token="test-token", cors_origins=["https://lab.example"])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(Request(f"{base}/healthz")) as response:
            health = json.loads(response.read())
            assert health["status"] == "ok"
            assert health["version"] == "1.1.0"

        with pytest.raises(HTTPError) as unauthorized:
            urlopen(Request(f"{base}/projects", headers={"Origin": "https://lab.example"}))
        assert unauthorized.value.code == 401
        assert json.loads(unauthorized.value.read()) == {"error": "authorization required"}
        assert unauthorized.value.headers["Access-Control-Allow-Origin"] == "https://lab.example"
        assert unauthorized.value.headers["WWW-Authenticate"] == 'Bearer realm="nomo-platform"'

        authorized = Request(f"{base}/projects", headers={"Authorization": "Bearer test-token", "Origin": "https://lab.example"})
        with urlopen(authorized) as response:
            assert response.headers["Access-Control-Allow-Origin"] == "https://lab.example"
            assert json.loads(response.read()) == []

        preflight = Request(f"{base}/projects", method="OPTIONS", headers={
            "Origin": "https://lab.example", "Access-Control-Request-Method": "POST",
        })
        with urlopen(preflight) as response:
            assert response.status == 204
            assert "PATCH" in response.headers["Access-Control-Allow-Methods"]

        with pytest.raises(HTTPError) as forbidden_origin:
            urlopen(Request(f"{base}/projects", method="OPTIONS", headers={"Origin": "https://other.example"}))
        assert forbidden_origin.value.code == 403

        with urlopen(Request(f"{base}/capabilities", headers={"Authorization": "Bearer test-token"})) as response:
            capabilities = json.loads(response.read())
            assert capabilities["authentication"] == "bearer"
            assert "simulations" in capabilities["operations"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_rejects_oversized_json_before_parsing(store):
    server = PlatformHTTPServer(("127.0.0.1", 0), store, max_body_bytes=32)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(f"http://127.0.0.1:{server.server_port}/projects", data=b'{"name":"' + (b"x" * 64) + b'"}', method="POST",
                          headers={"Content-Type": "application/json"})
        with pytest.raises(HTTPError) as oversized:
            urlopen(request)
        assert oversized.value.code == 413
        assert json.loads(oversized.value.read()) == {"error": "request body exceeds 32 bytes"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_cli_exposes_request_body_limit():
    args = build_parser().parse_args(["serve", "--max-body-bytes", "4096"])
    assert args.max_body_bytes == 4096


def test_mcp_json_rpc_tools_and_errors(store):
    service = PlatformService(store)
    listed = handle_mcp_message(service, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert any(tool["name"] == "reports.render" for tool in listed["result"]["tools"])
    created = handle_mcp_message(service, {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "projects.create", "arguments": {"name": "mcp"}}})
    project = json.loads(created["result"]["content"][0]["text"])
    assert project["name"] == "mcp"
    missing = handle_mcp_message(service, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "runs.get", "arguments": {"run_id": "absent"}}})
    assert missing["result"]["isError"] is True


def test_compare_runs_reports_numeric_and_timeline_deltas(store):
    project = store.create_project("comparison")
    first = store.create_run(project["id"], name="first", result={
        "metrics": {"latency_s": 2.0, "tokens_per_s": 10, "wrapped": {"value": 5}},
        "timeline": [
            {"kind": "compute", "start_s": 0, "end_s": 2},
            {"event": "queue", "time_s": 1},
        ],
    })
    second = store.create_run(project["id"], name="second", result={
        "metrics": {"latency_s": 1.5, "tokens_per_s": 15, "new": 9},
        "timeline": [
            {"kind": "compute", "start_s": 0, "end_s": 1.5},
            {"kind": "compute", "start_s": 1.5, "end_s": 2},
            {"event": "complete", "time_s": 2},
        ],
    })

    comparison = store.compare_runs(first["id"], second["id"])
    assert comparison["delta_direction"] == "run_b - run_a"
    assert comparison["metrics"]["latency_s"] == {
        "run_a": 2.0, "run_b": 1.5, "delta": -0.5, "delta_pct": -25.0}
    assert comparison["metrics"]["wrapped"]["delta"] is None
    deltas = {row["event"]: row for row in comparison["timeline_event_deltas"]}
    assert deltas["compute"]["count_delta"] == 1
    assert deltas["compute"]["duration_delta_s"] == 0.0
    assert deltas["queue"]["count_delta"] == -1
    assert deltas["complete"]["count_delta"] == 1


def test_training_simulation_uses_stored_artifacts_and_is_labeled_preview(store):
    project = store.create_project("preview sim")
    graph = store.create_artifact(project["id"], "graph.json", {
        "model_type": "llama", "num_hidden_layers": 1, "hidden_size": 8,
        "num_attention_heads": 2, "intermediate_size": 16, "vocab_size": 32,
    })
    topology = store.create_artifact(project["id"], "topology.json", {"devices": 1})
    config = store.create_artifact(project["id"], "config.json", {
        "sequence_length": 4, "batch_size": 1, "micro_batches": 1,
        "hardware": {"peak_flops": 1e12, "memory_bandwidth": 1e10},
    })

    result = PlatformService(store).dispatch("simulations.run", {
        "project_id": project["id"], "kind": "training", "name": "tiny-preview",
        "graph_artifact_id": graph["id"], "topology_artifact_id": topology["id"],
        "config_artifact_id": config["id"],
    })
    preview = result["simulation"]
    assert result["run"]["result"]["label"] == "Preview"
    assert preview["label"] == "Preview"
    assert preview["provenance"]["classification"] == "simulated analytical estimate"
    assert preview["provenance"]["inputs"]["graph"]["artifact_id"] == graph["id"]
    assert preview["metrics"]["step_time_s"] > 0
    assert preview["timeline"]


def test_model_artifact_inspection_is_available_through_the_service(store):
    service = PlatformService(store)
    inspected = service.dispatch("artifacts.inspect_model", {"content": {
        "model_type": "llama", "hidden_size": 8, "num_hidden_layers": 1,
        "num_attention_heads": 2, "intermediate_size": 16, "vocab_size": 32,
    }})
    assert inspected["format"] == "huggingface-config"
    assert inspected["status"] == "ready"
    assert inspected["validation"]["lowering"] == "transformer-skeleton-v1"
    assert inspected["graph_nodes"][0]["id"] == "embedding"

    project = store.create_project("artifact inspection")
    stored = store.create_artifact(project["id"], "config.json", {
        "model_type": "llama", "hidden_size": 8, "num_hidden_layers": 1,
        "num_attention_heads": 2, "intermediate_size": 16, "vocab_size": 32,
    })
    stored_result = service.dispatch("artifacts.inspect_model", {"artifact_id": stored["id"]})
    assert stored_result["stored_artifact"]["id"] == stored["id"]
    assert stored_result["artifact"]["status"] == "ready"
    assert stored_result["artifact"]["validation"]["valid"] is True


def test_sdk_inspects_inline_hf_and_nomo_json_over_http(store):
    server = PlatformHTTPServer(("127.0.0.1", 0), store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = PlatformClient(base_url=f"http://127.0.0.1:{server.server_port}")
        hf = client.inspect_model(content={
            "model_type": "llama", "hidden_size": 8, "num_hidden_layers": 1,
            "num_attention_heads": 2, "intermediate_size": 16, "vocab_size": 32,
        })
        assert hf["format"] == "huggingface-config"
        assert hf["status"] == "ready"
        assert hf["validation"]["lowering"] == "transformer-skeleton-v1"

        nomo = client.inspect_model(content={
            "schema_version": 1,
            "nodes": [
                {"id": "input", "kind": "input", "outputs": ["hidden"]},
                {"id": "output", "kind": "output", "inputs": ["hidden"]},
            ],
        })
        assert nomo["format"] == "nomo-graph-json"
        assert nomo["status"] == "ready"
        assert [node["id"] for node in nomo["graph_nodes"]] == ["input", "output"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_sdk_inspects_base64_onnx_envelope_locally_and_over_http(store, monkeypatch):
    node = types.SimpleNamespace(op_type="Relu", name="relu", input=["x"], output=["y"], attribute=[])
    graph_input = types.SimpleNamespace(name="x", type=types.SimpleNamespace(
        tensor_type=types.SimpleNamespace(elem_type=1, shape=types.SimpleNamespace(dim=[]))))
    graph_output = types.SimpleNamespace(name="y", type=types.SimpleNamespace(
        tensor_type=types.SimpleNamespace(elem_type=1, shape=types.SimpleNamespace(dim=[]))))
    model = types.SimpleNamespace(graph=types.SimpleNamespace(
        input=[graph_input], initializer=[], value_info=[], node=[node], output=[graph_output],
    ))
    fake_onnx = types.ModuleType("onnx")
    fake_onnx.load_model_from_string = lambda raw: model
    fake_onnx.checker = types.SimpleNamespace(check_model=lambda checked: None)
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)
    envelope = {
        "filename": "model.onnx",
        "encoding": "base64",
        "base64": base64.b64encode(b"safe test fixture").decode("ascii"),
    }

    local = PlatformClient(store=store)
    direct = local.inspect_model(content=envelope)
    assert direct["validation"]["lowering"] == "structural-graph-v1"
    project = local.create_project("stored onnx")
    stored = local.create_artifact(project["id"], "model.onnx", envelope)
    inspected = local.inspect_model(artifact_id=stored["id"])
    assert inspected["artifact"]["status"] == "graph-inspected"

    server = PlatformHTTPServer(("127.0.0.1", 0), store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        remote = PlatformClient(base_url=f"http://127.0.0.1:{server.server_port}")
        result = remote.inspect_model(content=envelope)
        assert result["format"] == "onnx"
        assert [node["kind"] for node in result["graph_nodes"]] == ["input", "Relu", "output"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_sdk_model_inspection_reports_missing_content_or_artifact_id(store):
    local = PlatformClient(store=store)
    with pytest.raises(ValueError, match="content or artifact_id is required"):
        local.inspect_model()

    server = PlatformHTTPServer(("127.0.0.1", 0), store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        remote = PlatformClient(base_url=f"http://127.0.0.1:{server.server_port}")
        with pytest.raises(RuntimeError, match="content or artifact_id is required"):
            remote.inspect_model()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_serving_simulation_and_http_surface_include_preview_and_comparison(store):
    local = PlatformClient(store=store)
    project = local.create_project("api sim")
    sim = local.run_simulation(project["id"], kind="serving", config={
        "request_count": 2, "arrival_process": "replay",
        "prefill_tokens_per_s": 1000, "decode_tokens_per_s": 100,
    }, trace=[
        {"request_id": "r1", "arrival_s": 0, "prompt_tokens": 4, "answer_tokens": 2},
        {"request_id": "r2", "arrival_s": 0.1, "prompt_tokens": 4, "answer_tokens": 2},
    ])
    assert sim["simulation"]["label"] == "Preview"
    assert sim["simulation"]["metrics"]["requests"] == 2
    baseline = local.create_run(project["id"], name="baseline", result={"metrics": {"score": 1}})
    followup = local.create_run(project["id"], name="followup", result={"metrics": {"score": 2}})

    server = PlatformHTTPServer(("127.0.0.1", 0), store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        remote = PlatformClient(base_url=f"http://127.0.0.1:{server.server_port}")
        compared = remote.compare_runs(baseline["id"], followup["id"])
        assert compared["metrics"]["score"]["delta"] == 1
        remote_preview = remote.run_simulation(project["id"], kind="training",
            graph={"num_hidden_layers": 1, "hidden_size": 8, "num_attention_heads": 2,
                   "intermediate_size": 16, "vocab_size": 32},
            topology={"devices": 1},
            config={"sequence_length": 2, "micro_batches": 1})
        assert remote_preview["simulation"]["label"] == "Preview"
        assert remote_preview["run"]["status"] == "completed"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_mcp_tools_expose_run_comparison_and_simulation(store):
    service = PlatformService(store)
    listed = handle_mcp_message(service, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    names = {tool["name"] for tool in listed["result"]["tools"]}
    assert {"runs.compare", "simulations.run"}.issubset(names)
