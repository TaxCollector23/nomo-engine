"""Logging, user tracking and admin API of the hosted backend (v3)."""
import importlib
import json
import time

import pytest
from fastapi.testclient import TestClient

TOKEN = "test-admin-token"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMO_ADMIN_TOKEN", TOKEN)
    monkeypatch.setenv("NOMO_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("NOMO_MAX_POP", "64")
    monkeypatch.setenv("NOMO_MAX_GENS", "40")
    monkeypatch.setenv("NOMO_LOG_GEN_EVERY", "1")
    import nomo.telemetry.observability as obs
    import nomo.telemetry.server as server
    import logging
    for h in list(logging.getLogger("nomo").handlers):           # fresh logging tree per test
        logging.getLogger("nomo").removeHandler(h)
    obs = importlib.reload(obs)
    server = importlib.reload(server)
    with TestClient(server.create_app()) as c:
        c.obs = obs
        yield c


def admin(c, path, **params):
    return c.get(path, params=params, headers={"Authorization": f"Bearer {TOKEN}"})


def run_to_completion(c, cid="browser-abc123"):
    rid = c.post("/runs", json={"model": "perception_cnn", "hardware": "akd1500", "pop_size": 24,
                                "generations": 6}, headers={"X-Nomo-Client": cid}).json()["run_id"]
    with c.websocket_connect(f"/ws/runs/{rid}?client={cid}") as ws:
        while ws.receive_json()["type"] not in ("run.completed", "run.failed"):
            pass
    for _ in range(50):                                           # worker logs completion after last envelope
        if any(r["event"] == "run.completed" for r in admin(c, "/admin/logs", stream="runs").json()["records"]):
            break
        time.sleep(0.05)
    return rid


def test_root_and_health(client):
    r = client.get("/").json()
    assert r["service"] == "nomo-backend" and r["version"] == "0.3.0"
    assert client.get("/healthz").json()["ok"]
    assert client.get("/healthz").headers["x-request-id"]


def test_admin_requires_token(client):
    assert client.get("/admin/stats").status_code == 401
    assert client.get("/admin/stats", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert admin(client, "/admin/stats").status_code == 200
    denied = admin(client, "/admin/logs", stream="backend").json()["records"]
    assert any(r["event"] == "admin.denied" for r in denied)


def test_every_stream_is_populated_by_a_real_session(client):
    rid = run_to_completion(client)
    logs = {s: admin(client, "/admin/logs", stream=s, limit=5000).json()["records"]
            for s in ("backend", "access", "users", "runs", "telemetry")}
    assert any(r["event"] == "backend.start" for r in logs["backend"])
    acc = [r for r in logs["access"] if r["event"] == "http.request" and r["path"] == "/runs"]
    assert acc and acc[0]["status"] == 200 and acc[0]["client_id"] == "browser-abc123" and acc[0]["duration_ms"] >= 0
    assert any(r["event"] == "user.first_seen" and r["client_id"] == "browser-abc123" for r in logs["users"])
    run_events = [r["event"] for r in logs["runs"] if r.get("run_id") == rid]
    assert run_events[0] == "run.created" and "run.started" in run_events and "run.progress" in run_events
    done = [r for r in logs["runs"] if r["event"] == "run.completed"][0]
    assert done["front"] > 0 and done["recommended"] and len(done["recommended_f"]) == 3
    ws = [r for r in logs["telemetry"] if r.get("run_id") == rid]
    assert [r["event"] for r in ws] == ["ws.open", "ws.close"]
    assert ws[0]["mode"] in ("full", "replay", "snapshot") and ws[1]["sent"] > 0


def test_users_and_runs_admin_views(client):
    rid = run_to_completion(client, cid="user-42")
    u = [x for x in admin(client, "/admin/users").json()["users"] if x["client_id"] == "user-42"][0]
    assert u["runs_started"] == 1 and u["ws_sessions"] == 1 and rid in u["run_ids"] and u["requests"] >= 1
    assert "." not in u["ip_hash"]                                # hashed, not a raw IP
    run = admin(client, "/admin/runs").json()["runs"][0]
    assert run["client_id"] == "user-42" and run["status"] == "completed" and run["envelopes_sent"] > 0
    st = admin(client, "/admin/stats").json()
    assert st["counters"]["runs_created"] == 1 and st["users"]["known"] >= 1


def test_rejections_are_logged(client):
    r = client.post("/runs", json={"model": "perception_cnn", "pop_size": 500}, headers={"X-Nomo-Client": "big"})
    assert r.status_code == 422
    assert client.post("/runs", json={"model": "nope"}).status_code == 404
    rej = [r for r in admin(client, "/admin/logs", stream="runs").json()["records"] if r["event"] == "run.rejected"]
    assert {r["status"] for r in rej} == {404, 422}


def test_filters_download_and_log_files(client, tmp_path):
    run_to_completion(client)
    only_ws = admin(client, "/admin/logs", stream="all", contains="ws.open").json()["records"]
    assert only_ws and all("ws.open" in json.dumps(r) for r in only_ws)
    warn = admin(client, "/admin/logs", stream="all", level="WARNING").json()["records"]
    assert all(r["level"] in ("WARNING", "ERROR", "CRITICAL") for r in warn)
    dl = admin(client, "/admin/logs/download", stream="runs")
    assert "attachment" in dl.headers["content-disposition"]
    lines = [json.loads(l) for l in dl.text.splitlines()]
    assert lines and all(l["stream"] == "runs" for l in lines)
    files = {p.name for p in (tmp_path / "logs").iterdir()}
    assert {"access.log", "runs.log", "telemetry.log", "users.log", "backend.log", "errors.log"} <= files
    rows = [json.loads(l) for l in (tmp_path / "logs" / "runs.log").read_text().splitlines()]
    assert rows and all(r["stream"] == "runs" for r in rows)
    assert admin(client, "/admin/logs", stream="bogus").status_code == 422


def test_failures_reach_errors_stream(client, monkeypatch):
    import nomo.telemetry.server as server
    def boom(*a, **k):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(server, "NeurosymbolicEvaluator", boom)
    rid = client.post("/runs", json={"model": "perception_cnn", "pop_size": 16, "generations": 2}).json()["run_id"]
    for _ in range(100):
        errs = admin(client, "/admin/logs", stream="errors").json()["records"]
        if errs:
            break
        time.sleep(0.02)
    e = errs[0]
    assert e["event"] == "run.failed" and e["run_id"] == rid and "synthetic failure" in e["exc"]
    assert client.get(f"/runs/{rid}").json()["status"] == "failed"
