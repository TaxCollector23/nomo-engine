from fastapi.testclient import TestClient

from nomo.telemetry.server import create_app


def test_websocket_stream_resume_and_snapshot():
    with TestClient(create_app()) as c:
        rid = c.post("/runs", json={"model": "perception_cnn", "hardware": "akd1500", "pop_size": 24,
                                    "generations": 6, "budgets": {"accuracy_drop_max": 4}}).json()["run_id"]
        seqs, final = [], None
        with c.websocket_connect(f"/ws/runs/{rid}") as ws:
            ws.send_json({"type": "ping"})
            while final is None:
                m = ws.receive_json()
                if m.get("type") == "pong":
                    continue
                assert m["v"] == 1 and m["run_id"] == rid
                seqs.append(m["seq"])
                if m["type"] in ("run.completed", "run.failed"):
                    final = m
        assert final["type"] == "run.completed"
        assert seqs == list(range(1, len(seqs) + 1))
        with c.websocket_connect(f"/ws/runs/{rid}?since={seqs[-3]}") as ws:
            assert [ws.receive_json()["seq"] for _ in range(2)] == seqs[-2:]
        with c.websocket_connect(f"/ws/runs/{rid}?since=10000000") as ws:
            snap = ws.receive_json()
            assert snap["type"] == "snapshot" and snap["data"]["items"]
        assert c.get(f"/runs/{rid}").json()["status"] == "completed"
        assert c.post("/runs", json={"model": "nope"}).status_code == 404
