import time

from fastapi.testclient import TestClient

from nomo.config import load_config
from nomo.telemetry.server import create_app


def _wait_done(client, run_id, headers):
    for _ in range(200):
        status = client.get(f"/runs/{run_id}", headers=headers).json()["status"]
        if status in {"completed", "failed", "stopped"}:
            return status
        time.sleep(0.01)
    raise AssertionError("run did not finish")


def test_profile_validation_co_design_and_emulation_endpoints():
    with TestClient(create_app()) as client:
        profile = load_config("nomo/enterprise/examples/hard_realtime_robotics.yaml")
        validation = client.post("/enterprise/profile/validate", json=profile)
        assert validation.status_code == 200
        assert validation.json()["profile"]["id"] == "hard-realtime-robotics"

        headers = {"X-Nomo-Client": "enterprise-test"}
        project = client.post("/projects", headers=headers, json={"name": "Validation project"}).json()["project"]
        started = client.post("/runs", headers=headers, json={
            "project_id": project["project_id"], "model": "attitude_policy", "hardware": "akd1500",
            "pop_size": 8, "generations": 1,
        })
        assert started.status_code == 200
        run_id = started.json()["run_id"]
        assert _wait_done(client, run_id, headers) == "completed"

        co = client.post(f"/runs/{run_id}/co-design", headers=headers, json={
            "pe_rows": [1], "pe_cols": [1, 2], "sram_bytes": [1048576],
            "memory_bandwidth_bytes_s": [1000000000.0], "precision_bits": [8],
        })
        assert co.status_code == 200
        assert co.json()["capability"] == "hardware/deployment co-search"
        assert co.json()["recommended"] is not None

        emulation = client.post(f"/runs/{run_id}/emulation", headers=headers,
                                json={"vectors": 1, "config": {"limits": {"max_vectors": 1}}})
        assert emulation.status_code == 200, emulation.text
        payload = emulation.json()
        assert payload["backend"]["capability"] == "simulated"
        assert payload["backend"]["physical_measurement"] is False
        assert payload["summary"]["cycles"] > 0
        assert client.get(f"/projects/{project['project_id']}/runs", headers=headers).json()["runs"]
