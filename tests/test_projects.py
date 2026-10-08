from nomo.telemetry.projects import ProjectStore
from fastapi.testclient import TestClient
from nomo.telemetry.server import create_app


def test_projects_are_owner_scoped_and_runs_are_linked():
    store = ProjectStore()
    project = store.create("Flight controller", "client-a", {"hardware": "akd1500"})
    assert project["metadata"] == {"hardware": "akd1500"}
    assert store.get(project["project_id"], "client-a")["name"] == "Flight controller"
    assert store.get(project["project_id"], "client-b") is None
    store.attach_run(project["project_id"], "run-1", "client-a")
    assert store.runs(project["project_id"], "client-a")[0]["run_id"] == "run-1"
    assert store.runs(project["project_id"], "client-b") == []


def test_project_store_round_trips_file(tmp_path):
    path = str(tmp_path / "nomo.db")
    first = ProjectStore(path)
    project = first.create("Persistent", "client-a")
    first.close()
    second = ProjectStore(path)
    assert second.list("client-a")[0]["project_id"] == project["project_id"]
    second.close()


def test_project_api_scopes_runs_to_client():
    with TestClient(create_app()) as client:
        headers = {"X-Nomo-Client": "team-a"}
        created = client.post("/projects", headers=headers,
                              json={"name": "Robotics release", "metadata": {"owner": "flight"}})
        assert created.status_code == 200
        project = created.json()["project"]
        assert client.get("/projects", headers=headers).json()["projects"][0]["project_id"] == project["project_id"]
        assert client.get("/projects/" + project["project_id"],
                          headers={"X-Nomo-Client": "team-b"}).status_code == 404
        assert client.post("/runs", headers=headers,
                           json={"project_id": project["project_id"], "pop_size": 8, "generations": 1}).status_code == 200


def test_run_and_enterprise_endpoints_are_owner_scoped():
    with TestClient(create_app()) as client:
        owner = {"X-Nomo-Client": "owner-client"}
        other = {"X-Nomo-Client": "other-client"}
        started = client.post("/runs", headers=owner, json={
            "model": "attitude_policy", "hardware": "akd1500", "pop_size": 8, "generations": 1,
        })
        assert started.status_code == 200
        run_id = started.json()["run_id"]
        assert client.get(f"/runs/{run_id}", headers=owner).status_code == 200
        assert client.get(f"/runs/{run_id}", headers=other).status_code == 404
        assert all(row["run_id"] != run_id for row in client.get("/runs", headers=other).json())
        assert client.post(f"/runs/{run_id}/emulation", headers=other, json={}).status_code == 404
        assert client.post(f"/runs/{run_id}/stop", headers=other).status_code == 404
