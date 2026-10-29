"""Dashboard <-> backend contract test.

The dashboard (../frontend) is a separate Next.js app that talks to this
FastAPI backend over a fixed HTTP/WS contract.  It was originally developed
against a hosted deployment, not against this repository, so nothing failed
when the two drifted.  This test pins every route the dashboard calls --
with the frontend file that issues the call -- plus the exact catalog and
preset strings the dashboard's default configuration depends on.

If a route or payload field changes, update the frontend AND this test in
the same commit.
"""
from __future__ import annotations

import time

from fastapi.testclient import TestClient

from nomo.telemetry.server import create_app

# (method, path, frontend call site) -- the routes the dashboard actually hits.
DASHBOARD_ROUTES = [
    ("GET", "/healthz", "frontend/src/lib/api.ts waitForBackend()"),
    ("GET", "/catalog", "frontend/src/app/page.tsx initial load"),
    ("GET", "/presets", "frontend/src/app/page.tsx initial load"),
    ("GET", "/runs", "frontend/src/app/page.tsx initial load"),
    ("POST", "/runs", "frontend/src/lib/runConfig.ts startRun()"),
    ("GET", "/runs/{run_id}", "frontend/src/app/runs/[run_id]/page.tsx"),
    ("POST", "/runs/{run_id}/stop", "frontend/src/lib/runConfig.ts stopRun()"),
    ("GET", "/runs/{run_id}/designs/{key:path}", "frontend design detail link"),
    ("GET", "/runs/{run_id}/workbench", "frontend/src/app/runs/[run_id]/workbench"),
    ("POST", "/runs/{run_id}/copilot", "frontend/src/lib/telemetry CopilotAnswer"),
]

# Auth-guarded admin surface shown by frontend/src/app/admin/page.tsx.
ADMIN_ROUTES = ["/admin/stats", "/admin/logs"]

# The default payload built by frontend/src/lib/runConfig.ts defaultConfig().
# pop_size/generations are shrunk to the smallest contract-valid values
# (RunIn: pop_size ge=8, generations ge=1) so the search finishes in seconds;
# every other field is sent exactly as the dashboard sends it.
DASHBOARD_DEFAULT_RUN = {
    "model": "perception_cnn",
    "hardware": "akd1500",
    "budgets": {"accuracy_drop_max": 3},
    "pop_size": 8,
    "generations": 2,
    "seed": 0,
    "search": {
        "allow_continuous": True, "allow_spiking": True, "allow_symbolic": True,
        "codings": ["rate", "ttfs"], "crossing_penalty": 0,
        "crossing_min_saving_pct": 0, "p_crossover": 0.9, "p_mutation": 1,
        "archive_capacity": 0, "patience": 12, "asf_weights": None,
    },
    "pins": {},
    "lock_symbolic": False,
    "hardware_overrides": None,
    "preset": "balanced_edge",
}


def _route_table(app) -> set[tuple[str, str]]:
    table = set()
    for r in app.routes:
        path = getattr(r, "path", "")
        methods = getattr(r, "methods", None)
        if methods:
            table.update((m, path) for m in methods if m in ("GET", "POST", "PUT", "DELETE"))
        else:
            table.add(("WS", path))
    return table


def test_every_dashboard_route_is_registered():
    app = create_app()
    table = _route_table(app)
    for method, path, site in DASHBOARD_ROUTES:
        assert (method, path) in table, f"{method} {path} (called by {site}) is missing from the backend"
    for path in ADMIN_ROUTES:
        assert any(p == path for _, p in table), f"{path} (called by admin page) is missing"
    # The telemetry stream the dashboard's WebSocket client subscribes to.
    assert ("WS", "/ws/runs/{run_id}") in table, "dashboard telemetry client needs /ws/runs/{run_id}"


def test_healthz_and_catalog_shape():
    with TestClient(create_app()) as c:
        h = c.get("/healthz").json()
        assert h["ok"] is True and isinstance(h["version"], str)
        cat = c.get("/catalog").json()
        # Exact strings the dashboard's defaultConfig() and preset flow depend on.
        assert "perception_cnn" in cat["models"]
        for hw in ("akd1500", "loihi2", "gpu"):
            assert hw in cat["hardware"], f"dashboard default hardware {hw} missing from catalog"
            entry = cat["hardware"][hw]
            assert "name" in entry and "bits" in entry
        assert "balanced_edge" in c.get("/presets").json()


def test_runs_list_matches_runs_index():
    with TestClient(create_app()) as c:
        rid = c.post("/runs", json=DASHBOARD_DEFAULT_RUN).json()["run_id"]
        deadline = time.time() + 180
        while time.time() < deadline:
            status = c.get(f"/runs/{rid}").json()["status"]
            if status in ("completed", "failed", "stopped"):
                break
            time.sleep(0.5)
        assert status == "completed", f"dashboard default payload must produce a completed run, got {status}"
        listed = c.get("/runs").json()
        assert any(row["run_id"] == rid for row in listed), "GET /runs must list the run it created"


def test_default_payload_with_all_three_domains_completes():
    # The dashboard ships allow_spiking + allow_symbolic on by default; this is
    # the exact tri-domain path, not just the continuous fallback.
    with TestClient(create_app()) as c:
        rid = c.post("/runs", json=DASHBOARD_DEFAULT_RUN).json()["run_id"]
        deadline = time.time() + 180
        while time.time() < deadline:
            body = c.get(f"/runs/{rid}").json()
            if body["status"] in ("completed", "failed", "stopped"):
                break
            time.sleep(0.5)
        assert body["status"] == "completed", body.get("error", body["status"])
        # The recommended design is fetched through the designs endpoint
        # (GET /runs/{id}/designs/recommended), as the dashboard does.
        rec = c.get(f"/runs/{rid}/designs/recommended")
        assert rec.status_code == 200, "completed tri-domain run must expose a recommended design"
        d = rec.json()
        assert d["design"]["design_key"] and d["summary"], "recommended design payload shape changed"
