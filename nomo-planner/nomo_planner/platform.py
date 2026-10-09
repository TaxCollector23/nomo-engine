"""Local project, artifact, run, and report storage for Nomo Planner.

The implementation uses only the Python standard library. A SQLite database
keeps metadata and JSON payloads together so a project can be copied or backed
up as a single file.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value must be JSON serializable: {exc}") from exc


def _object(value: Mapping[str, Any] | None, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    # Round-trip both validates JSON types and detaches caller-owned objects.
    return json.loads(_json(value))


class NotFoundError(LookupError):
    """Raised when a requested project, artifact, or run does not exist."""


class PlatformStore:
    """SQLite-backed platform store; safe to share across threads/processes."""

    def __init__(self, path: str | Path = ".nomo/platform.sqlite3") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        # Keep one connection for in-memory databases; normal stores open a
        # short-lived connection for each transaction.
        if self.path == ":memory:":
            self.path = f"file:nomo-{uuid.uuid4().hex}?mode=memory&cache=shared"
            self._uri = True
            self._memory_connection = sqlite3.connect(self.path, uri=True, timeout=30, check_same_thread=False)
            self._memory_connection.row_factory = sqlite3.Row
            self._memory_connection.execute("PRAGMA foreign_keys = ON")
            self._memory_connection.execute("PRAGMA busy_timeout = 30000")
            connection = self._memory_connection
        else:
            self._uri = False
            connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL,
                    metadata TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    name TEXT NOT NULL, status TEXT NOT NULL, config TEXT NOT NULL, result TEXT NOT NULL,
                    metadata TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS runs_project_created ON runs(project_id, created_at);
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    run_id TEXT REFERENCES runs(id) ON DELETE CASCADE, name TEXT NOT NULL,
                    media_type TEXT NOT NULL, content TEXT NOT NULL, sha256 TEXT NOT NULL,
                    metadata TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS artifacts_project_created ON artifacts(project_id, created_at);
                CREATE INDEX IF NOT EXISTS artifacts_run ON artifacts(run_id);
                """
            )
            connection.commit()
        finally:
            if connection is not getattr(self, "_memory_connection", None):
                connection.close()

    def _transaction(self):
        connection = self._memory_connection if self._uri else self._connect()
        return connection

    def create_project(
        self, name: str, *, description: str = "", metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        name = _required_text(name, "name")
        description = _text(description, "description")
        now, project_id = _now(), _id("prj")
        record = {"id": project_id, "name": name, "description": description,
                  "metadata": _object(metadata, "metadata"), "created_at": now, "updated_at": now}
        connection = self._transaction()
        try:
            connection.execute("INSERT INTO projects VALUES (?, ?, ?, ?, ?, ?)",
                               (project_id, name, description, _json(record["metadata"]), now, now))
            connection.commit()
        finally:
            if not self._uri:
                connection.close()
        return record

    def get_project(self, project_id: str) -> dict[str, Any]:
        row = self._one("SELECT * FROM projects WHERE id = ?", (project_id,))
        return _project(row)

    def list_projects(self) -> list[dict[str, Any]]:
        return [_project(row) for row in self._all("SELECT * FROM projects ORDER BY created_at, id")]

    def create_run(
        self, project_id: str, *, name: str = "run", config: Mapping[str, Any] | None = None,
        result: Mapping[str, Any] | None = None, status: str = "completed",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.get_project(project_id)
        name, status = _required_text(name, "name"), _required_text(status, "status")
        if status not in {"queued", "running", "completed", "failed", "cancelled"}:
            raise ValueError("status must be queued, running, completed, failed, or cancelled")
        now, run_id = _now(), _id("run")
        record = {"id": run_id, "project_id": project_id, "name": name, "status": status,
                  "config": _object(config, "config"), "result": _object(result, "result"),
                  "metadata": _object(metadata, "metadata"), "created_at": now, "updated_at": now}
        connection = self._transaction()
        try:
            connection.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                               (run_id, project_id, name, status, _json(record["config"]),
                                _json(record["result"]), _json(record["metadata"]), now, now))
            connection.commit()
        finally:
            if not self._uri:
                connection.close()
        return record

    def update_run(
        self, run_id: str, *, status: str | None = None,
        result: Mapping[str, Any] | None = None, metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = self.get_run(run_id)
        next_status = status or current["status"]
        if next_status not in {"queued", "running", "completed", "failed", "cancelled"}:
            raise ValueError("invalid run status")
        next_result = current["result"] if result is None else _object(result, "result")
        next_metadata = current["metadata"] if metadata is None else _object(metadata, "metadata")
        updated = _now()
        connection = self._transaction()
        try:
            connection.execute("UPDATE runs SET status=?, result=?, metadata=?, updated_at=? WHERE id=?",
                               (next_status, _json(next_result), _json(next_metadata), updated, run_id))
            connection.commit()
        finally:
            if not self._uri:
                connection.close()
        return {**current, "status": next_status, "result": next_result,
                "metadata": next_metadata, "updated_at": updated}

    def get_run(self, run_id: str) -> dict[str, Any]:
        row = self._one("SELECT * FROM runs WHERE id = ?", (run_id,))
        return _run(row)

    def list_runs(self, project_id: str) -> list[dict[str, Any]]:
        self.get_project(project_id)
        return [_run(row) for row in self._all(
            "SELECT * FROM runs WHERE project_id = ? ORDER BY created_at, id", (project_id,))]

    def compare_runs(self, run_id_a: str, run_id_b: str) -> dict[str, Any]:
        """Compare numeric result metrics and grouped timeline events.

        Metric deltas are ``run_b - run_a``. Event rows are grouped by their
        semantic event name/kind, so simulator-generated IDs do not obscure
        changes in event counts or accumulated duration.
        """
        run_a, run_b = self.get_run(run_id_a), self.get_run(run_id_b)
        metrics_a, metrics_b = _metric_values(run_a["result"]), _metric_values(run_b["result"])
        metrics: dict[str, Any] = {}
        for key in sorted(set(metrics_a) | set(metrics_b)):
            value_a, value_b = metrics_a.get(key), metrics_b.get(key)
            delta = value_b - value_a if value_a is not None and value_b is not None else None
            metrics[key] = {
                "run_a": value_a, "run_b": value_b, "delta": delta,
                "delta_pct": (delta / abs(value_a) * 100.0) if delta is not None and value_a != 0 else None,
            }
        events_a, events_b = _timeline_totals(run_a["result"]), _timeline_totals(run_b["result"])
        timeline_deltas = []
        for name in sorted(set(events_a) | set(events_b)):
            count_a, duration_a = events_a.get(name, (0, 0.0))
            count_b, duration_b = events_b.get(name, (0, 0.0))
            timeline_deltas.append({
                "event": name, "count_a": count_a, "count_b": count_b,
                "count_delta": count_b - count_a, "duration_a_s": duration_a,
                "duration_b_s": duration_b, "duration_delta_s": duration_b - duration_a,
            })
        return {
            "run_a": {"id": run_a["id"], "name": run_a["name"], "status": run_a["status"]},
            "run_b": {"id": run_b["id"], "name": run_b["name"], "status": run_b["status"]},
            "delta_direction": "run_b - run_a", "metrics": metrics,
            "timeline_event_deltas": timeline_deltas,
        }

    def create_artifact(
        self, project_id: str, name: str, content: Any, *, media_type: str = "application/json",
        run_id: str | None = None, metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.get_project(project_id)
        if run_id is not None and self.get_run(run_id)["project_id"] != project_id:
            raise ValueError("run_id must refer to a run in the same project")
        name, media_type = _required_text(name, "name"), _required_text(media_type, "media_type")
        content_json = _json(content)
        digest = hashlib.sha256(content_json.encode("utf-8")).hexdigest()
        artifact_id, created = _id("art"), _now()
        record = {"id": artifact_id, "project_id": project_id, "run_id": run_id,
                  "name": name, "media_type": media_type, "content": json.loads(content_json),
                  "sha256": digest, "metadata": _object(metadata, "metadata"), "created_at": created}
        connection = self._transaction()
        try:
            connection.execute("INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                               (artifact_id, project_id, run_id, name, media_type, content_json,
                                digest, _json(record["metadata"]), created))
            connection.commit()
        finally:
            if not self._uri:
                connection.close()
        return record

    def get_artifact(self, artifact_id: str) -> dict[str, Any]:
        row = self._one("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))
        return _artifact(row)

    def list_artifacts(self, project_id: str, *, run_id: str | None = None) -> list[dict[str, Any]]:
        self.get_project(project_id)
        if run_id is None:
            rows = self._all("SELECT * FROM artifacts WHERE project_id=? ORDER BY created_at, id", (project_id,))
        else:
            rows = self._all("SELECT * FROM artifacts WHERE project_id=? AND run_id=? ORDER BY created_at, id",
                             (project_id, run_id))
        return [_artifact(row) for row in rows]

    def render_report(self, run_id: str) -> dict[str, Any]:
        """Persist a JSON and an escaped HTML summary for a run."""
        run = self.get_run(run_id)
        project = self.get_project(run["project_id"])
        report = {"schema_version": 1, "project": {"id": project["id"], "name": project["name"]},
                  "run": run}
        json_artifact = self.create_artifact(
            project["id"], f"{run['name']}-report.json", report,
            media_type="application/json", run_id=run_id,
            metadata={"kind": "run-report", "format": "json"})
        html_content = render_html_report(report)
        html_artifact = self.create_artifact(
            project["id"], f"{run['name']}-report.html", html_content,
            media_type="text/html; charset=utf-8", run_id=run_id,
            metadata={"kind": "run-report", "format": "html"})
        return {"report": report, "json_artifact": json_artifact, "html_artifact": html_artifact}

    def _one(self, query: str, params: tuple[Any, ...]) -> sqlite3.Row:
        rows = self._all(query, params)
        if not rows:
            raise NotFoundError("record not found")
        return rows[0]

    def _all(self, query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        connection = self._transaction()
        try:
            return list(connection.execute(query, params).fetchall())
        finally:
            if not self._uri:
                connection.close()


def render_html_report(report: Mapping[str, Any]) -> str:
    """Render a self-contained, safely escaped HTML report."""
    run = report.get("run", {})
    project = report.get("project", {})
    details = html.escape(json.dumps(run, indent=2, sort_keys=True, ensure_ascii=False))
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{html.escape(str(run.get('name', 'Run report')))} · Nomo</title>"
            "<style>body{font:16px/1.55 system-ui,sans-serif;max-width:900px;margin:3rem auto;padding:0 1rem;color:#17212b}"
            "h1{margin-bottom:.25rem}p{color:#536270}pre{white-space:pre-wrap;background:#f3f6f8;padding:1rem;border-radius:8px}</style>"
            "</head><body><main><h1>Run report</h1>"
            f"<p>Project: {html.escape(str(project.get('name', '')))} · Run: {html.escape(str(run.get('id', '')))}</p>"
            f"<pre>{details}</pre></main></body></html>")


def _required_text(value: str, label: str) -> str:
    text = _text(value, label).strip()
    if not text:
        raise ValueError(f"{label} must not be empty")
    return text


def _text(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _project(row: sqlite3.Row) -> dict[str, Any]:
    return {"id": row["id"], "name": row["name"], "description": row["description"],
            "metadata": json.loads(row["metadata"]), "created_at": row["created_at"], "updated_at": row["updated_at"]}


def _run(row: sqlite3.Row) -> dict[str, Any]:
    return {"id": row["id"], "project_id": row["project_id"], "name": row["name"], "status": row["status"],
            "config": json.loads(row["config"]), "result": json.loads(row["result"]),
            "metadata": json.loads(row["metadata"]), "created_at": row["created_at"], "updated_at": row["updated_at"]}


def _artifact(row: sqlite3.Row) -> dict[str, Any]:
    return {"id": row["id"], "project_id": row["project_id"], "run_id": row["run_id"],
            "name": row["name"], "media_type": row["media_type"], "content": json.loads(row["content"]),
            "sha256": row["sha256"], "metadata": json.loads(row["metadata"]), "created_at": row["created_at"]}


def _simulation_result(result: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = result.get("simulation")
    return nested if isinstance(nested, Mapping) else result


def _metric_values(result: Mapping[str, Any]) -> dict[str, float]:
    source = _simulation_result(result)
    raw = source.get("metrics")
    if isinstance(raw, Mapping):
        candidates = raw
    else:
        candidates = {key: value for key, value in source.items()
                     if key not in {"timeline", "timelines", "events", "assumptions", "provenance", "metadata"}}
    values: dict[str, float] = {}
    for key, value in candidates.items():
        if isinstance(value, Mapping) and "value" in value:
            value = value["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number):
            values[str(key)] = number
    return values


def _timeline_totals(result: Mapping[str, Any]) -> dict[str, tuple[int, float]]:
    source = _simulation_result(result)
    timeline = source.get("timeline", source.get("events", ()))
    if isinstance(timeline, Mapping):
        timeline = timeline.get("events", ())
    if not isinstance(timeline, (list, tuple)):
        return {}
    totals: dict[str, tuple[int, float]] = {}
    for event in timeline:
        if not isinstance(event, Mapping):
            continue
        name = event.get("event", event.get("kind", event.get("type", event.get("name", "unknown"))))
        duration = event.get("duration_s")
        if duration is None and isinstance(event.get("start_s"), (int, float)) and isinstance(event.get("end_s"), (int, float)):
            duration = event["end_s"] - event["start_s"]
        try:
            seconds = float(duration or 0.0)
        except (TypeError, ValueError):
            seconds = 0.0
        if not math.isfinite(seconds):
            seconds = 0.0
        key = str(name)
        count, total = totals.get(key, (0, 0.0))
        totals[key] = (count + 1, total + seconds)
    return totals
