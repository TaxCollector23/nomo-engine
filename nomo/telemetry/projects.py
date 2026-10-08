"""Small, local-first project store for enterprise run organization.

The hosted service can point this store at a durable SQLite volume with
``NOMO_STATE_DB``.  With no path configured it stays in-process and explicitly
behaves as an ephemeral store, preserving the free-host behaviour.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


class ProjectStore:
    """Thread-safe SQLite-backed project and run-association store."""

    def __init__(self, path: str = ":memory:") -> None:
        self.path = str(Path(path).expanduser()) if path and path != ":memory:" else ":memory:"
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    owner_client_id TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS project_runs (
                    run_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
                    client_id TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_projects_owner ON projects(owner_client_id);
                CREATE INDEX IF NOT EXISTS idx_project_runs_project ON project_runs(project_id);
                """
            )
            self._conn.commit()

    @staticmethod
    def _metadata(value: Optional[Dict[str, Any]]) -> str:
        if value is None:
            return "{}"
        if not isinstance(value, dict):
            raise ValueError("project metadata must be an object")
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)

    @staticmethod
    def _row(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "project_id": row["project_id"],
            "name": row["name"],
            "owner_client_id": row["owner_client_id"],
            "metadata": json.loads(row["metadata_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def create(self, name: str, owner_client_id: str,
               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        name = str(name or "").strip()
        if not name or len(name) > 120:
            raise ValueError("project name must be between 1 and 120 characters")
        if not owner_client_id:
            raise ValueError("project owner is required")
        now = time.time()
        project_id = "project_" + uuid.uuid4().hex[:14]
        encoded = self._metadata(metadata)
        with self._lock:
            self._conn.execute(
                "INSERT INTO projects(project_id,name,owner_client_id,metadata_json,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                (project_id, name, owner_client_id, encoded, now, now),
            )
            self._conn.commit()
            row = self._conn.execute("SELECT * FROM projects WHERE project_id = ?", (project_id,)).fetchone()
        return self._row(row)

    def get(self, project_id: str, client_id: str, admin: bool = False) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM projects WHERE project_id = ?", (project_id,)).fetchone()
        if row is None or (not admin and row["owner_client_id"] != client_id):
            return None
        return self._row(row)

    def list(self, client_id: str, admin: bool = False) -> List[Dict[str, Any]]:
        with self._lock:
            if admin:
                rows = self._conn.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM projects WHERE owner_client_id = ? ORDER BY updated_at DESC", (client_id,)
                ).fetchall()
        return [self._row(row) for row in rows]

    def attach_run(self, project_id: str, run_id: str, client_id: str) -> None:
        if self.get(project_id, client_id) is None:
            raise KeyError(project_id)
        with self._lock:
            now = time.time()
            self._conn.execute(
                "INSERT OR REPLACE INTO project_runs(run_id,project_id,client_id,created_at) VALUES(?,?,?,?)",
                (run_id, project_id, client_id, now),
            )
            self._conn.execute("UPDATE projects SET updated_at = ? WHERE project_id = ?", (now, project_id))
            self._conn.commit()

    def runs(self, project_id: str, client_id: str, admin: bool = False) -> List[Dict[str, Any]]:
        if self.get(project_id, client_id, admin=admin) is None:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id, project_id, client_id, created_at FROM project_runs "
                "WHERE project_id = ? ORDER BY created_at DESC", (project_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
