"""Small Python SDK for local or HTTP-backed Nomo platform services."""

from __future__ import annotations

import json
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .platform import PlatformStore


class PlatformClient:
    """Client with a local-store mode and a standard-library HTTP mode.

    Pass ``store=PlatformStore(...)`` for embedded use, or ``base_url`` for a
    running :class:`nomo_planner.api.PlatformHTTPServer` service. Pass
    ``token=...`` when the HTTP service is configured with ``NOMO_API_TOKEN``;
    the SDK sends it as a bearer token and never stores it itself.
    """

    def __init__(self, *, store: PlatformStore | None = None, base_url: str | None = None,
                 timeout: float = 10.0, token: str | None = None) -> None:
        if (store is None) == (base_url is None):
            raise ValueError("provide exactly one of store or base_url")
        self.store, self.base_url, self.timeout, self.token = store, base_url.rstrip("/") if base_url else None, timeout, token

    def _request(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> Any:
        if self.base_url is None:
            raise RuntimeError("HTTP request requires base_url")
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(self.base_url + path, data=data, method=method, headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8")) if response.status != 204 else None
        except HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"platform API returned HTTP {exc.code}: {payload}") from exc
        except URLError as exc:
            raise RuntimeError(f"could not reach platform API: {exc.reason}") from exc

    def create_project(self, name: str, *, description: str = "", metadata: Mapping[str, Any] | None = None):
        if self.store:
            return self.store.create_project(name, description=description, metadata=metadata)
        return self._request("POST", "/projects", {"name": name, "description": description, "metadata": metadata or {}})

    def list_projects(self):
        return self.store.list_projects() if self.store else self._request("GET", "/projects")

    def create_artifact(self, project_id: str, name: str, content: Any, *, media_type: str = "application/json",
                        run_id: str | None = None, metadata: Mapping[str, Any] | None = None):
        if self.store:
            return self.store.create_artifact(project_id, name, content, media_type=media_type,
                                              run_id=run_id, metadata=metadata)
        return self._request("POST", f"/projects/{project_id}/artifacts", {
            "name": name, "content": content, "media_type": media_type, "run_id": run_id, "metadata": metadata or {}})

    def get_artifact(self, artifact_id: str):
        return self.store.get_artifact(artifact_id) if self.store else self._request("GET", f"/artifacts/{artifact_id}")

    def list_artifacts(self, project_id: str, *, run_id: str | None = None):
        if self.store:
            return self.store.list_artifacts(project_id, run_id=run_id)
        suffix = f"?run_id={run_id}" if run_id else ""
        return self._request("GET", f"/projects/{project_id}/artifacts{suffix}")

    def inspect_model(self, *, content: Any | None = None, artifact_id: str | None = None):
        """Inspect inline model JSON, a base64 binary envelope, or a stored artifact safely.

        The embedded client dispatches through the same service method as the
        HTTP API; the remote client uses ``POST /artifacts/inspect-model``.
        Binary pickle-backed weights are never deserialized implicitly by the
        inspection service. Binary ONNX/safetensors content uses a JSON
        envelope with ``filename``, ``encoding: "base64"``, and ``base64``.
        """
        params: dict[str, Any] = {}
        if content is not None:
            params["content"] = content
        if artifact_id is not None:
            params["artifact_id"] = artifact_id
        if self.store:
            from .api import PlatformService
            return PlatformService(self.store).dispatch("artifacts.inspect_model", params)
        return self._request("POST", "/artifacts/inspect-model", params)

    def create_run(self, project_id: str, *, name: str = "run", config: Mapping[str, Any] | None = None,
                   result: Mapping[str, Any] | None = None, status: str = "completed",
                   metadata: Mapping[str, Any] | None = None):
        if self.store:
            return self.store.create_run(project_id, name=name, config=config, result=result,
                                         status=status, metadata=metadata)
        return self._request("POST", f"/projects/{project_id}/runs", {
            "name": name, "config": config or {}, "result": result or {}, "status": status, "metadata": metadata or {}})

    def get_run(self, run_id: str):
        return self.store.get_run(run_id) if self.store else self._request("GET", f"/runs/{run_id}")

    def list_runs(self, project_id: str):
        return self.store.list_runs(project_id) if self.store else self._request("GET", f"/projects/{project_id}/runs")

    def update_run(self, run_id: str, *, status: str | None = None, result: Mapping[str, Any] | None = None,
                   metadata: Mapping[str, Any] | None = None):
        if self.store:
            return self.store.update_run(run_id, status=status, result=result, metadata=metadata)
        return self._request("PATCH", f"/runs/{run_id}", {"status": status, "result": result, "metadata": metadata})

    def compare_runs(self, run_id_a: str, run_id_b: str):
        if self.store:
            return self.store.compare_runs(run_id_a, run_id_b)
        return self._request("POST", "/runs/compare", {"run_id_a": run_id_a, "run_id_b": run_id_b})

    def run_simulation(self, project_id: str, *, kind: str = "training", **inputs: Any):
        """Execute a local preview simulation or request one from the HTTP API.

        ``inputs`` accepts inline JSON objects or ``*_artifact_id`` references.
        Training inputs use graph/topology/config; serving uses config and an
        optional trace. The service persists each result as a completed run.
        """
        params = {"project_id": project_id, "kind": kind, **inputs}
        if self.store:
            from .api import PlatformService
            return PlatformService(self.store).dispatch("simulations.run", params)
        return self._request("POST", "/simulations/run", params)

    def render_report(self, run_id: str):
        if self.store:
            return self.store.render_report(run_id)
        return self._request("POST", f"/runs/{run_id}/report", {})
