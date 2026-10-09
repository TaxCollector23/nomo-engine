"""Dependency-free HTTP and MCP-equivalent service interfaces."""

from __future__ import annotations

import json
import hmac
import os
from dataclasses import asdict, fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qs, unquote, urlsplit

from .platform import NotFoundError, PlatformStore


PLATFORM_API_VERSION = "1.1.0"


class PlatformService:
    """Shared service operations used by HTTP and MCP-style callers."""

    def __init__(self, store: PlatformStore) -> None:
        self.store = store

    def dispatch(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        args = dict(params or {})
        if method == "projects.create":
            return self.store.create_project(args["name"], description=args.get("description", ""), metadata=args.get("metadata"))
        if method == "projects.list":
            return self.store.list_projects()
        if method == "projects.get":
            return self.store.get_project(args["project_id"])
        if method == "artifacts.create":
            return self.store.create_artifact(args["project_id"], args["name"], args.get("content"),
                media_type=args.get("media_type", "application/json"), run_id=args.get("run_id"), metadata=args.get("metadata"))
        if method == "artifacts.get":
            return self.store.get_artifact(args["artifact_id"])
        if method == "artifacts.list":
            return self.store.list_artifacts(args["project_id"], run_id=args.get("run_id"))
        if method == "artifacts.inspect_model":
            from .artifact_ingestion import load_model_artifact

            artifact_id = args.get("artifact_id")
            if artifact_id:
                artifact = self.store.get_artifact(str(artifact_id))
                result = load_model_artifact(artifact["content"])
                return {
                    "artifact": result.as_dict(),
                    "stored_artifact": {"id": artifact["id"], "sha256": artifact["sha256"]},
                }
            if "content" not in args:
                raise ValueError("content or artifact_id is required")
            return load_model_artifact(args["content"]).as_dict()
        if method == "runs.create":
            return self.store.create_run(args["project_id"], name=args.get("name", "run"), config=args.get("config"),
                result=args.get("result"), status=args.get("status", "completed"), metadata=args.get("metadata"))
        if method == "runs.get":
            return self.store.get_run(args["run_id"])
        if method == "runs.list":
            return self.store.list_runs(args["project_id"])
        if method == "runs.update":
            return self.store.update_run(args["run_id"], status=args.get("status"), result=args.get("result"), metadata=args.get("metadata"))
        if method == "runs.compare":
            return self.store.compare_runs(args["run_id_a"], args["run_id_b"])
        if method == "simulations.run":
            return self._run_simulation(args)
        if method == "reports.render":
            return self.store.render_report(args["run_id"])
        raise ValueError(f"unknown platform method: {method}")

    def _run_simulation(self, args: Mapping[str, Any]) -> dict[str, Any]:
        kind = str(args.get("kind", args.get("simulation_type", "training"))).lower()
        if kind not in {"training", "serving"}:
            raise ValueError("kind must be training or serving")
        project_id = args.get("project_id")
        if not project_id:
            raise ValueError("project_id is required to create a simulation run")
        self.store.get_project(project_id)
        inputs: dict[str, Any] = {}
        provenance_inputs: dict[str, Any] = {}
        for name in ("graph", "topology", "config", "trace"):
            value, provenance = _resolve_simulation_input(self.store, args, name)
            inputs[name] = value
            if provenance is not None:
                provenance_inputs[name] = provenance
        config = inputs["config"] or {}
        if not isinstance(config, Mapping):
            raise ValueError("config must resolve to a JSON object")

        if kind == "training":
            from .simcore import (
                Link, Parallelism, RooflineHardware, Topology, TrainingOptions,
                build_operator_graph, simulate_training_step,
            )

            graph_config = inputs["graph"]
            if not isinstance(graph_config, Mapping):
                raise ValueError("training simulations require graph as a JSON object")
            if not isinstance(inputs["topology"] or {}, Mapping):
                raise ValueError("topology must resolve to a JSON object")
            topology_config = dict(inputs["topology"] or {})
            if "topology" in topology_config and isinstance(topology_config["topology"], Mapping):
                topology_config = {**topology_config["topology"], **{k: v for k, v in topology_config.items() if k != "topology"}}
            topology_hardware = topology_config.pop("hardware", {})
            topology_value = _make_topology(topology_config, Topology, Link)
            settings = dict(config)
            nested_options = settings.pop("options", {})
            if not isinstance(nested_options, Mapping):
                raise ValueError("config options must be a JSON object")
            settings = {**nested_options, **settings}
            option_overrides = args.get("options") or {}
            if not isinstance(option_overrides, Mapping):
                raise ValueError("options must be a JSON object")
            settings.update(option_overrides)
            hardware_values = args.get("hardware") or {}
            if not isinstance(hardware_values, Mapping):
                raise ValueError("hardware must be a JSON object")
            hardware_config = dict(settings.pop("hardware", {}) or {})
            hardware_config.update(hardware_values)
            if not hardware_config:
                hardware_config.update(topology_hardware or {})
            if "topology" in hardware_config:
                hardware_config.pop("topology")
            hardware = RooflineHardware(**{
                "peak_flops": hardware_config.get("peak_flops", 1e15),
                "memory_bandwidth": hardware_config.get("memory_bandwidth", 1e12),
                "compute_efficiency": hardware_config.get("compute_efficiency", 0.55),
                "memory_efficiency": hardware_config.get("memory_efficiency", 0.70),
                "name": hardware_config.get("name", "modeled hardware defaults"),
            })
            build_keys = {"sequence_length", "batch_size", "precision", "source"}
            graph_options = {key: settings.pop(key) for key in list(settings) if key in build_keys}
            graph_options.setdefault("precision", "bf16")
            graph = build_operator_graph(graph_config, **graph_options)
            option_fields = {field.name for field in fields(TrainingOptions)}
            unknown = set(settings) - option_fields
            if unknown:
                raise ValueError(f"unknown training config keys: {', '.join(sorted(unknown))}")
            options_value = dict(settings)
            if "parallelism" in options_value:
                parallelism = options_value["parallelism"]
                if isinstance(parallelism, Mapping):
                    options_value["parallelism"] = Parallelism(**parallelism)
                elif not isinstance(parallelism, Parallelism):
                    raise ValueError("parallelism must be an object with data/tensor/pipeline values")
            options_value.setdefault("precision", graph.precision)
            options = TrainingOptions(**options_value)
            simulation = simulate_training_step(graph, hardware, topology_value, options)
            raw = asdict(simulation)
            metrics = {"step_time_s": simulation.step_time_s,
                       "peak_memory_bytes": max(simulation.peak_memory_by_gpu, default=0.0),
                       "event_count": len(simulation.events)}
            timeline = raw["events"]
            assumptions = list(simulation.assumptions)
            sim_provenance = list(simulation.provenance)
            simulator = "nomo_planner.simcore.simulate_training_step"
        else:
            from .serving_sim import simulate_serving

            assumptions = dict(config)
            trace = inputs["trace"]
            if trace is not None and not isinstance(trace, (list, tuple)):
                raise ValueError("serving trace must be a JSON array of request rows")
            simulation = simulate_serving(trace, assumptions)
            raw = simulation.as_dict()
            metrics = {key: item["value"] for key, item in raw["metrics"].items()
                       if isinstance(item, Mapping) and isinstance(item.get("value"), (int, float))}
            timeline = raw["timeline"]
            assumptions = list(simulation.notes)
            sim_provenance = ["serving_sim uses deterministic simulated quantities and explicit assumptions"]
            simulator = "nomo_planner.serving_sim.simulate_serving"

        preview = {
            "label": "Preview", "simulated": True, "kind": kind, "simulator": simulator,
            "metrics": metrics, "metric_labels": {key: "Preview" for key in metrics},
            "timeline": timeline, "assumptions": assumptions,
            "provenance": {"label": "Preview", "classification": "simulated analytical estimate",
                           "implementation": simulator, "simulator_provenance": sim_provenance,
                           "inputs": provenance_inputs},
            "raw": raw,
        }
        run = self.store.create_run(
            project_id, name=str(args.get("name", f"{kind}-preview")), status="completed",
            config={"simulation_type": kind, "inputs": provenance_inputs}, result=preview,
            metadata={"label": "Preview", "operation": "simulations.run"})
        return {"run": run, "simulation": preview}


def _resolve_simulation_input(store: PlatformStore, args: Mapping[str, Any], name: str) -> tuple[Any, dict[str, Any] | None]:
    value = args.get(name)
    artifact_spec = args.get(f"{name}_artifact_id", args.get(f"{name}_artifact"))
    artifacts = args.get("artifacts")
    if artifact_spec is None and isinstance(artifacts, Mapping):
        artifact_spec = artifacts.get(name)
    if artifact_spec is None and isinstance(value, Mapping) and set(value).issubset({"artifact_id", "id"}) and (value.get("artifact_id") or value.get("id")):
        artifact_spec = value.get("artifact_id", value.get("id"))
        value = None
    if isinstance(artifact_spec, Mapping):
        artifact_spec = artifact_spec.get("artifact_id", artifact_spec.get("id"))
    if artifact_spec is not None:
        artifact = store.get_artifact(str(artifact_spec))
        payload = artifact["content"]
        if isinstance(payload, Mapping):
            payload = payload.get("payload", payload.get("data", payload))
        return payload, {"source": "stored-artifact", "artifact_id": artifact["id"], "sha256": artifact["sha256"]}
    if isinstance(value, Mapping) and ("payload" in value or "data" in value) and len(value) <= 5:
        value = value.get("payload", value.get("data"))
    if value is None:
        return None, None
    return value, {"source": "inline-json"}


def _make_topology(values: Mapping[str, Any], topology_type: Any, link_type: Any) -> Any:
    topology_fields = {field.name for field in fields(topology_type)}
    args: dict[str, Any] = {}
    for name in ("intra_node", "inter_node"):
        link = values.get(name)
        if link is not None:
            if not isinstance(link, Mapping):
                raise ValueError(f"topology {name} must be an object")
            args[name] = link_type(**link)
    for key, value in values.items():
        if key in topology_fields and key not in {"intra_node", "inter_node"}:
            args[key] = value
    args.setdefault("devices", 1)
    return topology_type(**args)


_TOOLS = [
    ("projects.create", "Create a project", {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}, "description": {"type": "string"}, "metadata": {"type": "object"}}}),
    ("projects.list", "List projects", {"type": "object", "properties": {}}),
    ("projects.get", "Get a project", {"type": "object", "required": ["project_id"], "properties": {"project_id": {"type": "string"}}}),
    ("artifacts.create", "Create a project artifact", {"type": "object", "required": ["project_id", "name", "content"], "properties": {"project_id": {"type": "string"}, "name": {"type": "string"}, "content": {}, "media_type": {"type": "string"}, "run_id": {"type": "string"}}}),
    ("artifacts.get", "Get an artifact", {"type": "object", "required": ["artifact_id"], "properties": {"artifact_id": {"type": "string"}}}),
    ("artifacts.list", "List project artifacts", {"type": "object", "required": ["project_id"], "properties": {"project_id": {"type": "string"}, "run_id": {"type": "string"}}}),
    ("artifacts.inspect_model", "Inspect a JSON model config, base64 binary model envelope, or stored artifact without unsafe deserialization", {"type": "object", "properties": {"artifact_id": {"type": "string"}, "content": {}}}),
    ("runs.create", "Create a run", {"type": "object", "required": ["project_id"], "properties": {"project_id": {"type": "string"}, "name": {"type": "string"}, "config": {"type": "object"}, "result": {"type": "object"}, "status": {"type": "string"}}}),
    ("runs.get", "Get a run", {"type": "object", "required": ["run_id"], "properties": {"run_id": {"type": "string"}}}),
    ("runs.list", "List project runs", {"type": "object", "required": ["project_id"], "properties": {"project_id": {"type": "string"}}}),
    ("runs.update", "Update run status or results", {"type": "object", "required": ["run_id"], "properties": {"run_id": {"type": "string"}, "status": {"type": "string"}, "result": {"type": "object"}, "metadata": {"type": "object"}}}),
    ("runs.compare", "Compare metrics and timeline event deltas between two runs", {"type": "object", "required": ["run_id_a", "run_id_b"], "properties": {"run_id_a": {"type": "string"}, "run_id_b": {"type": "string"}}}),
    ("simulations.run", "Run a labeled preview training or serving simulation from JSON inputs or stored artifacts", {"type": "object", "required": ["project_id"], "properties": {"project_id": {"type": "string"}, "kind": {"type": "string", "enum": ["training", "serving"]}, "graph": {"type": "object"}, "graph_artifact_id": {"type": "string"}, "topology": {"type": "object"}, "topology_artifact_id": {"type": "string"}, "config": {"type": "object"}, "config_artifact_id": {"type": "string"}, "trace": {"type": "array"}, "trace_artifact_id": {"type": "string"}}}),
    ("reports.render", "Create JSON and HTML report artifacts for a run", {"type": "object", "required": ["run_id"], "properties": {"run_id": {"type": "string"}}}),
]


def handle_mcp_message(service: PlatformService, message: Mapping[str, Any]) -> dict[str, Any] | None:
    """Handle the useful JSON-RPC 2.0 MCP lifecycle/tool subset."""
    request_id = message.get("id")
    method = message.get("method")
    if method == "notifications/initialized":
        return None
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": request_id, "result": {
            "protocolVersion": str(message.get("params", {}).get("protocolVersion", "2024-11-05")),
            "capabilities": {"tools": {}}, "serverInfo": {"name": "nomo-planner", "version": "1.0.0"}}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "tools/list":
        tools = [{"name": name, "description": description, "inputSchema": schema}
                 for name, description, schema in _TOOLS]
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": tools}}
    if method != "tools/call":
        return _rpc_error(request_id, -32601, f"method not found: {method}")
    params = message.get("params") or {}
    try:
        value = service.dispatch(params["name"], params.get("arguments", {}))
        return {"jsonrpc": "2.0", "id": request_id, "result": {
            "content": [{"type": "text", "text": json.dumps(value, sort_keys=True, ensure_ascii=False)}], "isError": False}}
    except (KeyError, ValueError, NotFoundError) as exc:
        return {"jsonrpc": "2.0", "id": request_id, "result": {
            "content": [{"type": "text", "text": str(exc)}], "isError": True}}


def _rpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


class PlatformHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], store: PlatformStore, *, token: str | None = None,
                 cors_origins: Iterable[str] | None = None):
        self.service = PlatformService(store)
        service = self.service
        auth_token = token or os.environ.get("NOMO_API_TOKEN")
        allowed_origins = frozenset(origin.strip() for origin in (cors_origins or ()) if origin.strip())

        class Handler(BaseHTTPRequestHandler):
            server_version = f"NomoPlatform/{PLATFORM_API_VERSION}"

            def _respond(self, status: int, value: Any, content_type: str = "application/json; charset=utf-8") -> None:
                body = value.encode("utf-8") if isinstance(value, str) else json.dumps(value, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                origin = self.headers.get("Origin")
                if origin and ("*" in allowed_origins or origin in allowed_origins):
                    self.send_header("Access-Control-Allow-Origin", "*" if "*" in allowed_origins else origin)
                    self.send_header("Vary", "Origin")
                self.end_headers()
                self.wfile.write(body)

            def _authorized(self) -> bool:
                if not auth_token:
                    return True
                request_path = urlsplit(self.path).path
                if self.command == "GET" and request_path in {"/health", "/healthz"}:
                    return True
                supplied = self.headers.get("Authorization", "")
                scheme, separator, value = supplied.partition(" ")
                if separator != " " or scheme.lower() != "bearer" or not hmac.compare_digest(value, auth_token):
                    self.send_response(401)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("WWW-Authenticate", 'Bearer realm="nomo-platform"')
                    self.end_headers()
                    return False
                return True

            def _preflight(self) -> None:
                origin = self.headers.get("Origin")
                if allowed_origins and origin not in allowed_origins and "*" not in allowed_origins:
                    self._respond(403, {"error": "origin is not allowed"})
                    return
                self.send_response(204)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
                self.send_header("Access-Control-Max-Age", "600")
                if origin and ("*" in allowed_origins or origin in allowed_origins):
                    self.send_header("Access-Control-Allow-Origin", "*" if "*" in allowed_origins else origin)
                    self.send_header("Vary", "Origin")
                self.end_headers()

            def _body(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 10_000_000:
                    raise ValueError("request body exceeds 10 MB")
                raw = self.rfile.read(length) if length else b"{}"
                value = json.loads(raw.decode("utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("request body must be a JSON object")
                return value

            def _route(self, method: str) -> None:
                path = [unquote(part) for part in urlsplit(self.path).path.strip("/").split("/") if part]
                body = self._body() if method in {"POST", "PATCH"} else {}
                if path in (["health"], ["healthz"]) and method == "GET":
                    self._respond(200, {"status": "ok", "service": "nomo-platform", "version": PLATFORM_API_VERSION}); return
                if path == ["capabilities"] and method == "GET":
                    self._respond(200, {"service": "nomo-platform", "version": PLATFORM_API_VERSION,
                                        "operations": ["projects", "runs", "artifacts", "simulations", "reports"],
                                        "authentication": "bearer" if auth_token else "none"}); return
                if path == ["projects"] and method == "GET":
                    self._respond(200, service.dispatch("projects.list")); return
                if path == ["projects"] and method == "POST":
                    self._respond(201, service.dispatch("projects.create", body)); return
                if len(path) == 2 and path[0] == "projects" and method == "GET":
                    self._respond(200, service.dispatch("projects.get", {"project_id": path[1]})); return
                if len(path) == 3 and path[0] == "projects" and path[2] == "runs":
                    if method == "GET":
                        self._respond(200, service.dispatch("runs.list", {"project_id": path[1]})); return
                    if method == "POST":
                        self._respond(201, service.dispatch("runs.create", {**body, "project_id": path[1]})); return
                if len(path) == 3 and path[0] == "projects" and path[2] == "artifacts" and method == "GET":
                    query = parse_qs(urlsplit(self.path).query)
                    args = {"project_id": path[1]}
                    if query.get("run_id"):
                        args["run_id"] = query["run_id"][0]
                    self._respond(200, service.dispatch("artifacts.list", args)); return
                if len(path) == 3 and path[0] == "projects" and path[2] == "artifacts" and method == "POST":
                    self._respond(201, service.dispatch("artifacts.create", {**body, "project_id": path[1]})); return
                if path == ["artifacts", "inspect-model"] and method == "POST":
                    self._respond(200, service.dispatch("artifacts.inspect_model", body)); return
                if len(path) == 2 and path[0] == "artifacts" and method == "GET":
                    self._respond(200, service.dispatch("artifacts.get", {"artifact_id": path[1]})); return
                if len(path) == 2 and path[0] == "runs":
                    if method == "GET":
                        self._respond(200, service.dispatch("runs.get", {"run_id": path[1]})); return
                    if method == "PATCH":
                        self._respond(200, service.dispatch("runs.update", {**body, "run_id": path[1]})); return
                if path == ["runs", "compare"] and method == "POST":
                    self._respond(200, service.dispatch("runs.compare", body)); return
                if path == ["simulations", "run"] and method == "POST":
                    self._respond(201, service.dispatch("simulations.run", body)); return
                if len(path) == 3 and path[0] == "runs" and path[2] == "report" and method == "POST":
                    self._respond(201, service.dispatch("reports.render", {"run_id": path[1]})); return
                self._respond(404, {"error": "route not found"})

            def _safe_route(self, method: str) -> None:
                try:
                    if method == "OPTIONS":
                        self._preflight(); return
                    if not self._authorized():
                        return
                    self._route(method)
                except NotFoundError as exc:
                    self._respond(404, {"error": str(exc)})
                except (KeyError, ValueError, json.JSONDecodeError) as exc:
                    self._respond(400, {"error": str(exc)})
                except Exception:
                    self._respond(500, {"error": "internal server error"})

            def do_GET(self) -> None: self._safe_route("GET")
            def do_POST(self) -> None: self._safe_route("POST")
            def do_PATCH(self) -> None: self._safe_route("PATCH")
            def do_OPTIONS(self) -> None: self._safe_route("OPTIONS")

            def log_message(self, format: str, *args: Any) -> None:
                return

        super().__init__(address, Handler)
