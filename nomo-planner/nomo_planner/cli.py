"""Command line interface for local Nomo Planner projects and runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from .api import PlatformHTTPServer, PlatformService, handle_mcp_message
from .platform import PlatformStore


def _json_arg(value: str) -> Any:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return parsed


def _json_value_arg(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nomo-platform", description="Local Nomo project, artifact, and run store")
    parser.add_argument("--store", default=".nomo/platform.sqlite3", help="SQLite store path")
    commands = parser.add_subparsers(dest="command", required=True)

    projects = commands.add_parser("projects", help="manage projects")
    project_actions = projects.add_subparsers(dest="action", required=True)
    create = project_actions.add_parser("create")
    create.add_argument("name"); create.add_argument("--description", default=""); create.add_argument("--metadata", type=_json_arg, default={})
    project_actions.add_parser("list")
    get = project_actions.add_parser("get"); get.add_argument("project_id")

    runs = commands.add_parser("runs", help="manage runs")
    run_actions = runs.add_subparsers(dest="action", required=True)
    create_run = run_actions.add_parser("create"); create_run.add_argument("project_id"); create_run.add_argument("--name", default="run")
    create_run.add_argument("--config", type=_json_arg, default={}); create_run.add_argument("--result", type=_json_arg, default={})
    create_run.add_argument("--status", choices=["queued", "running", "completed", "failed", "cancelled"], default="completed")
    list_runs = run_actions.add_parser("list"); list_runs.add_argument("project_id")
    get_run = run_actions.add_parser("get"); get_run.add_argument("run_id")
    update = run_actions.add_parser("update"); update.add_argument("run_id"); update.add_argument("--status")
    update.add_argument("--result", type=_json_arg); update.add_argument("--metadata", type=_json_arg)
    compare = run_actions.add_parser("compare"); compare.add_argument("run_id_a"); compare.add_argument("run_id_b")

    artifacts = commands.add_parser("artifacts", help="manage artifacts")
    artifact_actions = artifacts.add_subparsers(dest="action", required=True)
    add = artifact_actions.add_parser("add"); add.add_argument("project_id"); add.add_argument("name"); add.add_argument("content", help="JSON value")
    add.add_argument("--media-type", default="application/json"); add.add_argument("--run-id"); add.add_argument("--metadata", type=_json_arg, default={})
    artifact_get = artifact_actions.add_parser("get"); artifact_get.add_argument("artifact_id")
    artifact_list = artifact_actions.add_parser("list"); artifact_list.add_argument("project_id"); artifact_list.add_argument("--run-id")
    inspect_model = artifact_actions.add_parser("inspect-model", help="inspect a JSON model config or safe metadata boundary")
    inspect_model.add_argument("source", help="path to a model artifact")

    report = commands.add_parser("report", help="render run report artifacts")
    report.add_argument("run_id")
    serve = commands.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1"); serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--cors-origin", action="append", default=[], help="allowed browser origin; repeat for multiple origins (or use NOMO_CORS_ORIGINS)")
    commands.add_parser("mcp", help="serve MCP-style JSON-RPC over stdin/stdout")
    simulate = commands.add_parser("simulate", help="run a labeled training or serving preview simulation")
    simulate.add_argument("project_id")
    simulate.add_argument("--kind", choices=["training", "serving"], default="training")
    simulate.add_argument("--graph", type=_json_value_arg, help="training graph/config JSON")
    simulate.add_argument("--topology", type=_json_value_arg)
    simulate.add_argument("--config", type=_json_value_arg, default={})
    simulate.add_argument("--trace", type=_json_value_arg)
    simulate.add_argument("--hardware", type=_json_value_arg)
    simulate.add_argument("--options", type=_json_value_arg)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = PlatformStore(args.store)
    try:
        if args.command == "projects":
            if args.action == "create": value = store.create_project(args.name, description=args.description, metadata=args.metadata)
            elif args.action == "list": value = store.list_projects()
            else: value = store.get_project(args.project_id)
        elif args.command == "runs":
            if args.action == "create": value = store.create_run(args.project_id, name=args.name, config=args.config, result=args.result, status=args.status)
            elif args.action == "list": value = store.list_runs(args.project_id)
            elif args.action == "get": value = store.get_run(args.run_id)
            elif args.action == "compare": value = store.compare_runs(args.run_id_a, args.run_id_b)
            else: value = store.update_run(args.run_id, status=args.status, result=args.result, metadata=args.metadata)
        elif args.command == "artifacts":
            if args.action == "add":
                try: content = json.loads(args.content)
                except json.JSONDecodeError as exc: raise ValueError(f"content is not valid JSON: {exc}") from exc
                value = store.create_artifact(args.project_id, args.name, content, media_type=args.media_type, run_id=args.run_id, metadata=args.metadata)
            elif args.action == "get": value = store.get_artifact(args.artifact_id)
            elif args.action == "inspect-model":
                from .artifact_ingestion import load_model_artifact
                value = load_model_artifact(args.source).as_dict()
            else: value = store.list_artifacts(args.project_id, run_id=args.run_id)
        elif args.command == "report":
            value = store.render_report(args.run_id)
        elif args.command == "simulate":
            service = PlatformService(store)
            value = service.dispatch("simulations.run", {
                "project_id": args.project_id, "kind": args.kind, "graph": args.graph,
                "topology": args.topology, "config": args.config, "trace": args.trace,
                "hardware": args.hardware, "options": args.options,
            })
        elif args.command == "serve":
            configured_origins = args.cors_origin or [origin.strip() for origin in os.environ.get("NOMO_CORS_ORIGINS", "").split(",") if origin.strip()]
            server = PlatformHTTPServer((args.host, args.port), store, cors_origins=configured_origins)
            print(f"Nomo platform API listening at http://{args.host}:{args.port}", file=sys.stderr)
            try: server.serve_forever()
            except KeyboardInterrupt: pass
            finally: server.server_close()
            return 0
        else:
            service = PlatformService(store)
            for line in sys.stdin:
                try:
                    message = json.loads(line)
                    response = handle_mcp_message(service, message)
                    if response is not None:
                        print(json.dumps(response, ensure_ascii=False), flush=True)
                except (json.JSONDecodeError, TypeError) as exc:
                    print(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}}), flush=True)
            return 0
        print(json.dumps(value, indent=2, ensure_ascii=False))
        return 0
    except (ValueError, LookupError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
