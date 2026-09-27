"""Nomo backend: runs searches in worker threads, streams them over WebSockets, and logs everything.

Public API
    GET  /                          service info
    GET  /healthz                   liveness (used by the host's health check)
    GET  /catalog                   models and hardware profiles
    POST /runs                      start a search                       -> {run_id}
    GET  /runs                      list runs
    GET  /runs/{id}                 status + latest generation summary
    POST /runs/{id}/stop            cooperative stop at the next generation boundary
    GET  /presets                   one-click optimisation presets
    GET  /models/{id}               layer table of a built-in or uploaded model
    POST /models/upload?filename=&base_accuracy=&input_shape=   raw file body (.onnx/.pt/.pth/.json)
    GET  /runs/{id}/designs/{key}   one design: layers, metrics, summary, export capabilities
    POST /runs/{id}/export          {key, formats[]} -> zip download
    POST /runs/{id}/copilot         {question, key?} -> grounded answer + one-click actions
    WS   /ws/runs/{id}?since=<seq>&client=<id>   live envelopes (replay, or snapshot on gap)

Admin API (header `Authorization: Bearer $NOMO_ADMIN_TOKEN`; disabled when the token is unset)
    GET  /admin/stats               uptime, runs, users, sockets, request and error counters
    GET  /admin/logs                ?stream=all|backend|access|users|runs|telemetry|errors
                                    &limit=&level=&since_ts=&contains=
    GET  /admin/logs/download       same filters, NDJSON attachment
    GET  /admin/users               anonymous client registry
    GET  /admin/runs                every run with config, status, progress and requesting client

Clients identify themselves with header `X-Nomo-Client` (HTTP) or `?client=` (WebSocket); the
dashboard generates a random id per browser. See nomo/telemetry/observability.py for log schema.

Threading model: the optimizer runs in a daemon thread and calls `Run.publish` through
loop.call_soon_threadsafe, so subscriber state is only touched on the event loop. A slow consumer
that overflows its bounded queue is closed with code 1013 and resumes losslessly via ?since=.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import math
import os
import platform
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Deque, Dict, List, Optional, Set

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from .. import __version__
from ..hardware.profiles import PROFILES, get_profile
from ..hardware.custom import HardwareOverrides, apply_overrides
from ..modes import get_mode, mode_catalog, validate_mode
from ..models.zoo import MODELS, synthetic_weights
from ..search.evaluator import Budgets, NeurosymbolicEvaluator
from ..search.policy import intrinsic_domains, parse_policy, validate_policy
from ..search.presets import PRESETS
from ..search.nsga2 import NSGA2Config, TriDomainNSGA2Optimizer
from . import observability as obs
from .schema import envelope

try:                                            # Unix only; the admin stats degrade gracefully on Windows
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]

RING = 20000
SUB_QUEUE = 2048


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# Public-deployment guards. A hosted endpoint is reachable by anyone.
MAX_ACTIVE_RUNS = _env_int("NOMO_MAX_ACTIVE_RUNS", 2)
MAX_RUNS_KEPT = _env_int("NOMO_MAX_RUNS_KEPT", 50)
MAX_POP = _env_int("NOMO_MAX_POP", 128)
MAX_GENS = _env_int("NOMO_MAX_GENS", 150)
MAX_RUNS_PER_CLIENT_HOUR = _env_int("NOMO_MAX_RUNS_PER_CLIENT_HOUR", 30)
CORS_ORIGINS = [o.strip() for o in os.environ.get("NOMO_CORS_ORIGINS", "*").split(",") if o.strip()]
ADMIN_TOKEN = os.environ.get("NOMO_ADMIN_TOKEN", "")
MAX_UPLOADS = _env_int("NOMO_MAX_UPLOADS", 20)
MAX_UPLOAD_MB = _env_int("NOMO_MAX_UPLOAD_MB", 50)
LOG_HEALTH = os.environ.get("NOMO_LOG_HEALTH", "0") == "1"
GEN_LOG_EVERY = max(1, _env_int("NOMO_LOG_GEN_EVERY", 5))


class BudgetIn(BaseModel):
    energy_j: Optional[float] = None
    latency_s: Optional[float] = None
    accuracy_min: Optional[float] = None
    accuracy_drop_max: Optional[float] = Field(None, description="alternative to accuracy_min, in pp below baseline")
    period_s: Optional[float] = None
    min_plastic_params: int = 0


class SearchIn(BaseModel):
    allow_continuous: bool = True
    allow_spiking: bool = True
    allow_symbolic: bool = True
    codings: List[str] = ["rate", "ttfs"]
    crossing_penalty: float = Field(0.0, ge=0.0, le=10.0)
    crossing_min_saving_pct: float = Field(0.0, ge=0.0, le=95.0)
    p_crossover: float = Field(0.9, ge=0.0, le=1.0)
    p_mutation: float = Field(1.0, ge=0.0, le=1.0)
    archive_capacity: int = Field(0, ge=0, le=1000)
    patience: int = Field(12, ge=2, le=1000, description="generations without hypervolume gain before stopping")
    asf_weights: Optional[List[float]] = Field(None, description="recommendation weights (energy, latency, accuracy)")


class PinIn(BaseModel):
    domain: Optional[str] = None                   # ANN | SNN | SYM
    w_bits: Optional[Any] = None                   # int or INT16/INT8/INT4/INT2/BINARY
    a_bits: Optional[Any] = None


class HardwareIn(BaseModel):
    mac_energy_pj: Optional[float] = None
    sop_energy_pj: Optional[float] = None
    neuron_energy_pj: Optional[float] = None
    sram_kb_per_core: Optional[float] = None
    neurons_per_core: Optional[int] = None
    n_cores: Optional[int] = None
    bus_bandwidth_gbs: Optional[float] = None
    routing_latency_us: Optional[float] = None
    timestep_us: Optional[float] = None
    static_power_mw: Optional[float] = None
    clock_mhz: Optional[float] = None


class RunIn(BaseModel):
    model: str = "attitude_policy"                 # built-in name or an uploaded model id ("upload:...")
    hardware: str = "akd1500"
    budgets: BudgetIn = BudgetIn(accuracy_drop_max=4.0)
    pop_size: int = Field(64, ge=8, le=1024)
    generations: int = Field(60, ge=1, le=5000)
    seed: int = 0
    search: SearchIn = SearchIn()
    pins: Dict[str, PinIn] = {}
    lock_symbolic: bool = False
    hardware_overrides: Optional[HardwareIn] = None
    preset: Optional[str] = None                   # informational: which preset the UI applied
    mode: Optional[str] = None                     # low_power_neuromorphic | hard_realtime | radiation_hardened | on_chip_learning


class ExportIn(BaseModel):
    key: str
    formats: List[str]
    archive: str = Field("zip", pattern="^(zip|tar\\.gz)$")


class WorkbenchTargetsIn(BaseModel):
    energy: float = Field(1.0, gt=0)
    latency: float = Field(1.0, gt=0)
    accuracy: float = Field(1.0, gt=0)


class CopilotIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    key: Optional[str] = None


def _client_ip(req_headers, client) -> str:
    fwd = req_headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()        # first hop set by the host's proxy
    return client.host if client else ""


class Run:
    def __init__(self, run_id: str, cfg: RunIn, loop: asyncio.AbstractEventLoop, client_id: str) -> None:
        self.id, self.cfg, self.loop, self.client_id = run_id, cfg, loop, client_id
        self.seq = 0
        self.ring: Deque[Dict[str, Any]] = deque(maxlen=RING)
        self.subs: Set[asyncio.Queue] = set()
        self.status = "pending"
        self.started: Optional[dict] = None
        self.last_gen: Optional[dict] = None
        self.items: Dict[str, dict] = {}
        self.optimizer: Optional[TriDomainNSGA2Optimizer] = None
        self.error: Optional[str] = None
        self.created_at = time.time()
        self.finished_at: Optional[float] = None
        self.envelopes_sent = 0
        self.log = obs.get("runs")
        self.ctx = None                              # export.bundle.RunContext, set by the worker
        self.warnings: List[str] = []
        self.prepared: Optional[tuple] = None        # (model, weights, weights_source, hw, assumptions, policy, calibration)

    # called on the event loop
    def publish(self, type_: str, data: dict) -> None:
        self.seq += 1
        env = envelope(self.id, self.seq, type_, data)
        self.ring.append(env)
        if type_ == "run.started":
            self.started = data
        elif type_ == "eval.batch":
            for it in data["items"]:
                self.items[it["key"]] = it
        elif type_ == "gen.completed":
            self.last_gen = data
            if data["gen"] % GEN_LOG_EVERY == 0 or data["gen"] == 1:
                self.log.info("run.progress", extra={
                    "run_id": self.id, "gen": data["gen"], "hv": round(data["hv"], 5),
                    "front": len(data["front"]), "unique": data["unique"],
                    "feasible_fraction": round(data["feasible_fraction"], 3)})
        dead = []
        for q in self.subs:
            try:
                q.put_nowait(env)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self.subs.discard(q)
            q.put_nowait_overflow = True  # type: ignore[attr-defined]

    # called from the worker thread
    def emit(self, type_: str, data: dict) -> None:
        try:
            self.loop.call_soon_threadsafe(self.publish, type_, data)
        except RuntimeError:                   # event loop gone (server shutdown): stop searching
            if self.optimizer is not None:
                self.optimizer.stop()

    def set_status(self, status: str) -> None:
        try:
            self.loop.call_soon_threadsafe(setattr, self, "status", status)
        except RuntimeError:
            self.status = status

    def snapshot(self) -> Dict[str, Any]:
        return envelope(self.id, self.seq, "snapshot", {
            "status": self.status, "run": self.started, "items": list(self.items.values()), "last_gen": self.last_gen})

    def summary(self, admin: bool = False) -> Dict[str, Any]:
        d = {"run_id": self.id, "status": self.status, "seq": self.seq, "config": self.cfg.model_dump(),
             "last_gen": {k: v for k, v in (self.last_gen or {}).items() if k not in ("population", "front")},
             "error": self.error, "created_at": self.created_at, "finished_at": self.finished_at,
             "warnings": self.warnings}
        if admin:
            d.update({"client_id": self.client_id, "subscribers": len(self.subs),
                      "candidates": len(self.items), "envelopes_sent": self.envelopes_sent})
        return d


def _budgets(cfg: RunIn, model, hw) -> Budgets:
    b = cfg.budgets
    acc_min = b.accuracy_min if b.accuracy_min is not None else (
        model.base_accuracy - b.accuracy_drop_max if b.accuracy_drop_max is not None else 0.0)
    return Budgets(e_max_j=b.energy_j or math.inf, l_max_s=b.latency_s or math.inf, acc_min=acc_min,
                   period_max_s=b.period_s or math.inf, min_plastic_params=b.min_plastic_params,
                   crossing_penalty=cfg.search.crossing_penalty,
                   crossing_min_saving_pct=cfg.search.crossing_min_saving_pct)


def layer_table(model) -> List[Dict[str, Any]]:
    """Per-layer facts for the UI: shapes, size, compute, and which domains the layer can take."""
    out = []
    for spec in model.layers:
        out.append({"name": spec.name, "op": spec.op, "activation": spec.activation, "params": spec.params,
                    "macs": spec.macs, "fan_in": spec.fan_in, "fan_out": spec.out_neurons,
                    "weight_shape": list(spec.weight_shape or ()), "weight_kb_int8": round(spec.params / 1024, 1),
                    "can_be": [d.name for d in intrinsic_domains(spec)],
                    "in_shape": list(spec.attrs.get("in_shape", ())) or None,
                    "out_shape": list(spec.attrs.get("pooled_shape", spec.attrs.get("out_shape", ()))) or None,
                    "preserve_spatial": bool(spec.attrs.get("preserve_spatial", spec.op == "conv2d")),
                    "quantization_sensitivity": {"weight": spec.sensitivity.q_w, "activation": spec.sensitivity.q_a,
                                                   "rate": spec.sensitivity.c_rate, "ttfs": spec.sensitivity.c_ttfs}})
    return out


def _hw_defaults(p) -> Dict[str, float]:
    """Current values of every editable hardware parameter, in the units the UI uses."""
    return {"mac_energy_pj": p.ann_cost(8, 8)[0] * 1e12, "sop_energy_pj": p.sop_energy(4) * 1e12,
            "neuron_energy_pj": p.snn_e_neuron * 1e12, "sram_kb_per_core": p.syn_mem_bits_per_core / 8192,
            "neurons_per_core": p.neurons_per_core, "n_cores": p.n_cores,
            "bus_bandwidth_gbs": (p.links[0].bw_bytes_per_s / 1e9) if p.links else 0.0,
            "routing_latency_us": (p.links[0].lat_s * 1e6) if p.links else 0.0,
            "timestep_us": p.snn_t_step_min * 1e6, "static_power_mw": p.p_static_w * 1e3, "clock_mhz": p.clock_hz / 1e6}


def create_app() -> FastAPI:
    ring = obs.setup_logging()
    blog, alog, tlog, rlog = obs.get("backend"), obs.get("access"), obs.get("telemetry"), obs.get("runs")
    users = obs.UserRegistry()
    counters: Dict[str, int] = {"requests": 0, "errors_5xx": 0, "errors_4xx": 0, "ws_sessions": 0,
                                "ws_active": 0, "runs_created": 0, "runs_rejected": 0}
    per_client_runs: Dict[str, Deque[float]] = {}

    runs: Dict[str, Run] = {}
    uploads: Dict[str, Dict[str, Any]] = {}
    calibrations: Dict[str, Dict[str, Any]] = {}
    export_lock = threading.Lock()                 # exports are memory-heavy: one at a time

    def prepare(cfg: RunIn):
        """Resolve model/weights/hardware and apply the policy. Raises HTTPException(4xx) with reasons."""
        assumptions: List[str] = []
        try:
            mode = get_mode(cfg.mode)
            mode_warnings = validate_mode(cfg.mode, period_s=cfg.budgets.period_s,
                                          min_plastic_params=cfg.budgets.min_plastic_params,
                                          allow_spiking=cfg.search.allow_spiking)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        if cfg.model in MODELS:
            model = MODELS[cfg.model]()
            weights, wsrc = synthetic_weights(model), "synthetic (untrained demo weights for the built-in model)"
        elif cfg.model in uploads:
            up = uploads[cfg.model]
            model, weights, wsrc = up["build"](), up["weights"], up["weights_source"]
            assumptions += up["report"]["assumptions"]
            assumptions.append("uploaded model: per-layer accuracy sensitivities use Nomo's default estimates, "
                               "not measurements of your network")
        else:
            raise HTTPException(404, f"unknown model '{cfg.model}' (uploaded models are kept for a limited time; "
                                     "upload it again)")
        calibration = calibrations.get(cfg.model)
        if calibration:
            assumptions.append(f"uploaded calibration tensors: {calibration['report']['sample_count']} samples; PTQ ranges are data-driven")
        elif mode is not None and mode.id == "hard_realtime":
            assumptions.append("hard real-time mode has no hardware-in-the-loop measurement attached")
        if cfg.hardware not in PROFILES:
            raise HTTPException(404, f"unknown hardware {cfg.hardware}")
        hw = get_profile(cfg.hardware)
        if cfg.hardware_overrides is not None:
            try:
                hw = apply_overrides(hw, HardwareOverrides(**cfg.hardware_overrides.model_dump()))
            except (ValueError, TypeError) as exc:
                raise HTTPException(422, str(exc))
        pins = {k: v.model_dump() for k, v in cfg.pins.items()}
        if cfg.lock_symbolic:
            for spec in model.layers:
                if spec.symbolic_substitute and spec.name not in pins:
                    pins[spec.name] = {"domain": "SYM"}
        srch = cfg.search
        try:
            policy = parse_policy(model, {"continuous": srch.allow_continuous, "spiking": srch.allow_spiking,
                                          "symbolic": srch.allow_symbolic}, srch.codings, pins)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        rep = validate_policy(model, hw, policy)
        if not rep.ok:
            raise HTTPException(422, " ".join(rep.errors))
        model.policy = policy
        if srch.asf_weights is not None and (len(srch.asf_weights) != 3 or any(w <= 0 for w in srch.asf_weights)):
            raise HTTPException(422, "asf_weights needs three positive numbers (energy, latency, accuracy)")
        return model, weights, wsrc, hw, assumptions, rep.warnings + mode_warnings, policy, calibration

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        active = [r.id for r in runs.values() if r.status in ("pending", "running")]
        for r in runs.values():
            if r.optimizer is not None:
                r.optimizer.stop()
        blog.info("backend.stop", extra={"uptime_s": round(time.time() - obs.STARTED_AT, 1), "interrupted_runs": active})

    app = FastAPI(title="Nomo backend", version=__version__, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"],
                       expose_headers=["X-Request-ID", "Content-Disposition", "X-Nomo-Manifest"])
    app.state.runs, app.state.users, app.state.counters = runs, users, counters

    blog.info("backend.start", extra={
        "version": __version__, "python": platform.python_version(), "pid": os.getpid(),
        "config": {"max_active_runs": MAX_ACTIVE_RUNS, "max_runs_kept": MAX_RUNS_KEPT, "max_pop": MAX_POP,
                   "max_gens": MAX_GENS, "max_runs_per_client_hour": MAX_RUNS_PER_CLIENT_HOUR,
                   "cors_origins": CORS_ORIGINS, "admin_enabled": bool(ADMIN_TOKEN),
                   "log_dir": str(obs.file_dir()) if obs.file_dir() else None, "log_ring": obs.RING_SIZE}})
    if not ADMIN_TOKEN:
        blog.warning("admin.disabled", extra={"hint": "set NOMO_ADMIN_TOKEN to enable /admin endpoints"})
    if CORS_ORIGINS == ["*"]:
        blog.warning("cors.open", extra={"hint": "set NOMO_CORS_ORIGINS to your dashboard origin"})

    # ------------------------------------------------------------------ access log middleware
    @app.middleware("http")
    async def access_log(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        t0 = time.perf_counter()
        ip_hash = obs.hash_ip(_client_ip(request.headers, request.client))
        client_id = users.resolve_id(request.headers.get("x-nomo-client"), ip_hash)
        request.state.client_id, request.state.request_id, request.state.ip_hash = client_id, rid, ip_hash
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            alog.exception("http.unhandled", extra={"request_id": rid, "path": request.url.path})
            response = JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500)
        response.headers["X-Request-ID"] = rid
        path = request.url.path
        counters["requests"] += 1
        if status >= 500:
            counters["errors_5xx"] += 1
        elif status >= 400:
            counters["errors_4xx"] += 1
        quiet = (path == "/healthz" and not LOG_HEALTH) or request.method == "OPTIONS"
        if not quiet:
            users.touch(client_id, ip_hash, request.headers.get("user-agent", ""), request.headers.get("origin", ""))
            level = 40 if status >= 500 else 30 if status >= 400 else 20
            alog.log(level, "http.request", extra={
                "request_id": rid, "method": request.method, "path": path,
                "query": str(request.url.query)[:300], "status": status,
                "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
                "client_id": client_id, "ip_hash": ip_hash, "origin": request.headers.get("origin", "")})
        return response

    # ------------------------------------------------------------------ search worker
    def worker(run: Run) -> None:
        t0 = time.perf_counter()
        try:
            from ..export.bundle import RunContext
            model, weights, wsrc, hw, assumptions, policy, calibration = run.prepared
            s_ = run.cfg.search
            ev = NeurosymbolicEvaluator(model, hw, _budgets(run.cfg, model, hw))
            opt = TriDomainNSGA2Optimizer(model, hw, ev, NSGA2Config(
                pop_size=run.cfg.pop_size, generations=run.cfg.generations, seed=run.cfg.seed,
                p_crossover=s_.p_crossover, p_mutation=s_.p_mutation, archive_capacity=s_.archive_capacity,
                hv_window=s_.patience, asf_weights=tuple(s_.asf_weights) if s_.asf_weights else None), telemetry=run.emit)
            settings = run.cfg.model_dump()
            settings["policy"] = policy.to_dict([l.name for l in model.layers])
            calibration_data = calibration["data"] if calibration else None
            run.ctx = RunContext(model, weights, wsrc, hw, ev, settings, assumptions + run.warnings,
                                 opt.archive_front, opt.recommend, calibration_data=calibration_data,
                                 calibration_report=calibration["report"] if calibration else None)
            run.optimizer = opt
            run.set_status("running")
            rlog.info("run.started", extra={"run_id": run.id, "client_id": run.client_id})
            res = opt.run()
            final = "stopped" if opt._stop else "completed"
            run.finished_at = time.time()
            run.set_status(final)
            rec = res.recommended
            rlog.info(f"run.{final}", extra={
                "run_id": run.id, "client_id": run.client_id, "wall_s": round(time.perf_counter() - t0, 3),
                "generations": res.generations_run, "evaluations": res.evaluations, "unique": res.unique,
                "front": len(res.front), "hv": round(res.hv_history[-1], 5) if res.hv_history else 0.0,
                "recommended": rec.key if rec else None,
                "recommended_f": [float(rec.F[0]), float(rec.F[1]), float(rec.accuracy)] if rec else None})
        except Exception as exc:  # surfaced to clients and to the errors stream, never swallowed
            run.error = f"{type(exc).__name__}: {exc}"
            run.finished_at = time.time()
            run.set_status("failed")
            run.emit("run.failed", {"error": run.error})
            rlog.exception("run.failed", extra={"run_id": run.id, "client_id": run.client_id, "error": run.error})

    # ------------------------------------------------------------------ public routes
    @app.get("/")
    def root() -> Dict[str, Any]:
        return {"service": "nomo-backend", "version": __version__, "status": "ok",
                "uptime_s": round(time.time() - obs.STARTED_AT, 1),
                "endpoints": ["/healthz", "/catalog", "/presets", "/models/upload", "/models/{id}/calibration", "/runs",
                              "/ws/runs/{run_id}", "/runs/{id}/designs/{key}", "/runs/{id}/workbench",
                              "/runs/{id}/export", "/runs/{id}/copilot", "/docs"],
                "limits": {"max_active_runs": MAX_ACTIVE_RUNS, "max_pop": MAX_POP, "max_generations": MAX_GENS}}

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        return {"ok": True, "version": __version__,
                "active_runs": sum(r.status in ("pending", "running") for r in runs.values())}

    @app.get("/catalog")
    def catalog() -> Dict[str, Any]:
        return {"models": {k: {"layers": [l.name for l in f().layers], "base_accuracy": f().base_accuracy,
                               "layer_table": layer_table(f())} for k, f in MODELS.items()},
                "hardware": {k: {"name": p.name, "provenance": p.provenance, "defaults": _hw_defaults(p),
                                 "bits": {"continuous": list(p.ann_bits), "spiking": list(p.snn_w_bits)}}
                             for k, p in PROFILES.items()},
                "modes": mode_catalog(),
                "workbench": {"format": "nomo.workbench/1", "levels": 6},
                "limits": {"max_pop": MAX_POP, "max_generations": MAX_GENS}}

    def _reject(status: int, reason: str, client_id: str, **extra) -> HTTPException:
        counters["runs_rejected"] += 1
        rlog.warning("run.rejected", extra={"client_id": client_id, "reason": reason, "status": status, **extra})
        return HTTPException(status, reason)

    @app.post("/runs")
    async def start(cfg: RunIn, request: Request) -> Dict[str, Any]:
        cid = request.state.client_id
        try:
            model, weights, wsrc, hw, assumptions, warnings, policy, calibration = prepare(cfg)
        except HTTPException as exc:
            raise _reject(exc.status_code, exc.detail, cid)
        if cfg.pop_size > MAX_POP or cfg.generations > MAX_GENS:
            raise _reject(422, f"this server allows population <= {MAX_POP} and generations <= {MAX_GENS}", cid,
                          pop_size=cfg.pop_size, generations=cfg.generations)
        active = sum(r.status in ("pending", "running") for r in runs.values())
        if active >= MAX_ACTIVE_RUNS:
            raise _reject(429, f"{active} searches already running (limit {MAX_ACTIVE_RUNS}); try again shortly", cid)
        hist = per_client_runs.setdefault(cid, deque())
        now = time.time()
        while hist and now - hist[0] > 3600:
            hist.popleft()
        if len(hist) >= MAX_RUNS_PER_CLIENT_HOUR:
            raise _reject(429, f"hourly limit of {MAX_RUNS_PER_CLIENT_HOUR} searches reached for this client", cid)
        while len(runs) >= MAX_RUNS_KEPT:                     # evict oldest finished run (memory bound)
            done = [k for k, r in runs.items() if r.status not in ("pending", "running")]
            if not done:
                break
            rlog.info("run.evicted", extra={"run_id": done[0]})
            del runs[done[0]]
        run = Run(uuid.uuid4().hex[:12], cfg, asyncio.get_running_loop(), cid)
        run.prepared = (model, weights, wsrc, hw, assumptions, policy, calibration)
        run.warnings = warnings
        runs[run.id] = run
        hist.append(now)
        counters["runs_created"] += 1
        users.touch(cid, request.state.ip_hash, kind="run", run_id=run.id)
        rlog.info("run.created", extra={"run_id": run.id, "client_id": cid, "request_id": request.state.request_id,
                                        "config": cfg.model_dump()})
        threading.Thread(target=worker, args=(run,), daemon=True, name=f"nomo-run-{run.id}").start()
        return {"run_id": run.id, "warnings": warnings}

    @app.get("/runs")
    def list_runs() -> List[Dict[str, Any]]:
        return [r.summary() for r in runs.values()]

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> Dict[str, Any]:
        if run_id not in runs:
            raise HTTPException(404, "no such run")
        return runs[run_id].summary()

    @app.post("/runs/{run_id}/stop")
    def stop(run_id: str, request: Request) -> Dict[str, str]:
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(404, "no such run")
        if run.optimizer:
            run.optimizer.stop()
        rlog.info("run.stop_requested", extra={"run_id": run_id, "client_id": request.state.client_id,
                                               "owner": run.client_id})
        return {"status": "stopping"}

    # ------------------------------------------------------------------ v4: presets, uploads, designs, export, copilot
    @app.get("/presets")
    def presets() -> Dict[str, Any]:
        return PRESETS

    @app.post("/models/upload")
    async def upload_model(request: Request, filename: str = Query(..., max_length=200),
                           base_accuracy: float = Query(90.0, gt=0, le=100),
                           input_shape: Optional[str] = Query(None, description="e.g. 3,32,32")) -> Dict[str, Any]:
        from ..ingest import IngestError, ingest
        cid = request.state.client_id
        data = await request.body()
        if not data:
            raise HTTPException(400, "empty upload")
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"file is larger than this server's {MAX_UPLOAD_MB} MB limit")
        shape = None
        if input_shape:
            try:
                shape = tuple(int(v) for v in input_shape.replace("x", ",").split(",") if v.strip())
            except ValueError:
                raise HTTPException(422, "input_shape must look like 3,32,32")
        try:
            model, weights, rep = await asyncio.to_thread(ingest, filename, data, base_accuracy, shape)
        except IngestError as exc:
            rlog.warning("model.upload_rejected", extra={"client_id": cid, "upload_name": filename, "reason": str(exc)})
            raise HTTPException(422, str(exc))
        mid = "upload:" + uuid.uuid4().hex[:10]
        while len(uploads) >= MAX_UPLOADS:
            uploads.pop(next(iter(uploads)))
        spec_layers, name, acc, ishape = model.layers, model.name, model.base_accuracy, model.input_shape
        architecture_family, model_metadata = model.architecture_family, dict(model.metadata)

        def build(spec_layers=spec_layers, name=name, acc=acc, ishape=ishape,
                  architecture_family=architecture_family, model_metadata=model_metadata):
            from ..ir import ModelGraph
            return ModelGraph(name=name, input_shape=ishape, layers=list(spec_layers), base_accuracy=acc,
                              architecture_family=architecture_family, metadata=dict(model_metadata))

        uploads[mid] = {"build": build, "weights": weights, "report": rep.to_dict(), "client_id": cid,
                        "weights_source": {"file": "uploaded file",
                                           "partial": "uploaded file (some layers had no weights: random weights generated)",
                                           }.get(rep.weights_source, "synthetic (the uploaded graph had no weights)"),
                        "created_at": time.time(), "architecture_family": architecture_family}
        rlog.info("model.uploaded", extra={"client_id": cid, "model_id": mid, "upload_name": filename,
                                           "bytes": len(data), "layers": rep.layers, "format": rep.source_format})
        return {"model_id": mid, "name": name, "base_accuracy": acc, "input_shape": list(ishape),
                "report": rep.to_dict(), "layer_table": layer_table(model), "calibration": None}

    @app.post("/models/{model_id}/calibration")
    async def upload_calibration(model_id: str, request: Request,
                                 filename: str = Query("calibration.npz", max_length=200)) -> Dict[str, Any]:
        """Attach 100–500 calibration tensors used by PTQ and export validation."""
        from ..runtime.ptq import parse_calibration_bytes
        if model_id in MODELS:
            model = MODELS[model_id]()
        elif model_id in uploads:
            model = uploads[model_id]["build"]()
        else:
            raise HTTPException(404, "unknown model")
        data = await request.body()
        if not data:
            raise HTTPException(400, "empty calibration upload")
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"calibration is larger than this server's {MAX_UPLOAD_MB} MB limit")
        n_aux = max([model.constraints[s.constraint_id].n_aux for s in model.guard_sites] or [0])
        try:
            cal = await asyncio.to_thread(parse_calibration_bytes, filename, data, model.input_shape, n_aux)
        except ValueError as exc:
            rlog.warning("calibration.upload_rejected", extra={"client_id": request.state.client_id,
                                                                   "model_id": model_id, "reason": str(exc)})
            raise HTTPException(422, str(exc))
        report = cal.to_dict()
        calibrations[model_id] = {"data": (cal.inputs, cal.aux), "report": report,
                                   "client_id": request.state.client_id, "created_at": time.time()}
        rlog.info("calibration.uploaded", extra={"client_id": request.state.client_id, "model_id": model_id,
                                                   "upload_name": filename, "samples": cal.sample_count})
        return {"model_id": model_id, "calibration": report,
                "message": "Calibration attached. PTQ ranges will be selected for each design at export time."}

    @app.get("/models/{model_id}")
    def get_model(model_id: str) -> Dict[str, Any]:
        if model_id in MODELS:
            m = MODELS[model_id]()
            return {"model_id": model_id, "name": m.name, "base_accuracy": m.base_accuracy,
                    "input_shape": list(m.input_shape), "layer_table": layer_table(m), "report": None,
                    "calibration": calibrations.get(model_id, {}).get("report")}
        if model_id in uploads:
            m = uploads[model_id]["build"]()
            return {"model_id": model_id, "name": m.name, "base_accuracy": m.base_accuracy,
                    "input_shape": list(m.input_shape), "layer_table": layer_table(m), "report": uploads[model_id]["report"],
                    "calibration": calibrations.get(model_id, {}).get("report")}
        raise HTTPException(404, "unknown model")

    def _ctx_and_eval(run_id: str, key: Optional[str]):
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(404, "no such run")
        if run.status in ("pending", "running"):
            raise HTTPException(409, "the search is still running; wait for it to finish or press Stop")
        if run.ctx is None:
            raise HTTPException(409, "this run has no results")
        ctx = run.ctx
        if key is None:
            ev = ctx.recommend_fn()
            if ev is None:
                raise HTTPException(404, "no design met the budgets")
        else:
            ev = ctx.evaluator.cache.get(key)
            if ev is None:
                raise HTTPException(404, "unknown design key for this run")
        return run, ctx, ev

    @app.get("/runs/{run_id}/designs/{key:path}")
    def design(run_id: str, key: str) -> Dict[str, Any]:
        from ..copilot import summarize
        from ..export.bundle import capabilities, design_json
        run, ctx, ev = _ctx_and_eval(run_id, None if key == "recommended" else key)
        return {"design": design_json(ctx, ev), "summary": summarize(ctx, ev).to_dict(),
                "capabilities": capabilities(ctx, ev)}

    @app.get("/workbench/schema")
    def workbench_schema() -> Dict[str, Any]:
        from ..workbench import schema
        return schema()

    @app.get("/runs/{run_id}/workbench")
    def workbench(run_id: str, key: str = "recommended", energy_weight: float = Query(1.0, gt=0),
                  latency_weight: float = Query(1.0, gt=0), accuracy_weight: float = Query(1.0, gt=0)) -> Dict[str, Any]:
        from ..workbench import build_state
        run, ctx, ev = _ctx_and_eval(run_id, None if key == "recommended" else key)
        ctx.ptq_report_for(ev)
        return build_state(ctx.model, ev, plan=ctx.plan(ev), ptq_report=ctx.calibration_report,
                           hitl=ctx.hitl_measurement,
                           target_weights=(energy_weight, latency_weight, accuracy_weight),
                           mode=ctx.settings.get("mode"), hardware=ctx.hw)

    @app.post("/runs/{run_id}/workbench/targets")
    def workbench_targets(run_id: str, body: WorkbenchTargetsIn) -> Dict[str, Any]:
        from ..workbench import build_state, select_candidate
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(404, "no such run")
        if run.ctx is None or run.status in ("pending", "running"):
            raise HTTPException(409, "the search is still running; wait for it to finish")
        ctx = run.ctx
        ev = select_candidate((e for e in ctx.evaluator.cache.values() if e.feasible),
                              (body.energy, body.latency, body.accuracy))
        if ev is None:
            raise HTTPException(404, "no feasible design matches the requested target weights")
        ctx.ptq_report_for(ev)
        state = build_state(ctx.model, ev, plan=ctx.plan(ev), ptq_report=ctx.calibration_report,
                            hitl=ctx.hitl_measurement,
                            target_weights=(body.energy, body.latency, body.accuracy),
                            mode=ctx.settings.get("mode"), hardware=ctx.hw)
        state["selected_design_key"] = ev.key
        return state

    @app.post("/runs/{run_id}/export")
    def export(run_id: str, body: ExportIn, request: Request) -> Response:
        from ..export.bundle import build_archive
        run, ctx, ev = _ctx_and_eval(run_id, body.key)
        t0 = time.perf_counter()
        if not export_lock.acquire(timeout=120):
            raise HTTPException(503, "the server is busy preparing another export; try again in a minute")
        try:
            data, manifest = build_archive(ctx, ev, body.formats, body.archive)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        finally:
            ctx._plans.clear()                     # plans hold full weight copies: free them between exports
            import gc
            gc.collect()
            export_lock.release()
        rlog.info("run.exported", extra={"run_id": run_id, "client_id": request.state.client_id, "key": ev.key,
                                         "formats": body.formats, "bytes": len(data), "skipped": manifest["skipped"],
                                         "errors": manifest["errors"], "wall_s": round(time.perf_counter() - t0, 3)})
        is_tar = manifest.get("format") == "tar.gz"
        return Response(data, media_type="application/gzip" if is_tar else "application/zip", headers={
            "Content-Disposition": f'attachment; filename="{manifest["root"]}.tar.gz"' if is_tar else f'attachment; filename="{manifest["root"]}.zip"',
            "X-Nomo-Manifest": json.dumps({"files": manifest["files"], "skipped": manifest["skipped"]})[:7000],
            "Access-Control-Expose-Headers": "Content-Disposition, X-Nomo-Manifest"})

    @app.post("/runs/{run_id}/copilot")
    def copilot(run_id: str, body: CopilotIn, request: Request) -> Dict[str, Any]:
        from ..copilot import answer
        run, ctx, ev = _ctx_and_eval(run_id, body.key)
        ans = answer(ctx, ev, ctx.front_fn(), ctx.recommend_fn(), body.question)
        rlog.info("copilot.answer", extra={"run_id": run_id, "client_id": request.state.client_id,
                                           "question": body.question[:200], "source": ans.source,
                                           "actions": [a["type"] for a in ans.actions]})
        return ans.to_dict()

    # ------------------------------------------------------------------ websocket
    @app.websocket("/ws/runs/{run_id}")
    async def ws(websocket: WebSocket, run_id: str, since: int = 0, client: str = "") -> None:
        ip_hash = obs.hash_ip(_client_ip(websocket.headers, websocket.client))
        cid = users.resolve_id(client, ip_hash)
        run = runs.get(run_id)
        if run is None:
            tlog.warning("ws.not_found", extra={"run_id": run_id, "client_id": cid})
            await websocket.close(code=4404)
            return
        await websocket.accept()
        users.touch(cid, ip_hash, websocket.headers.get("user-agent", ""), websocket.headers.get("origin", ""), kind="ws")
        q: asyncio.Queue = asyncio.Queue(maxsize=SUB_QUEUE)
        oldest = run.ring[0]["seq"] if run.ring else run.seq + 1
        if since > run.seq:                       # client is ahead of this server (e.g. restart): resync
            backlog, mode = [run.snapshot()], "snapshot"
        elif since and since >= oldest - 1:
            backlog, mode = [e for e in run.ring if e["seq"] > since], "replay"
        elif since == 0 and (not run.ring or oldest == 1):
            backlog, mode = list(run.ring), "full"
        else:
            backlog, mode = [run.snapshot()], "snapshot"
        run.subs.add(q)
        sid = uuid.uuid4().hex[:10]
        t0 = time.perf_counter()
        sent, reason = 0, "client_closed"
        counters["ws_sessions"] += 1
        counters["ws_active"] += 1
        tlog.info("ws.open", extra={"session": sid, "run_id": run_id, "client_id": cid, "ip_hash": ip_hash,
                                    "since": since, "mode": mode, "backlog": len(backlog), "run_seq": run.seq,
                                    "run_status": run.status})

        async def pump_in() -> None:
            try:
                while True:
                    msg = await websocket.receive_json()
                    if msg.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
            except Exception:
                return                                 # disconnect or malformed frame ends the session

        reader = asyncio.create_task(pump_in())
        try:
            for env in backlog:
                await websocket.send_json(env)
                sent += 1
            while True:
                get = asyncio.create_task(q.get())
                done, _ = await asyncio.wait({get, reader}, return_when=asyncio.FIRST_COMPLETED)
                if reader in done:
                    get.cancel()
                    break
                env = get.result()
                await websocket.send_json(env)
                sent += 1
                if getattr(q, "put_nowait_overflow", False) and q.empty():
                    reason = "overflow"
                    tlog.warning("ws.overflow", extra={"session": sid, "run_id": run_id, "client_id": cid})
                    await websocket.close(code=1013)
                    break
        except (WebSocketDisconnect, RuntimeError):
            reason = "disconnect"
        except Exception:
            reason = "error"
            tlog.exception("ws.error", extra={"session": sid, "run_id": run_id, "client_id": cid})
        finally:
            reader.cancel()
            run.subs.discard(q)
            run.envelopes_sent += sent
            counters["ws_active"] -= 1
            tlog.info("ws.close", extra={"session": sid, "run_id": run_id, "client_id": cid, "reason": reason,
                                         "sent": sent, "duration_s": round(time.perf_counter() - t0, 2),
                                         "last_seq_sent": run.seq if sent else since})

    # ------------------------------------------------------------------ admin
    def require_admin(request: Request) -> None:
        if not ADMIN_TOKEN:
            raise HTTPException(403, "admin API disabled: set NOMO_ADMIN_TOKEN on the server")
        auth = request.headers.get("authorization", "")
        token = auth[7:] if auth.lower().startswith("bearer ") else ""
        if not hmac.compare_digest(token.encode(), ADMIN_TOKEN.encode()):
            blog.warning("admin.denied", extra={"client_id": request.state.client_id, "path": request.url.path,
                                                "ip_hash": request.state.ip_hash})
            raise HTTPException(401, "invalid admin token")

    @app.get("/admin/stats", dependencies=[Depends(require_admin)])
    def admin_stats() -> Dict[str, Any]:
        by_status: Dict[str, int] = {}
        for r in runs.values():
            by_status[r.status] = by_status.get(r.status, 0) + 1
        ru = resource.getrusage(resource.RUSAGE_SELF) if resource else None
        return {"version": __version__, "uptime_s": round(time.time() - obs.STARTED_AT, 1),
                "started_at": obs.STARTED_AT, "pid": os.getpid(),
                "max_rss_mb": round(ru.ru_maxrss / 1024, 1) if ru else None,
                "cpu_user_s": round(ru.ru_utime, 2) if ru else None,
                "runs": {"kept": len(runs), "by_status": by_status}, "counters": dict(counters),
                "users": {"known": len(users.users), "active_5m": users.active(300), "active_1h": users.active(3600)},
                "log_records": dict(ring.counts), "log_dir": str(obs.file_dir()) if obs.file_dir() else None,
                "limits": {"max_active_runs": MAX_ACTIVE_RUNS, "max_runs_kept": MAX_RUNS_KEPT, "max_pop": MAX_POP,
                           "max_gens": MAX_GENS, "max_runs_per_client_hour": MAX_RUNS_PER_CLIENT_HOUR}}

    def _logs(stream: str, limit: int, level: Optional[str], since_ts: Optional[float],
              contains: Optional[str]) -> List[Dict[str, Any]]:
        if stream != "all" and stream not in obs.STREAMS:
            raise HTTPException(422, f"stream must be 'all' or one of {list(obs.STREAMS)}")
        return ring.query(stream, limit, level, since_ts, contains)

    @app.get("/admin/logs", dependencies=[Depends(require_admin)])
    def admin_logs(stream: str = "all", limit: int = Query(200, ge=1, le=5000), level: Optional[str] = None,
                   since_ts: Optional[float] = None, contains: Optional[str] = None) -> Dict[str, Any]:
        rows = _logs(stream, limit, level, since_ts, contains)
        return {"stream": stream, "count": len(rows), "records": rows}

    @app.get("/admin/logs/download", dependencies=[Depends(require_admin)])
    def admin_logs_download(stream: str = "all", limit: int = Query(5000, ge=1, le=50000),
                            level: Optional[str] = None, since_ts: Optional[float] = None,
                            contains: Optional[str] = None) -> PlainTextResponse:
        rows = _logs(stream, limit, level, since_ts, contains)
        body = "\n".join(json.dumps(r, default=str) for r in rows) + ("\n" if rows else "")
        fname = f"nomo-{stream}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.ndjson"
        return PlainTextResponse(body, media_type="application/x-ndjson",
                                 headers={"Content-Disposition": f'attachment; filename="{fname}"'})

    @app.get("/admin/users", dependencies=[Depends(require_admin)])
    def admin_users() -> Dict[str, Any]:
        rows = users.snapshot()
        return {"count": len(rows), "users": rows}

    @app.get("/admin/runs", dependencies=[Depends(require_admin)])
    def admin_runs() -> Dict[str, Any]:
        rows = sorted((r.summary(admin=True) for r in runs.values()), key=lambda d: -d["created_at"])
        return {"count": len(rows), "runs": rows}

    return app


app = create_app()
