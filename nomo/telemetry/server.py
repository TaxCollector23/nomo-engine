"""Telemetry server: runs searches in worker threads and streams them over WebSockets.

    POST /runs                      start a run            -> {run_id}
    GET  /runs                      list runs
    GET  /runs/{id}                 status + latest generation summary
    POST /runs/{id}/stop            cooperative stop at the next generation boundary
    GET  /catalog                   available models and hardware profiles
    WS   /ws/runs/{id}?since=<seq>  live envelopes (replay from ring buffer, or snapshot on gap)

Threading model: the optimizer runs in a daemon thread and calls `Run.publish` through
loop.call_soon_threadsafe, so all subscriber state is touched only on the event loop.
Each subscriber has a bounded queue; a slow consumer that overflows it is disconnected
with close code 1013 and resumes via ?since=, which is lossless while the ring buffer covers it.
"""
from __future__ import annotations

import asyncio
import math
import threading
import uuid
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..hardware.profiles import PROFILES, get_profile
from ..models.zoo import MODELS
from ..search.evaluator import Budgets, NeurosymbolicEvaluator
from ..search.genome import Domain, uniform_genome
from ..search.nsga2 import NSGA2Config, TriDomainNSGA2Optimizer
from .schema import envelope

import os

RING = 20000
# Public-deployment guards (env-configurable). A hosted demo endpoint is reachable by anyone,
# so cap concurrent searches and restrict browser origins to the dashboard's domain.
MAX_ACTIVE_RUNS = int(os.environ.get("NOMO_MAX_ACTIVE_RUNS", "2"))
MAX_RUNS_KEPT = int(os.environ.get("NOMO_MAX_RUNS_KEPT", "50"))
CORS_ORIGINS = [o.strip() for o in os.environ.get("NOMO_CORS_ORIGINS", "*").split(",") if o.strip()]
SUB_QUEUE = 2048


class BudgetIn(BaseModel):
    energy_j: Optional[float] = None
    latency_s: Optional[float] = None
    accuracy_min: Optional[float] = None
    accuracy_drop_max: Optional[float] = Field(None, description="alternative to accuracy_min, in pp below baseline")
    period_s: Optional[float] = None
    min_plastic_params: int = 0


class RunIn(BaseModel):
    model: str = "attitude_policy"
    hardware: str = "akd1500"
    budgets: BudgetIn = BudgetIn(accuracy_drop_max=4.0)
    pop_size: int = Field(64, ge=8, le=1024)
    generations: int = Field(60, ge=1, le=5000)
    seed: int = 0


class Run:
    def __init__(self, run_id: str, cfg: RunIn, loop: asyncio.AbstractEventLoop) -> None:
        self.id, self.cfg, self.loop = run_id, cfg, loop
        self.seq = 0
        self.ring: Deque[Dict[str, Any]] = deque(maxlen=RING)
        self.subs: Set[asyncio.Queue] = set()
        self.status = "pending"
        self.started: Optional[dict] = None
        self.last_gen: Optional[dict] = None
        self.items: Dict[str, dict] = {}
        self.optimizer: Optional[TriDomainNSGA2Optimizer] = None
        self.error: Optional[str] = None

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

    def summary(self) -> Dict[str, Any]:
        return {"run_id": self.id, "status": self.status, "seq": self.seq, "config": self.cfg.model_dump(),
                "last_gen": {k: v for k, v in (self.last_gen or {}).items() if k not in ("population",)},
                "error": self.error}


def _budgets(cfg: RunIn, model, hw) -> Budgets:
    b = cfg.budgets
    acc_min = b.accuracy_min if b.accuracy_min is not None else (
        model.base_accuracy - b.accuracy_drop_max if b.accuracy_drop_max is not None else 0.0)
    return Budgets(e_max_j=b.energy_j or math.inf, l_max_s=b.latency_s or math.inf, acc_min=acc_min,
                   period_max_s=b.period_s or math.inf, min_plastic_params=b.min_plastic_params)


def create_app() -> FastAPI:
    app = FastAPI(title="Nomo telemetry", version="0.2.0")
    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
    runs: Dict[str, Run] = {}
    app.state.runs = runs

    def worker(run: Run) -> None:
        try:
            model = MODELS[run.cfg.model]()
            hw = get_profile(run.cfg.hardware)
            ev = NeurosymbolicEvaluator(model, hw, _budgets(run.cfg, model, hw))
            opt = TriDomainNSGA2Optimizer(model, hw, ev, NSGA2Config(pop_size=run.cfg.pop_size,
                                          generations=run.cfg.generations, seed=run.cfg.seed), telemetry=run.emit)
            run.optimizer = opt
            run.set_status("running")
            opt.run()
            run.set_status("stopped" if opt._stop else "completed")
        except Exception as exc:  # surfaced to clients, never swallowed
            run.error = f"{type(exc).__name__}: {exc}"
            run.set_status("failed")
            run.emit("run.failed", {"error": run.error})

    @app.get("/catalog")
    def catalog() -> Dict[str, Any]:
        return {"models": {k: {"layers": [l.name for l in f().layers], "base_accuracy": f().base_accuracy}
                           for k, f in MODELS.items()},
                "hardware": {k: {"name": p.name, "provenance": p.provenance} for k, p in PROFILES.items()}}

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        return {"ok": True, "active_runs": sum(r.status in ("pending", "running") for r in runs.values())}

    @app.post("/runs")
    async def start(cfg: RunIn) -> Dict[str, str]:
        active = sum(r.status in ("pending", "running") for r in runs.values())
        if active >= MAX_ACTIVE_RUNS:
            raise HTTPException(429, f"{active} searches already running (limit {MAX_ACTIVE_RUNS}); try again shortly")
        while len(runs) >= MAX_RUNS_KEPT:                     # evict oldest finished run (memory bound)
            done = [k for k, r in runs.items() if r.status not in ("pending", "running")]
            if not done:
                break
            del runs[done[0]]
        if cfg.model not in MODELS:
            raise HTTPException(404, f"unknown model {cfg.model}")
        if cfg.hardware not in PROFILES:
            raise HTTPException(404, f"unknown hardware {cfg.hardware}")
        run = Run(uuid.uuid4().hex[:12], cfg, asyncio.get_running_loop())
        runs[run.id] = run
        threading.Thread(target=worker, args=(run,), daemon=True, name=f"nomo-run-{run.id}").start()
        return {"run_id": run.id}

    @app.get("/runs")
    def list_runs() -> List[Dict[str, Any]]:
        return [r.summary() for r in runs.values()]

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> Dict[str, Any]:
        if run_id not in runs:
            raise HTTPException(404, "no such run")
        return runs[run_id].summary()

    @app.post("/runs/{run_id}/stop")
    def stop(run_id: str) -> Dict[str, str]:
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(404, "no such run")
        if run.optimizer:
            run.optimizer.stop()
        return {"status": "stopping"}

    @app.websocket("/ws/runs/{run_id}")
    async def ws(websocket: WebSocket, run_id: str, since: int = 0) -> None:
        run = runs.get(run_id)
        if run is None:
            await websocket.close(code=4404)
            return
        await websocket.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=SUB_QUEUE)
        # replay or snapshot, then subscribe; both happen on the loop so no envelope can slip between
        oldest = run.ring[0]["seq"] if run.ring else run.seq + 1
        if since > run.seq:                       # client is ahead of this server (e.g. restart): resync
            backlog = [run.snapshot()]
        elif since and since >= oldest - 1:
            backlog = [e for e in run.ring if e["seq"] > since]
        elif since == 0 and (not run.ring or oldest == 1):
            backlog = list(run.ring)
        else:
            backlog = [run.snapshot()]
        run.subs.add(q)

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
            while True:
                get = asyncio.create_task(q.get())
                done, _ = await asyncio.wait({get, reader}, return_when=asyncio.FIRST_COMPLETED)
                if reader in done:
                    get.cancel()
                    break
                env = get.result()
                await websocket.send_json(env)
                if getattr(q, "put_nowait_overflow", False) and q.empty():
                    await websocket.close(code=1013)
                    break
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            reader.cancel()
            run.subs.discard(q)

    return app


app = create_app()
