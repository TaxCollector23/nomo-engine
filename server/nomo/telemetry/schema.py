"""Telemetry wire protocol v1 (SPEC §9). Mirrored in frontend/src/lib/telemetry/protocol.ts.

Envelope (server -> client), one JSON object per WebSocket text frame:
    {"v": 1, "run_id": str, "seq": int, "ts": float, "type": str, "data": {...}}

seq is dense and strictly increasing per run, starting at 1. A client that reconnects with
?since=<last seq> receives every envelope with seq > since from the ring buffer, or a single
"snapshot" envelope (seq = current head) when `since` has fallen out of the buffer.

Event types:
    run.started     {model, hardware, layers[], guard_sites[], pop_size, generations, objectives[]}
    eval.batch      {gen, items: EvalItem[]}                    new unique candidates this generation
    gen.completed   {gen, hv, evaluations, unique, feasible_fraction, front: key[], population: key[],
                     recommended: EvalItem | null, operators: {name: prob}}
    run.completed   {recommended, front: EvalItem[], generations, evaluations, unique, hv, wall_s}
    run.failed      {error}
    snapshot        {status, run: run.started.data, items: EvalItem[], last_gen: gen.completed.data | null}

EvalItem:
    {key, f: [energy_j, latency_s, accuracy_pct], cv, feasible, rank, acc_src,
     genome: {layers: [[domain, w_bits, a_bits, coding, T, plastic], ...], guards: [[site, impl], ...]},
     crossings, cores, parent?, changed?: int[], ops?: [crossover|null, mutation]}

Client -> server frames: {"type": "ping"} -> {"type": "pong"} (not sequenced).
"""
from __future__ import annotations

import time
from typing import Any, Dict

PROTOCOL_VERSION = 1
EVENT_TYPES = ("run.started", "eval.batch", "gen.completed", "run.completed", "run.failed", "snapshot")


def envelope(run_id: str, seq: int, type_: str, data: Dict[str, Any]) -> Dict[str, Any]:
    if type_ not in EVENT_TYPES:
        raise ValueError(f"unknown event type {type_}")
    return {"v": PROTOCOL_VERSION, "run_id": run_id, "seq": seq, "ts": time.time(), "type": type_, "data": data}
