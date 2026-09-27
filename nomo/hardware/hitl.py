"""Hardware-in-the-loop benchmark clients.

The client speaks a small JSON protocol and never executes a remote command.
Targets run their own trusted agent, receive an artifact plus metadata, and
return measured latency and energy telemetry.
"""
from __future__ import annotations

import base64
import json
import socket
import struct
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class HITLError(RuntimeError):
    pass


@dataclass(frozen=True)
class HITLTarget:
    id: str
    endpoint: str
    token: Optional[str] = None
    timeout_s: float = 30.0
    protocol: str = "http"


@dataclass(frozen=True)
class HITLMeasurement:
    target_id: str
    latency_ms: float
    energy_uj: Optional[float]
    samples: int
    energy_source: str
    measured_at_utc: float
    raw: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"target_id": self.target_id, "latency_ms": self.latency_ms, "energy_uj": self.energy_uj,
                "samples": self.samples, "energy_source": self.energy_source,
                "measured_at_utc": self.measured_at_utc, "raw": self.raw}


def _measurement(target: HITLTarget, payload: Dict[str, Any]) -> HITLMeasurement:
    try:
        latency = float(payload["latency_ms"])
        energy = None if payload.get("energy_uj") is None else float(payload["energy_uj"])
        samples = int(payload.get("samples", 1))
    except (KeyError, TypeError, ValueError) as exc:
        raise HITLError("HITL response needs numeric latency_ms, optional energy_uj, and samples") from exc
    if latency < 0 or samples <= 0 or (energy is not None and energy < 0):
        raise HITLError("HITL telemetry must be non-negative and samples must be positive")
    return HITLMeasurement(target.id, latency, energy, samples, str(payload.get("energy_source", "unknown")),
                           time.time(), payload)


class HITLClient:
    """HTTP JSON client for a remote Nomo benchmark agent."""

    def __init__(self, target: HITLTarget) -> None:
        if target.protocol not in ("http", "https"):
            raise ValueError("HITLTarget protocol must be http or https")
        self.target = target

    def benchmark(self, artifact: bytes | str | Path, *, metadata: Optional[Dict[str, Any]] = None,
                  repeats: int = 20) -> HITLMeasurement:
        if isinstance(artifact, (str, Path)):
            blob = Path(artifact).read_bytes()
            artifact_name = Path(artifact).name
        else:
            blob = bytes(artifact)
            artifact_name = "nomo_artifact.bin"
        if len(blob) > 50 * 1024 * 1024:
            raise HITLError("HITL artifact exceeds the 50 MB client limit")
        body = {"protocol": "nomo.hitl/1", "request_id": uuid.uuid4().hex,
                "artifact_name": artifact_name, "artifact_b64": base64.b64encode(blob).decode("ascii"),
                "metadata": metadata or {}, "repeats": max(1, min(int(repeats), 1000))}
        request = urllib.request.Request(self.target.endpoint.rstrip("/") + "/v1/benchmark",
                                         data=json.dumps(body).encode("utf-8"), method="POST",
                                         headers={"Content-Type": "application/json", "X-Nomo-Protocol": "1"})
        if self.target.token:
            request.add_header("Authorization", f"Bearer {self.target.token}")
        try:
            with urllib.request.urlopen(request, timeout=self.target.timeout_s) as response:
                if response.status >= 300:
                    raise HITLError(f"HITL agent returned HTTP {response.status}")
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise HITLError(f"HITL request failed: {exc}") from exc
        return _measurement(self.target, payload)


class SocketBenchmarkClient:
    """Length-prefixed JSON protocol for LAN/serial bridge agents."""

    def benchmark(self, host: str, port: int, payload: Dict[str, Any], *, timeout_s: float = 30.0) -> Dict[str, Any]:
        data = json.dumps({"protocol": "nomo.hitl/1", **payload}).encode("utf-8")
        if len(data) > 10 * 1024 * 1024:
            raise HITLError("HITL socket payload exceeds the 10 MB limit")
        with socket.create_connection((host, int(port)), timeout=timeout_s) as sock:
            sock.sendall(struct.pack("!I", len(data)) + data)
            header = sock.recv(4)
            if len(header) != 4:
                raise HITLError("HITL agent closed before returning a response")
            size = struct.unpack("!I", header)[0]
            if size > 10 * 1024 * 1024:
                raise HITLError("HITL response exceeds the 10 MB limit")
            chunks, remaining = [], size
            while remaining:
                chunk = sock.recv(min(65536, remaining))
                if not chunk:
                    raise HITLError("HITL agent returned a truncated response")
                chunks.append(chunk)
                remaining -= len(chunk)
        try:
            return json.loads(b"".join(chunks).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HITLError("HITL agent response is not JSON") from exc


def measure_local(fn: Callable[[], Any], *, repeats: int = 20, warmup: int = 3,
                  energy_reader: Optional[Callable[[], float]] = None,
                  target_id: str = "local") -> HITLMeasurement:
    for _ in range(max(0, warmup)):
        fn()
    start_energy = energy_reader() if energy_reader else None
    t0 = time.perf_counter_ns()
    for _ in range(max(1, repeats)):
        fn()
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6 / max(1, repeats)
    end_energy = energy_reader() if energy_reader else None
    energy = None if start_energy is None or end_energy is None else max(0.0, end_energy - start_energy) / max(1, repeats)
    return HITLMeasurement(target_id, elapsed_ms, energy, max(1, repeats),
                           "energy_reader" if energy is not None else "unavailable", time.time(), {})
