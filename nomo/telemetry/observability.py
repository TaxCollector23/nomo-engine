"""Observability for the hosted backend.

Six log streams, every record a single JSON object:

    backend    process lifecycle, configuration, startup/shutdown
    access     one record per HTTP request (method, path, status, duration, client, request id)
    users      anonymous clients: first sighting, sessions, runs started
    runs       search lifecycle: created, started, per-generation progress, completed/failed/stopped
    telemetry  WebSocket sessions: open (replay/snapshot/full), close, overflow, envelopes sent
    errors     every record at level >= ERROR from any stream, with traceback

Each record goes to three sinks:
    1. stdout, one JSON line per record. Render (and any container host) captures it,
       so logs are visible in the host's log viewer and survive our process restarts.
    2. an in-memory ring per stream (NOMO_LOG_RING records), served by the admin API
       (GET /admin/logs), because free hosts have no persistent disk.
    3. optional rotating files in NOMO_LOG_DIR (<stream>.log, 5 MB x 3), when writable.

Privacy: client IPs are stored as salted SHA-256 prefixes unless NOMO_LOG_RAW_IP=1. The
salt is NOMO_LOG_SALT or a per-process random value (hashes then do not link across restarts).
Users are anonymous: a client id the dashboard generates and keeps in localStorage.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

STREAMS = ("backend", "access", "users", "runs", "telemetry", "errors")
RING_SIZE = int(os.environ.get("NOMO_LOG_RING", "5000"))
LOG_DIR = os.environ.get("NOMO_LOG_DIR", "logs")
LOG_LEVEL = os.environ.get("NOMO_LOG_LEVEL", "INFO").upper()
RAW_IP = os.environ.get("NOMO_LOG_RAW_IP", "0") == "1"
_SALT = os.environ.get("NOMO_LOG_SALT") or secrets.token_hex(16)
STARTED_AT = time.time()

_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}


def hash_ip(ip: Optional[str]) -> str:
    if not ip:
        return "unknown"
    if RAW_IP:
        return ip
    return hashlib.sha256(f"{_SALT}:{ip}".encode()).hexdigest()[:12]


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(record_to_dict(record), default=str, separators=(",", ":"))


def record_to_dict(record: logging.LogRecord) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "ts": round(record.created, 3),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
        "level": record.levelname,
        "stream": record.name.split(".", 1)[-1] if record.name.startswith("nomo.") else record.name,
        "event": record.getMessage(),
    }
    for k, v in record.__dict__.items():
        if k not in _RESERVED and not k.startswith("_"):
            d[k] = v
    if record.exc_info:
        d["exc"] = "".join(traceback.format_exception(*record.exc_info))[-4000:]
    return d


class RingHandler(logging.Handler):
    """Keeps the most recent records per stream for the admin API."""

    def __init__(self) -> None:
        super().__init__()
        self.rings: Dict[str, Deque[Dict[str, Any]]] = {s: deque(maxlen=RING_SIZE) for s in STREAMS}
        self.counts: Dict[str, int] = {s: 0 for s in STREAMS}
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            d = record_to_dict(record)
            stream = d["stream"] if d["stream"] in self.rings else "backend"
            with self._lock:
                self.rings[stream].append(d)
                self.counts[stream] += 1
                if record.levelno >= logging.ERROR and stream != "errors":
                    self.rings["errors"].append(d)
                    self.counts["errors"] += 1
        except Exception:  # a logging failure must never take down a request
            self.handleError(record)

    def query(self, stream: str, limit: int = 200, level: Optional[str] = None,
              since_ts: Optional[float] = None, contains: Optional[str] = None) -> List[Dict[str, Any]]:
        # "all" merges every primary stream; the errors ring only holds copies, so it is excluded there
        streams = tuple(x for x in STREAMS if x != "errors") if stream == "all" else (stream,)
        lvl = logging.getLevelName(level.upper()) if level else None
        with self._lock:
            rows = [r for s in streams for r in self.rings.get(s, ())]
        if lvl is not None and isinstance(lvl, int):
            rows = [r for r in rows if logging.getLevelName(r["level"]) >= lvl]
        if since_ts is not None:
            rows = [r for r in rows if r["ts"] > since_ts]
        if contains:
            c = contains.lower()
            rows = [r for r in rows if c in json.dumps(r, default=str).lower()]
        rows.sort(key=lambda r: r["ts"])
        return rows[-limit:]


class _StreamFilter(logging.Filter):
    def __init__(self, stream: str) -> None:
        super().__init__()
        self.stream = stream

    def filter(self, record: logging.LogRecord) -> bool:
        if self.stream == "errors":
            return record.levelno >= logging.ERROR
        return record.name == f"nomo.{self.stream}"


_ring: Optional[RingHandler] = None
_file_dir: Optional[Path] = None


def setup_logging() -> RingHandler:
    """Idempotent. Configures the `nomo.*` logger tree with stdout, ring and file sinks."""
    global _ring, _file_dir
    if _ring is not None:
        return _ring
    root = logging.getLogger("nomo")
    root.setLevel(LOG_LEVEL)
    root.propagate = False
    fmt = JsonFormatter()

    out = logging.StreamHandler(sys.stdout)
    out.setFormatter(fmt)
    root.addHandler(out)

    _ring = RingHandler()
    root.addHandler(_ring)

    try:
        d = Path(LOG_DIR)
        d.mkdir(parents=True, exist_ok=True)
        for s in STREAMS:
            fh = RotatingFileHandler(d / f"{s}.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
            fh.setFormatter(fmt)
            fh.addFilter(_StreamFilter(s))
            root.addHandler(fh)
        _file_dir = d.resolve()
    except OSError as exc:  # read-only filesystem: stdout + ring still work
        logging.getLogger("nomo.backend").warning("log.files_disabled", extra={"reason": str(exc)})
    return _ring


def ring() -> RingHandler:
    return setup_logging()


def file_dir() -> Optional[Path]:
    return _file_dir


def get(stream: str) -> logging.Logger:
    if stream not in STREAMS:
        raise ValueError(f"unknown log stream {stream}")
    return logging.getLogger(f"nomo.{stream}")


# ---------------------------------------------------------------------------
# anonymous user registry
# ---------------------------------------------------------------------------

@dataclass
class UserRecord:
    client_id: str
    first_seen: float
    last_seen: float
    ip_hash: str
    user_agent: str = ""
    origin: str = ""
    requests: int = 0
    runs_started: int = 0
    ws_sessions: int = 0
    run_ids: List[str] = field(default_factory=list)


class UserRegistry:
    """In-memory, bounded. Identifies clients by the dashboard's localStorage id, falling back
    to the hashed IP for clients that do not send one (curl, scripts)."""

    def __init__(self, max_users: int = 10_000) -> None:
        self.users: Dict[str, UserRecord] = {}
        self.max_users = max_users
        self._lock = threading.Lock()
        self.log = get("users")

    @staticmethod
    def resolve_id(client_header: Optional[str], ip_hash: str) -> str:
        cid = (client_header or "").strip()
        if cid and len(cid) <= 64 and all(c.isalnum() or c in "-_" for c in cid):
            return cid
        return f"anon-{ip_hash}"

    def touch(self, client_id: str, ip_hash: str, user_agent: str = "", origin: str = "",
              kind: str = "request", run_id: Optional[str] = None) -> UserRecord:
        now = time.time()
        with self._lock:
            u = self.users.get(client_id)
            new = u is None
            if new:
                if len(self.users) >= self.max_users:          # evict least recently seen
                    oldest = min(self.users.values(), key=lambda r: r.last_seen)
                    del self.users[oldest.client_id]
                u = UserRecord(client_id, now, now, ip_hash, user_agent[:200], origin[:200])
                self.users[client_id] = u
            u.last_seen = now
            if kind == "request":
                u.requests += 1
            elif kind == "ws":
                u.ws_sessions += 1
            elif kind == "run":
                u.runs_started += 1
                if run_id:
                    u.run_ids = (u.run_ids + [run_id])[-20:]
        if new:
            self.log.info("user.first_seen", extra={"client_id": client_id, "ip_hash": ip_hash,
                                                     "user_agent": user_agent[:200], "origin": origin[:200]})
        return u

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = [asdict(u) for u in self.users.values()]
        return sorted(rows, key=lambda r: -r["last_seen"])

    def active(self, window_s: float = 300.0) -> int:
        cutoff = time.time() - window_s
        with self._lock:
            return sum(u.last_seen >= cutoff for u in self.users.values())
