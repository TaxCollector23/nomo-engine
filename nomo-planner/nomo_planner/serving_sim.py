"""Standalone, deterministic reference simulator for autoregressive serving.

This module is intentionally dependency-free.  It is a reference model rather
than a GPU performance predictor: rates, batching gains, KV block sizes, cache
capacity, and SLO thresholds are explicit assumptions.  Every result emitted
by :func:`simulate_serving` is labelled ``"simulated"`` and request events are
retained so that a caller can inspect the path that produced a percentile.

The simulator supports two input modes:

* replay real trace rows (``timestamp``/``prompt_tokens``/``output_tokens``
  style fields are accepted, along with common aliases), or
* generate requests from deterministic Poisson or on/off bursty arrivals and
  configurable prompt/answer distributions.

The scheduling model is a small discrete-event model.  A non-disaggregated
replica interleaves continuous decode batches with prefill chunks.  A
disaggregated run simulates a prefill pool, an explicit KV transfer, and a
separate decode pool.  KV is accounted for in blocks; inactive prefix entries
are evicted by LRU and active requests may be preempted and re-prefilled when
capacity is exhausted.  The model is deliberately inspectable instead of
collapsing serving behavior into one latency formula.
"""

from __future__ import annotations

import csv
import datetime as _datetime
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SIMULATED_LABEL = "simulated"


def _finite(value: Any, default: float = 0.0) -> float:
    """Coerce a number while keeping malformed external trace data safe."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _positive_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        result = int(round(float(value)))
    except (TypeError, ValueError):
        return max(minimum, default)
    return max(minimum, result)


def _nonnegative_int(value: Any, default: int = 0) -> int:
    try:
        result = int(round(float(value)))
    except (TypeError, ValueError):
        return max(0, default)
    return max(0, result)


def _ceil_div(value: int, divisor: int) -> int:
    return 0 if value <= 0 else (value + divisor - 1) // divisor


@dataclass(frozen=True)
class DistributionSpec:
    """A small dependency-free distribution description.

    ``kind`` accepts ``fixed``/``constant``, ``uniform``, ``normal``,
    ``lognormal``, ``exponential``, ``gamma``, ``poisson``, ``choice`` and
    ``empirical``.  The named fields cover the common forms; ``params`` keeps
    JSON-like trace/config inputs extensible.
    """

    kind: str = "fixed"
    value: float = 1.0
    low: float | None = None
    high: float | None = None
    mean: float | None = None
    stddev: float | None = None
    shape: float | None = None
    scale: float | None = None
    values: tuple[Any, ...] = ()
    weights: tuple[float, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)

    def sample(self, rng: random.Random) -> float:
        """Draw one sample from this spec using only ``rng``."""

        kind = str(self.kind or "fixed").lower().replace("-", "_")
        params = dict(self.params or {})
        value = self.value
        if "value" in params:
            value = params["value"]
        if "mean" in params:
            mean = _finite(params["mean"], _finite(self.mean, _finite(value, 0.0)))
        else:
            mean = _finite(self.mean, _finite(value, 0.0))
        low = _finite(params.get("low", self.low), 0.0)
        high = _finite(params.get("high", self.high), low)
        if high < low:
            low, high = high, low
        stddev = max(0.0, _finite(params.get("stddev", params.get("std", self.stddev)), 0.0))
        shape = max(1e-12, _finite(params.get("shape", self.shape), 1.0))
        scale = max(1e-12, _finite(params.get("scale", self.scale), mean if mean > 0 else 1.0))
        values = tuple(params.get("values", self.values) or ())
        weights = tuple(_finite(item, 0.0) for item in (params.get("weights", self.weights) or ()))

        if kind in {"fixed", "constant", "deterministic"}:
            return _finite(value, mean)
        if kind in {"uniform", "flat"}:
            return rng.uniform(low, high)
        if kind in {"normal", "gaussian"}:
            return rng.gauss(mean, stddev)
        if kind in {"lognormal", "log_normal"}:
            # ``mean`` and ``stddev`` are interpreted in log-space, matching
            # random.Random.lognormvariate's contract.
            return rng.lognormvariate(mean, stddev)
        if kind in {"exponential", "exp"}:
            rate = _finite(params.get("rate_per_s", params.get("rate")), 0.0)
            mean_gap = _finite(params.get("mean_gap", params.get("mean")), 0.0)
            scale_value = 1.0 / rate if rate > 0 else max(mean_gap, scale, 1e-12)
            return rng.expovariate(1.0 / scale_value)
        if kind == "gamma":
            return rng.gammavariate(shape, scale)
        if kind in {"poisson", "pois"}:
            return float(_sample_poisson(rng, max(0.0, mean)))
        if kind in {"choice", "categorical"}:
            if not values:
                return _finite(value, mean)
            selected = _weighted_choice(rng, values, weights)
            return _finite(selected, mean)
        if kind in {"empirical", "sample", "samples"}:
            if not values:
                return _finite(value, mean)
            return _finite(rng.choice(values), mean)
        # Unknown distribution kinds are not allowed to become hidden model
        # behavior.  Treat them as fixed while leaving the kind visible in
        # the returned assumptions.
        return _finite(value, mean)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "low": self.low,
            "high": self.high,
            "mean": self.mean,
            "stddev": self.stddev,
            "shape": self.shape,
            "scale": self.scale,
            "values": list(self.values),
            "weights": list(self.weights),
            "params": dict(self.params or {}),
        }


@dataclass(frozen=True)
class ArrivalSpec:
    """Arrival process parameters used when trace replay is not selected."""

    kind: str = "replay"
    rate_per_s: float = 1.0
    burst_rate_per_s: float = 5.0
    burst_duration_s: float = 1.0
    quiet_duration_s: float = 1.0
    duration_s: float | None = None
    request_count: int = 32
    rows: tuple[Mapping[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "rate_per_s": self.rate_per_s,
            "burst_rate_per_s": self.burst_rate_per_s,
            "burst_duration_s": self.burst_duration_s,
            "quiet_duration_s": self.quiet_duration_s,
            "duration_s": self.duration_s,
            "request_count": self.request_count,
            "rows": len(self.rows),
        }


@dataclass(frozen=True)
class ServingAssumptions:
    """Explicit assumptions for the reference serving model.

    The rates are *simulation parameters*, not benchmark measurements.  A
    caller should replace them with calibrated values before using the output
    for capacity commitments.
    """

    seed: int = 0
    arrival_process: str | Mapping[str, Any] | ArrivalSpec = "replay"
    request_count: int = 32
    duration_s: float | None = None
    arrival_rate_per_s: float = 1.0
    burst_rate_per_s: float = 5.0
    burst_duration_s: float = 1.0
    quiet_duration_s: float = 1.0
    replay_rows: Sequence[Mapping[str, Any]] = ()
    trace_time_unit: str = "s"
    normalize_trace_timestamps: bool = True
    chars_per_token: float = 4.0

    prompt_distribution: Any = field(default_factory=lambda: DistributionSpec("fixed", value=256))
    answer_distribution: Any = field(default_factory=lambda: DistributionSpec("fixed", value=64))
    # Aliases kept as constructor fields for callers that naturally name the
    # input by its sampled quantity rather than by its distribution.
    prompt_tokens: Any = None
    answer_tokens: Any = None

    replicas: int = 1
    tensor_parallel: int = 1
    tp_efficiency: float = 0.90
    prefill_replicas: int | None = None
    decode_replicas: int | None = None
    disaggregate: bool = False
    prefill_decode_disaggregation: bool | None = None

    prefill_tokens_per_s: float = 10_000.0
    decode_tokens_per_s: float = 100.0
    draft_tokens_per_s: float = 400.0
    prefill_batch_overhead_s: float = 0.002
    decode_batch_overhead_s: float = 0.001
    transfer_latency_s: float = 0.0005
    transfer_tokens_per_s: float = 50_000.0
    prefill_batch_gain: float = 0.20
    decode_batch_gain: float = 0.15
    speculative_verification_factor: float = 0.15

    max_batch_size: int = 8
    chunked_prefill_tokens: int = 0
    max_decode_steps_before_prefill: int = 1
    continuous_batching: bool = True

    kv_block_tokens: int = 16
    kv_capacity_blocks: int | None = None
    kv_capacity_tokens: int = 65_536
    kv_bytes_per_token: float = 4.0
    eviction_policy: str = "lru"
    preemption_enabled: bool = True

    prefix_cache_enabled: bool = True
    prefix_cache_capacity_blocks: int | None = None
    prefix_cache_min_tokens: int = 1

    speculative_enabled: bool = False
    speculative_draft_tokens: int = 4
    speculative_acceptance: Any = 0.80

    slo_ttft_s: float | None = 1.0
    slo_itl_s: float | None = 0.20
    slo_e2e_s: float | None = 30.0

    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceRequest:
    """Normalized request input accepted by :func:`simulate_serving`."""

    request_id: str
    arrival_s: float
    prompt_tokens: int | None = None
    answer_tokens: int | None = None
    prefix_id: str | None = None
    prefix_tokens: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TimelineEvent:
    """One request-level event in simulated time."""

    time_s: float
    request_id: str
    event: str
    phase: str
    replica_id: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    label: str = SIMULATED_LABEL
    sequence: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "time_s": self.time_s,
            "request_id": self.request_id,
            "event": self.event,
            "phase": self.phase,
            "replica_id": self.replica_id,
            "details": dict(self.details),
            "label": self.label,
            "sequence": self.sequence,
        }


@dataclass(frozen=True)
class DistributionSummary:
    """Distribution values and selected percentiles, all labelled simulated."""

    name: str
    values: tuple[float, ...]
    p50: float | None
    p90: float | None
    p99: float | None
    label: str = SIMULATED_LABEL

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "values": list(self.values),
            "p50": self.p50,
            "p90": self.p90,
            "p99": self.p99,
            "label": self.label,
        }


@dataclass(frozen=True)
class RequestResult:
    """Outcome and timeline for one request."""

    request_id: str
    arrival_s: float
    prompt_tokens: int
    answer_tokens: int
    status: str
    first_token_s: float | None
    completion_s: float | None
    ttft_s: float | None
    e2e_s: float | None
    inter_token_latency_s: tuple[float, ...]
    average_itl_s: float | None
    max_itl_s: float | None
    output_tokens: int
    prefill_replica: str | None
    decode_replica: str | None
    prefix_cache_hit: bool
    prefix_tokens: int
    preemptions: int
    kv_peak_blocks: int
    error: str | None
    timeline: tuple[TimelineEvent, ...]
    label: str = SIMULATED_LABEL

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "arrival_s": self.arrival_s,
            "prompt_tokens": self.prompt_tokens,
            "answer_tokens": self.answer_tokens,
            "status": self.status,
            "first_token_s": self.first_token_s,
            "completion_s": self.completion_s,
            "ttft_s": self.ttft_s,
            "e2e_s": self.e2e_s,
            "inter_token_latency_s": list(self.inter_token_latency_s),
            "average_itl_s": self.average_itl_s,
            "max_itl_s": self.max_itl_s,
            "output_tokens": self.output_tokens,
            "prefill_replica": self.prefill_replica,
            "decode_replica": self.decode_replica,
            "prefix_cache_hit": self.prefix_cache_hit,
            "prefix_tokens": self.prefix_tokens,
            "preemptions": self.preemptions,
            "kv_peak_blocks": self.kv_peak_blocks,
            "error": self.error,
            "timeline": [event.as_dict() for event in self.timeline],
            "label": self.label,
        }


@dataclass(frozen=True)
class SimulationResult:
    """Complete serving-simulation output, including raw timelines."""

    seed: int
    simulated: bool
    assumptions: Mapping[str, Any]
    requests: tuple[RequestResult, ...]
    timeline: tuple[TimelineEvent, ...]
    timelines: Mapping[str, tuple[TimelineEvent, ...]]
    distributions: Mapping[str, DistributionSummary]
    metrics: Mapping[str, float | int | None]
    metric_labels: Mapping[str, str]
    replica_metrics: Mapping[str, Mapping[str, float | int | None]]
    notes: tuple[str, ...] = ()

    @property
    def request_results(self) -> tuple[RequestResult, ...]:
        """Alias useful to callers that prefer a more explicit field name."""

        return self.requests

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "simulated": self.simulated,
            "label": SIMULATED_LABEL,
            "assumptions": dict(self.assumptions),
            "requests": [request.as_dict() for request in self.requests],
            "timeline": [event.as_dict() for event in self.timeline],
            "timelines": {
                key: [event.as_dict() for event in value] for key, value in self.timelines.items()
            },
            "distributions": {
                key: value.as_dict() for key, value in self.distributions.items()
            },
            "metrics": {
                key: {"value": value, "label": self.metric_labels.get(key, SIMULATED_LABEL)}
                for key, value in self.metrics.items()
            },
            "replica_metrics": {key: dict(value) for key, value in self.replica_metrics.items()},
            "notes": list(self.notes),
        }

    to_dict = as_dict


@dataclass
class _MutableRequest:
    trace: TraceRequest
    prompt_tokens: int
    answer_tokens: int
    arrival_s: float
    prefix_key: str | None
    prefix_tokens_requested: int
    status: str = "new"
    first_token_s: float | None = None
    completion_s: float | None = None
    error: str | None = None
    prefill_replica: str | None = None
    decode_replica: str | None = None
    prefix_cache_hit: bool = False
    prefix_cache_ref: bool = False
    prefix_cache_stored: bool = False
    cached_prefix_tokens: int = 0
    preemptions: int = 0
    context_tokens: int = 0
    generated_tokens: int = 0
    prefill_remaining: int = 0
    prefill_initialized: bool = False
    private_blocks: int = 0
    kv_peak_blocks: int = 0
    last_service_time: float = 0.0
    token_times: list[float] = field(default_factory=list)
    timeline: list[TimelineEvent] = field(default_factory=list)


@dataclass
class _PrefixEntry:
    key: str
    prefix_tokens: int
    blocks: int
    last_used: int
    refs: int = 0


class _EventRecorder:
    def __init__(self) -> None:
        self.sequence = 0

    def add(
        self,
        request: _MutableRequest,
        event: str,
        time_s: float,
        phase: str,
        replica_id: str | None,
        details: Mapping[str, Any] | None = None,
    ) -> TimelineEvent:
        item = TimelineEvent(
            time_s=max(0.0, float(time_s)),
            request_id=request.trace.request_id,
            event=event,
            phase=phase,
            replica_id=replica_id,
            details=dict(details or {}),
            label=SIMULATED_LABEL,
            sequence=self.sequence,
        )
        self.sequence += 1
        request.timeline.append(item)
        return item


class _KVState:
    """Paged KV accounting for one simulated worker/replica."""

    def __init__(
        self,
        assumptions: ServingAssumptions,
        recorder: _EventRecorder,
        replica_id: str,
        phase: str,
    ) -> None:
        self.block_tokens = max(1, int(assumptions.kv_block_tokens))
        configured = assumptions.kv_capacity_blocks
        if configured is None:
            configured = _ceil_div(_nonnegative_int(assumptions.kv_capacity_tokens, 65_536), self.block_tokens)
        self.capacity_blocks = max(1, int(configured))
        configured_prefix = assumptions.prefix_cache_capacity_blocks
        if configured_prefix is None:
            configured_prefix = max(0, self.capacity_blocks // 4)
        self.prefix_capacity_blocks = max(0, min(self.capacity_blocks, int(configured_prefix)))
        self.enabled = bool(assumptions.prefix_cache_enabled and self.prefix_capacity_blocks > 0)
        self.prefix_min_tokens = max(1, int(assumptions.prefix_cache_min_tokens))
        self.eviction_policy = str(assumptions.eviction_policy or "lru").lower()
        self.recorder = recorder
        self.replica_id = replica_id
        self.phase = phase
        self.used_blocks = 0
        self.prefix_blocks = 0
        self.active_blocks: dict[str, int] = {}
        self.prefix: dict[str, _PrefixEntry] = {}
        self.clock = 0
        self.evictions = 0
        self.samples: list[tuple[float, int]] = [(0.0, 0)]
        self.peak_blocks = 0

    @property
    def utilization(self) -> float:
        return self.used_blocks / self.capacity_blocks if self.capacity_blocks else 0.0

    def sample(self, time_s: float) -> None:
        self.samples.append((max(0.0, time_s), self.used_blocks))
        self.peak_blocks = max(self.peak_blocks, self.used_blocks)

    def _touch(self, entry: _PrefixEntry) -> None:
        self.clock += 1
        entry.last_used = self.clock

    def lookup(
        self,
        request: _MutableRequest,
        time_s: float,
        phase: str,
        replica_id: str,
    ) -> int:
        if (
            not self.enabled
            or not request.prefix_key
            or request.prefix_tokens_requested < self.prefix_min_tokens
        ):
            return 0
        entry = self.prefix.get(request.prefix_key)
        if entry is None or entry.prefix_tokens < request.prefix_tokens_requested:
            self.recorder.add(
                request,
                "prefix_cache_miss",
                time_s,
                phase,
                replica_id,
                {"key": request.prefix_key, "requested_tokens": request.prefix_tokens_requested},
            )
            return 0
        self._touch(entry)
        entry.refs += 1
        request.prefix_cache_ref = True
        request.prefix_cache_hit = True
        request.cached_prefix_tokens = min(entry.prefix_tokens, request.prefix_tokens_requested)
        self.recorder.add(
            request,
            "prefix_cache_hit",
            time_s,
            phase,
            replica_id,
            {
                "key": request.prefix_key,
                "prefix_tokens": request.cached_prefix_tokens,
                "blocks": entry.blocks,
            },
        )
        return request.cached_prefix_tokens

    def release_prefix(self, request: _MutableRequest) -> None:
        if request.prefix_key and request.prefix_cache_ref:
            entry = self.prefix.get(request.prefix_key)
            if entry is not None and entry.refs > 0:
                entry.refs -= 1
            request.prefix_cache_ref = False

    def evict_inactive(
        self,
        required_extra: int,
        request: _MutableRequest,
        time_s: float,
        phase: str,
        replica_id: str,
    ) -> int:
        """Evict inactive prefix entries until ``required_extra`` fits."""

        needed = max(0, required_extra)
        while self.used_blocks + needed > self.capacity_blocks:
            candidates = [entry for entry in self.prefix.values() if entry.refs == 0]
            if not candidates:
                break
            if self.eviction_policy == "largest":
                victim = max(candidates, key=lambda item: (item.blocks, -item.last_used, item.key))
            else:
                victim = min(candidates, key=lambda item: (item.last_used, item.key))
            self.prefix.pop(victim.key, None)
            self.prefix_blocks -= victim.blocks
            self.used_blocks -= victim.blocks
            self.evictions += 1
            self.recorder.add(
                request,
                "kv_eviction",
                time_s,
                phase,
                replica_id,
                {"cache_key": victim.key, "blocks": victim.blocks, "policy": self.eviction_policy},
            )
        return max(0, self.capacity_blocks - self.used_blocks)

    def add_active_blocks(self, request: _MutableRequest, desired: int) -> int:
        desired = max(0, desired)
        additional = max(0, desired - request.private_blocks)
        request.private_blocks = desired
        self.active_blocks[request.trace.request_id] = desired
        self.used_blocks += additional
        self.peak_blocks = max(self.peak_blocks, self.used_blocks)
        return additional

    def free_active(self, request: _MutableRequest) -> None:
        old = self.active_blocks.pop(request.trace.request_id, request.private_blocks)
        self.used_blocks = max(0, self.used_blocks - old)
        request.private_blocks = 0

    def promote_prefix(
        self,
        request: _MutableRequest,
        time_s: float,
        phase: str,
        replica_id: str,
    ) -> bool:
        if (
            not self.enabled
            or not request.prefix_key
            or request.prefix_tokens_requested < self.prefix_min_tokens
            or request.generated_tokens > 0
            or request.prefix_cache_hit
        ):
            return False
        prefix_tokens = min(request.prefix_tokens_requested, request.context_tokens)
        if prefix_tokens <= 0:
            return False
        blocks = _ceil_div(prefix_tokens, self.block_tokens)
        if blocks > self.prefix_capacity_blocks:
            return False
        existing = self.prefix.get(request.prefix_key)
        if existing is not None and existing.prefix_tokens >= prefix_tokens:
            existing.refs += 1
            request.prefix_cache_ref = True
            request.prefix_cache_stored = True
            request.prefix_cache_hit = True
            request.cached_prefix_tokens = existing.prefix_tokens
            shared_blocks = min(existing.blocks, request.private_blocks)
            if shared_blocks:
                request.private_blocks -= shared_blocks
                self.active_blocks[request.trace.request_id] = request.private_blocks
                self.used_blocks = max(0, self.used_blocks - shared_blocks)
            self.recorder.add(
                request,
                "prefix_cache_hit",
                time_s,
                phase,
                replica_id,
                {
                    "key": request.prefix_key,
                    "prefix_tokens": existing.prefix_tokens,
                    "blocks": existing.blocks,
                    "same_batch_dedup": True,
                },
            )
            return True
        # The request's active blocks are already counted in used_blocks.  It
        # is safe to relabel the prefix portion as shared cache blocks only if
        # the cache has room after inactive entries are evicted.
        while self.prefix_blocks + blocks > self.prefix_capacity_blocks:
            candidates = [entry for entry in self.prefix.values() if entry.refs == 0]
            if not candidates:
                return False
            victim = min(candidates, key=lambda item: (item.last_used, item.key))
            self.prefix.pop(victim.key, None)
            self.prefix_blocks -= victim.blocks
            self.used_blocks -= victim.blocks
            self.evictions += 1
            self.recorder.add(
                request,
                "kv_eviction",
                time_s,
                phase,
                replica_id,
                {"cache_key": victim.key, "blocks": victim.blocks, "policy": self.eviction_policy},
            )
        if request.private_blocks < blocks:
            return False
        request.private_blocks -= blocks
        self.active_blocks[request.trace.request_id] = request.private_blocks
        self.prefix[request.prefix_key] = _PrefixEntry(
            key=request.prefix_key,
            prefix_tokens=prefix_tokens,
            blocks=blocks,
            last_used=self.clock + 1,
            refs=1,
        )
        self.clock += 1
        self.prefix_blocks += blocks
        request.prefix_cache_ref = True
        request.prefix_cache_stored = True
        request.cached_prefix_tokens = prefix_tokens
        self.recorder.add(
            request,
            "prefix_cache_store",
            time_s,
            phase,
            replica_id,
            {"key": request.prefix_key, "prefix_tokens": prefix_tokens, "blocks": blocks},
        )
        return True


class _Worker:
    """Discrete-event scheduler for one prefill/decode replica."""

    def __init__(
        self,
        requests: Sequence[_MutableRequest],
        assumptions: ServingAssumptions,
        recorder: _EventRecorder,
        rng: random.Random,
        replica_id: str,
        mode: str,
    ) -> None:
        self.requests = sorted(requests, key=lambda item: (item.arrival_s, item.trace.request_id))
        self.assumptions = assumptions
        self.recorder = recorder
        self.rng = rng
        self.replica_id = replica_id
        self.mode = mode
        self.kv = _KVState(assumptions, recorder, replica_id, mode)
        self.now = 0.0
        self.cursor = 0
        self.waiting: list[_MutableRequest] = []
        self.active: list[_MutableRequest] = []
        self.decode_steps_since_prefill = 0
        self.end_time = 0.0

    @property
    def tp_factor(self) -> float:
        return max(1.0, float(self.assumptions.tensor_parallel)) * max(0.01, float(self.assumptions.tp_efficiency))

    def _add_ready(self) -> None:
        while self.cursor < len(self.requests) and self.requests[self.cursor].arrival_s <= self.now + 1e-12:
            request = self.requests[self.cursor]
            self.cursor += 1
            if request.status in {"completed", "failed"}:
                continue
            if self.mode == "decode":
                request.status = "decoding"
                self.active.append(request)
                self.recorder.add(
                    request,
                    "decode_ready",
                    self.now,
                    "decode",
                    self.replica_id,
                    {"queue_wait_s": max(0.0, self.now - request.arrival_s)},
                )
            else:
                request.status = "waiting_prefill"
                self.waiting.append(request)
                self.recorder.add(
                    request,
                    "queued",
                    self.now,
                    "prefill" if self.mode != "decode" else "decode",
                    self.replica_id,
                    {"queue_wait_s": max(0.0, self.now - request.arrival_s)},
                )

    def _pending(self) -> bool:
        return bool(self.cursor < len(self.requests) or self.waiting or self.active)

    def _advance_if_idle(self) -> bool:
        if self.waiting or self.active:
            return False
        if self.cursor >= len(self.requests):
            return False
        self.now = max(self.now, self.requests[self.cursor].arrival_s)
        self._add_ready()
        return True

    def _batch_scale(self, batch_size: int, gain: float) -> float:
        return max(1.0, 1.0 + max(0.0, gain) * max(0, batch_size - 1))

    def _prefill_duration(self, total_tokens: int, batch_size: int) -> float:
        rate = max(1e-12, float(self.assumptions.prefill_tokens_per_s)) * self.tp_factor
        return max(0.0, float(self.assumptions.prefill_batch_overhead_s)) + (
            max(0, total_tokens) / (rate * self._batch_scale(batch_size, self.assumptions.prefill_batch_gain))
        )

    def _decode_duration(self, work: float, batch_size: int) -> float:
        rate = max(1e-12, float(self.assumptions.decode_tokens_per_s)) * self.tp_factor
        return max(0.0, float(self.assumptions.decode_batch_overhead_s)) + (
            max(0.0, work) / (rate * self._batch_scale(batch_size, self.assumptions.decode_batch_gain))
        )

    def _preempt(self, request: _MutableRequest, time_s: float) -> None:
        if request not in self.active:
            return
        self.active = [item for item in self.active if item is not request]
        self.kv.free_active(request)
        request.preemptions += 1
        request.status = "waiting_prefill"
        request.prefill_initialized = True
        recompute_tokens = max(0, request.context_tokens - request.cached_prefix_tokens)
        # Recompute starts from the shared prefix (or token zero).  Keeping
        # the old context as the prefill cursor would double-count it on the
        # next chunk and manufacture impossible KV growth.
        request.context_tokens = request.cached_prefix_tokens
        request.prefill_remaining = recompute_tokens
        request.last_service_time = time_s
        self.waiting.append(request)
        self.recorder.add(
            request,
            "preempt",
            time_s,
            "kv",
            self.replica_id,
            {
                "preemptions": request.preemptions,
                "recompute_tokens": recompute_tokens,
                "kv_capacity_blocks": self.kv.capacity_blocks,
                    "kv_used_blocks": self.kv.used_blocks,
                    "kv_utilization": self.kv.utilization,
            },
        )

    def _ensure_capacity(
        self,
        request: _MutableRequest,
        desired_private_blocks: int,
        time_s: float,
        protected: set[str],
        phase: str,
    ) -> bool:
        desired_private_blocks = max(0, desired_private_blocks)
        additional = max(0, desired_private_blocks - request.private_blocks)
        if self.kv.used_blocks + additional <= self.kv.capacity_blocks:
            self.kv.add_active_blocks(request, desired_private_blocks)
            request.kv_peak_blocks = max(request.kv_peak_blocks, self.kv.used_blocks)
            self.kv.sample(time_s)
            return True

        self.kv.evict_inactive(additional, request, time_s, phase, self.replica_id)
        while self.kv.used_blocks + additional > self.kv.capacity_blocks:
            if not self.assumptions.preemption_enabled:
                return False
            victims = [
                item
                for item in self.active
                if item.trace.request_id not in protected
                and item.status == "decoding"
                and item.private_blocks > 0
            ]
            if not victims:
                return False
            victim = min(
                victims,
                key=lambda item: (item.last_service_time, -item.private_blocks, item.trace.request_id),
            )
            self._preempt(victim, time_s)
            self.kv.evict_inactive(additional, request, time_s, phase, self.replica_id)
        self.kv.add_active_blocks(request, desired_private_blocks)
        request.kv_peak_blocks = max(request.kv_peak_blocks, self.kv.used_blocks)
        self.kv.sample(time_s)
        return True

    def _prepare_prefill(self, request: _MutableRequest, time_s: float) -> None:
        if request.prefill_initialized:
            return
        hit = self.kv.lookup(request, time_s, "prefill", self.replica_id)
        request.prefill_initialized = True
        request.context_tokens = hit
        request.prefill_remaining = max(0, request.prompt_tokens - hit)
        request.cached_prefix_tokens = hit
        if request.prefix_key and not hit:
            # The miss event is emitted by lookup for cache-enabled runs.  A
            # disabled/empty cache still deserves an explicit path marker.
            self.recorder.add(
                request,
                "prefix_cache_bypass",
                time_s,
                "prefill",
                self.replica_id,
                {"key": request.prefix_key, "requested_tokens": request.prefix_tokens_requested},
            )

    def _finish_prefill(self, request: _MutableRequest, time_s: float) -> None:
        request.prefill_remaining = 0
        request.last_service_time = time_s
        if request.generated_tokens == 0:
            self.kv.promote_prefix(request, time_s, "prefill", self.replica_id)
        self.recorder.add(
            request,
            "prefill_complete",
            time_s,
            "prefill",
            self.replica_id,
            {
                "prompt_tokens": request.prompt_tokens,
                "cached_prefix_tokens": request.cached_prefix_tokens,
                "prefix_cache_hit": request.prefix_cache_hit,
            },
        )
        if request.answer_tokens <= 0:
            self._finish(request, time_s)
            return
        request.status = "decoding"
        self.active.append(request)
        self.recorder.add(
            request,
            "decode_ready",
            time_s,
            "decode",
            self.replica_id,
            {"prefill_to_decode_s": max(0.0, time_s - request.arrival_s)},
        )

    def _run_prefill_batch(self) -> None:
        if not self.waiting:
            return
        # Preserve queue order.  A request that was preempted is placed at the
        # tail, making preemption visible instead of allowing it to spin.
        selected = self.waiting[: max(1, int(self.assumptions.max_batch_size))]
        self.waiting = self.waiting[len(selected) :]
        for request in selected:
            self._prepare_prefill(request, self.now)

        chunk_limit = int(self.assumptions.chunked_prefill_tokens)
        if chunk_limit <= 0:
            chunk_limit = max(1, max((request.prefill_remaining for request in selected), default=1))
        chunks: list[tuple[_MutableRequest, int]] = []
        for request in selected:
            if request.status in {"completed", "failed"}:
                continue
            chunk = min(max(0, request.prefill_remaining), chunk_limit)
            if chunk == 0:
                self._finish_prefill(request, self.now)
                continue
            desired_context = request.context_tokens + chunk
            desired_private = max(0, desired_context - request.cached_prefix_tokens)
            desired_blocks = _ceil_div(desired_private, self.kv.block_tokens)
            protected = {item.trace.request_id for item, _ in chunks}
            protected.update(item.trace.request_id for item in selected)
            if not self._ensure_capacity(request, desired_blocks, self.now, protected, "prefill"):
                request.status = "failed"
                request.error = "kv_capacity_exhausted"
                self.recorder.add(
                    request,
                    "kv_oom",
                    self.now,
                    "kv",
                    self.replica_id,
                    {"required_blocks": desired_blocks, "capacity_blocks": self.kv.capacity_blocks},
                )
                continue
            request.context_tokens = desired_context
            request.prefill_remaining -= chunk
            chunks.append((request, chunk))

        batch_size = max(1, len(chunks))
        total_tokens = sum(chunk for _, chunk in chunks)
        duration = self._prefill_duration(total_tokens, batch_size)
        start = self.now
        end = start + duration
        for request, chunk in chunks:
            self.recorder.add(
                request,
                "prefill_batch_start",
                start,
                "prefill",
                self.replica_id,
                {"batch_size": batch_size, "chunk_tokens": chunk, "total_batch_tokens": total_tokens},
            )
        self.now = end
        self.decode_steps_since_prefill = 0
        for request, chunk in chunks:
            request.last_service_time = end
            self.recorder.add(
                request,
                "prefill_chunk",
                end,
                "prefill",
                self.replica_id,
                {
                    "chunk_tokens": chunk,
                    "context_tokens": request.context_tokens,
                    "remaining_prefill_tokens": request.prefill_remaining,
                    "batch_size": batch_size,
                    "kv_used_blocks": self.kv.used_blocks,
                    "kv_capacity_blocks": self.kv.capacity_blocks,
                    "kv_utilization": self.kv.utilization,
                },
            )
            if request.prefill_remaining > 0:
                request.status = "waiting_prefill"
                self.waiting.append(request)
            else:
                self._finish_prefill(request, end)
        self._add_ready()

    def _sample_decode_work(self, request: _MutableRequest) -> tuple[int, int, int, float]:
        remaining = max(0, request.answer_tokens - request.generated_tokens)
        if remaining <= 0:
            return 0, 0, 0, 0.0
        if not self.assumptions.speculative_enabled:
            return 1, 0, 0, 1.0
        attempted = min(max(1, int(self.assumptions.speculative_draft_tokens)), remaining)
        acceptance_spec = _coerce_distribution(self.assumptions.speculative_acceptance, DistributionSpec("fixed", value=0.8))
        probability = min(1.0, max(0.0, acceptance_spec.sample(self.rng)))
        accepted = 0
        # The first rejected draft ends a speculative run.  This gives a
        # distribution of accepted prefixes rather than a fixed speedup.
        for _ in range(attempted):
            if self.rng.random() < probability:
                accepted += 1
            else:
                break
        emitted = min(remaining, accepted + 1)
        rejected = max(0, attempted - accepted)
        verification_work = 1.0 + max(0, emitted - 1) * max(0.0, float(self.assumptions.speculative_verification_factor))
        draft_work = attempted * (1.0 / max(1e-12, float(self.assumptions.draft_tokens_per_s)) * max(1e-12, float(self.assumptions.decode_tokens_per_s)))
        return emitted, accepted, rejected, verification_work + draft_work

    def _finish(self, request: _MutableRequest, time_s: float) -> None:
        if request.status == "completed":
            return
        request.status = "completed"
        request.completion_s = time_s
        self.kv.free_active(request)
        self.kv.release_prefix(request)
        self.active = [item for item in self.active if item is not request]
        self.recorder.add(
            request,
            "completed",
            time_s,
            "decode" if request.answer_tokens > 0 else "prefill",
            self.replica_id,
            {
                "output_tokens": request.generated_tokens,
                "e2e_s": max(0.0, time_s - request.arrival_s),
                "kv_used_blocks": self.kv.used_blocks,
                "kv_capacity_blocks": self.kv.capacity_blocks,
                "kv_utilization": self.kv.utilization,
            },
        )
        self.kv.sample(time_s)

    def _run_decode_batch(self) -> None:
        if not self.active:
            return
        active = [item for item in self.active if item.status == "decoding"]
        if not active:
            self.active = []
            return
        batch = active[: max(1, int(self.assumptions.max_batch_size))]
        works: list[tuple[_MutableRequest, int, int, int, float]] = []
        protected = {item.trace.request_id for item in batch}
        for request in batch:
            emitted, accepted, rejected, work = self._sample_decode_work(request)
            if emitted <= 0:
                continue
            desired_context = request.context_tokens + emitted
            desired_private = max(0, desired_context - request.cached_prefix_tokens)
            desired_blocks = _ceil_div(desired_private, self.kv.block_tokens)
            if not self._ensure_capacity(request, desired_blocks, self.now, protected, "decode"):
                request.status = "failed"
                request.error = "kv_capacity_exhausted"
                self.recorder.add(
                    request,
                    "kv_oom",
                    self.now,
                    "kv",
                    self.replica_id,
                    {"required_blocks": desired_blocks, "capacity_blocks": self.kv.capacity_blocks},
                )
                self.kv.free_active(request)
                self.active = [item for item in self.active if item is not request]
                continue
            works.append((request, emitted, accepted, rejected, work))
        if not works:
            self.now += max(0.0, float(self.assumptions.decode_batch_overhead_s))
            self._add_ready()
            return

        batch_size = len(works)
        total_work = sum(item[4] for item in works)
        # ``work`` is measured in target-token-equivalent units.  The target
        # rate then turns it into seconds; draft work is included only for
        # speculative batches.
        duration = self._decode_duration(total_work, batch_size)
        start = self.now
        end = start + duration
        for request, emitted, accepted, rejected, work in works:
            self.recorder.add(
                request,
                "decode_batch_start",
                start,
                "decode",
                self.replica_id,
                {
                    "batch_size": batch_size,
                    "speculative": bool(self.assumptions.speculative_enabled),
                    "emitted_tokens": emitted,
                    "accepted_draft_tokens": accepted,
                    "rejected_draft_tokens": rejected,
                    "target_work_units": work,
                },
            )
            if self.assumptions.speculative_enabled:
                self.recorder.add(
                    request,
                    "speculative_step",
                    start,
                    "decode",
                    self.replica_id,
                    {"accepted": accepted, "attempted": accepted + rejected, "rejected": rejected},
                )
            for index in range(emitted):
                token_time = start + duration * ((index + 1) / emitted)
                request.generated_tokens += 1
                request.context_tokens += 1
                request.token_times.append(token_time)
                if request.first_token_s is None:
                    request.first_token_s = token_time
                self.recorder.add(
                    request,
                    "decode_token",
                    token_time,
                    "decode",
                    self.replica_id,
                    {
                        "token_index": request.generated_tokens,
                        "speculative": bool(self.assumptions.speculative_enabled),
                        "accepted_draft": index < accepted,
                        "kv_used_blocks": self.kv.used_blocks,
                        "kv_capacity_blocks": self.kv.capacity_blocks,
                        "kv_utilization": self.kv.utilization,
                    },
                )
            request.last_service_time = end
            if request.generated_tokens >= request.answer_tokens:
                self._finish(request, end)
            else:
                self.recorder.add(
                    request,
                    "decode_batch_end",
                    end,
                    "decode",
                    self.replica_id,
                    {"generated_tokens": request.generated_tokens, "remaining_tokens": request.answer_tokens - request.generated_tokens},
                )
        self.now = end
        self.decode_steps_since_prefill += 1
        self._add_ready()

    def run(self) -> None:
        if self.mode == "prefill":
            while self._pending():
                self._add_ready()
                if not self.waiting and not self.active and self._advance_if_idle():
                    continue
                if self.waiting:
                    self._run_prefill_batch()
                elif self.cursor < len(self.requests):
                    # ``active`` contains requests whose prefill is complete
                    # and whose KV is waiting for the disaggregation handoff;
                    # it is not work for this prefill-only scheduler.  Move
                    # to the next trace arrival even while that bookkeeping
                    # list is populated.
                    self.now = max(self.now, self.requests[self.cursor].arrival_s)
                    self._add_ready()
                else:
                    break
            self.end_time = self.now
            return
        if self.mode == "decode":
            while self._pending():
                self._add_ready()
                if not self.active and self.cursor < len(self.requests):
                    self._advance_if_idle()
                    continue
                if self.active:
                    self._run_decode_batch()
                else:
                    break
            self.end_time = self.now
            return

        while self._pending():
            self._add_ready()
            if not self.waiting and not self.active and self.cursor < len(self.requests):
                self._advance_if_idle()
                continue
            if self.waiting and (
                not self.active
                or not self.assumptions.continuous_batching
                or self.decode_steps_since_prefill >= max(1, int(self.assumptions.max_decode_steps_before_prefill))
            ):
                self._run_prefill_batch()
            elif self.active:
                self._run_decode_batch()
            elif self.waiting:
                self._run_prefill_batch()
            else:
                self._advance_if_idle()
        self.end_time = self.now


def _sample_poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    if lam > 40:
        # Normal approximation avoids an O(lam) loop for large rate settings;
        # the seed and output remain deterministic.
        return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))
    threshold = math.exp(-lam)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1


def _weighted_choice(rng: random.Random, values: Sequence[Any], weights: Sequence[float]) -> Any:
    if not weights or len(weights) != len(values) or sum(max(0.0, item) for item in weights) <= 0:
        return rng.choice(list(values))
    total = sum(max(0.0, item) for item in weights)
    draw = rng.random() * total
    cumulative = 0.0
    for value, weight in zip(values, weights):
        cumulative += max(0.0, weight)
        if draw <= cumulative:
            return value
    return values[-1]


def _coerce_distribution(value: Any, default: DistributionSpec) -> DistributionSpec:
    if value is None:
        return default
    if isinstance(value, DistributionSpec):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return DistributionSpec("fixed", value=float(value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return DistributionSpec("empirical", values=tuple(value))
    if isinstance(value, Mapping):
        params = dict(value)
        kind = str(params.pop("kind", params.pop("type", params.pop("distribution", "fixed"))))
        return DistributionSpec(
            kind=kind,
            value=_finite(params.pop("value", params.pop("mean", default.value)), default.value),
            low=params.pop("low", None),
            high=params.pop("high", None),
            mean=params.pop("mean", None),
            stddev=params.pop("stddev", params.pop("std", None)),
            shape=params.pop("shape", None),
            scale=params.pop("scale", None),
            values=tuple(params.pop("values", ()) or ()),
            weights=tuple(_finite(item, 0.0) for item in (params.pop("weights", ()) or ())),
            params=params,
        )
    return default


def _mapping_value(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _coerce_assumptions(value: ServingAssumptions | Mapping[str, Any] | None) -> ServingAssumptions:
    if value is None:
        return ServingAssumptions()
    if isinstance(value, ServingAssumptions):
        direct = dict(value.__dict__)
        if isinstance(value.arrival_process, Mapping):
            arrival_mapping = dict(value.arrival_process)
            direct["arrival_process"] = str(arrival_mapping.get("kind", arrival_mapping.get("type", "replay")))
            for key in (
                "rate_per_s",
                "burst_rate_per_s",
                "burst_duration_s",
                "quiet_duration_s",
                "duration_s",
                "request_count",
            ):
                if key in arrival_mapping:
                    direct[key] = arrival_mapping[key]
            if "rows" in arrival_mapping:
                direct["replay_rows"] = tuple(arrival_mapping["rows"] or ())
        if value.prefill_decode_disaggregation is not None:
            direct["disaggregate"] = bool(value.prefill_decode_disaggregation)
        direct["prompt_distribution"] = _coerce_distribution(
            value.prompt_tokens if value.prompt_tokens is not None else value.prompt_distribution,
            DistributionSpec("fixed", value=256),
        )
        direct["answer_distribution"] = _coerce_distribution(
            value.answer_tokens if value.answer_tokens is not None else value.answer_distribution,
            DistributionSpec("fixed", value=64),
        )
        return ServingAssumptions(**direct)
    if not isinstance(value, Mapping):
        raise TypeError("assumptions must be ServingAssumptions, a mapping, or None")

    raw = dict(value)
    # Accept JSON-style nested sections while keeping the public dataclass
    # flat and easy to inspect.
    arrival = raw.get("arrival")
    if isinstance(arrival, Mapping):
        raw = {**dict(arrival), **{key: item for key, item in raw.items() if key != "arrival"}}
        raw.setdefault("arrival_process", arrival.get("kind", arrival.get("type", "replay")))
    for section_name, aliases in {
        "batching": (),
        "kv_cache": (),
        "prefix_cache": (),
        "speculative": ("speculative_decoding",),
        "disaggregation": ("prefill_decode",),
        "slo": (),
    }.items():
        section = raw.get(section_name)
        if section is None:
            for alias in aliases:
                if isinstance(raw.get(alias), Mapping):
                    section = raw[alias]
                    break
        if isinstance(section, Mapping):
            raw = {**dict(section), **{key: item for key, item in raw.items() if key != section_name and key not in aliases}}

    defaults = ServingAssumptions()
    fields = {field_info.name for field_info in ServingAssumptions.__dataclass_fields__.values()}
    kwargs: dict[str, Any] = {}
    for key in fields:
        if key in raw:
            kwargs[key] = raw[key]

    aliases = {
        "seed": ("random_seed",),
        "arrival_process": ("arrival_kind", "process"),
        "request_count": ("num_requests", "requests"),
        "arrival_rate_per_s": ("arrival_rate", "lambda_per_s", "lambda"),
        "quiet_duration_s": ("burst_gap_s", "off_duration_s"),
        "prompt_distribution": ("prompt_dist", "prompt_lengths", "prompt_length_distribution"),
        "answer_distribution": ("answer_dist", "answer_lengths", "output_distribution", "completion_distribution"),
        "replicas": ("num_replicas", "tp_replicas"),
        "tensor_parallel": ("tp", "tp_degree"),
        "disaggregate": ("prefill_decode_disaggregation", "disaggregated"),
        "kv_capacity_blocks": ("capacity_blocks",),
        "kv_capacity_tokens": ("capacity_tokens",),
        "kv_block_tokens": ("block_tokens",),
        "prefix_cache_capacity_blocks": ("prefix_capacity_blocks",),
        "prefix_cache_enabled": ("prefix_caching",),
        "speculative_enabled": ("speculation", "speculative_decoding_enabled"),
        "speculative_draft_tokens": ("draft_tokens",),
        "speculative_acceptance": ("draft_acceptance", "acceptance_rate"),
        "chunked_prefill_tokens": ("prefill_chunk_tokens",),
        "slo_ttft_s": ("ttft_slo_s",),
        "slo_itl_s": ("itl_slo_s",),
        "slo_e2e_s": ("e2e_slo_s",),
    }
    for canonical, names in aliases.items():
        if canonical in kwargs:
            continue
        found = _mapping_value(raw, *names)
        if found is not None:
            kwargs[canonical] = found

    # A nested speculative mapping often uses ``enabled`` rather than the
    # canonical field name.
    spec = value.get("speculative") if isinstance(value.get("speculative"), Mapping) else value.get("speculative_decoding")
    if isinstance(spec, Mapping):
        kwargs.setdefault("speculative_enabled", spec.get("enabled", spec.get("active", False)))
        kwargs.setdefault("speculative_draft_tokens", spec.get("draft_tokens", spec.get("k", 4)))
        kwargs.setdefault("speculative_acceptance", spec.get("acceptance", spec.get("acceptance_rate", 0.8)))
    disaggregation = value.get("disaggregation") if isinstance(value.get("disaggregation"), Mapping) else value.get("prefill_decode")
    if isinstance(disaggregation, Mapping):
        kwargs.setdefault("disaggregate", disaggregation.get("enabled", disaggregation.get("active", True)))
        kwargs.setdefault("prefill_replicas", disaggregation.get("prefill_replicas"))
        kwargs.setdefault("decode_replicas", disaggregation.get("decode_replicas"))
    prefix_cache = value.get("prefix_cache") if isinstance(value.get("prefix_cache"), Mapping) else None
    if isinstance(prefix_cache, Mapping):
        kwargs.setdefault("prefix_cache_enabled", prefix_cache.get("enabled", prefix_cache.get("active", True)))
        kwargs.setdefault("prefix_cache_capacity_blocks", prefix_cache.get("capacity_blocks"))
        kwargs.setdefault("prefix_cache_min_tokens", prefix_cache.get("min_tokens", 1))

    if "prompt_tokens" in kwargs and "prompt_distribution" not in kwargs:
        kwargs["prompt_distribution"] = kwargs["prompt_tokens"]
    if "answer_tokens" in kwargs and "answer_distribution" not in kwargs:
        kwargs["answer_distribution"] = kwargs["answer_tokens"]
    if "replay" in raw and "replay_rows" not in kwargs:
        kwargs["replay_rows"] = raw["replay"]

    kwargs["prompt_distribution"] = _coerce_distribution(
        kwargs.get("prompt_distribution"), _coerce_distribution(defaults.prompt_distribution, defaults.prompt_distribution)
    )
    kwargs["answer_distribution"] = _coerce_distribution(
        kwargs.get("answer_distribution"), _coerce_distribution(defaults.answer_distribution, defaults.answer_distribution)
    )
    if kwargs.get("prompt_tokens") is not None:
        kwargs["prompt_distribution"] = _coerce_distribution(kwargs["prompt_tokens"], kwargs["prompt_distribution"])
    if kwargs.get("answer_tokens") is not None:
        kwargs["answer_distribution"] = _coerce_distribution(kwargs["answer_tokens"], kwargs["answer_distribution"])
    if kwargs.get("prefill_decode_disaggregation") is not None and "disaggregate" not in raw:
        kwargs["disaggregate"] = bool(kwargs["prefill_decode_disaggregation"])
    if isinstance(kwargs.get("arrival_process"), Mapping):
        arrival_mapping = dict(kwargs["arrival_process"])
        kwargs["arrival_process"] = str(arrival_mapping.get("kind", arrival_mapping.get("type", "replay")))
        for key in ("rate_per_s", "burst_rate_per_s", "burst_duration_s", "quiet_duration_s", "duration_s", "request_count", "rows"):
            if key in arrival_mapping and key not in kwargs:
                kwargs[key if key != "rows" else "replay_rows"] = arrival_mapping[key]
    if isinstance(kwargs.get("speculative_enabled"), Mapping):
        speculative_mapping = kwargs["speculative_enabled"]
        kwargs["speculative_enabled"] = bool(speculative_mapping.get("enabled", speculative_mapping.get("active", False)))
        kwargs.setdefault("speculative_draft_tokens", speculative_mapping.get("draft_tokens", 4))
        kwargs.setdefault("speculative_acceptance", speculative_mapping.get("acceptance", 0.8))
    kwargs["replay_rows"] = tuple(kwargs.get("replay_rows") or ())
    kwargs["seed"] = int(_finite(kwargs.get("seed", defaults.seed), defaults.seed))
    kwargs["request_count"] = _nonnegative_int(kwargs.get("request_count", defaults.request_count), defaults.request_count)
    kwargs["replicas"] = _positive_int(kwargs.get("replicas", defaults.replicas), defaults.replicas)
    kwargs["tensor_parallel"] = _positive_int(kwargs.get("tensor_parallel", defaults.tensor_parallel), defaults.tensor_parallel)
    kwargs["max_batch_size"] = _positive_int(kwargs.get("max_batch_size", defaults.max_batch_size), defaults.max_batch_size)
    kwargs["kv_block_tokens"] = _positive_int(kwargs.get("kv_block_tokens", defaults.kv_block_tokens), defaults.kv_block_tokens)
    kwargs["max_decode_steps_before_prefill"] = _positive_int(
        kwargs.get("max_decode_steps_before_prefill", defaults.max_decode_steps_before_prefill),
        defaults.max_decode_steps_before_prefill,
    )
    kwargs["chunked_prefill_tokens"] = _nonnegative_int(kwargs.get("chunked_prefill_tokens", defaults.chunked_prefill_tokens))
    kwargs["speculative_draft_tokens"] = _positive_int(kwargs.get("speculative_draft_tokens", defaults.speculative_draft_tokens), defaults.speculative_draft_tokens)
    return ServingAssumptions(**kwargs)


def _parse_time(value: Any, unit: str) -> float:
    if isinstance(value, _datetime.datetime):
        if value.tzinfo is None:
            return value.timestamp()
        return value.timestamp()
    if isinstance(value, str):
        text = value.strip()
        try:
            return float(text)
        except ValueError:
            try:
                parsed = _datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
                return parsed.timestamp()
            except ValueError:
                return 0.0
    number = _finite(value, 0.0)
    normalized = str(unit or "s").lower()
    factor = {"s": 1.0, "sec": 1.0, "ms": 1e-3, "millisecond": 1e-3, "us": 1e-6, "ns": 1e-9}.get(normalized, 1.0)
    return number * factor


def _extract_row_value(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        if name in row and row[name] is not None and row[name] != "":
            return row[name]
        if name.lower() in lowered and lowered[name.lower()] is not None and lowered[name.lower()] != "":
            return lowered[name.lower()]
    return None


def load_trace_rows(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load CSV rows without imposing a vendor-specific trace schema."""

    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


def _row_to_trace(
    row: TraceRequest | Mapping[str, Any],
    index: int,
    assumptions: ServingAssumptions,
    rng: random.Random,
) -> TraceRequest:
    if isinstance(row, TraceRequest):
        request_id = str(row.request_id or f"request-{index}")
        arrival = row.arrival_s
        prompt = row.prompt_tokens
        answer = row.answer_tokens
        prefix_id = row.prefix_id
        prefix_tokens = row.prefix_tokens
        metadata = dict(row.metadata or {})
    else:
        request_id = str(_extract_row_value(row, ("request_id", "id", "request", "uid")) or f"request-{index}")
        arrival_value = _extract_row_value(
            row,
            (
                "arrival_s",
                "arrival",
                "arrival_time_s",
                "arrival_time",
                "timestamp",
                "time",
                "request_time",
                "created_at",
                "timestamp_ms",
                "time_ms",
            ),
        )
        arrival_unit = assumptions.trace_time_unit
        if _extract_row_value(row, ("timestamp_ms", "time_ms", "arrival_ms")) is not None:
            arrival_unit = "ms"
        arrival = _parse_time(arrival_value, arrival_unit) if arrival_value is not None else 0.0
        prompt = _extract_row_value(
            row,
            ("prompt_tokens", "input_tokens", "input_length", "prompt_len", "prompt_length", "num_prompt_tokens", "prefill_tokens"),
        )
        answer = _extract_row_value(
            row,
            ("answer_tokens", "output_tokens", "completion_tokens", "generated_tokens", "output_len", "completion_length", "num_output_tokens"),
        )
        prompt_text = _extract_row_value(row, ("prompt", "input_text", "text"))
        answer_text = _extract_row_value(row, ("answer", "output_text", "completion"))
        if prompt is None and isinstance(prompt_text, str) and prompt_text:
            prompt = math.ceil(len(prompt_text) / max(1e-12, assumptions.chars_per_token))
        if answer is None and isinstance(answer_text, str) and answer_text:
            answer = math.ceil(len(answer_text) / max(1e-12, assumptions.chars_per_token))
        prefix_id_value = _extract_row_value(
            row,
            ("prefix_id", "prefix_key", "prefix", "prefix_hash", "prompt_hash", "cache_key", "system_prompt_id"),
        )
        prefix_id = None if prefix_id_value is None else str(prefix_id_value)
        prefix_tokens_value = _extract_row_value(row, ("prefix_tokens", "cached_prefix_tokens", "shared_prefix_tokens"))
        prefix_tokens = _nonnegative_int(prefix_tokens_value, 0) if prefix_tokens_value is not None else None
        metadata = dict(row)
    if prompt is None:
        prompt = assumptions.prompt_distribution.sample(rng)
    if answer is None:
        answer = assumptions.answer_distribution.sample(rng)
    prompt_tokens = max(1, int(round(_finite(prompt, 1.0))))
    answer_tokens = max(0, int(round(_finite(answer, 0.0))))
    if prefix_id is None:
        prefix_tokens = 0
    elif prefix_tokens is None:
        prefix_tokens = prompt_tokens
    prefix_tokens = min(prompt_tokens, max(0, int(prefix_tokens or 0)))
    return TraceRequest(
        request_id=request_id,
        arrival_s=max(0.0, float(arrival)),
        prompt_tokens=prompt_tokens,
        answer_tokens=answer_tokens,
        prefix_id=prefix_id,
        prefix_tokens=prefix_tokens,
        metadata=metadata,
    )


def _generated_trace(assumptions: ServingAssumptions, rng: random.Random) -> tuple[TraceRequest, ...]:
    process = assumptions.arrival_process
    if isinstance(process, ArrivalSpec):
        kind = str(process.kind or "replay").lower().replace("-", "_")
        rate = max(0.0, float(process.rate_per_s))
        burst_rate = max(0.0, float(process.burst_rate_per_s))
        on_duration = max(1e-12, float(process.burst_duration_s))
        off_duration = max(0.0, float(process.quiet_duration_s))
        horizon = process.duration_s if process.duration_s is not None else assumptions.duration_s
        target = _nonnegative_int(process.request_count, assumptions.request_count)
        replay_rows = process.rows or assumptions.replay_rows
    else:
        kind = str(process or "replay").lower().replace("-", "_")
        rate = max(0.0, float(assumptions.arrival_rate_per_s))
        burst_rate = max(0.0, float(assumptions.burst_rate_per_s))
        on_duration = max(1e-12, float(assumptions.burst_duration_s))
        off_duration = max(0.0, float(assumptions.quiet_duration_s))
        horizon = assumptions.duration_s
        target = _nonnegative_int(assumptions.request_count, 0)
        replay_rows = assumptions.replay_rows
    if kind in {"replay", "trace"}:
        return tuple(_row_to_trace(row, index, assumptions, rng) for index, row in enumerate(replay_rows))
    safety_limit = max(1, target if target > 0 else 10_000)
    horizon = None if horizon is None else max(0.0, float(horizon))
    arrivals: list[float] = []
    time_s = 0.0
    if kind in {"poisson", "exp", "exponential"}:
        while len(arrivals) < safety_limit and (horizon is None or time_s <= horizon):
            if rate <= 0:
                break
            time_s += rng.expovariate(rate)
            if horizon is not None and time_s > horizon:
                break
            arrivals.append(time_s)
    elif kind in {"bursty", "burst", "on_off", "onoff"}:
        cycle_start = 0.0
        while len(arrivals) < safety_limit and (horizon is None or cycle_start <= horizon):
            cursor = cycle_start
            while burst_rate > 0 and len(arrivals) < safety_limit:
                cursor += rng.expovariate(burst_rate)
                if cursor >= cycle_start + on_duration:
                    break
                if horizon is not None and cursor > horizon:
                    break
                arrivals.append(cursor)
            cycle_start += on_duration + off_duration
            if on_duration + off_duration <= 1e-12:
                break
    else:
        raise ValueError(f"unsupported arrival process: {assumptions.arrival_process!r}")
    rows = [
        TraceRequest(
            request_id=f"request-{index}",
            arrival_s=arrival,
            prompt_tokens=None,
            answer_tokens=None,
        )
        for index, arrival in enumerate(arrivals)
    ]
    return tuple(_row_to_trace(row, index, assumptions, rng) for index, row in enumerate(rows))


def normalize_trace_rows(
    rows: Iterable[TraceRequest | Mapping[str, Any]] | None,
    assumptions: ServingAssumptions | Mapping[str, Any] | None = None,
    *,
    seed: int | None = None,
) -> tuple[TraceRequest, ...]:
    """Normalize real trace rows and sample missing lengths deterministically."""

    config = _coerce_assumptions(assumptions)
    config = ServingAssumptions(
        **{
            **config.__dict__,
            "prompt_distribution": _coerce_distribution(config.prompt_distribution, DistributionSpec("fixed", value=256)),
            "answer_distribution": _coerce_distribution(config.answer_distribution, DistributionSpec("fixed", value=64)),
        }
    )
    if seed is not None:
        config = ServingAssumptions(**{**config.__dict__, "seed": int(seed)})
    rng = random.Random(config.seed)
    if rows is None:
        result = list(_generated_trace(config, rng))
    else:
        if isinstance(rows, Mapping):
            rows = (rows,)
        result = [_row_to_trace(row, index, config, rng) for index, row in enumerate(rows)]
    if not result:
        return ()
    if config.normalize_trace_timestamps:
        origin = min(item.arrival_s for item in result)
        result = [
            TraceRequest(
                request_id=item.request_id,
                arrival_s=max(0.0, item.arrival_s - origin),
                prompt_tokens=item.prompt_tokens,
                answer_tokens=item.answer_tokens,
                prefix_id=item.prefix_id,
                prefix_tokens=item.prefix_tokens,
                metadata=item.metadata,
            )
            for item in result
        ]
    seen: dict[str, int] = {}
    unique: list[TraceRequest] = []
    for item in sorted(enumerate(result), key=lambda pair: (pair[1].arrival_s, pair[0])):
        request = item[1]
        count = seen.get(request.request_id, 0)
        seen[request.request_id] = count + 1
        request_id = request.request_id if count == 0 else f"{request.request_id}-{count + 1}"
        unique.append(
            TraceRequest(
                request_id=request_id,
                arrival_s=request.arrival_s,
                prompt_tokens=request.prompt_tokens,
                answer_tokens=request.answer_tokens,
                prefix_id=request.prefix_id,
                prefix_tokens=request.prefix_tokens,
                metadata=request.metadata,
            )
        )
    return tuple(unique)


def _assign_requests(
    requests: Sequence[_MutableRequest],
    count: int,
    assumptions: ServingAssumptions,
    *,
    workload: str,
) -> list[list[_MutableRequest]]:
    pools: list[list[_MutableRequest]] = [[] for _ in range(max(1, count))]
    loads = [0.0 for _ in pools]
    prefill_rate = max(1e-12, assumptions.prefill_tokens_per_s)
    decode_rate = max(1e-12, assumptions.decode_tokens_per_s)
    for request in sorted(requests, key=lambda item: (item.arrival_s, item.trace.request_id)):
        if workload == "prefill":
            estimate = request.prompt_tokens / prefill_rate
        elif workload == "decode":
            estimate = request.answer_tokens / decode_rate
        else:
            estimate = request.prompt_tokens / prefill_rate + request.answer_tokens / decode_rate
        selected = min(range(len(pools)), key=lambda index: (loads[index], index))
        pools[selected].append(request)
        loads[selected] += max(0.000001, estimate)
    return pools


def _record_arrivals(requests: Sequence[_MutableRequest], recorder: _EventRecorder) -> None:
    for request in requests:
        recorder.add(
            request,
            "arrival",
            request.arrival_s,
            "arrival",
            request.prefill_replica,
            {
                "prompt_tokens": request.prompt_tokens,
                "answer_tokens": request.answer_tokens,
                "prefix_id": request.prefix_key,
                "prefix_tokens": request.prefix_tokens_requested,
            },
        )


def _make_request_states(trace: Sequence[TraceRequest]) -> list[_MutableRequest]:
    return [
        _MutableRequest(
            trace=item,
            prompt_tokens=max(1, int(item.prompt_tokens or 1)),
            answer_tokens=max(0, int(item.answer_tokens or 0)),
            arrival_s=max(0.0, float(item.arrival_s)),
            prefix_key=item.prefix_id,
            prefix_tokens_requested=max(0, int(item.prefix_tokens or 0)),
        )
        for item in trace
    ]


def _run_disaggregated(
    requests: list[_MutableRequest],
    assumptions: ServingAssumptions,
    recorder: _EventRecorder,
    rng: random.Random,
) -> tuple[list[_Worker], list[_Worker]]:
    prefill_count = max(1, int(assumptions.prefill_replicas or assumptions.replicas))
    decode_count = max(1, int(assumptions.decode_replicas or assumptions.replicas))
    prefill_pools = _assign_requests(requests, prefill_count, assumptions, workload="prefill")
    for index, pool in enumerate(prefill_pools):
        for request in pool:
            request.prefill_replica = f"prefill-{index}"
    prefill_workers: list[_Worker] = []
    for index, pool in enumerate(prefill_pools):
        worker = _Worker(pool, assumptions, recorder, rng, f"prefill-{index}", "prefill")
        prefill_workers.append(worker)
        worker.run()

    decode_ready: list[_MutableRequest] = []
    for request in requests:
        if request.status == "failed":
            continue
        prefill_end = request.completion_s if request.answer_tokens == 0 else request.last_service_time
        if prefill_end is None:
            prefill_end = request.arrival_s
        if request.answer_tokens <= 0:
            # A zero-token answer is complete after prefill.  Do not enqueue
            # it into the decode-only pool, where an empty decode batch would
            # otherwise make no progress.
            for worker in prefill_workers:
                if request in worker.active:
                    worker.active = [item for item in worker.active if item is not request]
                    worker.kv.free_active(request)
                    worker.kv.release_prefix(request)
                    worker.kv.sample(float(prefill_end))
            continue
        request.completion_s = None
        transfer_start = max(request.arrival_s, float(prefill_end))
        transfer_tokens = max(0, request.context_tokens)
        transfer_rate = max(1e-12, float(assumptions.transfer_tokens_per_s))
        transfer_duration = max(0.0, float(assumptions.transfer_latency_s)) + transfer_tokens / transfer_rate
        recorder.add(
            request,
            "kv_transfer_start",
            transfer_start,
            "transfer",
            request.prefill_replica,
            {"tokens": transfer_tokens, "to_decode_pool": True},
        )
        transfer_end = transfer_start + transfer_duration
        recorder.add(
            request,
            "kv_transfer",
            transfer_end,
            "transfer",
            request.prefill_replica,
            {"tokens": transfer_tokens, "duration_s": transfer_duration},
        )
        recorder.add(
            request,
            "kv_transfer_complete",
            transfer_end,
            "transfer",
            request.prefill_replica,
            {"tokens": transfer_tokens, "duration_s": transfer_duration},
        )
        # The decode pool sees the transferred context.  Prefix cache blocks
        # are not double-counted here; the decode worker accounts for the
        # private suffix and keeps the hit as a request-level fact.
        request.arrival_s = transfer_end
        request.decode_replica = None
        request.status = "new"
        request.prefill_remaining = 0
        request.prefill_initialized = True
        for worker in prefill_workers:
            if request in worker.active:
                worker.active = [item for item in worker.active if item is not request]
                worker.kv.free_active(request)
                worker.kv.release_prefix(request)
                worker.kv.sample(transfer_start)
        decode_ready.append(request)

    decode_pools = _assign_requests(decode_ready, decode_count, assumptions, workload="decode")
    for index, pool in enumerate(decode_pools):
        for request in pool:
            request.decode_replica = f"decode-{index}"
            request.status = "new"
    decode_workers: list[_Worker] = []
    for index, pool in enumerate(decode_pools):
        worker = _Worker(pool, assumptions, recorder, rng, f"decode-{index}", "decode")
        decode_workers.append(worker)
        # Decode workers receive an already-prefilled context.  Seed each
        # request's private block count before the first decode batch, while
        # still allowing normal paged-cache eviction/preemption thereafter.
        for request in pool:
            request.context_tokens = max(request.context_tokens, request.prompt_tokens)
            request.cached_prefix_tokens = min(request.cached_prefix_tokens, request.context_tokens)
            request.private_blocks = 0
            request.status = "new"
        worker.run()
    return prefill_workers, decode_workers


def _run_monolithic(
    requests: list[_MutableRequest],
    assumptions: ServingAssumptions,
    recorder: _EventRecorder,
    rng: random.Random,
) -> list[_Worker]:
    pools = _assign_requests(requests, max(1, assumptions.replicas), assumptions, workload="mixed")
    workers: list[_Worker] = []
    for index, pool in enumerate(pools):
        replica_id = f"replica-{index}"
        for request in pool:
            request.prefill_replica = replica_id
            request.decode_replica = replica_id
        worker = _Worker(pool, assumptions, recorder, rng, replica_id, "mixed")
        workers.append(worker)
    for worker in workers:
        worker.run()
    return workers


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summary(name: str, values: Iterable[float]) -> DistributionSummary:
    data = tuple(float(value) for value in values if value is not None and math.isfinite(float(value)))
    return DistributionSummary(name, data, _percentile(data, 50), _percentile(data, 90), _percentile(data, 99))


def _time_weighted_utilization(workers: Sequence[_Worker]) -> tuple[float, float, list[float]]:
    averages: list[float] = []
    peaks: list[float] = []
    samples: list[float] = []
    for worker in workers:
        kv = worker.kv
        if not kv.samples:
            averages.append(0.0)
            peaks.append(0.0)
            continue
        ordered = sorted(kv.samples, key=lambda item: item[0])
        local_samples = [used / kv.capacity_blocks for _, used in ordered]
        end = max(worker.end_time, ordered[-1][0])
        integral = 0.0
        previous_time = 0.0
        previous_used = ordered[0][1]
        for time_s, used in ordered:
            if time_s > previous_time:
                integral += (time_s - previous_time) * (previous_used / kv.capacity_blocks)
            previous_time = max(previous_time, time_s)
            previous_used = used
            samples.append(used / kv.capacity_blocks)
        if end > previous_time:
            integral += (end - previous_time) * (previous_used / kv.capacity_blocks)
        averages.append(integral / end if end > 0 else previous_used / kv.capacity_blocks)
        peaks.append(max(local_samples or [0.0]))
    return (
        sum(averages) / len(averages) if averages else 0.0,
        max(peaks) if peaks else 0.0,
        samples,
    )


def _request_result(request: _MutableRequest) -> RequestResult:
    itl = tuple(
        max(0.0, later - earlier)
        for earlier, later in zip(request.token_times, request.token_times[1:])
    )
    ttft = None if request.first_token_s is None else max(0.0, request.first_token_s - request.trace.arrival_s)
    e2e = None if request.completion_s is None else max(0.0, request.completion_s - request.trace.arrival_s)
    return RequestResult(
        request_id=request.trace.request_id,
        arrival_s=request.trace.arrival_s,
        prompt_tokens=request.prompt_tokens,
        answer_tokens=request.answer_tokens,
        status=request.status,
        first_token_s=request.first_token_s,
        completion_s=request.completion_s,
        ttft_s=ttft,
        e2e_s=e2e,
        inter_token_latency_s=itl,
        average_itl_s=(sum(itl) / len(itl) if itl else None),
        max_itl_s=(max(itl) if itl else None),
        output_tokens=request.generated_tokens,
        prefill_replica=request.prefill_replica,
        decode_replica=request.decode_replica,
        prefix_cache_hit=request.prefix_cache_hit,
        prefix_tokens=request.prefix_tokens_requested,
        preemptions=request.preemptions,
        kv_peak_blocks=request.kv_peak_blocks,
        error=request.error,
        timeline=tuple(sorted(request.timeline, key=lambda event: (event.time_s, event.sequence))),
    )


def _effective_assumptions(config: ServingAssumptions, seed: int, input_mode: str) -> dict[str, Any]:
    process_kind = config.arrival_process.kind if isinstance(config.arrival_process, ArrivalSpec) else config.arrival_process
    return {
        "label": "assumption",
        "simulator": "nomo_planner.serving_sim standalone Python reference",
        "input_mode": input_mode,
        "seed": seed,
        "arrival_process": process_kind if isinstance(process_kind, str) else str(process_kind),
        "request_count": config.request_count,
        "duration_s": config.duration_s,
        "arrival_rate_per_s": config.arrival_rate_per_s,
        "burst_rate_per_s": config.burst_rate_per_s,
        "burst_duration_s": config.burst_duration_s,
        "quiet_duration_s": config.quiet_duration_s,
        "trace_time_unit": config.trace_time_unit,
        "prompt_distribution": _coerce_distribution(config.prompt_distribution, DistributionSpec()).as_dict(),
        "answer_distribution": _coerce_distribution(config.answer_distribution, DistributionSpec()).as_dict(),
        "replicas": config.replicas,
        "tensor_parallel": config.tensor_parallel,
        "tp_efficiency": config.tp_efficiency,
        "prefill_replicas": config.prefill_replicas or config.replicas,
        "decode_replicas": config.decode_replicas or config.replicas,
        "disaggregate": bool(config.disaggregate),
        "prefill_tokens_per_s": config.prefill_tokens_per_s,
        "decode_tokens_per_s": config.decode_tokens_per_s,
        "draft_tokens_per_s": config.draft_tokens_per_s,
        "prefill_batch_overhead_s": config.prefill_batch_overhead_s,
        "decode_batch_overhead_s": config.decode_batch_overhead_s,
        "transfer_latency_s": config.transfer_latency_s,
        "transfer_tokens_per_s": config.transfer_tokens_per_s,
        "prefill_batch_gain": config.prefill_batch_gain,
        "decode_batch_gain": config.decode_batch_gain,
        "speculative_verification_factor": config.speculative_verification_factor,
        "max_batch_size": config.max_batch_size,
        "chunked_prefill_tokens": config.chunked_prefill_tokens,
        "max_decode_steps_before_prefill": config.max_decode_steps_before_prefill,
        "continuous_batching": config.continuous_batching,
        "kv_block_tokens": config.kv_block_tokens,
        "kv_capacity_blocks": config.kv_capacity_blocks,
        "kv_capacity_tokens": config.kv_capacity_tokens,
        "kv_bytes_per_token": config.kv_bytes_per_token,
        "eviction_policy": config.eviction_policy,
        "preemption_enabled": config.preemption_enabled,
        "prefix_cache_enabled": config.prefix_cache_enabled,
        "prefix_cache_capacity_blocks": config.prefix_cache_capacity_blocks,
        "prefix_cache_min_tokens": config.prefix_cache_min_tokens,
        "speculative_enabled": config.speculative_enabled,
        "speculative_draft_tokens": config.speculative_draft_tokens,
        "speculative_acceptance": _coerce_distribution(config.speculative_acceptance, DistributionSpec("fixed", value=0.8)).as_dict(),
        "slo_ttft_s": config.slo_ttft_s,
        "slo_itl_s": config.slo_itl_s,
        "slo_e2e_s": config.slo_e2e_s,
    }


def simulate_serving(
    trace_rows: Iterable[TraceRequest | Mapping[str, Any]] | None = None,
    assumptions: ServingAssumptions | Mapping[str, Any] | None = None,
    *,
    seed: int | None = None,
) -> SimulationResult:
    """Run the deterministic serving simulation.

    ``trace_rows`` may be a list of dictionaries, a ``csv.DictReader``, or
    normalized :class:`TraceRequest` objects.  If omitted, arrivals are
    generated according to ``assumptions.arrival_process``.  ``seed`` is an
    optional run-level override and does not mutate the caller's assumptions.
    """

    config = _coerce_assumptions(assumptions)
    actual_seed = config.seed if seed is None else int(seed)
    if seed is not None:
        config = ServingAssumptions(**{**config.__dict__, "seed": actual_seed})
    # Distribution coercion happens here as well for a directly constructed
    # dataclass that used numeric aliases in its fields.
    prompt_default = DistributionSpec("fixed", value=256)
    answer_default = DistributionSpec("fixed", value=64)
    config = ServingAssumptions(
        **{
            **config.__dict__,
            "prompt_distribution": _coerce_distribution(config.prompt_distribution, prompt_default),
            "answer_distribution": _coerce_distribution(config.answer_distribution, answer_default),
        }
    )
    rng = random.Random(actual_seed)
    if trace_rows is not None or config.replay_rows:
        input_mode = "replay"
    elif isinstance(config.arrival_process, ArrivalSpec):
        input_mode = str(config.arrival_process.kind)
    else:
        input_mode = str(config.arrival_process)
    normalized = normalize_trace_rows(trace_rows, config, seed=actual_seed) if trace_rows is not None else _generated_trace(config, rng)
    requests = _make_request_states(normalized)
    recorder = _EventRecorder()

    # Assign routes before recording arrival events so each event identifies
    # the selected replica.  Disaggregated routing is assigned in two stages.
    if config.disaggregate:
        prefill_count = max(1, int(config.prefill_replicas or config.replicas))
        prefill_pools = _assign_requests(requests, prefill_count, config, workload="prefill")
        for index, pool in enumerate(prefill_pools):
            for request in pool:
                request.prefill_replica = f"prefill-{index}"
    else:
        mixed_pools = _assign_requests(requests, max(1, config.replicas), config, workload="mixed")
        for index, pool in enumerate(mixed_pools):
            for request in pool:
                request.prefill_replica = f"replica-{index}"
                request.decode_replica = f"replica-{index}"
    _record_arrivals(requests, recorder)

    prefill_workers: list[_Worker] = []
    decode_workers: list[_Worker] = []
    if config.disaggregate:
        prefill_workers, decode_workers = _run_disaggregated(requests, config, recorder, rng)
        all_workers = [*prefill_workers, *decode_workers]
    else:
        all_workers = _run_monolithic(requests, config, recorder, rng)

    results = tuple(
        _request_result(request)
        for request in sorted(requests, key=lambda item: (item.trace.arrival_s, item.trace.request_id))
    )
    all_events = tuple(sorted((event for request in requests for event in request.timeline), key=lambda event: (event.time_s, event.sequence)))
    timelines = {request.request_id: request.timeline for request in results}

    ttft_values = [request.ttft_s for request in results if request.ttft_s is not None]
    itl_values = [value for request in results for value in request.inter_token_latency_s]
    e2e_values = [request.e2e_s for request in results if request.e2e_s is not None]
    completed = [request for request in results if request.status == "completed"]
    elapsed_start = min((request.arrival_s for request in results), default=0.0)
    elapsed_end = max(
        max((request.completion_s or request.first_token_s or request.arrival_s for request in results), default=elapsed_start),
        max((event.time_s for event in all_events), default=elapsed_start),
    )
    elapsed = max(0.0, elapsed_end - elapsed_start)
    slo_good = []
    for request in completed:
        good = True
        if config.slo_ttft_s is not None:
            good = good and request.ttft_s is not None and request.ttft_s <= float(config.slo_ttft_s)
        if config.slo_itl_s is not None and request.inter_token_latency_s:
            good = good and request.max_itl_s is not None and request.max_itl_s <= float(config.slo_itl_s)
        if config.slo_e2e_s is not None:
            good = good and request.e2e_s is not None and request.e2e_s <= float(config.slo_e2e_s)
        if good:
            slo_good.append(request)

    average_kv, peak_kv, kv_samples = _time_weighted_utilization(all_workers)
    speculation_attempted = 0
    speculation_accepted = 0
    for event in all_events:
        if event.event == "speculative_step":
            details = event.details
            speculation_attempted += int(details.get("attempted", 0))
            speculation_accepted += int(details.get("accepted", 0))
    prefix_candidates = [request for request in results if request.prefix_tokens > 0 or request.timeline and any(item.event == "prefix_cache_miss" for item in request.timeline)]
    prefix_hits = sum(1 for request in results if request.prefix_cache_hit)
    preemptions = sum(request.preemptions for request in results)
    evictions = sum(1 for event in all_events if event.event == "kv_eviction")
    output_tokens = sum(request.output_tokens for request in completed)
    metrics: dict[str, float | int | None] = {
        "requests": len(results),
        "completed_requests": len(completed),
        "failed_requests": sum(1 for request in results if request.status == "failed"),
        "wall_time_s": elapsed,
        "throughput_requests_per_s": len(completed) / elapsed if elapsed > 0 else 0.0,
        "throughput_output_tokens_per_s": output_tokens / elapsed if elapsed > 0 else 0.0,
        "output_tokens": output_tokens,
        "ttft_p50_s": _percentile(ttft_values, 50),
        "ttft_p90_s": _percentile(ttft_values, 90),
        "ttft_p99_s": _percentile(ttft_values, 99),
        "inter_token_latency_p50_s": _percentile(itl_values, 50),
        "inter_token_latency_p90_s": _percentile(itl_values, 90),
        "inter_token_latency_p99_s": _percentile(itl_values, 99),
        "itl_p50_s": _percentile(itl_values, 50),
        "itl_p90_s": _percentile(itl_values, 90),
        "itl_p99_s": _percentile(itl_values, 99),
        "e2e_p50_s": _percentile(e2e_values, 50),
        "e2e_p90_s": _percentile(e2e_values, 90),
        "e2e_p99_s": _percentile(e2e_values, 99),
        "slo_goodput_requests": len(slo_good),
        "slo_goodput_ratio": len(slo_good) / len(results) if results else 0.0,
        "slo_goodput_requests_per_s": len(slo_good) / elapsed if elapsed > 0 else 0.0,
        "kv_utilization_avg": average_kv,
        "kv_utilization_peak": peak_kv,
        "kv_utilization_p50": _percentile(kv_samples, 50),
        "prefix_cache_hits": prefix_hits,
        "prefix_cache_hit_rate": prefix_hits / len(prefix_candidates) if prefix_candidates else 0.0,
        "prefix_cache_evictions": evictions,
        "preemptions": preemptions,
        "speculative_attempted_tokens": speculation_attempted,
        "speculative_accepted_tokens": speculation_accepted,
        "speculative_acceptance_rate": speculation_accepted / speculation_attempted if speculation_attempted else None,
        "tensor_parallel": config.tensor_parallel,
        "replicas": config.replicas,
    }
    metrics.update({
        "p50_ttft_s": metrics["ttft_p50_s"],
        "p90_ttft_s": metrics["ttft_p90_s"],
        "p99_ttft_s": metrics["ttft_p99_s"],
        "throughput_tokens_per_s": metrics["throughput_output_tokens_per_s"],
    })
    distributions = {
        "ttft_s": _summary("ttft_s", ttft_values),
        "inter_token_latency_s": _summary("inter_token_latency_s", itl_values),
        "e2e_s": _summary("e2e_s", e2e_values),
        "prompt_tokens": _summary("prompt_tokens", (request.prompt_tokens for request in results)),
        "answer_tokens": _summary("answer_tokens", (request.answer_tokens for request in results)),
        "kv_utilization": _summary("kv_utilization", kv_samples),
    }
    replica_metrics: dict[str, Mapping[str, float | int | None]] = {}
    for worker in all_workers:
        worker_requests = [request for request in results if worker.replica_id in {request.prefill_replica, request.decode_replica}]
        worker_completed = [request for request in worker_requests if request.status == "completed"]
        worker_avg, worker_peak, _ = _time_weighted_utilization([worker])
        replica_metrics[worker.replica_id] = {
            "requests": len(worker_requests),
            "completed_requests": len(worker_completed),
            "capacity_blocks": worker.kv.capacity_blocks,
            "kv_utilization_avg": worker_avg,
            "kv_utilization_peak": worker_peak,
            "kv_evictions": worker.kv.evictions,
            "end_time_s": worker.end_time,
            "label": SIMULATED_LABEL,
        }

    notes = (
        "All latency, throughput, goodput, cache, and utilization values are simulated quantities.",
        "Prefill/decode rates, batching gains, KV capacity, cache policy, and SLO thresholds are explicit assumptions.",
        "Request timelines retain arrival, queue, prefill, decode, cache, preemption, transfer, and completion events.",
    )
    return SimulationResult(
        seed=actual_seed,
        simulated=True,
        assumptions=_effective_assumptions(config, actual_seed, input_mode),
        requests=results,
        timeline=all_events,
        timelines=timelines,
        distributions=distributions,
        metrics=metrics,
        metric_labels={key: SIMULATED_LABEL for key in metrics},
        replica_metrics=replica_metrics,
        notes=notes,
    )


def run_serving_simulation(
    trace_rows: Iterable[TraceRequest | Mapping[str, Any]] | None = None,
    assumptions: ServingAssumptions | Mapping[str, Any] | None = None,
    *,
    seed: int | None = None,
) -> SimulationResult:
    """Readable alias for :func:`simulate_serving`."""

    return simulate_serving(trace_rows, assumptions, seed=seed)


class ServingSimulator:
    """Small state-free facade for applications that prefer an object API."""

    def __init__(self, assumptions: ServingAssumptions | Mapping[str, Any] | None = None, *, seed: int | None = None) -> None:
        self.assumptions = _coerce_assumptions(assumptions)
        self.seed = self.assumptions.seed if seed is None else int(seed)

    def run(self, trace_rows: Iterable[TraceRequest | Mapping[str, Any]] | None = None) -> SimulationResult:
        return simulate_serving(trace_rows, self.assumptions, seed=self.seed)


# Short aliases make the reference convenient in notebooks without hiding the
# more descriptive names used in the public documentation.
SimulationConfig = ServingAssumptions
ServingConfig = ServingAssumptions
Request = TraceRequest
simulate = simulate_serving


__all__ = [
    "SIMULATED_LABEL",
    "ArrivalSpec",
    "DistributionSpec",
    "DistributionSummary",
    "RequestResult",
    "ServingAssumptions",
    "ServingConfig",
    "ServingSimulator",
    "SimulationConfig",
    "SimulationResult",
    "TimelineEvent",
    "TraceRequest",
    "Request",
    "load_trace_rows",
    "normalize_trace_rows",
    "run_serving_simulation",
    "simulate",
    "simulate_serving",
]
