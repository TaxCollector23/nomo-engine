"""Bounded, deterministic cycle-level evidence for Nomo integer graphs.

This module profiles the canonical :class:`nomo.runtime.qgraph.QGraph` using a
small abstract machine.  It counts integer work, memory references, a bounded
LRU cache, and spike/event activity.  The result is explicitly labelled
``simulated``.  It is useful for comparing designs and finding regressions;
it is not a physical latency, energy, or PPA measurement.

The public entry points are:

``CycleEmulator(config).run(qgraph, inputs, aux)``
    Run a graph already in memory.
``emulate(qgraph, inputs, aux, config)``
    Convenience wrapper around ``CycleEmulator``.
``run_artifact(artifact, config)``
    Load the stable JSON artifact/config boundary and run it.
``artifact_to_dict(qgraph, inputs, aux)`` / ``artifact_from_dict(...)``
    Serialize and restore a bounded emulation artifact without introducing a
    new dependency or changing the deployment runtime.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..runtime.qgraph import (
    QDecoder,
    QDense,
    QEncoder,
    QFromQ16,
    QGraph,
    QGuard,
    QLIF,
    QSymLinear,
    QToQ16,
    run as run_qgraph,
)
from ..symbolic.constraints import BoxBound, RotationalRateBound, SymbolicConstraint, ThrustLimit
from .schema import ARTIFACT_SCHEMA, CONFIG_SCHEMA, RESULT_SCHEMA


SUPPORTED_STAGE_KINDS = (
    "dense",
    "encoder",
    "lif",
    "decoder",
    "to_q16",
    "from_q16",
    "sym_linear",
    "guard",
)


class EmulationError(ValueError):
    """Base class for invalid or unsafe emulation requests."""


class UnsupportedGraphOperation(EmulationError):
    """Raised when a graph contains an operation outside the bounded backend."""


class EmulationLimitError(EmulationError):
    """Raised before execution would exceed a configured bound."""


class ArtifactError(EmulationError):
    """Raised when an artifact/config JSON document is malformed."""


def _integer(value: Any, name: str, *, minimum: Optional[int] = None,
             maximum: Optional[int] = None) -> int:
    if isinstance(value, bool):
        raise EmulationError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise EmulationError(f"{name} must be an integer") from exc
    if result != value:
        raise EmulationError(f"{name} must be an integer")
    if minimum is not None and result < minimum:
        raise EmulationError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise EmulationError(f"{name} must be <= {maximum}")
    return result


def _finite_number(value: Any, name: str, *, minimum: Optional[float] = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EmulationError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise EmulationError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise EmulationError(f"{name} must be >= {minimum}")
    return result


def _object(data: Optional[Mapping[str, Any]], name: str) -> Dict[str, Any]:
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise EmulationError(f"{name} must be an object")
    return dict(data)


@dataclass(frozen=True)
class EmulationLimits:
    """Hard bounds that keep a request deterministic and resource-bounded."""

    max_vectors: int = 64
    max_stages: int = 128
    max_timesteps: int = 4096
    max_graph_elements: int = 2_000_000
    max_memory_accesses: int = 5_000_000
    max_cycles: int = 1_000_000_000_000
    max_trace_events: int = 100_000

    def __post_init__(self) -> None:
        ceilings = {
            "max_vectors": 4096,
            "max_stages": 4096,
            "max_timesteps": 1_000_000,
            "max_graph_elements": 100_000_000,
            "max_memory_accesses": 1_000_000_000,
            "max_cycles": 10**15,
            "max_trace_events": 10_000_000,
        }
        for name, ceiling in ceilings.items():
            _integer(getattr(self, name), name, minimum=1, maximum=ceiling)

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "EmulationLimits":
        data = _object(data, "emulation limits")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise EmulationError(f"unknown emulation limit(s): {', '.join(unknown)}")
        values = {name: _integer(data[name], name, minimum=1) for name in data}
        return cls(**values)

    def to_dict(self) -> Dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class EmulationHardware:
    """Parameters for the abstract deterministic machine.

    These are cycle-model parameters, not a description of a particular chip.
    No wall-clock latency or energy value is derived from them.
    """

    macs_per_cycle: int = 16
    vector_lanes: int = 16
    memory_hit_cycles: int = 1
    memory_miss_cycles: int = 10
    cache_line_bytes: int = 64
    cache_capacity_lines: int = 256
    cache_write_allocate: bool = True
    lif_mode: str = "event_driven"

    def __post_init__(self) -> None:
        _integer(self.macs_per_cycle, "macs_per_cycle", minimum=1, maximum=1_000_000)
        _integer(self.vector_lanes, "vector_lanes", minimum=1, maximum=1_000_000)
        _integer(self.memory_hit_cycles, "memory_hit_cycles", minimum=1, maximum=1_000_000)
        _integer(self.memory_miss_cycles, "memory_miss_cycles", minimum=1, maximum=1_000_000)
        _integer(self.cache_line_bytes, "cache_line_bytes", minimum=1, maximum=4096)
        _integer(self.cache_capacity_lines, "cache_capacity_lines", minimum=1, maximum=1_000_000)
        if self.lif_mode not in ("event_driven", "dense"):
            raise EmulationError("lif_mode must be 'event_driven' or 'dense'")

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "EmulationHardware":
        data = _object(data, "emulation hardware")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise EmulationError(f"unknown emulation hardware option(s): {', '.join(unknown)}")
        if "cache_write_allocate" in data and not isinstance(data["cache_write_allocate"], bool):
            raise EmulationError("cache_write_allocate must be a boolean")
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class EmulationTrace:
    """Trace controls; traces are intervals, never one record per scalar MAC."""

    granularity: str = "stage"

    def __post_init__(self) -> None:
        if self.granularity not in ("stage", "timestep"):
            raise EmulationError("trace granularity must be 'stage' or 'timestep'")

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "EmulationTrace":
        data = _object(data, "emulation trace")
        unknown = sorted(set(data) - {"granularity"})
        if unknown:
            raise EmulationError(f"unknown emulation trace option(s): {', '.join(unknown)}")
        return cls(str(data.get("granularity", "stage")))

    def to_dict(self) -> Dict[str, str]:
        return {"granularity": self.granularity}


@dataclass(frozen=True)
class EmulationConfig:
    hardware: EmulationHardware = field(default_factory=EmulationHardware)
    limits: EmulationLimits = field(default_factory=EmulationLimits)
    trace: EmulationTrace = field(default_factory=EmulationTrace)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "EmulationConfig":
        data = _object(data, "emulation config")
        schema = data.get("schema")
        if schema is not None and schema != CONFIG_SCHEMA:
            raise EmulationError(f"unsupported emulation config schema {schema!r}")
        unknown = sorted(set(data) - {"schema", "hardware", "limits", "trace", "metadata", "inputs", "aux"})
        if unknown:
            raise EmulationError(f"unknown emulation config field(s): {', '.join(unknown)}")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise EmulationError("emulation config metadata must be an object")
        return cls(
            hardware=EmulationHardware.from_dict(data.get("hardware")),
            limits=EmulationLimits.from_dict(data.get("limits")),
            trace=EmulationTrace.from_dict(data.get("trace")),
            metadata=dict(metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": CONFIG_SCHEMA,
            "hardware": self.hardware.to_dict(),
            "limits": self.limits.to_dict(),
            "trace": self.trace.to_dict(),
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True)
class EmulationArtifact:
    graph: QGraph
    inputs: Optional[np.ndarray] = None
    aux: Optional[np.ndarray] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmulationResult:
    """JSON-ready result with no wall-clock fields or random run identifier."""

    schema: str
    backend: Mapping[str, Any]
    graph: Mapping[str, Any]
    configuration: Mapping[str, Any]
    summary: Mapping[str, Any]
    layers: Sequence[Mapping[str, Any]]
    trace: Sequence[Mapping[str, Any]]
    vectors: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    warnings: Sequence[str] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy({
            "schema": self.schema,
            "backend": self.backend,
            "graph": self.graph,
            "configuration": self.configuration,
            "summary": self.summary,
            "layers": list(self.layers),
            "vectors": list(self.vectors),
            "trace": list(self.trace),
            "warnings": list(self.warnings),
        })

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, separators=(",", ": "))


@dataclass(frozen=True)
class _Edge:
    kind: str
    n: int
    count: int
    element_bytes: int
    timesteps: Optional[int] = None


@dataclass(frozen=True)
class _StageShape:
    input_edge: _Edge
    output_edge: _Edge


@dataclass(frozen=True)
class _GraphInfo:
    stages: Tuple[_StageShape, ...]
    edges: Tuple[_Edge, ...]
    graph_elements: int


@dataclass(frozen=True)
class _Region:
    base: int
    count: int
    element_bytes: int
    label: str


@dataclass(frozen=True)
class _Layout:
    edges: Tuple[_Region, ...]
    stages: Tuple[Mapping[str, _Region], ...]
    aux: _Region


@dataclass
class _Cache:
    line_bytes: int
    capacity_lines: int
    lines: OrderedDict = field(default_factory=OrderedDict)
    hits: int = 0
    misses: int = 0

    def reset(self) -> None:
        self.lines.clear()
        self.hits = 0
        self.misses = 0

    def access(self, address: int, *, allocate: bool = True) -> bool:
        line = int(address) // self.line_bytes
        if line in self.lines:
            self.lines.move_to_end(line)
            self.hits += 1
            return True
        self.misses += 1
        if allocate:
            self.lines[line] = None
            self.lines.move_to_end(line)
            while len(self.lines) > self.capacity_lines:
                self.lines.popitem(last=False)
        return False


@dataclass
class _Memory:
    cache: _Cache
    write_allocate: bool
    reads: int = 0
    writes: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    start_hits: int = 0
    start_misses: int = 0

    def begin_scope(self) -> Tuple[int, int, int, int, int, int]:
        return (self.reads, self.writes, self.read_bytes, self.write_bytes, self.cache.hits, self.cache.misses)

    def delta(self, start: Tuple[int, int, int, int, int, int]) -> Dict[str, int]:
        r, w, rb, wb, h, m = start
        return {
            "memory_reads": self.reads - r,
            "memory_writes": self.writes - w,
            "memory_accesses": (self.reads - r) + (self.writes - w),
            "read_bytes": self.read_bytes - rb,
            "write_bytes": self.write_bytes - wb,
            "cache_hits": self.cache.hits - h,
            "cache_misses": self.cache.misses - m,
        }

    def read(self, region: _Region, index: int) -> None:
        if index < 0 or index >= region.count:
            raise EmulationError(f"internal read outside {region.label}")
        self.cache.access(region.base + index * region.element_bytes)
        self.reads += 1
        self.read_bytes += region.element_bytes

    def write(self, region: _Region, index: int) -> None:
        if index < 0 or index >= region.count:
            raise EmulationError(f"internal write outside {region.label}")
        self.cache.access(region.base + index * region.element_bytes, allocate=self.write_allocate)
        self.writes += 1
        self.write_bytes += region.element_bytes


@dataclass
class _Slice:
    timestep: int
    cycles: int
    memory_accesses: int
    cache_hits: int
    cache_misses: int
    event_count: Optional[int]
    spike_count: Optional[int]


@dataclass
class _Profile:
    compute_cycles: int
    memory_cycles: int
    op_count: int
    memory: Dict[str, int]
    event_count: Optional[int]
    spike_count: Optional[int]
    input_event_count: Optional[int]
    output_event_count: Optional[int]
    slices: List[_Slice] = field(default_factory=list)

    @property
    def cycles(self) -> int:
        return self.compute_cycles + self.memory_cycles


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        result = float(value)
        if not math.isfinite(result):
            raise ArtifactError("JSON metadata cannot contain non-finite numbers")
        return result
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ArtifactError("JSON metadata cannot contain non-finite numbers")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ArtifactError(f"value of type {type(value).__name__} is not JSON-safe")


def _array(data: Any, *, name: str, dtype: np.dtype, ndim: int) -> np.ndarray:
    try:
        arr = np.asarray(data)
    except Exception as exc:  # pragma: no cover - numpy owns the exact error text
        raise ArtifactError(f"{name} must be an array") from exc
    if arr.ndim != ndim:
        raise ArtifactError(f"{name} must have {ndim} dimension(s)")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ArtifactError(f"{name} must contain integers")
    try:
        cast = arr.astype(dtype, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ArtifactError(f"{name} cannot be represented as {dtype}") from exc
    if not np.array_equal(arr, cast):
        raise ArtifactError(f"{name} contains values outside {dtype}")
    return np.array(cast, copy=True)


def _edge(kind: str, n: int, *, timesteps: Optional[int] = None) -> _Edge:
    if kind == "spk":
        if timesteps is None:
            raise EmulationError("spike edge is missing its timestep count")
        return _Edge(kind, n, n * timesteps, 1, timesteps)
    return _Edge(kind, n, n, 1 if kind == "i8" else 4, None)


def _known_constraint(constraint: Any, *, stage_name: str) -> SymbolicConstraint:
    if not isinstance(constraint, SymbolicConstraint):
        raise UnsupportedGraphOperation(f"{stage_name}: guard constraint is not a supported SymbolicConstraint")
    for term in constraint.terms:
        if not isinstance(term, (BoxBound, ThrustLimit, RotationalRateBound)):
            raise UnsupportedGraphOperation(
                f"{stage_name}: unsupported symbolic constraint term {type(term).__name__}"
            )
    return constraint


def _validate_graph(qg: QGraph, limits: EmulationLimits) -> _GraphInfo:
    if not isinstance(qg, QGraph):
        raise EmulationError("emulator expects a nomo.runtime.qgraph.QGraph")
    if not isinstance(qg.name, str) or not qg.name:
        raise EmulationError("graph name must be a non-empty string")
    in_size = _integer(qg.in_size, "graph.in_size", minimum=1)
    out_size = _integer(qg.out_size, "graph.out_size", minimum=1)
    n_aux = _integer(qg.n_aux, "graph.n_aux", minimum=0)
    _finite_number(qg.in_scale, "graph.in_scale", minimum=0.0)
    if len(qg.stages) == 0:
        raise EmulationError("graph must contain at least one stage")
    if len(qg.stages) > limits.max_stages:
        raise EmulationLimitError(f"graph has {len(qg.stages)} stages; limit is {limits.max_stages}")

    current = _edge("i8", in_size)
    edges = [current]
    shapes: List[_StageShape] = []
    elements = current.count

    for index, st in enumerate(qg.stages):
        name = getattr(st, "name", f"stage-{index}")
        kind = getattr(st, "kind", None)
        if kind not in SUPPORTED_STAGE_KINDS:
            raise UnsupportedGraphOperation(
                f"stage {index} ({name!r}) has unsupported operation {kind!r}; "
                f"supported: {', '.join(SUPPORTED_STAGE_KINDS)}"
            )

        if isinstance(st, QDense):
            if current.kind != "i8":
                raise UnsupportedGraphOperation(f"{name}: dense requires an i8 input edge")
            W = _array(st.W, name=f"{name}.W", dtype=np.int8, ndim=2)
            b = _array(st.b, name=f"{name}.b", dtype=np.int32, ndim=1)
            if W.shape[1] != current.n or W.shape[0] != b.size:
                raise EmulationError(f"{name}: dense W/b shape does not match its input edge")
            if not 1 <= _integer(st.w_bits, f"{name}.w_bits") <= 8:
                raise EmulationError(f"{name}.w_bits must be between 1 and 8")
            _integer(st.m0, f"{name}.m0")
            _integer(st.shift, f"{name}.shift")
            output = _edge("i8", int(W.shape[0]))
            elements += int(W.size + b.size)
        elif isinstance(st, QEncoder):
            if current.kind != "i8":
                raise UnsupportedGraphOperation(f"{name}: rate encoder requires an i8 input edge")
            n = _integer(st.n, f"{name}.n", minimum=1)
            T = _integer(st.T, f"{name}.T", minimum=1, maximum=limits.max_timesteps)
            if n != current.n:
                raise EmulationError(f"{name}: encoder n={n} does not match input width {current.n}")
            _integer(st.theta, f"{name}.theta", minimum=1)
            output = _edge("spk", n, timesteps=T)
            elements += output.count
        elif isinstance(st, QLIF):
            if current.kind != "spk":
                raise UnsupportedGraphOperation(f"{name}: rate-LIF requires a spike raster input")
            W = _array(st.W, name=f"{name}.W", dtype=np.int8, ndim=2)
            b = _array(st.b, name=f"{name}.b", dtype=np.int32, ndim=1)
            T = _integer(st.T, f"{name}.T", minimum=1, maximum=limits.max_timesteps)
            if current.timesteps != T:
                raise EmulationError(f"{name}: LIF T={T} does not match input raster T={current.timesteps}")
            if W.shape[1] != current.n or W.shape[0] != b.size:
                raise EmulationError(f"{name}: LIF W/b shape does not match its input edge")
            _integer(st.theta, f"{name}.theta", minimum=1)
            _integer(st.leak_shift, f"{name}.leak_shift", minimum=0, maximum=62)
            _integer(st.v_bits, f"{name}.v_bits", minimum=2, maximum=62)
            _integer(st.w_bits, f"{name}.w_bits", minimum=1, maximum=8)
            output = _edge("spk", int(W.shape[0]), timesteps=T)
            elements += int(W.size + b.size + output.count)
        elif isinstance(st, QDecoder):
            if current.kind != "spk":
                raise UnsupportedGraphOperation(f"{name}: decoder requires a spike raster input")
            n = _integer(st.n, f"{name}.n", minimum=1)
            T = _integer(st.T, f"{name}.T", minimum=1, maximum=limits.max_timesteps)
            if current.n != n or current.timesteps != T:
                raise EmulationError(f"{name}: decoder shape does not match its input raster")
            if st.out not in ("i8", "q16"):
                raise UnsupportedGraphOperation(f"{name}: decoder output {st.out!r} is unsupported")
            output = _edge(st.out, n)
        elif isinstance(st, QToQ16):
            if current.kind != "i8":
                raise UnsupportedGraphOperation(f"{name}: to_q16 requires an i8 input edge")
            n = _integer(st.n, f"{name}.n", minimum=1)
            if n != current.n:
                raise EmulationError(f"{name}: n={n} does not match input width {current.n}")
            output = _edge("q16", n)
        elif isinstance(st, QFromQ16):
            if current.kind != "q16":
                raise UnsupportedGraphOperation(f"{name}: from_q16 requires a q16 input edge")
            n = _integer(st.n, f"{name}.n", minimum=1)
            if n != current.n:
                raise EmulationError(f"{name}: n={n} does not match input width {current.n}")
            _integer(st.m0, f"{name}.m0")
            _integer(st.shift, f"{name}.shift")
            output = _edge("i8", n)
        elif isinstance(st, QSymLinear):
            if current.kind != "q16":
                raise UnsupportedGraphOperation(f"{name}: symbolic linear stage requires a q16 input edge")
            phi = _array(st.Phi, name=f"{name}.Phi", dtype=np.int64, ndim=2)
            if phi.shape[1] != current.n:
                raise EmulationError(f"{name}: Phi input width does not match input edge")
            output = _edge("q16", int(phi.shape[0]))
            elements += int(phi.size)
        elif isinstance(st, QGuard):
            if current.kind != "q16":
                raise UnsupportedGraphOperation(f"{name}: guard requires a q16 input edge")
            n = _integer(st.n, f"{name}.n", minimum=1)
            n_aux_stage = _integer(st.n_aux, f"{name}.n_aux", minimum=0)
            constraint = _known_constraint(st.constraint, stage_name=name)
            if n != current.n or n != constraint.n_out:
                raise EmulationError(f"{name}: guard width does not match its input/constraint")
            if n_aux_stage != constraint.n_aux or n_aux_stage > n_aux:
                raise EmulationError(f"{name}: guard auxiliary width is inconsistent with graph.n_aux")
            for term in constraint.terms:
                if isinstance(term, BoxBound):
                    if any(c < 0 or c >= n for c in term.channels):
                        raise EmulationError(f"{name}: box constraint channel is outside the graph")
                elif isinstance(term, ThrustLimit):
                    if term.channel < 0 or term.channel >= n or term.aux_density_ratio < 0 or term.aux_density_ratio >= n_aux:
                        raise EmulationError(f"{name}: thrust constraint index is outside the graph")
                elif isinstance(term, RotationalRateBound):
                    if any(c < 0 or c >= n for c in term.channels) or any(i < 0 or i >= n_aux for i in term.aux_omega):
                        raise EmulationError(f"{name}: rotational constraint index is outside the graph")
            output = _edge("q16", n)
        else:  # pragma: no cover - the kind/type checks above are defensive
            raise UnsupportedGraphOperation(f"stage {index} ({name!r}) is not a supported QGraph stage")

        shapes.append(_StageShape(current, output))
        edges.append(output)
        current = output
        elements += output.count

    if current.kind != "q16" or current.n != out_size:
        raise UnsupportedGraphOperation("graph must terminate in a q16 output edge")
    if elements > limits.max_graph_elements:
        raise EmulationLimitError(
            f"graph contains {elements} runtime/constant elements; limit is {limits.max_graph_elements}"
        )
    return _GraphInfo(tuple(shapes), tuple(edges), elements)


def _align(value: int, alignment: int = 64) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _layout(qg: QGraph, info: _GraphInfo) -> _Layout:
    cursor = 0x10000
    edge_regions: List[_Region] = []
    for index, edge in enumerate(info.edges):
        cursor = _align(cursor)
        edge_regions.append(_Region(cursor, edge.count, edge.element_bytes, f"edge[{index}]"))
        cursor += max(1, edge.count * edge.element_bytes)

    stage_regions: List[Mapping[str, _Region]] = []
    for index, st in enumerate(qg.stages):
        regions: Dict[str, _Region] = {}

        def add(label: str, count: int, element_bytes: int) -> None:
            nonlocal cursor
            cursor = _align(cursor)
            regions[label] = _Region(cursor, int(count), int(element_bytes), f"stage[{index}].{label}")
            cursor += max(1, int(count) * int(element_bytes))

        if isinstance(st, QDense):
            add("weights", int(np.asarray(st.W).size), 1)
            add("bias", int(np.asarray(st.b).size), 4)
        elif isinstance(st, QEncoder):
            add("accumulator", st.n, 8)
        elif isinstance(st, QLIF):
            add("weights", int(np.asarray(st.W).size), 1)
            add("bias", int(np.asarray(st.b).size), 4)
            add("state", st.W.shape[0], 8)
        elif isinstance(st, QSymLinear):
            add("weights", int(np.asarray(st.Phi).size), 4)
        stage_regions.append(regions)
    cursor = _align(cursor)
    aux = _Region(cursor, max(1, qg.n_aux), 4, "aux_q16")
    return _Layout(tuple(edge_regions), tuple(stage_regions), aux)


def _estimate_request(info: _GraphInfo, qg: QGraph, vectors: int, cfg: EmulationConfig) -> None:
    max_accesses_per_vector = 0
    max_compute_per_vector = 0
    temporal_trace_per_vector = 0
    for st, shape in zip(qg.stages, info.stages):
        in_n = shape.input_edge.n
        out_n = shape.output_edge.n
        T = shape.input_edge.timesteps or shape.output_edge.timesteps or 1
        if isinstance(st, QDense):
            max_accesses_per_vector += int(st.W.size) * 2 + out_n * 2
            max_compute_per_vector += int(st.W.size) + out_n
        elif isinstance(st, QEncoder):
            # input read + accumulator read/write + raster write per value.
            max_accesses_per_vector += T * (in_n * 4)
            max_compute_per_vector += T * out_n * 2
            temporal_trace_per_vector += T
        elif isinstance(st, QLIF):
            max_accesses_per_vector += T * (in_n + int(st.W.size) + out_n * 4)
            max_compute_per_vector += T * (int(st.W.size) + out_n * 2)
            temporal_trace_per_vector += T
        elif isinstance(st, QDecoder):
            max_accesses_per_vector += T * in_n + out_n
            max_compute_per_vector += T * in_n + out_n
            temporal_trace_per_vector += T
        elif isinstance(st, (QToQ16, QFromQ16)):
            max_accesses_per_vector += in_n * 2
            max_compute_per_vector += in_n
        elif isinstance(st, QSymLinear):
            max_accesses_per_vector += int(st.Phi.size) + in_n * out_n + out_n
            max_compute_per_vector += int(st.Phi.size) + out_n
        elif isinstance(st, QGuard):
            max_accesses_per_vector += in_n * 2 + qg.n_aux
            max_compute_per_vector += in_n + max(1, 16 * len(st.constraint.terms))
    worst_accesses = max_accesses_per_vector * vectors
    if worst_accesses > cfg.limits.max_memory_accesses:
        raise EmulationLimitError(
            f"request may perform up to {worst_accesses} memory accesses; "
            f"limit is {cfg.limits.max_memory_accesses}"
        )
    worst_cycles = worst_accesses * cfg.hardware.memory_miss_cycles + max_compute_per_vector * vectors
    if worst_cycles > cfg.limits.max_cycles:
        raise EmulationLimitError(f"request may require up to {worst_cycles} cycles; limit is {cfg.limits.max_cycles}")
    trace_events = vectors * len(qg.stages)
    if cfg.trace.granularity == "timestep":
        trace_events += vectors * temporal_trace_per_vector
    if trace_events > cfg.limits.max_trace_events:
        raise EmulationLimitError(
            f"request may emit {trace_events} trace events; limit is {cfg.limits.max_trace_events}"
        )


def _coerce_vectors(data: Any, *, name: str, width: int, vectors_limit: int,
                    signed_bits: Optional[int] = None) -> np.ndarray:
    try:
        arr = np.asarray(data)
    except Exception as exc:
        raise EmulationError(f"{name} must be an array of vectors") from exc
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != width:
        raise EmulationError(f"{name} must have shape [vectors, {width}]")
    if arr.shape[0] < 1 or arr.shape[0] > vectors_limit:
        raise EmulationLimitError(f"{name} contains {arr.shape[0]} vectors; limit is {vectors_limit}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise EmulationError(f"{name} must contain integers")
    result = arr.astype(np.int64, copy=True)
    if signed_bits is not None:
        lo, hi = -(1 << (signed_bits - 1)), (1 << (signed_bits - 1)) - 1
        if result.min(initial=0) < lo or result.max(initial=0) > hi:
            raise EmulationError(f"{name} values must fit signed {signed_bits}-bit integers")
    return result


def _profile_stage(stage: Any, input_value: np.ndarray, output_value: np.ndarray,
                   regions: Mapping[str, _Region], input_region: _Region, output_region: _Region,
                   aux_region: _Region, memory: _Memory, hw: EmulationHardware) -> _Profile:
    start = memory.begin_scope()
    compute_cycles = 0
    op_count = 0
    event_count: Optional[int] = None
    spike_count: Optional[int] = None
    input_event_count: Optional[int] = None
    output_event_count: Optional[int] = None
    slices: List[_Slice] = []

    def finish_slice(timestep: int, compute: int, event_in: Optional[int], event_out: Optional[int], spikes: Optional[int], scope: Tuple[int, int, int, int, int, int]) -> None:
        delta = memory.delta(scope)
        mem_cycles = delta["cache_hits"] * hw.memory_hit_cycles + delta["cache_misses"] * hw.memory_miss_cycles
        slices.append(_Slice(timestep, int(compute + mem_cycles), delta["memory_accesses"],
                             delta["cache_hits"], delta["cache_misses"],
                             None if event_in is None and event_out is None else int((event_in or 0) + (event_out or 0)),
                             spikes))

    if isinstance(stage, QDense):
        no, ni = stage.W.shape
        for row in range(no):
            for col in range(ni):
                memory.read(input_region, col)
                memory.read(regions["weights"], row * ni + col)
            memory.read(regions["bias"], row)
            memory.write(output_region, row)
        op_count = int(no * ni + no)
        compute_cycles = _ceil_div(int(no * ni), hw.macs_per_cycle) + _ceil_div(int(no), hw.vector_lanes)
    elif isinstance(stage, QEncoder):
        n, T = stage.n, stage.T
        output = np.asarray(output_value).reshape(T, n)
        input_flat = np.asarray(input_value).reshape(n)
        for t in range(T):
            scope = memory.begin_scope()
            for i in range(n):
                memory.read(input_region, i)
                memory.read(regions["accumulator"], i)
                memory.write(regions["accumulator"], i)
                memory.write(output_region, t * n + i)
            out_events = int(np.count_nonzero(output[t]))
            compute = _ceil_div(n * 2, hw.vector_lanes)
            finish_slice(t, compute, None, out_events, out_events, scope)
            compute_cycles += compute
            op_count += n * 2
        output_event_count = int(np.count_nonzero(output))
        event_count = output_event_count
        spike_count = output_event_count
    elif isinstance(stage, QLIF):
        no, ni = stage.W.shape
        T = stage.T
        raster_in = np.asarray(input_value).reshape(T, ni)
        raster_out = np.asarray(output_value).reshape(T, no)
        total_in = total_out = 0
        for t in range(T):
            scope = memory.begin_scope()
            active = int(np.count_nonzero(raster_in[t]))
            for i in range(ni):
                memory.read(input_region, t * ni + i)
            if hw.lif_mode == "event_driven":
                for i in range(ni):
                    if raster_in[t, i] != 0:
                        for row in range(no):
                            memory.read(regions["weights"], row * ni + i)
                syn_ops = active * no
            else:
                for idx in range(no * ni):
                    memory.read(regions["weights"], idx)
                syn_ops = no * ni
            for row in range(no):
                memory.read(regions["bias"], row)
                memory.read(regions["state"], row)
                memory.write(regions["state"], row)
                memory.write(output_region, t * no + row)
            out_spikes = int(np.count_nonzero(raster_out[t]))
            compute = _ceil_div(syn_ops, hw.macs_per_cycle) + _ceil_div(no * 2, hw.vector_lanes)
            finish_slice(t, compute, active, out_spikes, out_spikes, scope)
            compute_cycles += compute
            op_count += syn_ops + no * 2
            total_in += active
            total_out += out_spikes
        input_event_count = total_in
        output_event_count = total_out
        event_count = total_out
        spike_count = total_out
    elif isinstance(stage, QDecoder):
        n, T = stage.n, stage.T
        raster = np.asarray(input_value).reshape(T, n)
        input_events = int(np.count_nonzero(raster))
        for t in range(T):
            scope = memory.begin_scope()
            for i in range(n):
                memory.read(input_region, t * n + i)
            if t == T - 1:
                for i in range(n):
                    memory.write(output_region, i)
            compute = _ceil_div(n, hw.macs_per_cycle) + (_ceil_div(n, hw.vector_lanes) if t == T - 1 else 0)
            finish_slice(t, compute, int(np.count_nonzero(raster[t])), None, int(np.count_nonzero(raster[t])), scope)
            compute_cycles += compute
            op_count += n + (n if t == T - 1 else 0)
        input_event_count = input_events
        event_count = input_events
        spike_count = input_events
    elif isinstance(stage, (QToQ16, QFromQ16)):
        n = int(stage.n)
        for i in range(n):
            memory.read(input_region, i)
            memory.write(output_region, i)
        op_count = n
        compute_cycles = _ceil_div(n, hw.vector_lanes)
    elif isinstance(stage, QSymLinear):
        no, ni = stage.Phi.shape
        for row in range(no):
            for col in range(ni):
                memory.read(input_region, col)
                memory.read(regions["weights"], row * ni + col)
            memory.write(output_region, row)
        op_count = int(no * ni + no)
        compute_cycles = _ceil_div(int(no * ni), hw.macs_per_cycle) + _ceil_div(int(no), hw.vector_lanes)
    elif isinstance(stage, QGuard):
        n = int(stage.n)
        for i in range(n):
            memory.read(input_region, i)
            memory.write(output_region, i)
        aux_indices: List[int] = []
        for term in stage.constraint.terms:
            if isinstance(term, ThrustLimit):
                aux_indices.append(term.aux_density_ratio)
            elif isinstance(term, RotationalRateBound):
                aux_indices.extend(term.aux_omega)
        for i in sorted(set(aux_indices)):
            memory.read(aux_region, i)
        op_count = n + max(1, 16 * len(stage.constraint.terms))
        compute_cycles = _ceil_div(op_count, hw.vector_lanes)
    else:  # pragma: no cover - graph validation rejects this first
        raise UnsupportedGraphOperation(f"unsupported stage {type(stage).__name__}")

    memory_delta = memory.delta(start)
    memory_cycles = memory_delta["cache_hits"] * hw.memory_hit_cycles + memory_delta["cache_misses"] * hw.memory_miss_cycles
    return _Profile(compute_cycles, memory_cycles, op_count, memory_delta,
                    event_count, spike_count, input_event_count, output_event_count, slices)


def _layer_report(index: int, stage: Any, shape: _StageShape, profile: _Profile, vector_count: int) -> Dict[str, Any]:
    accesses = profile.memory["memory_accesses"]
    hit_ratio = (profile.memory["cache_hits"] / accesses) if accesses else 1.0
    return {
        "index": index,
        "name": str(stage.name),
        "kind": str(stage.kind),
        "capability": "simulated",
        "vectors": vector_count,
        "input_kind": shape.input_edge.kind,
        "output_kind": shape.output_edge.kind,
        "input_elements_per_vector": shape.input_edge.count,
        "output_elements_per_vector": shape.output_edge.count,
        "cycles": int(profile.cycles),
        "compute_cycles": int(profile.compute_cycles),
        "memory_cycles": int(profile.memory_cycles),
        "op_count": int(profile.op_count),
        "memory_accesses": int(accesses),
        "memory_reads": int(profile.memory["memory_reads"]),
        "memory_writes": int(profile.memory["memory_writes"]),
        "read_bytes": int(profile.memory["read_bytes"]),
        "write_bytes": int(profile.memory["write_bytes"]),
        "cache_hits": int(profile.memory["cache_hits"]),
        "cache_misses": int(profile.memory["cache_misses"]),
        "cache_hit_ratio": float(hit_ratio),
        "input_event_count": profile.input_event_count,
        "output_event_count": profile.output_event_count,
        "event_count": profile.event_count,
        "spike_count": profile.spike_count,
    }


class CycleEmulator:
    """Run a QGraph through the deterministic bounded cycle/cache model."""

    BACKEND_ID = "nomo-cycle-emulator"
    BACKEND_VERSION = "1"

    def __init__(self, config: Optional[EmulationConfig] = None) -> None:
        self.config = config or EmulationConfig()

    def run(self, qgraph: QGraph, inputs: Any, aux: Any = None) -> EmulationResult:
        info = _validate_graph(qgraph, self.config.limits)
        vectors = _coerce_vectors(inputs, name="inputs", width=int(qgraph.in_size),
                                   vectors_limit=self.config.limits.max_vectors, signed_bits=8)
        if qgraph.n_aux:
            if aux is None:
                raise EmulationError(f"graph requires {qgraph.n_aux} auxiliary Q16.16 values per vector")
            aux_vectors = _coerce_vectors(aux, name="aux", width=int(qgraph.n_aux),
                                          vectors_limit=self.config.limits.max_vectors)
            if aux_vectors.shape[0] != vectors.shape[0]:
                raise EmulationError("aux vector count must match inputs")
            if aux_vectors.min(initial=0) < -(1 << 31) or aux_vectors.max(initial=0) > (1 << 31) - 1:
                raise EmulationError("aux values must fit signed Q16.16 int32")
        else:
            if aux is not None:
                aux_vectors = _coerce_vectors(aux, name="aux", width=0,
                                              vectors_limit=self.config.limits.max_vectors)
                if aux_vectors.shape[0] != vectors.shape[0]:
                    raise EmulationError("aux vector count must match inputs")
            else:
                aux_vectors = np.empty((vectors.shape[0], 0), dtype=np.int64)
        _estimate_request(info, qgraph, int(vectors.shape[0]), self.config)
        layout = _layout(qgraph, info)
        cache = _Cache(self.config.hardware.cache_line_bytes, self.config.hardware.cache_capacity_lines)
        memory = _Memory(cache, self.config.hardware.cache_write_allocate)

        aggregate: List[Optional[Dict[str, Any]]] = [None] * len(qgraph.stages)
        trace: List[Dict[str, Any]] = []
        vector_reports: List[Dict[str, Any]] = []
        output_digest = hashlib.sha256()
        total_cycles = total_accesses = total_hits = total_misses = 0
        total_events = total_spikes = 0

        for vector_index, input_vector in enumerate(vectors):
            cache.reset()
            stage_values: List[Tuple[str, str, np.ndarray]] = []
            aux_row = None if qgraph.n_aux == 0 else aux_vectors[vector_index].tolist()
            output = run_qgraph(qgraph, input_vector, aux_row, trace=stage_values)
            values = [np.asarray(input_vector, dtype=np.int64)] + [np.asarray(item[2], dtype=np.int64) for item in stage_values]
            if len(values) != len(qgraph.stages) + 1:
                raise EmulationError("integer runtime did not return one value edge per stage")
            output_digest.update(np.asarray(output, dtype=np.int32).tobytes(order="C"))

            vector_cycle = 0
            vector_accesses = vector_hits = vector_misses = 0
            vector_events = vector_spikes = 0
            vector_trace_start = len(trace)
            for index, (stage, shape) in enumerate(zip(qgraph.stages, info.stages)):
                profile = _profile_stage(stage, values[index], values[index + 1], layout.stages[index],
                                         layout.edges[index], layout.edges[index + 1], layout.aux,
                                         memory, self.config.hardware)
                start_cycle = vector_cycle
                end_cycle = start_cycle + profile.cycles
                report = _layer_report(index, stage, shape, profile, 1)
                if aggregate[index] is None:
                    aggregate[index] = report
                else:
                    current = aggregate[index]
                    assert current is not None
                    for key in ("cycles", "compute_cycles", "memory_cycles", "op_count", "memory_accesses",
                                "memory_reads", "memory_writes", "read_bytes", "write_bytes", "cache_hits",
                                "cache_misses"):
                        current[key] += report[key]
                    current["vectors"] += 1
                    current["cache_hit_ratio"] = current["cache_hits"] / max(1, current["memory_accesses"])
                    for key in ("input_event_count", "output_event_count", "event_count", "spike_count"):
                        if report[key] is not None:
                            current[key] = (current[key] or 0) + report[key]

                trace.append({
                    "sequence": len(trace),
                    "vector": vector_index,
                    "stage_index": index,
                    "stage": str(stage.name),
                    "kind": str(stage.kind),
                    "event_type": "stage",
                    "phase": "execute",
                    "cycle_start": int(start_cycle),
                    "cycle_end": int(end_cycle),
                    "duration_cycles": int(profile.cycles),
                    "memory_accesses": int(report["memory_accesses"]),
                    "cache_hits": int(report["cache_hits"]),
                    "cache_misses": int(report["cache_misses"]),
                    "input_event_count": report["input_event_count"],
                    "output_event_count": report["output_event_count"],
                    "event_count": report["event_count"],
                    "spike_count": report["spike_count"],
                    "capability": "simulated",
                })
                if self.config.trace.granularity == "timestep":
                    slice_cycle = start_cycle
                    for time_slice in profile.slices:
                        trace.append({
                            "sequence": len(trace),
                            "vector": vector_index,
                            "stage_index": index,
                            "stage": str(stage.name),
                            "kind": str(stage.kind),
                            "event_type": "timestep",
                            "phase": "execute",
                            "timestep": int(time_slice.timestep),
                            "cycle_start": int(slice_cycle),
                            "cycle_end": int(slice_cycle + time_slice.cycles),
                            "duration_cycles": int(time_slice.cycles),
                            "memory_accesses": int(time_slice.memory_accesses),
                            "cache_hits": int(time_slice.cache_hits),
                            "cache_misses": int(time_slice.cache_misses),
                            "event_count": time_slice.event_count,
                            "spike_count": time_slice.spike_count,
                            "capability": "simulated",
                        })
                        slice_cycle += time_slice.cycles
                    if profile.slices and slice_cycle != end_cycle:
                        raise EmulationError(f"internal trace cycle mismatch at stage {index}")

                vector_cycle = end_cycle
                vector_accesses += int(report["memory_accesses"])
                vector_hits += int(report["cache_hits"])
                vector_misses += int(report["cache_misses"])
                if report["event_count"] is not None:
                    vector_events += int(report["event_count"])
                if report["spike_count"] is not None:
                    vector_spikes += int(report["spike_count"])

            vector_trace_end = len(trace)
            if vector_cycle > self.config.limits.max_cycles:
                raise EmulationLimitError(f"vector {vector_index} exceeds the {self.config.limits.max_cycles}-cycle limit")
            vector_reports.append({
                "index": vector_index,
                "cycles": int(vector_cycle),
                "memory_accesses": int(vector_accesses),
                "cache_hits": int(vector_hits),
                "cache_misses": int(vector_misses),
                "event_count": int(vector_events),
                "spike_count": int(vector_spikes),
                "trace_start": vector_trace_start,
                "trace_end": vector_trace_end,
                "output_sha256": hashlib.sha256(np.asarray(output, dtype=np.int32).tobytes(order="C")).hexdigest(),
                "capability": "simulated",
            })
            total_cycles += vector_cycle
            total_accesses += vector_accesses
            total_hits += vector_hits
            total_misses += vector_misses
            total_events += vector_events
            total_spikes += vector_spikes

        if total_cycles > self.config.limits.max_cycles:
            raise EmulationLimitError(f"request exceeds the {self.config.limits.max_cycles}-cycle limit")
        if len(trace) > self.config.limits.max_trace_events:
            raise EmulationLimitError(f"request emitted {len(trace)} trace events; limit is {self.config.limits.max_trace_events}")
        if total_accesses > self.config.limits.max_memory_accesses:
            raise EmulationLimitError(
                f"request performed {total_accesses} memory accesses; limit is {self.config.limits.max_memory_accesses}"
            )

        layers = [item for item in aggregate if item is not None]
        layer_cycles = sum(int(item["cycles"]) for item in layers)
        layer_accesses = sum(int(item["memory_accesses"]) for item in layers)
        layer_hits = sum(int(item["cache_hits"]) for item in layers)
        layer_misses = sum(int(item["cache_misses"]) for item in layers)
        stage_trace_duration = sum(int(event["duration_cycles"]) for event in trace if event["event_type"] == "stage")
        if layer_cycles != total_cycles or layer_accesses != total_accesses or layer_hits != total_hits or layer_misses != total_misses:
            raise EmulationError("internal aggregate accounting mismatch")
        if stage_trace_duration != total_cycles:
            raise EmulationError("internal stage trace accounting mismatch")

        hit_ratio = total_hits / max(1, total_accesses)
        graph_summary = {
            "name": qgraph.name,
            "in_size": int(qgraph.in_size),
            "out_size": int(qgraph.out_size),
            "n_aux": int(qgraph.n_aux),
            "stage_count": len(qgraph.stages),
            "stages": [{"index": i, "name": str(st.name), "kind": str(st.kind)} for i, st in enumerate(qgraph.stages)],
            "supported_stage_kinds": list(SUPPORTED_STAGE_KINDS),
            "meta": _json_safe(qgraph.meta),
        }
        metrics = {
            "cycles": {"value": int(total_cycles), "capability": "simulated", "available": True},
            "memory_accesses": {"value": int(total_accesses), "capability": "simulated", "available": True},
            "cache_hit_ratio": {"value": float(hit_ratio), "capability": "simulated", "available": True},
            "energy_uj": {
                "value": None,
                "capability": "proxy",
                "available": False,
                "note": "requires a trusted physical measurement agent; this emulator does not estimate energy",
            },
            "latency_ms": {
                "value": None,
                "capability": "proxy",
                "available": False,
                "note": "cycle counts are simulated; wall-clock latency requires a target-specific measurement",
            },
        }
        summary = {
            "vector_count": len(vector_reports),
            "cycles": int(total_cycles),
            "memory_accesses": int(total_accesses),
            "memory_reads": int(sum(int(item["memory_reads"]) for item in layers)),
            "memory_writes": int(sum(int(item["memory_writes"]) for item in layers)),
            "cache_hits": int(total_hits),
            "cache_misses": int(total_misses),
            "cache_hit_ratio": float(hit_ratio),
            "event_count": int(total_events),
            "spike_count": int(total_spikes),
            "output_sha256": output_digest.hexdigest(),
            "metrics": metrics,
            "cycle_accounting": {
                "layer_cycles": int(layer_cycles),
                "stage_trace_duration_cycles": int(stage_trace_duration),
                "balanced": True,
            },
            "measurement": {
                "capability": "simulated",
                "physical_measurement": False,
                "note": "No physical device, SystemC, Verilator, or Gem5 execution occurred.",
            },
        }
        backend = {
            "id": self.BACKEND_ID,
            "version": self.BACKEND_VERSION,
            "capability": "simulated",
            "physical_measurement": False,
            "description": "Deterministic bounded cycle/cache model over the Nomo integer QGraph.",
            "integration_boundary": "Use nomo.hardware.hitl for target-specific measured latency and energy.",
        }
        return EmulationResult(
            RESULT_SCHEMA,
            backend,
            graph_summary,
            self.config.to_dict(),
            summary,
            tuple(layers),
            tuple(trace),
            tuple(vector_reports),
            (
                "Simulated cycle/cache evidence only; it is not a physical hardware measurement.",
                "Energy and wall-clock latency are intentionally unavailable until a target measurement is supplied.",
            ),
        )


def emulate(qgraph: QGraph, inputs: Any, aux: Any = None,
            config: Optional[EmulationConfig] = None) -> EmulationResult:
    """Convenience library entry point for the bounded simulated backend."""
    return CycleEmulator(config).run(qgraph, inputs, aux)


simulate = emulate


def _constraint_to_dict(constraint: SymbolicConstraint) -> Dict[str, Any]:
    terms: List[Dict[str, Any]] = []
    for term in constraint.terms:
        if isinstance(term, BoxBound):
            terms.append({"kind": "box", "channels": list(term.channels), "lo": list(term.lo), "hi": list(term.hi)})
        elif isinstance(term, ThrustLimit):
            terms.append({"kind": "thrust", "channel": term.channel, "t_max_n": term.t_max_n,
                          "aux_density_ratio": term.aux_density_ratio})
        elif isinstance(term, RotationalRateBound):
            terms.append({"kind": "rotational_rate", "channels": list(term.channels),
                          "inertia": list(term.inertia), "dt": term.dt, "w_max": term.w_max,
                          "tau_max": term.tau_max, "aux_omega": list(term.aux_omega)})
        else:
            raise UnsupportedGraphOperation(f"cannot serialize constraint term {type(term).__name__}")
    return {"id": constraint.id, "terms": terms, "n_out": constraint.n_out, "n_aux": constraint.n_aux,
            "description": constraint.description}


def _constraint_from_dict(data: Mapping[str, Any]) -> SymbolicConstraint:
    if not isinstance(data, Mapping):
        raise ArtifactError("guard constraint must be an object")
    terms: List[Any] = []
    for term in data.get("terms", []):
        if not isinstance(term, Mapping):
            raise ArtifactError("constraint terms must be objects")
        kind = term.get("kind")
        if kind == "box":
            terms.append(BoxBound(tuple(int(v) for v in term["channels"]), tuple(float(v) for v in term["lo"]),
                                  tuple(float(v) for v in term["hi"])))
        elif kind == "thrust":
            terms.append(ThrustLimit(int(term["channel"]), float(term["t_max_n"]), int(term["aux_density_ratio"])))
        elif kind == "rotational_rate":
            terms.append(RotationalRateBound(tuple(int(v) for v in term["channels"]),
                                              tuple(float(v) for v in term["inertia"]), float(term["dt"]),
                                              float(term["w_max"]), float(term["tau_max"]),
                                              tuple(int(v) for v in term["aux_omega"])))
        else:
            raise UnsupportedGraphOperation(f"unsupported serialized constraint term {kind!r}")
    return SymbolicConstraint(str(data["id"]), tuple(terms), int(data["n_out"]), int(data.get("n_aux", 0)),
                              str(data.get("description", "")))


def qgraph_to_dict(qgraph: QGraph) -> Dict[str, Any]:
    """Serialize a QGraph into the stable graph portion of an artifact."""
    stages: List[Dict[str, Any]] = []
    for stage in qgraph.stages:
        item: Dict[str, Any] = {"kind": stage.kind, "name": stage.name}
        if isinstance(stage, QDense):
            item.update({"W": np.asarray(stage.W).tolist(), "b": np.asarray(stage.b).tolist(), "m0": stage.m0,
                         "shift": stage.shift, "relu": stage.relu, "w_bits": stage.w_bits,
                         "w_scale": stage.w_scale, "in_scale": stage.in_scale, "out_scale": stage.out_scale})
        elif isinstance(stage, QEncoder):
            item.update({"theta": stage.theta, "T": stage.T, "n": stage.n, "in_scale": stage.in_scale, "lam": stage.lam})
        elif isinstance(stage, QLIF):
            item.update({"W": np.asarray(stage.W).tolist(), "b": np.asarray(stage.b).tolist(), "theta": stage.theta,
                         "leak_shift": stage.leak_shift, "v_bits": stage.v_bits, "T": stage.T, "w_bits": stage.w_bits,
                         "v_scale": stage.v_scale, "lam_in": stage.lam_in, "lam_out": stage.lam_out, "plastic": stage.plastic})
        elif isinstance(stage, QDecoder):
            item.update({"T": stage.T, "n": stage.n, "out": stage.out, "m0": stage.m0, "shift": stage.shift,
                         "k_q16": stage.k_q16, "lam": stage.lam, "out_scale": stage.out_scale})
        elif isinstance(stage, QToQ16):
            item.update({"k_q16": stage.k_q16, "n": stage.n, "in_scale": stage.in_scale})
        elif isinstance(stage, QFromQ16):
            item.update({"m0": stage.m0, "shift": stage.shift, "n": stage.n, "out_scale": stage.out_scale})
        elif isinstance(stage, QSymLinear):
            item.update({"Phi": np.asarray(stage.Phi).tolist(), "substitute_id": stage.substitute_id})
        elif isinstance(stage, QGuard):
            item.update({"n": stage.n, "n_aux": stage.n_aux, "constraint": _constraint_to_dict(stage.constraint)})
        else:
            raise UnsupportedGraphOperation(f"cannot serialize unsupported stage {type(stage).__name__}")
        stages.append(_json_safe(item))
    return {"name": qgraph.name, "in_size": qgraph.in_size, "in_scale": qgraph.in_scale,
            "out_size": qgraph.out_size, "n_aux": qgraph.n_aux, "meta": _json_safe(qgraph.meta), "stages": stages}


def artifact_to_dict(qgraph: QGraph, inputs: Any = None, aux: Any = None,
                     metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Create a stable JSON-ready artifact document."""
    document: Dict[str, Any] = {"schema": ARTIFACT_SCHEMA, "graph": qgraph_to_dict(qgraph)}
    if inputs is not None:
        document["inputs"] = _json_safe(np.asarray(inputs))
    if aux is not None:
        document["aux"] = _json_safe(np.asarray(aux))
    if metadata:
        document["metadata"] = _json_safe(dict(metadata))
    return document


def _stage_from_dict(item: Mapping[str, Any]) -> Any:
    kind = item.get("kind")
    name = str(item.get("name", kind or "stage"))
    if kind == "dense":
        return QDense(name, _array(item["W"], name=f"{name}.W", dtype=np.int8, ndim=2),
                      _array(item["b"], name=f"{name}.b", dtype=np.int32, ndim=1),
                      int(item["m0"]), int(item["shift"]), bool(item["relu"]), int(item.get("w_bits", 8)),
                      float(item.get("w_scale", 1.0)), float(item.get("in_scale", 1.0)), float(item.get("out_scale", 1.0)))
    if kind == "encoder":
        return QEncoder(name, int(item["theta"]), int(item["T"]), int(item["n"]), float(item.get("in_scale", 1.0)),
                        float(item.get("lam", 1.0)))
    if kind == "lif":
        return QLIF(name, _array(item["W"], name=f"{name}.W", dtype=np.int8, ndim=2),
                    _array(item["b"], name=f"{name}.b", dtype=np.int32, ndim=1), int(item["theta"]),
                    int(item["leak_shift"]), int(item["v_bits"]), int(item["T"]), int(item.get("w_bits", 4)),
                    float(item.get("v_scale", 1.0)), float(item.get("lam_in", 1.0)), float(item.get("lam_out", 1.0)),
                    bool(item.get("plastic", False)))
    if kind == "decoder":
        return QDecoder(name, int(item["T"]), int(item["n"]), str(item["out"]), int(item.get("m0", 0)),
                        int(item.get("shift", 0)), int(item.get("k_q16", 0)), float(item.get("lam", 1.0)),
                        float(item.get("out_scale", 1.0)))
    if kind == "to_q16":
        return QToQ16(name, int(item["k_q16"]), int(item["n"]), float(item.get("in_scale", 1.0)))
    if kind == "from_q16":
        return QFromQ16(name, int(item["m0"]), int(item["shift"]), int(item["n"]), float(item.get("out_scale", 1.0)))
    if kind == "sym_linear":
        return QSymLinear(name, _array(item["Phi"], name=f"{name}.Phi", dtype=np.int64, ndim=2),
                          str(item.get("substitute_id", "")))
    if kind == "guard":
        return QGuard(name, _constraint_from_dict(item["constraint"]), int(item["n"]), int(item["n_aux"]))
    raise UnsupportedGraphOperation(f"unsupported serialized graph operation {kind!r}")


def qgraph_from_dict(data: Mapping[str, Any]) -> QGraph:
    """Restore and structurally validate the graph portion of an artifact."""
    if not isinstance(data, Mapping):
        raise ArtifactError("artifact.graph must be an object")
    stages_data = data.get("stages")
    if not isinstance(stages_data, list):
        raise ArtifactError("artifact.graph.stages must be an array")
    try:
        stages = [_stage_from_dict(item) for item in stages_data]
        return QGraph(str(data["name"]), stages, int(data["in_size"]), float(data["in_scale"]),
                      int(data["out_size"]), int(data.get("n_aux", 0)), dict(data.get("meta", {})))
    except KeyError as exc:
        raise ArtifactError(f"artifact graph is missing {exc.args[0]!r}") from exc
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"artifact graph is malformed: {exc}") from exc


def artifact_from_dict(data: Mapping[str, Any]) -> EmulationArtifact:
    if not isinstance(data, Mapping):
        raise ArtifactError("emulation artifact must be an object")
    if data.get("schema") != ARTIFACT_SCHEMA:
        raise ArtifactError(f"artifact schema must be {ARTIFACT_SCHEMA!r}")
    graph = qgraph_from_dict(data.get("graph", {}))
    inputs = None if "inputs" not in data else np.asarray(data["inputs"])
    aux = None if "aux" not in data else np.asarray(data["aux"])
    metadata = data.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ArtifactError("artifact.metadata must be an object")
    return EmulationArtifact(graph, inputs, aux, dict(metadata))


def _load_json(value: Any, *, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    try:
        payload = json.loads(Path(value).read_text())
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"could not read {name} JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ArtifactError(f"{name} JSON root must be an object")
    return payload


def run_artifact(artifact: Any, config: Any = None, *, inputs: Any = None, aux: Any = None) -> EmulationResult:
    """Run an artifact JSON object/path with a config object/path.

    Inputs and auxiliary vectors may live in the artifact or in the config;
    explicit keyword arguments take precedence.  The graph itself is always
    restored through the stable artifact schema before execution.
    """
    artifact_doc = _load_json(artifact, name="artifact")
    loaded = artifact_from_dict(artifact_doc)
    config_doc = {} if config is None else _load_json(config, name="config")
    cfg = EmulationConfig.from_dict(config_doc)
    selected_inputs = inputs if inputs is not None else config_doc.get("inputs", loaded.inputs)
    selected_aux = aux if aux is not None else config_doc.get("aux", loaded.aux)
    if selected_inputs is None:
        raise EmulationError("no inputs supplied; put inputs in the artifact/config or pass inputs=")
    return CycleEmulator(cfg).run(loaded.graph, selected_inputs, selected_aux)


# Short aliases for callers that prefer a noun-like serializer name.
serialize_artifact = artifact_to_dict
deserialize_artifact = artifact_from_dict
