"""Hardware/deployment co-search.

This module adds an explicit accelerator-architecture dimension next to the
existing :class:`~nomo.search.genome.Genome`.  It is intentionally a bounded
co-search, not neural-topology NAS: a deployment genome is held fixed while
the search explores PE array shape/count, SRAM, memory bandwidth, and
datapath precision.

The calculations here are analytic priors.  They are useful for making the
architecture trade-off visible and for narrowing a measurement plan; they are
not a claim of measured silicon PPA.  ``evidence_sources`` is carried through
every result so a caller can distinguish these priors from an oracle accuracy
result or future hardware measurements.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .pareto import asf_select, crowding_distance, fast_non_dominated_sort, normalise


CO_DESIGN_CAPABILITY = "hardware/deployment co-search"
"""Stable capability label.  This is not a neural-topology NAS claim."""

CO_DESIGN_OBJECTIVES = ("energy_j", "latency_s", "accuracy_loss_pp", "area_proxy")
"""Objective order used by ``CoDesignEvaluation.F`` and its wire form."""

EVIDENCE_ANALYTIC = "analytic_prior"
EVIDENCE_MODEL = "model_graph"
EVIDENCE_DEPLOYMENT_PROXY = "deployment_proxy"
EVIDENCE_ORACLE = "deployment_oracle"

_PRECISION_BITS = (1, 2, 4, 8, 16, 32, 64)
_MAX_ARRAY_DIMENSION = 4096
_MAX_PE_COUNT = 1_048_576
_MIN_SRAM_BYTES = 256
_MAX_SRAM_BYTES = 1 << 40
_MAX_BANDWIDTH_BYTES_S = 1.0e15
_MAX_CANDIDATES = 100_000


def _int_value(name: str, value: Any, *, minimum: int, maximum: int) -> int:
    """Validate an integer hardware field without silently truncating it."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    out = int(value)
    if out < minimum or out > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {out}")
    return out


def _real_value(name: str, value: Any, *, minimum: float, maximum: float) -> float:
    """Validate a finite positive hardware field."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    out = float(value)
    if not math.isfinite(out) or out < minimum or out > maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}], got {value!r}")
    return out


def _budget_value(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a positive finite number or infinity, got {value!r}")
    out = float(value)
    if math.isnan(out) or out <= 0.0:
        raise ValueError(f"{name} must be positive or infinity, got {value!r}")
    if not math.isinf(out) and not math.isfinite(out):
        raise ValueError(f"{name} must be finite or infinity, got {value!r}")
    return out


def _unique_sorted(values: Iterable[Any], name: str) -> Tuple[Any, ...]:
    values = tuple(values)
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    try:
        return tuple(sorted(set(values)))
    except TypeError as exc:
        raise ValueError(f"{name} must contain comparable scalar values") from exc


def _bandwidth_key(value: float) -> str:
    return format(float(value), ".12g")


@dataclass(frozen=True, slots=True)
class AcceleratorArchitecture:
    """One bounded accelerator point in the co-design search space.

    ``pe_rows * pe_cols`` is the parallel PE count.  SRAM is the shared
    on-chip working memory available to this accelerator candidate and
    ``memory_bandwidth_bytes_s`` is its sustained analytic bandwidth prior.
    """

    pe_rows: int
    pe_cols: int
    sram_bytes: int
    memory_bandwidth_bytes_s: float
    precision_bits: int

    def __post_init__(self) -> None:
        rows = _int_value("pe_rows", self.pe_rows, minimum=1, maximum=_MAX_ARRAY_DIMENSION)
        cols = _int_value("pe_cols", self.pe_cols, minimum=1, maximum=_MAX_ARRAY_DIMENSION)
        sram = _int_value("sram_bytes", self.sram_bytes, minimum=_MIN_SRAM_BYTES, maximum=_MAX_SRAM_BYTES)
        bw = _real_value("memory_bandwidth_bytes_s", self.memory_bandwidth_bytes_s,
                         minimum=1.0, maximum=_MAX_BANDWIDTH_BYTES_S)
        precision = _int_value("precision_bits", self.precision_bits, minimum=1, maximum=64)
        if precision not in _PRECISION_BITS:
            raise ValueError(f"precision_bits must be one of {_PRECISION_BITS}, got {precision}")
        if rows * cols > _MAX_PE_COUNT:
            raise ValueError(f"PE array has {rows * cols} PEs; maximum is {_MAX_PE_COUNT}")
        object.__setattr__(self, "pe_rows", rows)
        object.__setattr__(self, "pe_cols", cols)
        object.__setattr__(self, "sram_bytes", sram)
        object.__setattr__(self, "memory_bandwidth_bytes_s", bw)
        object.__setattr__(self, "precision_bits", precision)

    @property
    def pe_count(self) -> int:
        return self.pe_rows * self.pe_cols

    @property
    def array_shape(self) -> Tuple[int, int]:
        return self.pe_rows, self.pe_cols

    @property
    def key(self) -> str:
        return (f"pe{self.pe_rows}x{self.pe_cols}-sram{self.sram_bytes}"
                f"-bw{_bandwidth_key(self.memory_bandwidth_bytes_s)}"
                f"-p{self.precision_bits}")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "AcceleratorArchitecture":
        """Build a candidate from a strict mapping, rejecting typos and aliases.

        ``bandwidth_bytes_s`` is accepted as a short alias for the canonical
        ``memory_bandwidth_bytes_s`` field because it is common in hardware
        configuration files.
        """
        allowed = {"pe_rows", "pe_cols", "sram_bytes", "memory_bandwidth_bytes_s",
                   "bandwidth_bytes_s", "precision_bits"}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown accelerator field(s): {', '.join(unknown)}")
        if "memory_bandwidth_bytes_s" in values and "bandwidth_bytes_s" in values:
            raise ValueError("set only one of memory_bandwidth_bytes_s and bandwidth_bytes_s")
        try:
            bandwidth = (values["memory_bandwidth_bytes_s"] if "memory_bandwidth_bytes_s" in values
                         else values["bandwidth_bytes_s"])
            return cls(
                pe_rows=values["pe_rows"],
                pe_cols=values["pe_cols"],
                sram_bytes=values["sram_bytes"],
                memory_bandwidth_bytes_s=bandwidth,
                precision_bits=values["precision_bits"],
            )
        except KeyError as exc:
            raise ValueError(f"missing accelerator field: {exc.args[0]}") from exc

    def to_wire(self) -> dict:
        return {
            "pe_rows": self.pe_rows,
            "pe_cols": self.pe_cols,
            "pe_count": self.pe_count,
            "sram_bytes": self.sram_bytes,
            "memory_bandwidth_bytes_s": self.memory_bandwidth_bytes_s,
            "precision_bits": self.precision_bits,
            "key": self.key,
        }


@dataclass(frozen=True, slots=True)
class CoDesignConstraints:
    """Static hardware limits and optional deployment budgets.

    Static limits reject malformed or unsafe custom hardware values before
    evaluation.  Deployment budgets remain ordinary constrained objectives so
    an infeasible point can still be explained in a Pareto report.
    """

    max_pe_count: int = _MAX_PE_COUNT
    min_sram_bytes: int = _MIN_SRAM_BYTES
    max_sram_bytes: int = _MAX_SRAM_BYTES
    min_memory_bandwidth_bytes_s: float = 1.0
    max_memory_bandwidth_bytes_s: float = _MAX_BANDWIDTH_BYTES_S
    allowed_precision_bits: Tuple[int, ...] = _PRECISION_BITS
    max_array_aspect_ratio: float = 4096.0
    max_latency_s: float = math.inf
    max_energy_j: float = math.inf

    def __post_init__(self) -> None:
        max_pes = _int_value("max_pe_count", self.max_pe_count, minimum=1, maximum=_MAX_PE_COUNT)
        min_sram = _int_value("min_sram_bytes", self.min_sram_bytes,
                              minimum=_MIN_SRAM_BYTES, maximum=_MAX_SRAM_BYTES)
        max_sram = _int_value("max_sram_bytes", self.max_sram_bytes,
                              minimum=_MIN_SRAM_BYTES, maximum=_MAX_SRAM_BYTES)
        if min_sram > max_sram:
            raise ValueError("min_sram_bytes cannot exceed max_sram_bytes")
        min_bw = _real_value("min_memory_bandwidth_bytes_s", self.min_memory_bandwidth_bytes_s,
                             minimum=1.0, maximum=_MAX_BANDWIDTH_BYTES_S)
        max_bw = _real_value("max_memory_bandwidth_bytes_s", self.max_memory_bandwidth_bytes_s,
                             minimum=1.0, maximum=_MAX_BANDWIDTH_BYTES_S)
        if min_bw > max_bw:
            raise ValueError("min_memory_bandwidth_bytes_s cannot exceed max_memory_bandwidth_bytes_s")
        precisions = _unique_sorted(self.allowed_precision_bits, "allowed_precision_bits")
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in precisions):
            raise ValueError("allowed_precision_bits must contain integers")
        if any(int(v) not in _PRECISION_BITS for v in precisions):
            raise ValueError(f"allowed_precision_bits must be drawn from {_PRECISION_BITS}")
        aspect = _real_value("max_array_aspect_ratio", self.max_array_aspect_ratio,
                             minimum=1.0, maximum=float(_MAX_ARRAY_DIMENSION))
        max_latency = _budget_value("max_latency_s", self.max_latency_s)
        max_energy = _budget_value("max_energy_j", self.max_energy_j)
        object.__setattr__(self, "max_pe_count", max_pes)
        object.__setattr__(self, "min_sram_bytes", min_sram)
        object.__setattr__(self, "max_sram_bytes", max_sram)
        object.__setattr__(self, "min_memory_bandwidth_bytes_s", min_bw)
        object.__setattr__(self, "max_memory_bandwidth_bytes_s", max_bw)
        object.__setattr__(self, "allowed_precision_bits", tuple(int(v) for v in precisions))
        object.__setattr__(self, "max_array_aspect_ratio", aspect)
        object.__setattr__(self, "max_latency_s", max_latency)
        object.__setattr__(self, "max_energy_j", max_energy)

    def violations(self, architecture: AcceleratorArchitecture) -> Tuple[str, ...]:
        """Return static violations without hiding the reason behind a boolean."""
        reasons = []
        if architecture.pe_count > self.max_pe_count:
            reasons.append(f"PE count {architecture.pe_count} > limit {self.max_pe_count}")
        if architecture.sram_bytes < self.min_sram_bytes:
            reasons.append(f"SRAM {architecture.sram_bytes} B < minimum {self.min_sram_bytes} B")
        if architecture.sram_bytes > self.max_sram_bytes:
            reasons.append(f"SRAM {architecture.sram_bytes} B > limit {self.max_sram_bytes} B")
        bw = architecture.memory_bandwidth_bytes_s
        if bw < self.min_memory_bandwidth_bytes_s:
            reasons.append(f"bandwidth {bw:g} B/s < minimum {self.min_memory_bandwidth_bytes_s:g} B/s")
        if bw > self.max_memory_bandwidth_bytes_s:
            reasons.append(f"bandwidth {bw:g} B/s > limit {self.max_memory_bandwidth_bytes_s:g} B/s")
        if architecture.precision_bits not in self.allowed_precision_bits:
            reasons.append(f"precision {architecture.precision_bits} not in {self.allowed_precision_bits}")
        aspect = max(architecture.pe_rows, architecture.pe_cols) / min(architecture.pe_rows, architecture.pe_cols)
        if aspect > self.max_array_aspect_ratio:
            reasons.append(f"array aspect ratio {aspect:g} > limit {self.max_array_aspect_ratio:g}")
        return tuple(reasons)

    def validate_architecture(self, architecture: AcceleratorArchitecture) -> None:
        if not isinstance(architecture, AcceleratorArchitecture):
            raise ValueError("architecture must be an AcceleratorArchitecture")
        violations = self.violations(architecture)
        if violations:
            raise ValueError("accelerator configuration rejected: " + "; ".join(violations))


@dataclass(frozen=True, slots=True)
class CoDesignSpace:
    """Deterministic finite architecture choices.

    Values are canonicalised and deduplicated at construction, so the same
    custom space always produces the same candidate order independent of input
    ordering.
    """

    pe_rows: Tuple[int, ...] = (1, 2, 4, 8, 16)
    pe_cols: Tuple[int, ...] = (1, 2, 4, 8, 16)
    sram_bytes: Tuple[int, ...] = (32 * 1024, 128 * 1024, 512 * 1024)
    memory_bandwidth_bytes_s: Tuple[float, ...] = (1e9, 4e9, 16e9, 64e9)
    precision_bits: Tuple[int, ...] = (4, 8, 16)
    constraints: CoDesignConstraints = field(default_factory=CoDesignConstraints)

    def __post_init__(self) -> None:
        rows = tuple(_int_value("pe_rows", v, minimum=1, maximum=_MAX_ARRAY_DIMENSION)
                     for v in _unique_sorted(self.pe_rows, "pe_rows"))
        cols = tuple(_int_value("pe_cols", v, minimum=1, maximum=_MAX_ARRAY_DIMENSION)
                     for v in _unique_sorted(self.pe_cols, "pe_cols"))
        srams = tuple(_int_value("sram_bytes", v, minimum=_MIN_SRAM_BYTES, maximum=_MAX_SRAM_BYTES)
                      for v in _unique_sorted(self.sram_bytes, "sram_bytes"))
        bws = tuple(_real_value("memory_bandwidth_bytes_s", v, minimum=1.0, maximum=_MAX_BANDWIDTH_BYTES_S)
                    for v in _unique_sorted(self.memory_bandwidth_bytes_s, "memory_bandwidth_bytes_s"))
        precisions = tuple(_int_value("precision_bits", v, minimum=1, maximum=64)
                           for v in _unique_sorted(self.precision_bits, "precision_bits"))
        if any(v not in _PRECISION_BITS for v in precisions):
            raise ValueError(f"precision_bits must be drawn from {_PRECISION_BITS}")
        cardinality = len(rows) * len(cols) * len(srams) * len(bws) * len(precisions)
        if cardinality > _MAX_CANDIDATES:
            raise ValueError(f"co-design space has {cardinality} combinations; maximum is {_MAX_CANDIDATES}")
        if not isinstance(self.constraints, CoDesignConstraints):
            raise ValueError("constraints must be a CoDesignConstraints instance")
        object.__setattr__(self, "pe_rows", rows)
        object.__setattr__(self, "pe_cols", cols)
        object.__setattr__(self, "sram_bytes", srams)
        object.__setattr__(self, "memory_bandwidth_bytes_s", bws)
        object.__setattr__(self, "precision_bits", precisions)

    @property
    def cardinality(self) -> int:
        return (len(self.pe_rows) * len(self.pe_cols) * len(self.sram_bytes)
                * len(self.memory_bandwidth_bytes_s) * len(self.precision_bits))

    def candidates(self, deployment_key: str = "deployment",
                   constraints: Optional[CoDesignConstraints] = None) -> Tuple["CoDesignCandidate", ...]:
        return generate_candidates(self, deployment_key, constraints=constraints)


@dataclass(frozen=True, slots=True)
class CoDesignCandidate:
    """A deployment genome key paired with one accelerator architecture."""

    deployment_key: str
    architecture: AcceleratorArchitecture

    def __post_init__(self) -> None:
        if not isinstance(self.deployment_key, str) or not self.deployment_key.strip():
            raise ValueError("deployment_key must be a non-empty string")
        if not isinstance(self.architecture, AcceleratorArchitecture):
            raise ValueError("architecture must be an AcceleratorArchitecture")

    @property
    def key(self) -> str:
        return f"{self.deployment_key}@@{self.architecture.key}"

    def to_wire(self) -> dict:
        return {"key": self.key, "deployment_key": self.deployment_key,
                "architecture": self.architecture.to_wire(),
                "capability": CO_DESIGN_CAPABILITY}


def generate_candidates(space: CoDesignSpace, deployment_key: str = "deployment",
                        *, constraints: Optional[CoDesignConstraints] = None) -> Tuple[CoDesignCandidate, ...]:
    """Generate a deterministic, statically valid architecture candidate set."""
    if not isinstance(space, CoDesignSpace):
        raise ValueError("space must be a CoDesignSpace")
    limits = constraints or space.constraints
    if not isinstance(limits, CoDesignConstraints):
        raise ValueError("constraints must be a CoDesignConstraints instance")
    if not isinstance(deployment_key, str) or not deployment_key.strip():
        raise ValueError("deployment_key must be a non-empty string")
    out = []
    for rows, cols, sram, bandwidth, precision in product(
            space.pe_rows, space.pe_cols, space.sram_bytes,
            space.memory_bandwidth_bytes_s, space.precision_bits):
        architecture = AcceleratorArchitecture(rows, cols, sram, bandwidth, precision)
        if not limits.violations(architecture):
            out.append(CoDesignCandidate(deployment_key, architecture))
    if not out:
        raise ValueError("co-design space contains no candidates within the supplied constraints")
    return tuple(sorted(out, key=lambda candidate: candidate.architecture.key))


def generate_co_design_candidates(deployment_keys: Iterable[Any], space: CoDesignSpace,
                                  *, constraints: Optional[CoDesignConstraints] = None) -> Tuple[CoDesignCandidate, ...]:
    """Generate deterministic pairs for several existing deployment genomes/evaluations."""
    keys = set()
    for deployment in deployment_keys:
        key = deployment if isinstance(deployment, str) else getattr(deployment, "key", None)
        if not isinstance(key, str) or not key.strip():
            raise ValueError("each deployment must be a genome/evaluation with a non-empty key")
        keys.add(key)
    if not keys:
        raise ValueError("at least one deployment is required for co-search")
    out = []
    for key in sorted(keys):
        out.extend(generate_candidates(space, key, constraints=constraints))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class CoDesignWorkload:
    """Shape/workload facts used by the bounded architecture evaluator."""

    macs: int
    parameters: int
    activation_values: int
    working_set_values: int
    required_precision_bits: int = 1
    base_accuracy_loss_pp: float = 0.0
    accuracy_loss_per_bit_pp: float = 0.15
    reference_precision_bits: int = 8
    compute_rate_per_pe_s: float = 1e9
    mac_energy_j_at_reference: float = 1e-12
    memory_energy_j_per_byte: float = 20e-12
    pe_static_power_w: float = 1e-6
    evidence_source: str = EVIDENCE_ANALYTIC
    accuracy_source: str = EVIDENCE_DEPLOYMENT_PROXY
    baseline_energy_j: Optional[float] = None
    baseline_latency_s: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("macs", "parameters", "activation_values", "working_set_values"):
            value = _int_value(name, getattr(self, name), minimum=1, maximum=10 ** 18)
            object.__setattr__(self, name, value)
        required = _int_value("required_precision_bits", self.required_precision_bits, minimum=1, maximum=64)
        reference = _int_value("reference_precision_bits", self.reference_precision_bits, minimum=1, maximum=64)
        if required not in _PRECISION_BITS or reference not in _PRECISION_BITS:
            raise ValueError(f"precision fields must be drawn from {_PRECISION_BITS}")
        object.__setattr__(self, "required_precision_bits", required)
        object.__setattr__(self, "reference_precision_bits", reference)
        for name in ("base_accuracy_loss_pp", "accuracy_loss_per_bit_pp"):
            value = _real_value(name, getattr(self, name), minimum=0.0, maximum=100.0)
            object.__setattr__(self, name, value)
        for name in ("compute_rate_per_pe_s", "mac_energy_j_at_reference", "memory_energy_j_per_byte",
                     "pe_static_power_w"):
            value = _real_value(name, getattr(self, name), minimum=1e-30, maximum=1e30)
            object.__setattr__(self, name, value)
        for name in ("evidence_source", "accuracy_source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty evidence label")
        for name in ("baseline_energy_j", "baseline_latency_s"):
            value = getattr(self, name)
            if value is not None:
                value = _real_value(name, value, minimum=1e-30, maximum=1e30)
                object.__setattr__(self, name, value)

    @classmethod
    def from_model(cls, model: Any, genome: Optional[Any] = None,
                   base_evaluation: Optional[Any] = None,
                   hardware: Optional[Any] = None) -> "CoDesignWorkload":
        """Derive conservative workload facts from the existing model/genome.

        This adapter deliberately uses model shape facts and the deployment
        genome only; it does not claim topology search or measured hardware
        behavior.
        """
        layers = getattr(model, "layers", None)
        if not layers:
            raise ValueError("model must provide at least one layer")
        macs = max(1, int(sum(int(layer.macs) for layer in layers)))
        parameters = max(1, int(sum(int(layer.params) for layer in layers)))
        input_size = max(1, int(model.input_size()))
        activation_values = max(input_size, max(int(layer.out_neurons) for layer in layers))
        working_set_values = max(
            1,
            max(int(layer.params) + int(layer.fan_in) + int(layer.out_neurons) for layer in layers),
        )
        required_precision = 1
        if genome is not None:
            for gene in getattr(genome, "layers", ()):
                domain = getattr(gene, "domain", None)
                # Symbolic stages execute outside the accelerator datapath.
                if getattr(domain, "name", None) != "SYM":
                    required_precision = max(required_precision, int(gene.w_bits), int(gene.a_bits))

        compute_rate = 1e9
        mac_energy = 1e-12
        memory_energy = 20e-12
        pe_static = 1e-6
        evidence = EVIDENCE_ANALYTIC
        if hardware is not None:
            try:
                mac_energy, compute_rate = hardware.ann_cost(8, 8)
            except (AttributeError, KeyError, ValueError):
                pass
            memory_energy = float(getattr(hardware, "ann_e_byte", memory_energy))
            pe_static = float(getattr(hardware, "p_static_w", pe_static)) / max(1, int(getattr(hardware, "n_cores", 1)))
            provenance = getattr(hardware, "provenance", {}) or {}
            if any(str(v).lower() in {"measured", "calibrated", "user"} for v in provenance.values()):
                evidence = "calibrated_hardware_prior"

        base_accuracy_loss = 0.0
        accuracy_source = EVIDENCE_DEPLOYMENT_PROXY
        baseline_energy = baseline_latency = None
        if base_evaluation is not None:
            base_accuracy_loss = max(0.0, 100.0 - float(base_evaluation.accuracy))
            accuracy_source = (EVIDENCE_ORACLE if getattr(base_evaluation, "accuracy_source", "proxy") == "oracle"
                               else EVIDENCE_DEPLOYMENT_PROXY)
            cost = getattr(base_evaluation, "cost", None)
            baseline_energy = getattr(cost, "energy_j", None)
            baseline_latency = getattr(cost, "latency_s", None)

        return cls(
            macs=macs,
            parameters=parameters,
            activation_values=activation_values,
            working_set_values=working_set_values,
            required_precision_bits=required_precision,
            base_accuracy_loss_pp=base_accuracy_loss,
            compute_rate_per_pe_s=max(compute_rate, 1e-30),
            mac_energy_j_at_reference=max(mac_energy, 1e-30),
            memory_energy_j_per_byte=max(memory_energy, 1e-30),
            pe_static_power_w=max(pe_static, 1e-30),
            evidence_source=evidence,
            accuracy_source=accuracy_source,
            baseline_energy_j=baseline_energy,
            baseline_latency_s=baseline_latency,
        )


@dataclass
class CoDesignEvaluation:
    """Constrained architecture result paired with one deployment genome."""

    candidate: CoDesignCandidate
    F: np.ndarray                         # [energy, latency, accuracy loss, area proxy]
    cv: float
    g: dict
    metrics: dict
    reasons: Tuple[str, ...]
    evidence_sources: dict
    deployment_evaluation: Any = field(default=None, repr=False, compare=False)
    rank: int = 0
    crowd: float = 0.0

    @property
    def feasible(self) -> bool:
        return self.cv == 0.0

    @property
    def key(self) -> str:
        return self.candidate.key

    def to_wire(self) -> dict:
        return {
            "capability": CO_DESIGN_CAPABILITY,
            "key": self.key,
            "deployment_key": self.candidate.deployment_key,
            "architecture": self.candidate.architecture.to_wire(),
            "objective_names": list(CO_DESIGN_OBJECTIVES),
            "f": [float(v) for v in self.F],
            "cv": float(self.cv),
            "feasible": self.feasible,
            "rank": int(self.rank),
            "crowd": float(self.crowd) if math.isfinite(self.crowd) else "inf",
            "constraints": {k: float(v) for k, v in self.g.items()},
            "metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                        for k, v in self.metrics.items()},
            "selection_reasons": list(self.reasons),
            "evidence_sources": dict(self.evidence_sources),
        }


class CoDesignEvaluator:
    """Evaluate a fixed deployment workload on candidate accelerators."""

    def __init__(self, workload: CoDesignWorkload,
                 constraints: Optional[CoDesignConstraints] = None) -> None:
        if not isinstance(workload, CoDesignWorkload):
            raise ValueError("workload must be a CoDesignWorkload")
        self.workload = workload
        self.constraints = constraints or CoDesignConstraints()
        if not isinstance(self.constraints, CoDesignConstraints):
            raise ValueError("constraints must be a CoDesignConstraints instance")

    def evaluate(self, candidate: CoDesignCandidate, deployment_evaluation: Optional[Any] = None) -> CoDesignEvaluation:
        if not isinstance(candidate, CoDesignCandidate):
            raise ValueError("candidate must be a CoDesignCandidate")
        # Static custom-hardware errors are rejected.  Workload budgets below
        # are represented as constrained violations so Pareto ranking can show
        # why a candidate was not selected.
        self.constraints.validate_architecture(candidate.architecture)
        a, w = candidate.architecture, self.workload
        precision = a.precision_bits
        reference = w.reference_precision_bits
        bit_scale = precision / reference
        throughput_scale = max(0.25, reference / precision)
        aspect = min(a.pe_rows, a.pe_cols) / max(a.pe_rows, a.pe_cols)
        array_efficiency = 0.75 + 0.25 * aspect
        effective_pes = a.pe_count * array_efficiency
        traffic_bytes = math.ceil(w.parameters * precision / 8.0) + math.ceil(w.activation_values * precision / 8.0)
        required_sram_bytes = math.ceil(w.working_set_values * precision / 8.0)
        compute_latency = w.macs / max(w.compute_rate_per_pe_s * effective_pes * throughput_scale, 1e-30)
        memory_latency = traffic_bytes / a.memory_bandwidth_bytes_s
        latency = max(compute_latency, memory_latency)
        energy = (
            w.macs * w.mac_energy_j_at_reference * (bit_scale ** 0.8)
            + traffic_bytes * w.memory_energy_j_per_byte
            + a.pe_count * w.pe_static_power_w * latency
        )
        precision_penalty = max(0, reference - precision) * w.accuracy_loss_per_bit_pp
        accuracy_loss = w.base_accuracy_loss_pp + precision_penalty
        area_proxy = float(a.pe_count + math.ceil(a.sram_bytes / 1024.0))
        g = {
            "sram_capacity": required_sram_bytes / a.sram_bytes - 1.0,
            "precision_support": w.required_precision_bits / precision - 1.0,
            "latency": latency / self.constraints.max_latency_s - 1.0
            if math.isfinite(self.constraints.max_latency_s) else -1.0,
            "energy": energy / self.constraints.max_energy_j - 1.0
            if math.isfinite(self.constraints.max_energy_j) else -1.0,
        }
        cv = float(sum(max(0.0, value) for value in g.values()))
        reasons = [
            f"{CO_DESIGN_CAPABILITY}: deployment genome is held fixed; this is not neural-topology NAS.",
            (f"{a.pe_rows}x{a.pe_cols} array provides {a.pe_count} PEs with "
             f"{array_efficiency:.0%} analytic shape efficiency."),
        ]
        if memory_latency >= compute_latency:
            reasons.append("memory-bound: bandwidth is the limiting latency term")
        else:
            reasons.append("compute-bound: PE count/precision is the limiting latency term")
        if required_sram_bytes <= a.sram_bytes:
            reasons.append(f"working set fits in {a.sram_bytes} B SRAM")
        else:
            reasons.append(f"infeasible: working set needs {required_sram_bytes} B but SRAM is {a.sram_bytes} B")
        if precision < w.required_precision_bits:
            reasons.append(f"infeasible: deployment requires at least {w.required_precision_bits}-bit precision")
        else:
            reasons.append(f"{precision}-bit datapath has an estimated {precision_penalty:.2f} pp precision penalty")
        if cv:
            reasons.append("candidate is constraint-infeasible and is ranked by constrained violation")

        evidence = {
            "energy_j": w.evidence_source,
            "latency_s": w.evidence_source,
            "accuracy_loss_pp": w.accuracy_source,
            "area_proxy": w.evidence_source,
            "sram_capacity": EVIDENCE_MODEL,
            "selection": "constrained_pareto",
        }
        metrics = {
            "estimated_energy_j": float(energy),
            "estimated_latency_s": float(latency),
            "accuracy_loss_pp": float(accuracy_loss),
            "area_proxy": area_proxy,
            "compute_latency_s": float(compute_latency),
            "memory_latency_s": float(memory_latency),
            "traffic_bytes": float(traffic_bytes),
            "required_sram_bytes": float(required_sram_bytes),
            "array_efficiency": float(array_efficiency),
            "effective_pes": float(effective_pes),
            "baseline_deployment_energy_j": w.baseline_energy_j,
            "baseline_deployment_latency_s": w.baseline_latency_s,
        }
        return CoDesignEvaluation(
            candidate=candidate,
            F=np.asarray([energy, latency, accuracy_loss, area_proxy], dtype=np.float64),
            cv=cv,
            g=g,
            metrics=metrics,
            reasons=tuple(reasons),
            evidence_sources=evidence,
            deployment_evaluation=deployment_evaluation,
        )


def pareto_rank(evaluations: Sequence[CoDesignEvaluation]) -> Tuple[CoDesignEvaluation, ...]:
    """Assign constrained Pareto rank/crowding and return stable rank order."""
    if not evaluations:
        return ()
    F = np.stack([evaluation.F for evaluation in evaluations])
    cv = np.asarray([evaluation.cv for evaluation in evaluations], dtype=np.float64)
    fronts = fast_non_dominated_sort(F, cv)
    lo, hi = F.min(axis=0), F.max(axis=0)
    Fn = normalise(F, lo, hi, log_axes=(0, 1))
    for rank, indices in enumerate(fronts):
        distances = crowding_distance(Fn[indices])
        for local, index in enumerate(indices):
            evaluations[int(index)].rank = rank
            evaluations[int(index)].crowd = float(distances[local])
            if rank == 0 and evaluations[int(index)].feasible:
                evaluations[int(index)].reasons += (
                    "Pareto rank 0: no feasible candidate is better on all reported objectives.",
                )
            elif rank == 0:
                evaluations[int(index)].reasons += (
                    "Pareto rank 0 among infeasible points; no feasible point was available.",
                )
            else:
                evaluations[int(index)].reasons += (
                    f"Pareto rank {rank}: dominated or less competitive than rank-0 trade-offs.",
                )
    return tuple(sorted(evaluations, key=lambda e: (e.rank, -e.crowd, e.key)))


@dataclass(frozen=True, slots=True)
class CoDesignSearchResult:
    evaluations: Tuple[CoDesignEvaluation, ...]
    front: Tuple[CoDesignEvaluation, ...]
    recommended: Optional[CoDesignEvaluation]
    capability: str = CO_DESIGN_CAPABILITY

    def to_wire(self) -> dict:
        return {
            "capability": self.capability,
            "evaluations": [evaluation.to_wire() for evaluation in self.evaluations],
            "front": [evaluation.to_wire() for evaluation in self.front],
            "recommended": self.recommended.to_wire() if self.recommended else None,
        }


class CoDesignSearch:
    """Run bounded hardware/deployment co-search over existing deployments.

    ``deployments`` can be existing ``Genome`` objects or cached
    ``Evaluation`` objects from ``NeurosymbolicEvaluator``.  The normal
    deployment search remains the source of deployment genomes; this class
    adds the accelerator dimension without changing that API.
    """

    def __init__(self, evaluator: Any, space: Optional[CoDesignSpace] = None,
                 constraints: Optional[CoDesignConstraints] = None) -> None:
        if not hasattr(evaluator, "model") or not hasattr(evaluator, "evaluate"):
            raise ValueError("evaluator must be a NeurosymbolicEvaluator-compatible object")
        self.evaluator = evaluator
        self.space = space or CoDesignSpace()
        self.constraints = constraints or self.space.constraints
        if not isinstance(self.space, CoDesignSpace):
            raise ValueError("space must be a CoDesignSpace")
        if not isinstance(self.constraints, CoDesignConstraints):
            raise ValueError("constraints must be a CoDesignConstraints instance")
        # Fail early for a custom space that has no statically valid point.
        self._architectures = tuple(
            candidate.architecture for candidate in generate_candidates(
                self.space, "_validation_", constraints=self.constraints
            )
        )

    @staticmethod
    def _base_evaluation(deployment: Any, evaluator: Any) -> Any:
        if hasattr(deployment, "genome") and hasattr(deployment, "accuracy"):
            return deployment
        if hasattr(deployment, "layers") and hasattr(deployment, "key"):
            return evaluator.evaluate(deployment)
        raise ValueError("deployments must be Genome or Evaluation objects")

    def candidate_pairs(self, deployments: Sequence[Any]) -> Tuple[CoDesignCandidate, ...]:
        keys = set()
        for deployment in deployments:
            base = self._base_evaluation(deployment, self.evaluator)
            keys.add(str(base.key))
        if not keys:
            raise ValueError("at least one deployment is required for co-search")
        out = []
        for key in sorted(keys):
            out.extend(CoDesignCandidate(key, architecture) for architecture in self._architectures)
        return tuple(out)

    def run(self, deployments: Sequence[Any]) -> CoDesignSearchResult:
        if not deployments:
            raise ValueError("at least one deployment is required for co-search")
        bases = {}
        for deployment in deployments:
            base = self._base_evaluation(deployment, self.evaluator)
            bases[str(base.key)] = base
        results = []
        for key in sorted(bases):
            base = bases[key]
            workload = CoDesignWorkload.from_model(
                self.evaluator.model, base.genome, base_evaluation=base,
                hardware=getattr(self.evaluator, "hw", None),
            )
            architecture_evaluator = CoDesignEvaluator(workload, self.constraints)
            for architecture in self._architectures:
                candidate = CoDesignCandidate(key, architecture)
                results.append(architecture_evaluator.evaluate(candidate, base))
        ranked = pareto_rank(results)
        feasible = tuple(evaluation for evaluation in ranked if evaluation.rank == 0 and evaluation.feasible)
        if not feasible:
            recommended = min(ranked, key=lambda e: (e.cv, tuple(float(v) for v in e.F), e.key))
            recommended.reasons += ("Recommended fallback: lowest constrained violation; no feasible point was found.",)
        else:
            front_F = np.stack([evaluation.F for evaluation in feasible])
            front_norm = normalise(front_F, front_F.min(axis=0), front_F.max(axis=0), log_axes=(0, 1))
            recommended = feasible[asf_select(front_norm)]
            recommended.reasons += (
                "Recommended by the equal-weight ASF knee across estimated energy, latency, and accuracy loss.",
            )
        return CoDesignSearchResult(ranked, feasible, recommended)


__all__ = [
    "AcceleratorArchitecture",
    "CoDesignCandidate",
    "CoDesignConstraints",
    "CoDesignEvaluation",
    "CoDesignEvaluator",
    "CoDesignSearch",
    "CoDesignSearchResult",
    "CoDesignSpace",
    "CoDesignWorkload",
    "CO_DESIGN_CAPABILITY",
    "CO_DESIGN_OBJECTIVES",
    "EVIDENCE_ANALYTIC",
    "EVIDENCE_DEPLOYMENT_PROXY",
    "EVIDENCE_MODEL",
    "EVIDENCE_ORACLE",
    "generate_candidates",
    "generate_co_design_candidates",
    "pareto_rank",
]
