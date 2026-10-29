"""Model-level intermediate representation consumed by the search engine.

The search operates on a *chain* of schedulable units (a layer or a fused block).
Each unit carries the shape facts the cost model needs and the calibrated
sensitivities the accuracy model needs. Branching topologies are collapsed into
single units by the ingestion frontend (torch.fx tracing) before search.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class LayerSensitivity:
    """Calibrated accuracy sensitivities, in accuracy points (pp).

    Each coefficient is the drop measured on the calibration set when *only this*
    layer is perturbed to the reference setting (one-at-a-time sweep, see SPEC §5.2):

      q_w     drop at the minimum weight precision b_min
      q_a     drop at the minimum activation precision b_min
      c_rate  drop when converted to rate-coded LIF at T_ref timesteps
      c_ttfs  drop when converted to TTFS coding
      sym     drop when replaced by its symbolic substitute (model mismatch)
    """

    q_w: float = 0.3
    q_a: float = 0.3
    c_rate: float = 1.0
    c_ttfs: float = 1.2
    sym: float = 0.0
    t_ref: int = 8
    b_min: int = 4


@dataclass(frozen=True)
class LayerSpec:
    name: str
    op: str                      # dense | conv2d | block
    fan_in: int                  # N_in: flattened input neurons
    out_neurons: int             # N_out: flattened output neurons
    macs: int                    # multiply-accumulates per dense inference
    params: int
    weight_shape: Tuple[int, ...]
    activation: str = "relu"     # relu | linear. Unsigned LIF spikes cannot carry signed outputs,
                                 # so linear layers are not spiking-admissible (dual-rail coding: roadmap)
    spiking_ok: bool = True
    symbolic_substitute: Optional[str] = None
    sensitivity: LayerSensitivity = field(default_factory=LayerSensitivity)
    base_rate: float = 0.12      # mean spike probability per neuron per timestep
    transient: float = 0.6       # kappa in r(t) = r (1 + kappa e^{-t/tau})
    attrs: Dict[str, Any] = field(default_factory=dict, compare=False, hash=False)
    # geometry for exporters: conv {in_shape, out_shape, kernel, stride, padding};
    # dense {flatten_input}; either {pool: {type: max|avg|global_avg, kernel, stride}}

    @property
    def spiking_admissible(self) -> bool:
        return self.spiking_ok and self.activation == "relu"

    @property
    def fanout_per_input(self) -> float:
        """Average synaptic fan-out of one presynaptic neuron (synaptic ops per input spike)."""
        return self.macs / max(1, self.fan_in)


@dataclass(frozen=True)
class GuardSite:
    """A point in the chain where a symbolic constraint must be enforced.

    after_layer = i places the guard on the output edge of layer i.
    """

    after_layer: int
    constraint_id: str
    mandatory: bool = True


@dataclass
class ModelGraph:
    name: str
    input_shape: Tuple[int, ...]
    layers: List[LayerSpec]
    base_accuracy: float                              # pp, full-precision continuous baseline
    guard_sites: List[GuardSite] = field(default_factory=list)
    constraints: Dict[str, object] = field(default_factory=dict)   # id -> SymbolicConstraint
    substitutes: Dict[str, object] = field(default_factory=dict)   # id -> LinearODESubstitute
    input_rate: float = 0.2                           # mean encoder spike rate on raw input
    interaction: float = 0.004                        # rho, second-order accuracy interaction
    policy: Optional[object] = None                   # search.policy.SearchPolicy (user locks/toggles)
    architecture_family: Optional[str] = None         # mlp | cnn | fno | vit | gnn | hybrid
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.layers)

    def total_macs(self) -> int:
        return int(sum(l.macs for l in self.layers))

    def total_params(self) -> int:
        return int(sum(l.params for l in self.layers))

    def input_size(self) -> int:
        return int(np.prod(self.input_shape))

    def edge_size(self, i: int) -> int:
        """Number of values on the edge entering layer i (i == n is the network output)."""
        return self.input_size() if i == 0 else self.layers[i - 1].out_neurons

    def guard_at(self, layer: int) -> Optional[GuardSite]:
        for g in self.guard_sites:
            if g.after_layer == layer:
                return g
        return None


def dense(name: str, n_in: int, n_out: int, **kw) -> LayerSpec:
    return LayerSpec(name=name, op="dense", fan_in=n_in, out_neurons=n_out,
                     macs=n_in * n_out, params=n_in * n_out + n_out,
                     weight_shape=(n_out, n_in), **kw)


def conv2d(name: str, c_in: int, c_out: int, k: int, h: int, w: int, stride: int = 1,
           padding: Optional[int] = None, **kw) -> LayerSpec:
    pad = k // 2 if padding is None else padding
    oh, ow = (h + 2 * pad - k) // stride + 1, (w + 2 * pad - k) // stride + 1
    params = c_in * c_out * k * k
    attrs = dict(kw.pop("attrs", {}))
    attrs.update({"in_shape": (c_in, h, w), "out_shape": (c_out, oh, ow), "kernel": k, "stride": stride, "padding": pad})
    return LayerSpec(name=name, op="conv2d", fan_in=c_in * h * w, out_neurons=c_out * oh * ow,
                     macs=params * oh * ow, params=params + c_out,
                     weight_shape=(c_out, c_in, k, k), attrs=attrs, **kw)


def operator_block(name: str, family: str, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...],
                   macs: int, params: int, weight_shape: Tuple[int, ...] = (), **kw) -> LayerSpec:
    """Create an architecture-level block without flattening its geometry.

    Operator-family adapters may provide a backend-specific lowering later.  The
    search engine can still reason about the block today while the ``attrs``
    contract keeps spatial, token, or graph dimensions explicit.
    """
    attrs = dict(kw.pop("attrs", {}))
    attrs.update({"in_shape": tuple(input_shape), "out_shape": tuple(output_shape),
                  "preserve_spatial": True, "operator_family": family})
    return LayerSpec(name=name, op=family, fan_in=int(np.prod(input_shape)),
                     out_neurons=int(np.prod(output_shape)), macs=int(macs), params=int(params),
                     weight_shape=tuple(weight_shape), attrs=attrs, **kw)
