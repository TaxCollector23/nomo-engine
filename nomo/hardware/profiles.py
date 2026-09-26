"""Silicon profiles.

A profile is the *analytic prior* of the cost model. Every scalar is a
placeholder until overwritten by the calibration LUT (hardware/lut.py) built
from on-device microbenchmarks. `provenance` records which fields were measured.

Architectural counts that are publicly documented are marked `# public`; all
energy/latency coefficients are illustrative placeholders and MUST be replaced
by measurements before any number from this engine is reported externally.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple

Plasticity = Literal["none", "final_layer", "any"]


@dataclass(frozen=True)
class Link:
    src: str
    dst: str
    bw_bytes_per_s: float
    e_per_byte: float        # J / byte
    lat_s: float             # fixed per-transfer latency (hop + protocol)
    bidirectional: bool = True


@dataclass(frozen=True)
class SiliconProfile:
    id: str
    name: str

    # ---- continuous (ANN) engine: keyed by (w_bits, a_bits)
    ann_e_mac: Dict[Tuple[int, int], float]
    ann_macs_per_s: Dict[Tuple[int, int], float]
    ann_mem_bw: float                    # bytes / s for weight + activation streaming
    ann_e_byte: float                    # J / byte of weight/activation traffic
    ann_launch_s: float                  # per-layer dispatch overhead

    # ---- spiking engine
    snn_e_sop: Dict[int, float]          # J per synaptic op, keyed by weight bits
    snn_e_neuron: float                  # J per neuron update per timestep
    snn_e_learn: float                   # J per plastic synapse update (on-device learning)
    snn_sops_per_s_core: float
    snn_t_step_min: float                # s, minimum barrier-synchronised timestep
    snn_dense: bool                      # True: spikes simulated densely (GPU)
    neurons_per_core: int
    syn_mem_bits_per_core: int
    n_cores: int
    trace_bits: int                      # eligibility-trace bits per plastic synapse

    # ---- symbolic engine (host CPU / embedded core)
    sym_e_op: float                      # J per fixed/float op
    sym_ops_per_s: float
    sym_invoke_s: float
    sym_mem_bytes: int

    # ---- domain conversion
    e_encode: float                      # J per value per timestep (ANN -> spikes)
    e_decode: float                      # J per value per timestep (spikes -> values)

    # ---- platform
    p_static_w: float
    ann_mem_bytes: int

    # ---- capabilities
    ann_bits: Tuple[int, ...]
    snn_w_bits: Tuple[int, ...]
    snn_mem_bits: Tuple[int, ...]
    timesteps: Tuple[int, ...]
    plasticity: Plasticity
    fused_guard: bool                    # guard can run fused on the ANN engine

    # ---- interconnect: which unit executes each domain, and the link graph
    units: Dict[str, str]                # {"ANN": unit, "SNN": unit, "SYM": unit}
    links: Tuple[Link, ...]
    provenance: Dict[str, str] = field(default_factory=lambda: {"*": "placeholder"})

    def ann_cost(self, w: int, a: int) -> Tuple[float, float]:
        key = (w, a) if (w, a) in self.ann_e_mac else min(self.ann_e_mac, key=lambda k: abs(k[0] - w) + abs(k[1] - a))
        return self.ann_e_mac[key], self.ann_macs_per_s[key]

    def sop_energy(self, w: int) -> float:
        key = w if w in self.snn_e_sop else min(self.snn_e_sop, key=lambda k: abs(k - w))
        return self.snn_e_sop[key]


def _ann_table(e8: float, r8: float) -> Tuple[Dict, Dict]:
    """Precision scaling prior: energy ~ (w*a)^0.8, throughput ~ 1/max(w,a) (bit-serial-ish)."""
    e, r = {}, {}
    for w in (4, 8, 16):
        for a in (4, 8, 16):
            e[(w, a)] = e8 * ((w * a) / 64.0) ** 0.8
            r[(w, a)] = r8 * 8.0 / max(w, a)
    return e, r


_e_l2, _r_l2 = _ann_table(9.0e-12, 0.4e12)
LOIHI2 = SiliconProfile(
    id="loihi2", name="Intel Loihi 2",
    ann_e_mac=_e_l2, ann_macs_per_s=_r_l2, ann_mem_bw=10e9, ann_e_byte=20e-12, ann_launch_s=40e-6,
    snn_e_sop={1: 0.9e-12, 2: 1.1e-12, 4: 1.5e-12, 8: 2.4e-12},
    snn_e_neuron=0.8e-12, snn_e_learn=6e-12, snn_sops_per_s_core=0.25e9, snn_t_step_min=10e-6,
    snn_dense=False, neurons_per_core=8192, syn_mem_bits_per_core=8 * 128 * 1024, n_cores=128,  # cores: public
    trace_bits=8,
    sym_e_op=40e-12, sym_ops_per_s=0.2e9, sym_invoke_s=5e-6, sym_mem_bytes=512 * 1024,
    e_encode=1.0e-12, e_decode=1.0e-12,
    p_static_w=0.08, ann_mem_bytes=2 * 1024 ** 3,
    ann_bits=(8, 16), snn_w_bits=(1, 2, 4, 8), snn_mem_bits=(16, 24), timesteps=(2, 4, 8, 16, 32),
    plasticity="any", fused_guard=False,
    units={"ANN": "host_cpu", "SNN": "nc_mesh", "SYM": "embedded_x86"},
    links=(
        Link("nc_mesh", "embedded_x86", 2.0e9, 2e-12, 1e-6),
        Link("embedded_x86", "host_cpu", 0.12e9, 60e-12, 50e-6),     # host I/O path
        Link("nc_mesh", "io_fabric", 0.8e9, 10e-12, 5e-6),
        Link("io_fabric", "host_cpu", 0.4e9, 30e-12, 20e-6),         # alternate path
    ),
)

_e_ak, _r_ak = _ann_table(6.0e-12, 0.3e12)
AKD1500 = SiliconProfile(
    id="akd1500", name="BrainChip AKD1500",
    ann_e_mac=_e_ak, ann_macs_per_s=_r_ak, ann_mem_bw=4e9, ann_e_byte=30e-12, ann_launch_s=60e-6,
    snn_e_sop={1: 0.5e-12, 2: 0.6e-12, 4: 0.9e-12},
    snn_e_neuron=0.4e-12, snn_e_learn=4e-12, snn_sops_per_s_core=0.4e9, snn_t_step_min=20e-6,
    snn_dense=False, neurons_per_core=4096, syn_mem_bits_per_core=8 * 64 * 1024, n_cores=64,
    trace_bits=4,
    sym_e_op=25e-12, sym_ops_per_s=0.3e9, sym_invoke_s=4e-6, sym_mem_bytes=256 * 1024,
    e_encode=0.8e-12, e_decode=0.8e-12,
    p_static_w=0.05, ann_mem_bytes=512 * 1024 ** 2,
    ann_bits=(8,), snn_w_bits=(1, 2, 4), snn_mem_bits=(16,), timesteps=(1, 2, 4, 8, 16),
    plasticity="final_layer", fused_guard=False,
    units={"ANN": "host_mcu", "SNN": "npu", "SYM": "host_mcu"},
    links=(
        Link("npu", "pcie", 1.0e9, 15e-12, 3e-6),
        Link("pcie", "host_mcu", 1.0e9, 15e-12, 3e-6),
        Link("npu", "spi", 0.05e9, 5e-12, 30e-6),
        Link("spi", "host_mcu", 0.05e9, 5e-12, 30e-6),
    ),
)

_e_gpu, _r_gpu = _ann_table(1.2e-12, 40e12)
EDGE_GPU = SiliconProfile(
    id="gpu", name="Conventional edge GPU",
    ann_e_mac=_e_gpu, ann_macs_per_s=_r_gpu, ann_mem_bw=100e9, ann_e_byte=8e-12, ann_launch_s=8e-6,
    snn_e_sop={4: 1.2e-12, 8: 1.2e-12},
    snn_e_neuron=2e-12, snn_e_learn=3e-12, snn_sops_per_s_core=40e12, snn_t_step_min=8e-6,
    snn_dense=True, neurons_per_core=10 ** 9, syn_mem_bits_per_core=8 * 8 * 1024 ** 3, n_cores=1,
    trace_bits=16,
    sym_e_op=50e-12, sym_ops_per_s=2e9, sym_invoke_s=2e-6, sym_mem_bytes=1024 ** 3,
    e_encode=0.05e-12, e_decode=0.05e-12,
    p_static_w=4.0, ann_mem_bytes=8 * 1024 ** 3,
    ann_bits=(8, 16), snn_w_bits=(4, 8), snn_mem_bits=(16, 24), timesteps=(2, 4, 8, 16),
    plasticity="any", fused_guard=True,
    units={"ANN": "sm", "SNN": "sm", "SYM": "cpu"},
    links=(
        Link("sm", "unified_mem", 60e9, 4e-12, 1e-6),
        Link("unified_mem", "cpu", 30e9, 6e-12, 2e-6),
    ),
)

PROFILES: Dict[str, SiliconProfile] = {p.id: p for p in (LOIHI2, AKD1500, EDGE_GPU)}


def get_profile(pid: str) -> SiliconProfile:
    try:
        return PROFILES[pid]
    except KeyError as exc:
        raise KeyError(f"unknown hardware profile '{pid}'; available: {sorted(PROFILES)}") from exc
