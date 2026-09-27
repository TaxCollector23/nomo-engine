"""User hardware overrides (SPEC §3.10): derive a custom SiliconProfile from a base profile.

Every field is optional; only the ones supplied change, and each changed field is recorded in
`provenance` as "user" so reports can distinguish user-entered numbers from placeholders.

    mac_energy_pj         energy of one 8x8-bit MAC; the whole precision table is rescaled so the
                          relative cost of other bit widths is preserved
    sop_energy_pj         energy of one synaptic operation at 4-bit weights (table rescaled likewise)
    neuron_energy_pj      energy of one neuron update per timestep
    sram_kb_per_core      synaptic memory per neuromorphic core
    neurons_per_core, n_cores
    bus_bandwidth_gbs     bandwidth of every interconnect link, GB/s
    routing_latency_us    fixed latency of every interconnect hop, microseconds
    timestep_us           minimum spiking timestep (barrier) duration
    static_power_mw       always-on power, milliwatts
    clock_mhz             scales every throughput figure by clock / nominal clock; energy per operation
                          is unchanged (first-order CMOS model), so a lower clock lengthens latency and
                          increases static energy per inference
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Dict, Optional

from .profiles import Link, SiliconProfile


@dataclass(frozen=True)
class HardwareOverrides:
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

    def is_empty(self) -> bool:
        return all(getattr(self, f.name) is None for f in fields(self))

    def validate(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if v is not None and not (v > 0):
                raise ValueError(f"hardware override '{f.name}' must be positive, got {v}")


def apply_overrides(base: SiliconProfile, ov: HardwareOverrides) -> SiliconProfile:
    ov.validate()
    if ov.is_empty():
        return base
    kw: Dict[str, object] = {}
    prov = dict(base.provenance)

    def mark(*names: str) -> None:
        for n in names:
            prov[n] = "user"

    if ov.mac_energy_pj is not None:
        ref = base.ann_cost(8, 8)[0]
        k = ov.mac_energy_pj * 1e-12 / ref
        kw["ann_e_mac"] = {key: v * k for key, v in base.ann_e_mac.items()}
        mark("ann_e_mac")
    if ov.sop_energy_pj is not None:
        ref = base.sop_energy(4)
        k = ov.sop_energy_pj * 1e-12 / ref
        kw["snn_e_sop"] = {key: v * k for key, v in base.snn_e_sop.items()}
        mark("snn_e_sop")
    if ov.neuron_energy_pj is not None:
        kw["snn_e_neuron"] = ov.neuron_energy_pj * 1e-12
        mark("snn_e_neuron")
    if ov.sram_kb_per_core is not None:
        kw["syn_mem_bits_per_core"] = int(ov.sram_kb_per_core * 1024 * 8)
        mark("syn_mem_bits_per_core")
    if ov.neurons_per_core is not None:
        kw["neurons_per_core"] = int(ov.neurons_per_core)
        mark("neurons_per_core")
    if ov.n_cores is not None:
        kw["n_cores"] = int(ov.n_cores)
        mark("n_cores")
    if ov.timestep_us is not None:
        kw["snn_t_step_min"] = ov.timestep_us * 1e-6
        mark("snn_t_step_min")
    if ov.static_power_mw is not None:
        kw["p_static_w"] = ov.static_power_mw * 1e-3
        mark("p_static_w")
    if ov.bus_bandwidth_gbs is not None or ov.routing_latency_us is not None:
        kw["links"] = tuple(Link(l.src, l.dst,
                                 ov.bus_bandwidth_gbs * 1e9 if ov.bus_bandwidth_gbs is not None else l.bw_bytes_per_s,
                                 l.e_per_byte,
                                 ov.routing_latency_us * 1e-6 if ov.routing_latency_us is not None else l.lat_s,
                                 l.bidirectional) for l in base.links)
        mark("links")
    if ov.clock_mhz is not None:
        k = ov.clock_mhz * 1e6 / base.clock_hz
        kw["ann_macs_per_s"] = {key: v * k for key, v in base.ann_macs_per_s.items()}
        kw["snn_sops_per_s_core"] = base.snn_sops_per_s_core * k
        kw["sym_ops_per_s"] = base.sym_ops_per_s * k
        kw["clock_hz"] = ov.clock_mhz * 1e6
        mark("clock_hz", "ann_macs_per_s", "snn_sops_per_s_core", "sym_ops_per_s")
    return replace(base, id=f"{base.id}+custom", name=f"{base.name} (custom)", provenance=prov, **kw)
