"""HardwareCostModel: maps a (model, genome) pair to energy, latency, memory and
core occupancy on one silicon profile. Formulas are in SPEC §3; symbol names in
comments match the spec.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..ir import ModelGraph
from ..search.genome import Coding, Crossing, Domain, Genome, GuardImpl, Stage
from .lut import CostLUT
from .profiles import SiliconProfile
from .routing import Commodity, CrossingRouter
from .surrogate import BayesianResidualSurrogate, features

DNAME = {Domain.ANN: "ANN", Domain.SNN: "SNN", Domain.SYM: "SYM"}


# ---------------------------------------------------------------------------
# Activity (spike-rate) model
# ---------------------------------------------------------------------------

class ActivityModel:
    """r_{l,t}: per-neuron spike probability of layer l's outputs at timestep t.

    Prior:  rate coding  r_{l,t} = min(1, r_l (1 + kappa_l e^{-t / tau_tr}))
            TTFS         each neuron fires at most once: r_{l,t} = p_l / T, p_l = min(1, 2 r_l)
    Calibrated rates (from runtime/golden.py simulation of calibration data, or on-chip
    probes) override the prior per (layer, coding, T).
    """

    def __init__(self, model: ModelGraph, tau_tr: float = 2.0) -> None:
        self.model = model
        self.tau_tr = tau_tr
        self._cal: Dict[Tuple[int, int, int], np.ndarray] = {}

    def calibrate(self, layer: int, coding: Coding, T: int, rates: np.ndarray) -> None:
        rates = np.asarray(rates, float)
        if rates.shape != (T,):
            raise ValueError(f"expected {T} per-timestep rates, got shape {rates.shape}")
        self._cal[(layer, int(coding), T)] = np.clip(rates, 0.0, 1.0)

    def out_rates(self, layer: int, coding: Coding, T: int) -> np.ndarray:
        hit = self._cal.get((layer, int(coding), T))
        if hit is not None:
            return hit
        spec = self.model.layers[layer]
        if coding == Coding.TTFS:
            return np.full(T, min(1.0, 2.0 * spec.base_rate) / T)
        t = np.arange(T)
        return np.minimum(1.0, spec.base_rate * (1.0 + spec.transient * np.exp(-t / self.tau_tr)))

    def encoder_rates(self, source_layer: int, coding: Coding, T: int) -> np.ndarray:
        """Spike rates produced by encoding the values on the edge entering a spiking segment."""
        hit = self._cal.get((-1 - source_layer, int(coding), T))
        if hit is not None:
            return hit
        base = self.model.input_rate if source_layer < 0 else 0.25
        if coding == Coding.TTFS:
            return np.full(T, min(1.0, 2.0 * base) / T)
        return np.full(T, base)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@dataclass
class StageCost:
    name: str
    kind: str                         # layer | guard
    domain: str
    unit: str
    energy_j: float
    latency_s: float                  # contribution to end-to-end latency (0 for pipelined SNN layers)
    busy_s: float                     # occupancy of the executing unit per frame
    mem_bytes: float
    cores: int = 0
    per_timestep_energy_j: Optional[List[float]] = None
    per_timestep_s: Optional[List[float]] = None
    source: str = "analytic"


@dataclass
class CrossingCost:
    src: str
    dst: str
    values: int
    timesteps: int
    payload_bytes: float
    payload_kind: str                 # values | spikes-dense | spikes-aer
    conversion_energy_j: float
    conversion_s: float
    transfer_energy_j: float
    transfer_s: float
    src_unit: str
    dst_unit: str


@dataclass
class CostReport:
    energy_j: float
    latency_s: float
    frame_period_s: float
    memory_bytes: float
    cores_used: int
    core_budget: int
    stages: List[StageCost]
    crossings: List[CrossingCost]
    energy_by_domain: Dict[str, float]
    crossing_energy_j: float
    static_energy_j: float
    routing_tau_s: float
    max_link_utilisation: float
    unit_memory: Dict[str, float]
    unit_capacity: Dict[str, float]
    measured_energy_fraction: float
    capacity_violations: List[str] = field(default_factory=list)
    surrogate_applied: bool = False


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

class HardwareCostModel:
    def __init__(self, hw: SiliconProfile, model: ModelGraph, lut: Optional[CostLUT] = None,
                 activity: Optional[ActivityModel] = None,
                 energy_surrogate: Optional[BayesianResidualSurrogate] = None,
                 latency_surrogate: Optional[BayesianResidualSurrogate] = None,
                 time_price_j_per_s: float = 0.0) -> None:
        self.hw = hw
        self.model = model
        self.lut = lut or CostLUT()
        self.activity = activity or ActivityModel(model)
        self.e_sur = energy_surrogate
        self.l_sur = latency_surrogate
        self.router = CrossingRouter(hw.links, lam_j_per_s=hw.p_static_w + time_price_j_per_s)

    # ------------------------------------------------------------- per stage
    def _ann_layer(self, i: int, w: int, a: int) -> StageCost:
        hw, spec = self.hw, self.model.layers[i]
        e_mac, rate = hw.ann_cost(w, a)
        traffic = spec.params * w / 8 + (spec.fan_in + spec.out_neurons) * a / 8        # bytes
        E = spec.macs * e_mac + traffic * hw.ann_e_byte
        L = max(spec.macs / rate, traffic / hw.ann_mem_bw) + hw.ann_launch_s             # roofline
        src = "analytic"
        hit = self.lut.query((hw.id, spec.op, "ANN", w, a, 0), spec.macs)
        if hit:
            E, L, src = hit.energy_j, hit.latency_s, hit.source
        mem = spec.params * w / 8 + max(spec.fan_in, spec.out_neurons) * a / 8
        return StageCost(spec.name, "layer", "ANN", hw.units["ANN"], E, L, L, mem, source=src)

    def _snn_layer(self, i: int, gene, r_in: np.ndarray) -> Tuple[StageCost, np.ndarray]:
        hw, spec = self.hw, self.model.layers[i]
        T = gene.timesteps
        fan = spec.fanout_per_input
        if hw.snn_dense:
            sops = np.full(T, float(spec.macs))                     # dense per-step simulation
        else:
            sops = r_in * spec.fan_in * fan                         # events_t x fan-out
        learn = sops if gene.plastic else np.zeros(T)
        e_sop = hw.sop_energy(gene.w_bits)
        E_t = sops * e_sop + spec.out_neurons * hw.snn_e_neuron + learn * hw.snn_e_learn
        src = "analytic"
        if not hw.snn_dense:
            hit = self.lut.query((hw.id, "sop", "SNN", gene.w_bits, gene.a_bits, 0), max(1.0, float(sops.mean())))
            if hit:
                scale = hit.energy_j / max(1e-30, float(sops.mean()) * e_sop)
                E_t = E_t * scale
                src = hit.source

        syn_bits = spec.params * (gene.w_bits + (hw.trace_bits if gene.plastic else 0))
        if hw.snn_dense:
            cores = 0
            tau_t = np.maximum(sops / hw.snn_sops_per_s_core, hw.snn_t_step_min)
        else:
            cores = max(math.ceil(spec.out_neurons / hw.neurons_per_core),
                        math.ceil(syn_bits / hw.syn_mem_bits_per_core), 1)
            tau_t = np.maximum(sops / (cores * hw.snn_sops_per_s_core), hw.snn_t_step_min)
        mem = syn_bits / 8 + spec.out_neurons * gene.a_bits / 8
        if hw.snn_dense:
            # dense simulation streams weights once per inference and spike tensors every step
            traffic = spec.params * gene.w_bits / 8 + (spec.fan_in + spec.out_neurons) * T / 8
            E_t = E_t + traffic * hw.ann_e_byte / T
            tau_t = np.maximum(tau_t, traffic / T / hw.ann_mem_bw) + hw.ann_launch_s
        r_out = self.activity.out_rates(i, gene.coding, T)
        st = StageCost(spec.name, "layer", "SNN", hw.units["SNN"], float(E_t.sum()), 0.0,
                       float(tau_t.sum()), mem, cores, E_t.tolist(), tau_t.tolist(), src)
        return st, r_out

    def _sym_stage(self, name: str, kind: str, flops: float, mem: float, fused: bool) -> StageCost:
        hw = self.hw
        if fused:
            e_mac, rate = hw.ann_cost(8, 8)
            E, L, unit, dom = flops * e_mac, flops / rate, hw.units["ANN"], "ANN"
        else:
            E, L, unit, dom = flops * hw.sym_e_op, flops / hw.sym_ops_per_s + hw.sym_invoke_s, hw.units["SYM"], "SYM"
        return StageCost(name, kind, dom, unit, E, L, L, mem)

    @staticmethod
    def pipeline_latency(tau: List[np.ndarray]) -> float:
        """Latency of a D-layer spiking segment, T steps, with layer l processing step t at clock k = t + l.

        L_seg = sum_{k=0}^{T+D-2} max_{l : 0 <= k-l < T} tau_{l, k-l}
        """
        D, T = len(tau), len(tau[0])
        total = 0.0
        for k in range(T + D - 1):
            total += max(tau[l][k - l] for l in range(D) if 0 <= k - l < T)
        return total

    # ------------------------------------------------------------- crossings
    def _crossing_payload(self, c: Crossing, rates: Optional[np.ndarray], src_bits: int) -> Tuple[float, str]:
        """Bytes per frame on the wire. Encoders/decoders are placed on whichever side
        minimises payload: multi-bit values vs dense spike bitmap vs AER event list."""
        if c.src == Domain.SYM or c.dst == Domain.SYM and c.src != Domain.SNN:
            return c.values * 4.0, "values"                       # Q16.16
        if Domain.SNN not in (c.src, c.dst):
            return c.values * src_bits / 8.0, "values"
        T = max(1, c.timesteps)
        value_bits = src_bits if c.src != Domain.SNN else math.ceil(math.log2(T + 1))
        if c.dst == Domain.SYM:
            value_bits = 32
        values_bytes = c.values * value_bits / 8.0
        dense = c.values * T / 8.0
        events = float(np.sum(rates)) * c.values if rates is not None else 0.25 * c.values * T
        addr = math.ceil((math.log2(max(2, c.values)) + math.log2(max(2, T))) / 8.0)
        aer = events * addr
        best = min((values_bytes, "values"), (dense, "spikes-dense"), (aer, "spikes-aer"))
        return best

    # ------------------------------------------------------------- evaluate
    def evaluate(self, g: Genome) -> CostReport:
        hw, model = self.hw, self.model
        stages: List[StageCost] = []
        stream: List[Stage] = g.stages()
        rates_by_stage: Dict[int, np.ndarray] = {}

        # layers and guards, SNN segments pipelined
        k = 0
        seg_latency: List[float] = []
        while k < len(stream):
            st = stream[k]
            if st.kind == "layer" and st.domain == Domain.SNN:
                # collect maximal run of SNN layers (guards on SNN outputs always break the run: HOST impl)
                run = [k]
                while k + 1 < len(stream) and stream[k + 1].kind == "layer" and stream[k + 1].domain == Domain.SNN:
                    k += 1
                    run.append(k)
                first = stream[run[0]].index
                gene0 = g.layers[first]
                r_in = self.activity.encoder_rates(first - 1, gene0.coding, gene0.timesteps) if (
                    run[0] == 0 or stream[run[0] - 1].domain != Domain.SNN) else rates_by_stage[run[0] - 1]
                taus = []
                for s_idx in run:
                    li = stream[s_idx].index
                    sc, r_out = self._snn_layer(li, g.layers[li], r_in)
                    stages.append(sc)
                    taus.append(np.asarray(sc.per_timestep_s))
                    rates_by_stage[s_idx] = r_out
                    r_in = r_out
                lat = self.pipeline_latency(taus)
                seg_latency.append(lat)
                stages[-1].latency_s = lat          # attribute segment latency to its last layer
            elif st.kind == "layer" and st.domain == Domain.ANN:
                gene = g.layers[st.index]
                stages.append(self._ann_layer(st.index, gene.w_bits, gene.a_bits))
            elif st.kind == "layer":
                spec = model.layers[st.index]
                sub = model.substitutes[spec.symbolic_substitute]
                stages.append(self._sym_stage(spec.name, "layer", sub.flops(), sub.memory_bytes(), fused=False))
            else:
                site = model.guard_at(st.index)
                con = model.constraints[site.constraint_id]
                fused = g.guard(st.index).impl == GuardImpl.FUSED
                stages.append(self._sym_stage(f"guard:{site.constraint_id}", "guard", con.flops(),
                                              con.memory_bytes(), fused))
            k += 1

        # crossings
        crossings = g.crossings(model)
        cc: List[CrossingCost] = []
        commodities: List[Commodity] = []
        for c in crossings:
            src_stage_rates = rates_by_stage.get(c.edge) if c.src == Domain.SNN else None
            if c.dst == Domain.SNN:
                nxt = stream[c.edge + 1].index if c.edge + 1 < len(stream) else 0
                gene = g.layers[nxt]
                src_stage_rates = self.activity.encoder_rates(nxt - 1, gene.coding, gene.timesteps)
            src_bits = 8
            if c.edge >= 0 and stream[c.edge].kind == "layer":
                src_bits = g.layers[stream[c.edge].index].a_bits if c.src == Domain.ANN else 32
            payload, kind = self._crossing_payload(c, src_stage_rates, src_bits)
            T = max(1, c.timesteps)
            if c.dst == Domain.SNN:
                e_conv, t_conv = c.values * T * hw.e_encode, c.values * T / hw.snn_sops_per_s_core
            elif c.src == Domain.SNN:
                e_conv, t_conv = c.values * T * hw.e_decode, c.values * T / hw.snn_sops_per_s_core
            else:                                   # ANN <-> SYM requantisation on the symbolic engine
                e_conv, t_conv = c.values * hw.sym_e_op, c.values / hw.sym_ops_per_s
            hit = self.lut.query((hw.id, f"x:{DNAME[c.src]}>{DNAME[c.dst]}", "X", 0, 0, T), c.values * T)
            if hit:
                e_conv, t_conv = hit.energy_j, hit.latency_s
            su, du = hw.units[DNAME[c.src]], hw.units[DNAME[c.dst]]
            commodities.append(Commodity(su, du, payload))
            cc.append(CrossingCost(DNAME[c.src], DNAME[c.dst], c.values, c.timesteps, payload, kind,
                                   e_conv, t_conv, 0.0, 0.0, su, du))

        routing = self.router.route(commodities)
        per_c = routing.per_commodity_latency_s
        # distribute routed energy in proportion to bytes x path energy (exact per-commodity split is
        # not identifiable from aggregate arc loads when paths are shared)
        tot_bytes = sum(c.payload_bytes for c in cc if c.src_unit != c.dst_unit) or 1.0
        for i, c in enumerate(cc):
            c.transfer_s = per_c[i]
            if c.src_unit != c.dst_unit:
                c.transfer_energy_j = routing.energy_j * c.payload_bytes / tot_bytes

        # aggregate
        e_by_dom: Dict[str, float] = {}
        for s in stages:
            e_by_dom[s.domain] = e_by_dom.get(s.domain, 0.0) + s.energy_j
        crossing_e = sum(c.conversion_energy_j + c.transfer_energy_j for c in cc)
        latency = sum(s.latency_s for s in stages) + sum(c.conversion_s + c.transfer_s for c in cc)
        static = hw.p_static_w * latency
        energy = sum(e_by_dom.values()) + crossing_e + static

        busy: Dict[str, float] = {}
        mem: Dict[str, float] = {}
        for s in stages:
            busy[s.unit] = busy.get(s.unit, 0.0) + s.busy_s
            if s.domain != "SNN":
                mem[s.unit] = mem.get(s.unit, 0.0) + s.mem_bytes
        period = max([routing.tau_s] + list(busy.values()))

        cap: Dict[str, float] = {}
        for dom, unit in hw.units.items():
            c_ = {"ANN": hw.ann_mem_bytes, "SYM": hw.sym_mem_bytes,
                  "SNN": hw.n_cores * hw.syn_mem_bits_per_core / 8}[dom]
            cap[unit] = max(cap.get(unit, 0.0), c_)
        cores = sum(s.cores for s in stages)
        snn_mem = sum(s.mem_bytes for s in stages if s.domain == "SNN")
        mem[hw.units["SNN"]] = mem.get(hw.units["SNN"], 0.0) + snn_mem
        violations = [f"{u}: {mem[u]:.0f} B > {cap[u]:.0f} B" for u in mem if mem[u] > cap.get(u, float("inf"))]
        if not hw.snn_dense and cores > hw.n_cores:
            violations.append(f"cores: {cores} > {hw.n_cores}")

        measured = sum(s.energy_j for s in stages if s.source.startswith("lut"))
        report = CostReport(
            energy_j=energy, latency_s=latency, frame_period_s=period,
            memory_bytes=float(sum(mem.values())), cores_used=cores,
            core_budget=hw.n_cores, stages=stages, crossings=cc, energy_by_domain=e_by_dom,
            crossing_energy_j=crossing_e, static_energy_j=static, routing_tau_s=routing.tau_s,
            max_link_utilisation=(routing.tau_s / latency) if latency > 0 else 0.0,
            unit_memory=mem, unit_capacity=cap, measured_energy_fraction=measured / max(energy, 1e-30),
            capacity_violations=violations,
        )
        if (self.e_sur and self.e_sur.n_obs) or (self.l_sur and self.l_sur.n_obs):
            phi = features(g, report)
            if self.e_sur and self.e_sur.n_obs:
                report.energy_j = self.e_sur.correct(phi, report.energy_j)
            if self.l_sur and self.l_sur.n_obs:
                report.latency_s = self.l_sur.correct(phi, report.latency_s)
            report.surrogate_applied = True
        return report
