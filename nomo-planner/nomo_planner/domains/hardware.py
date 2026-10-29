"""Hardware catalog, model presets, and cost models.

Restored Python reference for the browser engine (src/planner/hardware.ts).
Every value and formula is a line-for-line match so the browser engine can be
verified against this module: studies/export_golden.py writes
scripts/golden.json from this code, and `npm run verify` compares the two.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

EFF = {"matmul": 0.55, "memory": 0.8}


@dataclass(frozen=True)
class Device:
    name: str
    peak_flops: Dict[str, float]  # dtype -> dense FLOP/s
    mem_bytes: float
    mem_bw: float  # bytes/s
    power_w: float
    efficiency: Dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.efficiency is None:
            object.__setattr__(self, "efficiency", dict(EFF))


H100 = Device("NVIDIA H100 SXM", {"bf16": 989e12, "fp8": 1979e12, "fp32": 67e12}, 80e9, 3.35e12, 700)
A100 = Device("NVIDIA A100 SXM 80GB", {"bf16": 312e12, "fp32": 19.5e12}, 80e9, 2.039e12, 400)
L4 = Device("NVIDIA L4", {"bf16": 121e12, "fp8": 242e12}, 24e9, 0.3e12, 72)

# AMD Instinct (CDNA3/CDNA4), datasheet dense (non-sparse) rates.
# Sources: amd.com MI300X / MI325X / MI355X product pages (peak matrix performance, dense).
#   MI300X: BF16 1307.4 TF, FP8 2614.9 TF, FP32 163.4 TF; 192 GB HBM3 @ 5.3 TB/s; 750 W TBP.
#   MI325X: same compute as MI300X; 256 GB HBM3E @ 6.0 TB/s; 1000 W TBP.
#   MI355X: BF16 2500 TF, FP8 5000 TF (OCP FP8), FP32 157.3 TF; 288 GB HBM3E @ 8 TB/s; 1400 W TBP.
# MXFP6/MXFP4 (MI355X, 10.1 PF) are outside the planner's dtype set (bf16/fp8 only).
MI300X = Device("AMD Instinct MI300X", {"bf16": 1307.4e12, "fp8": 2614.9e12, "fp32": 163.4e12}, 192e9, 5.3e12, 750)
MI325X = Device("AMD Instinct MI325X", {"bf16": 1307.4e12, "fp8": 2614.9e12, "fp32": 163.4e12}, 256e9, 6.0e12, 1000)
MI355X = Device("AMD Instinct MI355X", {"bf16": 2500e12, "fp8": 5000e12, "fp32": 157.3e12}, 288e9, 8.0e12, 1400)


@dataclass(frozen=True)
class Link:
    bw: float
    latency_s: float


@dataclass(frozen=True)
class Cluster:
    key: str
    label: str
    device: Device
    devices_per_node: int
    intra: Link
    inter: Link
    cost_per_device_hour: float
    provenance: Dict[str, str]


CLUSTERS: Dict[str, Cluster] = {
    "h100_nvlink_ib": Cluster(
        "h100_nvlink_ib", "H100 SXM · NVLink + InfiniBand", H100, 8,
        Link(450e9, 3e-6), Link(50e9, 10e-6), 2.5,
        {"peak": "spec", "intra.bw": "spec", "inter.bw": "spec", "latency": "placeholder", "price": "placeholder", "efficiency": "placeholder"},
    ),
    "a100_nvlink_ib": Cluster(
        "a100_nvlink_ib", "A100 80GB · NVLink + InfiniBand", A100, 8,
        Link(300e9, 3e-6), Link(25e9, 10e-6), 1.5,
        {"peak": "spec", "intra.bw": "spec", "inter.bw": "spec", "latency": "placeholder", "price": "placeholder", "efficiency": "calibrated (training)"},
    ),
    "l4_pcie": Cluster(
        "l4_pcie", "L4 · PCIe", L4, 8,
        Link(32e9, 5e-6), Link(12.5e9, 15e-6), 0.8,
        {"peak": "spec", "intra.bw": "spec (PCIe Gen4 x16)", "inter.bw": "placeholder", "latency": "placeholder", "price": "placeholder", "efficiency": "placeholder"},
    ),
    # AMD Instinct platforms: 8 OAM modules per UBB node, fully connected over AMD Infinity
    # Fabric (full mesh); scale-out over standard 400GbE Ethernet (RoCEv2).
    # intra bw = uni-directional aggregate per GPU: MI300X/MI325X 7 peer links x 128 GB/s / 2
    # = 448 GB/s; MI355X 7 links x 153 GB/s / 2 ~= 535 GB/s (CDNA4 whitepaper: >1 TB/s bidir).
    # Fabric hop latency and cloud price are engineering placeholders until measured.
    "mi300x_if_roce": Cluster(
        "mi300x_if_roce", "MI300X · Infinity Fabric + 400GbE", MI300X, 8,
        Link(448e9, 5e-6), Link(50e9, 10e-6), 2.0,
        {"peak": "spec (AMD datasheet, dense)", "intra.bw": "spec (7x128 GB/s IF full mesh, uni-dir)", "inter.bw": "placeholder (400GbE RoCE est.)", "latency": "placeholder", "price": "placeholder (market est.)", "efficiency": "placeholder"},
    ),
    "mi325x_if_roce": Cluster(
        "mi325x_if_roce", "MI325X · Infinity Fabric + 400GbE", MI325X, 8,
        Link(448e9, 5e-6), Link(50e9, 10e-6), 2.5,
        {"peak": "spec (AMD datasheet, dense)", "intra.bw": "spec (7x128 GB/s IF full mesh, uni-dir)", "inter.bw": "placeholder (400GbE RoCE est.)", "latency": "placeholder", "price": "placeholder (market est.)", "efficiency": "placeholder"},
    ),
    "mi355x_if_roce": Cluster(
        "mi355x_if_roce", "MI355X · Infinity Fabric + 400GbE", MI355X, 8,
        Link(535e9, 5e-6), Link(50e9, 10e-6), 3.0,
        {"peak": "spec (AMD datasheet, dense)", "intra.bw": "spec (7x153 GB/s IF full mesh, uni-dir)", "inter.bw": "placeholder (400GbE RoCE est.)", "latency": "placeholder", "price": "placeholder (market est.)", "efficiency": "placeholder"},
    ),
}


def flops(d: Device, dtype: str) -> float:
    if dtype not in d.peak_flops:
        raise ValueError(f"{d.name} has no {dtype} rate")
    return d.peak_flops[dtype]


def link_for_group(c: Cluster, size: float, stride: float = 1) -> Link:
    return c.intra if size * stride <= c.devices_per_node else c.inter


# ---- cost models (ring collectives, alpha-beta; roofline) -------------------------------------


def roofline_time(fl: float, byts: float, d: Device, dtype: str = "bf16") -> float:
    tc = fl / (flops(d, dtype) * d.efficiency["matmul"]) if fl else 0.0
    tm = byts / (d.mem_bw * d.efficiency["memory"]) if byts else 0.0
    return max(tc, tm)


def all_reduce(s_bytes: float, n: float, link: Link) -> float:
    if n <= 1 or s_bytes <= 0:
        return 0
    return 2 * (n - 1) * link.latency_s + ((2 * (n - 1)) / n) * s_bytes / link.bw


def all_gather(s_bytes: float, n: float, link: Link) -> float:
    if n <= 1 or s_bytes <= 0:
        return 0
    return (n - 1) * link.latency_s + ((n - 1) / n) * s_bytes / link.bw


def p2p(s_bytes: float, link: Link) -> float:
    return link.latency_s + s_bytes / link.bw if s_bytes > 0 else 0


# ---- transformer description -------------------------------------------------------------------


@dataclass(frozen=True)
class Transformer:
    name: str
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    ffn: int
    vocab: int
    gated_mlp: bool = True
    tied_embeddings: bool = False


def head_dim(m: Transformer) -> float:
    return math.floor(m.hidden / m.heads)


def params_per_layer(m: Transformer) -> float:
    h = m.hidden
    kv = m.kv_heads * head_dim(m)
    return h * h + 2 * h * kv + h * h + (3 if m.gated_mlp else 2) * h * m.ffn + 2 * h


def embedding_params(m: Transformer) -> float:
    return m.vocab * m.hidden * (1 if m.tied_embeddings else 2)


def params(m: Transformer) -> float:
    return m.layers * params_per_layer(m) + embedding_params(m) + m.hidden


def _t(name, layers, hidden, heads, kv_heads, ffn, vocab, gated_mlp=True, tied_embeddings=False):
    return Transformer(name, layers, hidden, heads, kv_heads, ffn, vocab, gated_mlp, tied_embeddings)


MODELS: Dict[str, Transformer] = {
    "llama2-7b": _t("Llama 2 7B", 32, 4096, 32, 32, 11008, 32000),
    "llama3-8b": _t("Llama 3 8B", 32, 4096, 32, 8, 14336, 128256),
    "llama2-70b": _t("Llama 2 70B", 80, 8192, 64, 8, 28672, 32000),
    "gpt3-175b": _t("GPT-3 175B", 96, 12288, 96, 96, 49152, 50257, False, True),
}
for _k, _heads, _hidden, _layers in [
    ("1.7b", 24, 2304, 24), ("3.6b", 32, 3072, 30), ("7.5b", 32, 4096, 36),
    ("18.4b", 48, 6144, 40), ("39.1b", 64, 8192, 48), ("76.1b", 80, 10240, 60),
    ("145.6b", 96, 12288, 80), ("310.1b", 128, 16384, 96), ("529.6b", 128, 20480, 105),
    ("1008b", 160, 25600, 128), ("174.6b", 96, 12288, 96),
]:
    MODELS[f"megatron-gpt-{_k}"] = _t(f"Megatron GPT {_k.upper()}", _layers, _hidden, _heads, _heads, 4 * _hidden, 51200, False, True)
