"""Domain packs: training, serving, co-design.

Restored Python reference for src/planner/packs.ts (which says, in its header:
"Port of nomo_planner/domains/*.py"). Formula citations: PaLM FLOPs accounting
(Chowdhery et al. 2022), activation memory after Korthikanti et al.
(arXiv:2205.05198), 1F1B pipeline bubble after Chen et al. (arXiv:1604.06174),
ring collectives after Thakur et al. (2005), roofline (Williams et al. 2009),
scaling laws after Hoffmann et al. 2022 and Besiroglu et al. 2024.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .hardware import CLUSTERS, Cluster, Device, Link, Transformer, all_gather, all_reduce, flops, head_dim, link_for_group, p2p, params, params_per_layer, roofline_time

# ================================================================== shared types


def violation(metrics: dict) -> float:
    return sum(max(0.0, v) for v in metrics["constraints"].values())


def feasible(metrics: dict) -> bool:
    return violation(metrics) <= 0


# ================================================================== training

TRAINING_DEFAULTS: Dict[str, float] = {
    "e_matmul": 0.55, "dp_overlap": 0.7, "overhead_gb": 3.0, "bubble_scale": 1.0,
    "zero3_overlap": 0.7, "gemm_h_half": 0.0, "zero3_latency_us": -1.0,
}


@dataclass
class TrainingProblem:
    model: Transformer
    cluster: Cluster
    seq_len: int
    global_batch_tokens: int
    total_tokens: int
    device_counts: List[int]
    usable_memory: float


class TrainingPack:
    name = "llm_training"
    description = "Plan the parallel layout of an LLM training run for minimum time and cost."
    calibration = None

    def __init__(self, problem: TrainingProblem, params_: Optional[Dict[str, float]] = None):
        self.p = problem
        self.params = {**TRAINING_DEFAULTS, **(params_ or {})}

    def variables(self):
        dtypes = [d for d in ["bf16", "fp8"] if d in self.p.cluster.device.peak_flops]
        return [
            {"name": "devices", "choices": list(self.p.device_counts), "label": "number of GPUs"},
            {"name": "tp", "choices": [1, 2, 4, 8], "label": "tensor parallel size", "help": "splits each layer's matrices across GPUs"},
            {"name": "pp", "choices": [1, 2, 4, 8, 16], "label": "pipeline stages", "help": "splits the layers into consecutive stages"},
            {"name": "zero", "choices": [0, 1, 2, 3], "label": "ZeRO stage", "help": "shards optimizer / gradients / weights across data parallel"},
            {"name": "recompute", "choices": ["none", "selective", "full"], "label": "activation recomputation"},
            {"name": "micro_batch", "choices": [1, 2, 4, 8], "label": "micro-batch size (sequences)"},
            {"name": "matmul_dtype", "choices": dtypes, "label": "matmul precision"},
        ]

    def objectives(self):
        return [
            {"name": "days", "unit": "days", "label": "training time"},
            {"name": "cost_usd", "unit": "USD", "label": "training cost"},
        ]

    def describe(self, plan):
        d = math.floor(plan["devices"] / max(1, plan["tp"] * plan["pp"]))
        rc = {"none": "no", "selective": "selective", "full": "full"}[plan["recompute"]]
        return (f"{plan['devices']} GPUs as TP{plan['tp']} x PP{plan['pp']} x DP{d}, ZeRO-{plan['zero']}, "
                f"{rc} recompute, micro-batch {plan['micro_batch']}, {plan['matmul_dtype']}")

    def evaluate(self, plan: dict) -> dict:
        P, M, prm = self.p, self.p.model, self.params
        G, t, p, z = int(plan["devices"]), int(plan["tp"]), int(plan["pp"]), int(plan["zero"])
        b, rc, dt = int(plan["micro_batch"]), plan["recompute"], plan["matmul_dtype"]
        cl = P.cluster
        dev = cl.device
        s, h, a, L = P.seq_len, M.hidden, M.heads, M.layers
        notes: List[str] = []
        problems: List[str] = []
        if G % (t * p):
            problems.append("GPU count not divisible by TP x PP")
        d = max(1, math.floor(G / (t * p)))
        if a % t:
            problems.append("attention heads not divisible by TP")
        if M.kv_heads % t and t % M.kv_heads:
            problems.append("KV heads incompatible with TP")
        if L % p:
            problems.append("layers not divisible by PP")
        seqs = math.floor(P.global_batch_tokens / s)
        if seqs % (d * b):
            problems.append("global batch not divisible by DP x micro-batch")
        m = max(1, math.floor(seqs / (d * b)))

        per_layer = params_per_layer(M)
        layers_per_stage = math.floor(L / max(p, 1))
        stage_params = layers_per_stage * per_layer + M.vocab * h * (1 if (p > 1 or M.tied_embeddings) else 2)
        n_local = stage_params / t
        w = (2 * n_local) / (d if z >= 3 else 1)
        g = (2 * n_local) / (d if z >= 2 else 1)
        o = (12 * n_local) / (d if z >= 1 else 1)
        if rc == "none":
            act_layer = (s * b * h * (34 + (5 * a * s) / h)) / t
        elif rc == "selective":
            act_layer = (s * b * h * 34) / t
        else:
            act_layer = (2 * s * b * h) / t
        act = act_layer * layers_per_stage * min(p, m)
        logits = (4 * s * b * M.vocab) / t if p == 1 else 0
        mem = w + g + o + act + logits + prm["overhead_gb"] * 1e9
        cap = dev.mem_bytes * P.usable_memory

        N = params(M)
        attn = 12 * L * s * h
        extra = 0 if rc == "none" else (4 * L * s * h if rc == "selective" else 2 * N + 4 * L * s * h)
        hw_tok = 6 * N + attn + extra
        model_tok = 6 * N + attn
        tokens = P.global_batch_tokens
        peak = flops(dev, dt)
        w_local = h / t
        e_eff = (prm["e_matmul"] * w_local) / (w_local + prm["gemm_h_half"])
        compute = (hw_tok * tokens) / (G * peak * e_eff)
        bubble = 1 + (prm["bubble_scale"] * (p - 1)) / m
        msg = b * s * h * 2.0
        n_ar = 4 + (2 if rc == "full" else 0)
        t_tp = layers_per_stage * m * n_ar * all_reduce(msg, t, link_for_group(cl, t)) if t > 1 else 0
        t_pp = 2 * m * p2p(msg / t, link_for_group(cl, p, t)) if p > 1 else 0
        dp_link = link_for_group(cl, d, t * p)
        dp_bytes = 2 * n_local
        t_dp = all_reduce(dp_bytes, d, dp_link) * (1 - prm["dp_overlap"])
        if z >= 3:
            ls = max(1, layers_per_stage)
            z3 = Link(dp_link.bw, prm["zero3_latency_us"] * 1e-6) if prm["zero3_latency_us"] >= 0 else dp_link
            t_dp += 2 * m * ls * all_gather(dp_bytes / ls, d, z3) * (1 - prm["zero3_overlap"])
        step = compute * bubble + t_tp + t_pp + t_dp
        seconds = step * (P.total_tokens / tokens)
        cost = (seconds / 3600) * G * cl.cost_per_device_hour
        if dt == "fp8":
            notes.append("fp8 matmuls: speed modelled; training-quality effects are not modelled")
        return {
            "objectives": {"days": seconds / 86400, "cost_usd": cost},
            "constraints": {"valid_layout": float(len(problems)) if problems else -1.0, "memory": mem / cap - 1.0},
            "breakdown": {
                "params": N, "dp": d, "micro_batches": m, "step_s": step, "tokens_per_s": tokens / step,
                "mfu": (model_tok * tokens) / (step * G * peak), "energy_mwh": (G * dev.power_w * seconds) / 3.6e9,
                "memory_gb": mem / 1e9, "memory_cap_gb": cap / 1e9,
            },
            "notes": notes,
            "_breakdown_full": {
                "memory_parts_gb": {"weights": w / 1e9, "grads": g / 1e9, "optimizer": o / 1e9, "activations": act / 1e9, "logits": logits / 1e9, "overhead": prm["overhead_gb"]},
                "time_parts_s": {"compute": compute, "pipeline_bubble": compute * (bubble - 1), "tensor_parallel_comm": t_tp, "pipeline_comm": t_pp, "data_parallel_comm_exposed": t_dp},
                "layout_problems": problems,
            },
        }


# ================================================================== serving

BYTES: Dict[str, float] = {"bf16": 2, "fp8": 1, "int4": 0.5}
DEFAULT_QUALITY_LOSS: Dict[str, Dict[str, float]] = {
    "weights": {"bf16": 0, "fp8": 0.2, "int4": 1.0},
    "kv": {"bf16": 0, "fp8": 0.1, "int4": 0.8},
}


@dataclass
class ServingProblem:
    model: Transformer
    cluster: Cluster
    prompt_tokens: int
    output_tokens: int
    max_ms_per_token: Optional[int]
    max_quality_loss: float
    usable_memory: float
    batch_sizes: List[int]
    quality_loss: Dict[str, Dict[str, float]]
    paged_kv_block_tokens: int = 16  # vLLM-style paged KV block; 0 disables paging


class InferencePack:
    name = "llm_inference"
    description = "Choose precision, parallelism and batch size to serve an LLM at minimum cost per token."
    calibration = None
    overhead_gb = 2.0

    def __init__(self, problem: ServingProblem):
        self.p = problem

    def variables(self):
        return [
            {"name": "weights", "choices": ["bf16", "fp8", "int4"], "label": "weight precision"},
            {"name": "kv", "choices": ["bf16", "fp8", "int4"], "label": "KV-cache precision"},
            {"name": "tp", "choices": [1, 2, 4, 8], "label": "GPUs per model copy (tensor parallel)"},
            {"name": "batch", "choices": list(self.p.batch_sizes), "label": "concurrent sequences"},
        ]

    def objectives(self):
        return [
            {"name": "usd_per_mtok", "unit": "USD / 1M output tokens", "label": "cost per million tokens"},
            {"name": "ms_per_token", "unit": "ms", "label": "time per generated token"},
            {"name": "quality_loss", "unit": "points", "label": "assumed quality loss"},
        ]

    def describe(self, plan):
        tp = int(plan["tp"])
        return f"{plan['weights']} weights, {plan['kv']} KV cache, {tp} GPU{'' if tp == 1 else 's'} per copy, batch {plan['batch']}"

    def evaluate(self, plan: dict) -> dict:
        P, M, cl, dev = self.p, self.p.model, self.p.cluster, self.p.cluster.device
        t, B, wq, kq = int(plan["tp"]), int(plan["batch"]), plan["weights"], plan["kv"]
        problems: List[str] = []
        if t > cl.devices_per_node:
            problems.append("tensor parallel larger than a node")
        if M.heads % t or (M.kv_heads % t and t % M.kv_heads):
            problems.append("heads not divisible by tensor parallel")
        if wq == "fp8" and "fp8" not in dev.peak_flops:
            problems.append(f"{dev.name} has no fp8 support")
        cdt = "fp8" if (wq == "fp8" and "fp8" in dev.peak_flops) else "bf16"
        N, L, h = params(M), M.layers, M.hidden
        s_in, s_out = P.prompt_tokens, P.output_tokens
        w_bytes = N * BYTES[wq]
        kv_tok = 2 * L * M.kv_heads * head_dim(M) * BYTES[kq]
        kv_full = B * (s_in + s_out) * kv_tok
        # Paged KV (vLLM-style): allocation happens in fixed-size blocks, so the cache
        # carries expected padding of about half a block per sequence.
        kv_padding = 0.5 / P.paged_kv_block_tokens if P.paged_kv_block_tokens > 0 else 0.0
        kv_allocated = kv_full * (1.0 + kv_padding)
        mem = (w_bytes + kv_allocated) / t + self.overhead_gb * 1e9
        cap = dev.mem_bytes * P.usable_memory
        # Attention prefill FLOPs: 4·L·s²·h (2 matmuls × 2 FLOPs/MAC) — consistent with the
        # decode term below and with layers.py. The historical 2·L·s²·h undercounted attention 2×.
        prefill_flops = ((2 * N * s_in + 4 * L * s_in * s_in * h) * B) / t
        prefill = roofline_time(prefill_flops, w_bytes / t, dev, cdt)
        kv_avg = B * (s_in + s_out / 2) * kv_tok
        dec_flops = ((2 * N + 4 * L * (s_in + s_out / 2) * h) * B) / t
        link = link_for_group(cl, t)
        decode = roofline_time(dec_flops, (w_bytes + kv_avg) / t, dev, cdt)
        if t > 1:
            decode += 2 * L * all_reduce(B * h * 2.0, t, link)
        cycle = prefill + s_out * decode
        tok_s = (B * s_out) / cycle
        usd = ((cl.cost_per_device_hour * t) / 3600 / tok_s) * 1e6
        # First-order latency decomposition (static batching): a decode step waits for the
        # whole batch, so time-between-tokens carries the amortized prefill; TTFT is the prefill.
        tbt_static_ms = (decode + prefill / s_out) * 1e3
        ttft_static_ms = prefill * 1e3
        q = P.quality_loss["weights"][wq] + P.quality_loss["kv"][kq]
        cons = {
            "valid": float(len(problems)) if problems else -1.0,
            "memory": mem / cap - 1.0,
            "quality": (q - P.max_quality_loss) / max(P.max_quality_loss, 1e-9),
        }
        if P.max_ms_per_token is not None:
            cons["latency"] = (decode * 1e3) / P.max_ms_per_token - 1.0
        mem_t = (w_bytes + kv_avg) / (dev.mem_bw * dev.efficiency["memory"])
        cmp_t = dec_flops / (flops(dev, cdt) * dev.efficiency["matmul"])
        notes = ["quality_loss values are assumptions until replaced by your own evaluations"]
        return {
            "objectives": {"usd_per_mtok": usd, "ms_per_token": decode * 1e3, "quality_loss": q},
            "constraints": cons,
            "breakdown": {
                "tokens_per_s": tok_s, "tokens_per_s_per_gpu": tok_s / t, "memory_gb": mem / 1e9, "memory_cap_gb": cap / 1e9,
                "prefill_s": prefill, "decode_step_s": decode, "kv_gb": kv_full / 1e9 / t, "weights_gb": w_bytes / 1e9 / t,
                "tbt_static_ms": tbt_static_ms, "ttft_static_ms": ttft_static_ms,
            },
            "notes": notes + [f"KV cache is paged in {P.paged_kv_block_tokens}-token blocks with ~{kv_padding * 100:.1f}% expected padding" if P.paged_kv_block_tokens > 0 else "paged KV disabled"],
            "_breakdown_full": {"decode_bound": "memory" if mem_t > cmp_t else "compute", "layout_problems": problems},
        }


# ================================================================== co-design

SCALING_LAWS: Dict[str, dict] = {
    "besiroglu2024": {"E": 1.8172, "A": 482.01, "B": 2085.43, "alpha": 0.3478, "beta": 0.3658, "label": "Besiroglu et al. 2024 (replication)", "cite": "arXiv:2404.10102"},
    "hoffmann2022": {"E": 1.69, "A": 406.4, "B": 410.7, "alpha": 0.34, "beta": 0.28, "label": "Hoffmann et al. 2022 (Chinchilla)", "cite": "arXiv:2203.15556"},
}
DEFAULT_ATTENTION_PENALTY: Dict[str, float] = {"MHA": 0, "GQA-8": 0.002, "MQA": 0.01}


@dataclass
class CodesignProblem:
    train_cluster: Cluster
    serve_cluster: Cluster
    tokens_served: float
    prompt_tokens: int
    output_tokens: int
    max_ms_per_token: Optional[int]
    budget_usd: Optional[float]
    max_train_tokens: Optional[float]
    seq_len: int
    vocab: int
    scaling_law: str
    train_mfu: float
    attention_loss_penalty: Dict[str, float]
    layers: List[int]
    hidden: List[int]
    tokens_per_param: List[int]


class CodesignPack:
    name = "arch_codesign"
    description = "Choose the model to build: size, shape, attention type and training length, for quality vs lifetime cost."
    calibration = None

    def __init__(self, problem: CodesignProblem):
        self.p = problem
        self._serve_cache: Dict[str, Optional[dict]] = {}
        if problem.scaling_law not in SCALING_LAWS:
            raise ValueError(f"unknown scaling law '{problem.scaling_law}'")

    def variables(self):
        return [
            {"name": "layers", "choices": list(self.p.layers), "label": "layers (depth)"},
            {"name": "hidden", "choices": list(self.p.hidden), "label": "hidden size (width)"},
            {"name": "attention", "choices": ["MHA", "GQA-8", "MQA"], "label": "attention type", "help": "fewer key/value heads = smaller KV cache = cheaper serving"},
            {"name": "tokens_per_param", "choices": list(self.p.tokens_per_param), "label": "training tokens per parameter"},
        ]

    def objectives(self):
        return [
            {"name": "loss", "unit": "nats/token", "label": "predicted loss"},
            {"name": "total_cost_usd", "unit": "USD", "label": "lifetime cost"},
        ]

    def model(self, plan) -> Transformer:
        h = int(plan["hidden"])
        L = int(plan["layers"])
        heads = math.floor(h / 128)
        kv = heads if plan["attention"] == "MHA" else min(8, heads) if plan["attention"] == "GQA-8" else 1
        ffn = int(math.floor(((8 / 3) * h) / 256 + 0.5)) * 256
        return Transformer(f"{L}x{h} {plan['attention']}", L, h, heads, kv, ffn, self.p.vocab, True, False)

    def describe(self, plan):
        N = params(self.model(plan))
        return (f"{N / 1e9:.1f}B params ({plan['layers']} layers x {plan['hidden']} wide, {plan['attention']}), "
                f"trained on {plan['tokens_per_param']} tokens/param ({(int(plan['tokens_per_param']) * N) / 1e12:.2f}T tokens)")

    def _serving(self, M: Transformer) -> Optional[dict]:
        key = f"{M.layers}/{M.hidden}/{M.kv_heads}"
        if key in self._serve_cache:
            return self._serve_cache[key]
        P = self.p
        pack = InferencePack(ServingProblem(M, P.serve_cluster, P.prompt_tokens, P.output_tokens, P.max_ms_per_token,
                                            1e9, 0.9, [1, 8, 32, 64, 128, 256], DEFAULT_QUALITY_LOSS))
        best = None
        for tp in [1, 2, 4, 8]:
            for batch in [1, 8, 32, 64, 128, 256]:
                mtr = pack.evaluate({"weights": "bf16", "kv": "bf16", "tp": tp, "batch": batch})
                if feasible(mtr) and (best is None or mtr["objectives"]["usd_per_mtok"] < best["usd"]):
                    best = {"usd": mtr["objectives"]["usd_per_mtok"], "ms": mtr["objectives"]["ms_per_token"], "tp": tp, "batch": batch}
        self._serve_cache[key] = best
        return best

    def evaluate(self, plan: dict) -> dict:
        P = self.p
        M = self.model(plan)
        N = params(M)
        tpp = int(plan["tokens_per_param"])
        D = tpp * N
        law = SCALING_LAWS[P.scaling_law]
        loss = law["E"] + law["A"] / N ** law["alpha"] + law["B"] / D ** law["beta"]
        loss += P.attention_loss_penalty.get(str(plan["attention"]), 0)
        dev = P.train_cluster.device
        train_flops = 6 * N * D + 12 * M.layers * P.seq_len * M.hidden * D
        gpu_hours = train_flops / (flops(dev, "bf16") * P.train_mfu) / 3600
        train_cost = gpu_hours * P.train_cluster.cost_per_device_hour
        serve = self._serving(M)
        serve_cost = (serve["usd"] * P.tokens_served) / 1e6 if serve else math.inf
        total = train_cost + serve_cost
        ratio = M.hidden / M.layers
        cons = {
            "shape": -1.0 if 32 <= ratio <= 256 else max(32 - ratio, ratio - 256) / 32,
            "servable": -1.0 if serve else 1.0,
        }
        if P.budget_usd:
            cons["budget"] = total / P.budget_usd - 1.0
        if P.max_train_tokens:
            cons["data"] = D / P.max_train_tokens - 1.0
        notes = ["attention-type quality penalties and train_mfu are assumptions until measured"]
        if tpp > 100:
            notes.append("more than ~100 tokens/parameter: the scaling law is extrapolated here (Sardana et al. 2024)")
        return {
            "objectives": {"loss": loss, "total_cost_usd": total if serve else 1e30},
            "constraints": cons,
            "breakdown": {
                "params": N, "train_tokens": D, "train_cost_usd": train_cost,
                "serve_cost_usd": serve_cost if serve else None, "train_gpu_hours": gpu_hours,
                "kv_heads": M.kv_heads, "serve_share": (serve_cost / total) if (serve and total > 0) else None,
            },
            "notes": notes,
            "_breakdown_full": {},
        }
