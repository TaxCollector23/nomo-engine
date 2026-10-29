"""Spec-range Monte Carlo golden for the browser port (src/planner/specUncertainty.ts).

For uncalibrated clusters (H100, AMD Instinct) the browser draws a mulberry32
LCG over disclosed spec ranges (achievable MFU / overlap / pipeline bubble for
training; datasheet flops + HBM derating for serving) and re-evaluates one plan
through the same evaluator. This study replays the exact same LCG and ranges in
Python and pins the resulting percentiles plus tornado swings, so `npm run
verify` can hold both implementations to identical numbers.

Usage (from nomo-planner/): PYTHONPATH=. python3 studies/export_spec_mc.py ../scripts/spec-mc-golden.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import dataclasses

from nomo_planner.domains import build
from nomo_planner.domains.packs import InferencePack, TrainingPack

MASK32 = 0xFFFFFFFF
SAMPLES = 400


def mulberry32(seed: int):
    state = seed & MASK32

    def draw() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & MASK32
        t = state
        t = ((t ^ (t >> 15)) * (t | 1)) & MASK32
        t = (t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & MASK32)) & MASK32)) & MASK32
        value = ((t ^ (t >> 14)) & MASK32) / 4294967296.0
        return value

    return draw


def spec_mc_seed(text: str) -> int:
    h = 1779033703
    for ch in text:
        h = (h ^ ord(ch)) & MASK32
        h = (h * 3432918353) & MASK32
        h = ((h << 13) & MASK32) | (h >> 19)
    return h & MASK32


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def plan_key(plan: dict) -> str:
    # byte-identical to JS JSON.stringify(sorted entries): no whitespace
    return json.dumps([[k, plan[k]] for k in sorted(plan)], separators=(",", ":"))


TRAIN_INPUTS = [
    {"name": "e_matmul", "low": 0.35, "high": 0.65, "default": 0.55},
    {"name": "dp_overlap", "low": 0.5, "high": 0.85, "default": 0.7},
    {"name": "bubble_scale", "low": 0.85, "high": 1.15, "default": 1.0},
]
INFER_INPUTS = [
    {"name": "compute_eff", "low": 0.85, "high": 1.0, "default": 0.95},
    {"name": "mem_eff", "low": 0.8, "high": 1.0, "default": 0.92},
]


def train_case_spec() -> dict:
    return {
        "name": "train_llama3_8b_mi355x",
        "domain": "llm_training",
        "problem": {
            "model": "llama3-8b", "cluster": "mi355x_if_roce", "seq_len": 4096,
            "global_batch_tokens": 4194304, "total_tokens": 1e12,
            "device_counts": [256, 512, 1024], "usable_memory": 0.92,
        },
        "plan": {"devices": 512, "tp": 4, "pp": 4, "zero": 1, "recompute": "selective", "micro_batch": 2, "matmul_dtype": "bf16"},
    }


def infer_case_spec() -> dict:
    return {
        "name": "serve_llama3_8b_mi300x",
        "domain": "llm_inference",
        "problem": {
            "model": "llama3-8b", "cluster": "mi300x_if_roce", "prompt_tokens": 2048,
            "output_tokens": 256, "max_ms_per_token": 50, "max_quality_loss": 1.0,
            "usable_memory": 0.9, "batch_sizes": [1, 8, 32, 64, 128, 256],
        },
        "plan": {"weights": "fp8", "kv": "int4", "tp": 1, "batch": 256},
    }


def percentile_block(values: list[float]) -> dict:
    return {"p10": quantile(values, 0.1), "p50": quantile(values, 0.5), "p90": quantile(values, 0.9)}


def export_train(case: dict) -> dict:
    pack = build("llm_training", {**case["problem"], "device_counts": [int(x) for x in case["problem"]["device_counts"]]})
    problem = pack.p
    plan = {key: (int(value) if key not in ("recompute", "matmul_dtype") else value) for key, value in case["plan"].items()}
    objectives = [o["name"] for o in pack.objectives()]
    rng = mulberry32(spec_mc_seed(f"{pack.name}:{plan_key(case['plan'])}"))
    draws = {name: [] for name in objectives}
    for _ in range(SAMPLES):
        params = {
            "e_matmul": TRAIN_INPUTS[0]["low"] + (TRAIN_INPUTS[0]["high"] - TRAIN_INPUTS[0]["low"]) * rng(),
            "dp_overlap": TRAIN_INPUTS[1]["low"] + (TRAIN_INPUTS[1]["high"] - TRAIN_INPUTS[1]["low"]) * rng(),
            "bubble_scale": TRAIN_INPUTS[2]["low"] + (TRAIN_INPUTS[2]["high"] - TRAIN_INPUTS[2]["low"]) * rng(),
        }
        metrics = TrainingPack(problem, params).evaluate(plan)
        for name in objectives:
            value = metrics["objectives"][name]
            if math.isfinite(value) and value > 0:
                draws[name].append(value)
    intervals = {name: percentile_block(values) for name, values in draws.items()}
    base = TrainingPack(problem).evaluate(plan)["objectives"]
    tornado = {}
    for name in objectives:
        rows = []
        for index, spec in enumerate(TRAIN_INPUTS):
            low_params = {s["name"]: (s["low"] if index == i else s["default"]) for i, s in enumerate(TRAIN_INPUTS)}
            high_params = {s["name"]: (s["high"] if index == i else s["default"]) for i, s in enumerate(TRAIN_INPUTS)}
            low_value = TrainingPack(problem, low_params).evaluate(plan)["objectives"][name]
            high_value = TrainingPack(problem, high_params).evaluate(plan)["objectives"][name]
            rows.append({"name": spec["name"], "swingPct": abs(high_value - low_value) / max(abs(base[name]), 1e-12) * 100})
        tornado[name] = sorted(rows, key=lambda row: -row["swingPct"])
    return {"name": case["name"], "samples": SAMPLES, "inputs": TRAIN_INPUTS, "objectives": objectives, "case_settings": case["problem"], "plan": case["plan"],
            "intervals": intervals, "tornado": tornado}


def export_infer(case: dict) -> dict:
    pack = build("llm_inference", {**case["problem"], "batch_sizes": [int(x) for x in case["problem"]["batch_sizes"]]})
    problem = pack.p
    cluster = problem.cluster
    plan = {key: (int(value) if key in ("tp", "batch") else value) for key, value in case["plan"].items()}
    objectives = [o["name"] for o in pack.objectives()]

    def with_eff(compute_eff: float, mem_eff: float) -> InferencePack:
        device = dataclasses.replace(
            cluster.device,
            peak_flops={key: value * compute_eff for key, value in cluster.device.peak_flops.items()},
            mem_bw=cluster.device.mem_bw * mem_eff,
        )
        perturbed = dataclasses.replace(cluster, device=device)
        return InferencePack(dataclasses.replace(problem, cluster=perturbed))

    rng = mulberry32(spec_mc_seed(f"{pack.name}:{plan_key(case['plan'])}"))
    draws = {name: [] for name in objectives}
    for _ in range(SAMPLES):
        compute_eff = INFER_INPUTS[0]["low"] + (INFER_INPUTS[0]["high"] - INFER_INPUTS[0]["low"]) * rng()
        mem_eff = INFER_INPUTS[1]["low"] + (INFER_INPUTS[1]["high"] - INFER_INPUTS[1]["low"]) * rng()
        metrics = with_eff(compute_eff, mem_eff).evaluate(plan)
        for name in objectives:
            value = metrics["objectives"][name]
            if math.isfinite(value) and value > 0:
                draws[name].append(value)
    intervals = {name: percentile_block(values) for name, values in draws.items()}
    base = with_eff(INFER_INPUTS[0]["default"], INFER_INPUTS[1]["default"]).evaluate(plan)["objectives"]
    tornado = {}
    for name in objectives:
        rows = []
        for index, spec in enumerate(INFER_INPUTS):
            low_value = with_eff(
                spec["low"] if index == 0 else INFER_INPUTS[0]["default"],
                spec["low"] if index == 1 else INFER_INPUTS[1]["default"],
            ).evaluate(plan)["objectives"][name]
            high_value = with_eff(
                spec["high"] if index == 0 else INFER_INPUTS[0]["default"],
                spec["high"] if index == 1 else INFER_INPUTS[1]["default"],
            ).evaluate(plan)["objectives"][name]
            rows.append({"name": spec["name"], "swingPct": abs(high_value - low_value) / max(abs(base[name]), 1e-12) * 100})
        tornado[name] = sorted(rows, key=lambda row: -row["swingPct"])
    return {"name": case["name"], "samples": SAMPLES, "inputs": INFER_INPUTS, "objectives": objectives, "case_settings": case["problem"], "plan": case["plan"],
            "intervals": intervals, "tornado": tornado}


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../scripts/spec-mc-golden.json")
    payload = {
        "contract": "spec-mc-golden-v1",
        "method": "spec-range Monte Carlo",
        "cases": [export_train(train_case_spec()), export_infer(infer_case_spec())],
    }
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {len(payload['cases'])} spec-mc golden cases to {out}")


if __name__ == "__main__":
    main()
