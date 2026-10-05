"""A1 uncertainty reference implementation.

This module intentionally uses only the Python standard library.  It fits the
same A100 training-step equations used by the browser port, then bootstraps
the published rows by strategy and refits the six calibrated parameters in
log-error space.  It is not a Bayesian posterior; samples are an empirical
bootstrap distribution and are labelled that way in reports and exports.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


CALIBRATED_PARAMS: dict[str, float] = {
    "zero3_overlap": 0.0,
    "e_matmul": 0.6400824774976149,
    "bubble_scale": 0.41036560876264244,
    "gemm_h_half": 320.5480034537571,
    "dp_overlap": 0.9999999999999999,
    "zero3_latency_us": 78.40485850566189,
}

PARAM_ORDER = tuple(CALIBRATED_PARAMS)
BOUNDS: dict[str, tuple[float, float]] = {
    "zero3_overlap": (0.0, 0.99),
    "e_matmul": (0.1, 1.2),
    "bubble_scale": (0.05, 2.0),
    "gemm_h_half": (0.0, 20_000.0),
    "dp_overlap": (0.0, 0.99),
    "zero3_latency_us": (0.0, 1_000.0),
}
INITIAL_STEPS = {
    "zero3_overlap": 0.1,
    "e_matmul": 0.08,
    "bubble_scale": 0.08,
    "gemm_h_half": 80.0,
    "dp_overlap": 0.08,
    "zero3_latency_us": 20.0,
}


@dataclass(frozen=True)
class Observation:
    row_id: str
    source: str
    url: str
    model: str
    seq_len: int
    global_batch_tokens: int
    cluster: str
    devices: int
    tp: int
    pp: int
    zero: int
    recompute: str
    micro_batch: int
    precision: str
    metric: str
    value: float
    notes: str

    @property
    def strategy(self) -> str:
        return "zero3" if self.zero == 3 else "tensor_pipeline"


@dataclass(frozen=True)
class PosteriorSample:
    params: dict[str, float]
    residual_sigma: float
    fit_error: float
    source_rows: int

    def as_dict(self) -> dict[str, object]:
        return {"params": self.params, "residual_sigma": self.residual_sigma,
                "fit_error": self.fit_error, "source_rows": self.source_rows}


MODELS: dict[str, tuple[int, int, int, int, int, bool]] = {
    # layers, hidden, heads, ffn, vocab, tied embeddings
    "megatron-gpt-1.7b": (24, 2304, 24, 4 * 2304, 51200, True),
    "megatron-gpt-3.6b": (30, 3072, 32, 4 * 3072, 51200, True),
    "megatron-gpt-7.5b": (36, 4096, 32, 4 * 4096, 51200, True),
    "megatron-gpt-18.4b": (40, 6144, 48, 4 * 6144, 51200, True),
    "megatron-gpt-39.1b": (48, 8192, 64, 4 * 8192, 51200, True),
    "megatron-gpt-76.1b": (60, 10240, 80, 4 * 10240, 51200, True),
    "megatron-gpt-145.6b": (80, 12288, 96, 4 * 12288, 51200, True),
    "megatron-gpt-174.6b": (96, 12288, 96, 4 * 12288, 51200, True),
    "megatron-gpt-310.1b": (96, 16384, 128, 4 * 16384, 51200, True),
    "megatron-gpt-529.6b": (105, 20480, 128, 4 * 20480, 51200, True),
    "megatron-gpt-1008b": (128, 25600, 160, 4 * 25600, 51200, True),
}


def load_observations(path: str | Path) -> list[Observation]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(Observation(
                row_id=row["row_id"], source=row["source"], url=row["url"], model=row["model"],
                seq_len=int(row["seq_len"]), global_batch_tokens=int(row["global_batch_tokens"]),
                cluster=row["cluster"], devices=int(row["devices"]), tp=int(row["tp"]), pp=int(row["pp"]),
                zero=int(row["zero"]), recompute=row["recompute"], micro_batch=int(row["micro_batch"]),
                precision=row["precision"], metric=row["metric"], value=float(row["value"]), notes=row["notes"],
            ))
    return rows


def _params(model: tuple[int, int, int, int, int, bool]) -> int:
    layers, hidden, heads, ffn, vocab, tied = model
    head_dim = hidden // heads
    kv = heads * head_dim
    per_layer = hidden * hidden + 2 * hidden * kv + hidden * hidden + 2 * hidden * ffn + 2 * hidden
    return layers * per_layer + vocab * hidden * (1 if tied else 2) + hidden


def measured_step(row: Observation) -> float:
    layers, hidden, _heads, _ffn, vocab, _tied = MODELS[row.model]
    batch = row.global_batch_tokens // row.seq_len
    flops = 96 * batch * row.seq_len * layers * hidden**2 * (
        1 + row.seq_len / (6 * hidden) + vocab / (16 * layers * hidden)
    )
    return flops / (row.devices * row.value * 1e12)


def _link(group_size: int, stride: int) -> tuple[float, float]:
    # A100 cluster: 300 GB/s intra-node, 25 GB/s inter-node; 3/10 us latency.
    return (300e9, 3e-6) if group_size * stride <= 8 else (25e9, 10e-6)


def _all_reduce(size: float, n: int, link: tuple[float, float]) -> float:
    if n <= 1 or size <= 0:
        return 0.0
    bw, latency = link
    return 2 * (n - 1) * latency + (2 * (n - 1) / n) * size / bw


def _all_gather(size: float, n: int, link: tuple[float, float]) -> float:
    if n <= 1 or size <= 0:
        return 0.0
    bw, latency = link
    return (n - 1) * latency + ((n - 1) / n) * size / bw


def _p2p(size: float, link: tuple[float, float]) -> float:
    bw, latency = link
    return latency + size / bw if size > 0 else 0.0


def predict_step(row: Observation, params: Mapping[str, float]) -> float:
    layers, hidden, heads, ffn, _vocab, _tied = MODELS[row.model]
    g, tp, pp, zero, batch_size = row.devices, row.tp, row.pp, row.zero, row.micro_batch
    seqs = row.global_batch_tokens // row.seq_len
    micro_batches = max(1, seqs // (max(1, g // (tp * pp)) * batch_size))
    layers_per_stage = layers // max(pp, 1)
    stage_params = layers_per_stage * (_params(MODELS[row.model]) - 51200 * hidden - hidden) // layers + 51200 * hidden
    n_local = stage_params / tp
    matmul = params["e_matmul"]
    w_local = hidden / tp
    e_eff = matmul * w_local / (w_local + params["gemm_h_half"])
    model_params = _params(MODELS[row.model])
    attn = 12 * layers * row.seq_len * hidden
    extra = 2 * model_params + 4 * layers * row.seq_len * hidden
    hw_tokens = 6 * model_params + attn + extra
    compute = (hw_tokens * row.global_batch_tokens) / (g * 312e12 * e_eff)
    bubble = 1 + params["bubble_scale"] * (pp - 1) / micro_batches
    msg = batch_size * row.seq_len * hidden * 2.0
    n_ar = 6
    t_tp = layers_per_stage * micro_batches * n_ar * _all_reduce(msg, tp, _link(tp, 1)) if tp > 1 else 0.0
    t_pp = 2 * micro_batches * _p2p(msg / tp, _link(pp, tp)) if pp > 1 else 0.0
    data_parallel = max(1, g // (tp * pp))
    dp_link = _link(data_parallel, tp * pp)
    dp_bytes = 2 * n_local
    t_dp = _all_reduce(dp_bytes, data_parallel, dp_link) * (1 - params["dp_overlap"])
    if zero >= 3:
        zero_link = (dp_link[0], params["zero3_latency_us"] * 1e-6)
        t_dp += 2 * micro_batches * layers_per_stage * _all_gather(
            dp_bytes / max(1, layers_per_stage), data_parallel, zero_link
        ) * (1 - params["zero3_overlap"])
    return compute * bubble + t_tp + t_pp + t_dp


def _loss(params: Mapping[str, float], rows: Sequence[Observation]) -> float:
    errors = []
    for row in rows:
        predicted = predict_step(row, params)
        measured = measured_step(row)
        if not math.isfinite(predicted) or predicted <= 0:
            return float("inf")
        errors.append(math.log(predicted / measured))
    return sum(error * error for error in errors) / max(1, len(errors))


def fit_parameters(rows: Sequence[Observation], start: Mapping[str, float] | None = None) -> tuple[dict[str, float], float]:
    current = {name: float((start or CALIBRATED_PARAMS)[name]) for name in PARAM_ORDER}
    best_loss = _loss(current, rows)
    steps = dict(INITIAL_STEPS)
    # Coordinate descent keeps the reference deterministic and avoids adding a
    # numerical dependency just to fit six scalar hardware parameters.
    for _ in range(28):
        improved = False
        for name in PARAM_ORDER:
            low, high = BOUNDS[name]
            candidates = (current[name] - steps[name], current[name], current[name] + steps[name])
            for candidate in candidates:
                candidate = min(high, max(low, candidate))
                trial = dict(current)
                trial[name] = candidate
                loss = _loss(trial, rows)
                if loss + 1e-14 < best_loss:
                    current, best_loss = trial, loss
                    improved = True
        if not improved:
            steps = {name: step * 0.55 for name, step in steps.items()}
    sigma = math.sqrt(best_loss)
    return current, sigma


def fit_bootstrap(rows: Sequence[Observation], replicates: int = 256, seed: int = 20260929) -> list[PosteriorSample]:
    if not rows:
        raise ValueError("cannot bootstrap an empty observation set")
    groups: dict[str, list[Observation]] = {}
    for row in rows:
        groups.setdefault(row.strategy, []).append(row)
    rng = random.Random(seed)
    samples: list[PosteriorSample] = []
    for _ in range(replicates):
        resampled = [rng.choice(group) for group in groups.values() for _ in range(len(group))]
        params, sigma = fit_parameters(resampled)
        samples.append(PosteriorSample(params=params, residual_sigma=sigma,
                                       fit_error=_loss(params, resampled), source_rows=len(resampled)))
    return samples


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile of empty values")
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def predictive_interval(row: Observation, samples: Sequence[PosteriorSample], seed: int = 20260929) -> dict[str, float]:
    rng = random.Random(seed)
    draws = []
    for sample in samples:
        mean = predict_step(row, sample.params)
        draws.append(math.exp(math.log(mean) + rng.gauss(0.0, sample.residual_sigma)))
    return {"median": _quantile(draws, 0.5), "low": _quantile(draws, 0.05), "high": _quantile(draws, 0.95)}


def safest_plan(plans: Sequence[Mapping[str, object]], intervals: Mapping[str, Mapping[str, Mapping[str, float]]],
                objectives: Sequence[tuple[str, bool]]) -> Mapping[str, object] | None:
    """Choose the plan with the smallest worst normalized interval-regret bound.

    ``objectives`` contains ``(name, maximize)`` pairs. This is intentionally
    an interval safeguard, not a claim that correlated posterior regret has
    been observed; callers should label it accordingly.
    """
    if not plans or not objectives:
        return None
    rows = [(plan, intervals.get(str(plan.get("key", "")))) for plan in plans]
    if any(interval is None for _, interval in rows):
        return None
    best: Mapping[str, object] | None = None
    best_regret = float("inf")
    for plan, interval in rows:
        assert interval is not None
        worst = 0.0
        for name, maximize in objectives:
            values = [candidate[name] for _, candidate in rows if candidate is not None and name in candidate]
            current = interval.get(name)
            if len(values) != len(rows) or current is None:
                return None
            scale = max(max(abs(value["median"]) for value in values), 1e-12)
            comparator = max(value["high"] for value in values) if maximize else min(value["low"] for value in values)
            regret = max(0.0, (comparator - current["low"]) / scale) if maximize else max(0.0, (current["high"] - comparator) / scale)
            worst = max(worst, regret)
        key = str(plan.get("key", ""))
        best_key = str(best.get("key", "")) if best else ""
        if worst < best_regret - 1e-12 or (abs(worst - best_regret) <= 1e-12 and key < best_key):
            best, best_regret = plan, worst
    return best


def sample_manifest(samples: Iterable[PosteriorSample], *, cluster: str, seed: int, source: str) -> dict[str, object]:
    materialized = list(samples)
    return {
        "method": "stratified non-parametric bootstrap over published runs; log-error refit",
        "cluster": cluster,
        "seed": seed,
        "source": source,
        "sample_count": len(materialized),
        "parameter_order": list(PARAM_ORDER),
        "samples": [sample.as_dict() for sample in materialized],
    }
