"""Domain packs registry: build packs from plain settings.

Restored Python reference for src/planner/registry.ts (header: "mirrors
nomo_planner/registry.py and the Python dataclass defaults").
"""
from __future__ import annotations

from typing import Dict, Optional

from .calibration import CALIBRATED
from .hardware import CLUSTERS, MODELS
from .packs import (DEFAULT_ATTENTION_PENALTY, DEFAULT_QUALITY_LOSS, CodesignPack, CodesignProblem, InferencePack,
                    ServingProblem, TrainingPack, TrainingProblem)

DomainId = str  # "llm_training" | "llm_inference" | "arch_codesign"

DEFAULT_SETTINGS: Dict[str, dict] = {
    "llm_training": {"model": "llama3-8b", "cluster": "h100_nvlink_ib", "seq_len": 4096, "global_batch_tokens": 4194304, "total_tokens": 1e12,
                     "device_counts": [64, 128, 256, 512], "usable_memory": 0.92},
    "llm_inference": {"model": "llama3-8b", "cluster": "h100_nvlink_ib", "prompt_tokens": 2048, "output_tokens": 256, "max_ms_per_token": 50,
                      "max_quality_loss": 1.0, "usable_memory": 0.9, "batch_sizes": [1, 8, 32, 64, 128, 256], "paged_kv_block_tokens": 16},
    "arch_codesign": {"train_cluster": "h100_nvlink_ib", "serve_cluster": "h100_nvlink_ib", "tokens_served": 1e12, "prompt_tokens": 1024,
                      "output_tokens": 256, "max_ms_per_token": 50, "budget_usd": None, "max_train_tokens": None, "seq_len": 4096, "vocab": 128256,
                      "scaling_law": "besiroglu2024", "train_mfu": 0.4, "layers": [16, 24, 32, 40, 48, 64, 80, 96, 120],
                      "hidden": [2048, 3072, 4096, 5120, 6144, 8192, 10240, 12288, 16384], "tokens_per_param": [10, 20, 40, 80, 160]},
}


def _num(v, d: float) -> float:
    return d if v is None or v == "" else float(v)


def _opt(v):
    return None if v is None or v == "" else float(v)


def build(domain: DomainId, s0: Optional[dict] = None):
    s = {**DEFAULT_SETTINGS[domain], **(s0 or {})}
    if domain == "llm_training":
        model = MODELS[str(s["model"])]
        cluster = CLUSTERS[str(s["cluster"])]
        if model is None or cluster is None:
            raise ValueError("unknown model or hardware")
        cal = CALIBRATED.get(cluster.key)
        pack = TrainingPack(
            TrainingProblem(model=model, cluster=cluster, seq_len=int(_num(s["seq_len"], 4096)),
                            global_batch_tokens=int(_num(s["global_batch_tokens"], 4194304)),
                            total_tokens=_num(s["total_tokens"], 1e12),
                            device_counts=[int(x) for x in s["device_counts"]],
                            usable_memory=_num(s["usable_memory"], 0.92)),
            cal["params"] if cal else None)
        if cal:
            pack.calibration = {"observations": cal["observations"], "loo_mape_pct": cal["loo_mape_pct"], "source": cal["source"]}
        return pack
    if domain == "llm_inference":
        model = MODELS[str(s["model"])]
        cluster = CLUSTERS[str(s["cluster"])]
        if model is None or cluster is None:
            raise ValueError("unknown model or hardware")
        return InferencePack(ServingProblem(
            model=model, cluster=cluster, prompt_tokens=int(_num(s["prompt_tokens"], 2048)), output_tokens=int(_num(s["output_tokens"], 256)),
            max_ms_per_token=_opt(s["max_ms_per_token"]), max_quality_loss=_num(s["max_quality_loss"], 1),
            usable_memory=_num(s["usable_memory"], 0.9), batch_sizes=[int(x) for x in s["batch_sizes"]],
            quality_loss=s.get("quality_loss") or DEFAULT_QUALITY_LOSS,
            paged_kv_block_tokens=int(_num(s.get("paged_kv_block_tokens", 16), 16))))
    tc = CLUSTERS[str(s["train_cluster"])]
    sc = CLUSTERS[str(s["serve_cluster"])]
    if tc is None or sc is None:
        raise ValueError("unknown hardware")
    return CodesignPack(CodesignProblem(
        train_cluster=tc, serve_cluster=sc, tokens_served=_num(s["tokens_served"], 1e12),
        prompt_tokens=int(_num(s["prompt_tokens"], 1024)), output_tokens=int(_num(s["output_tokens"], 256)),
        max_ms_per_token=_opt(s["max_ms_per_token"]), budget_usd=_opt(s["budget_usd"]), max_train_tokens=_opt(s["max_train_tokens"]),
        seq_len=int(_num(s["seq_len"], 4096)), vocab=int(_num(s["vocab"], 128256)), scaling_law=str(s["scaling_law"]),
        train_mfu=_num(s["train_mfu"], 0.4), attention_loss_penalty=s.get("attention_loss_penalty") or DEFAULT_ATTENTION_PENALTY,
        layers=[int(x) for x in s["layers"]], hidden=[int(x) for x in s["hidden"]],
        tokens_per_param=[int(x) for x in s["tokens_per_param"]]))


def from_python_problem(domain: DomainId, p: dict) -> dict:
    """Python-style problem dict (golden files) -> settings (identity, as in the TS mirror)."""
    return dict(p or {})
