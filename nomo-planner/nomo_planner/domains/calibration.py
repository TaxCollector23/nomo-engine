"""Published measurements (Narayanan et al. 2021, arXiv:2104.04473, Tables 1-2)
and the parameters fitted to them.

Restored Python reference for src/planner/calibration.ts. The 22 observations
are transcribed from the paper; measured step time uses the paper's eq. (3).
"""
from __future__ import annotations

import math
from typing import Dict, List

from .hardware import CLUSTERS, MODELS
from .packs import TrainingPack, TrainingProblem

OBSERVATIONS: List[dict] = [
    {"source": "Narayanan et al. 2021 Table 1", "url": "https://arxiv.org/abs/2104.04473", "model": "megatron-gpt-1.7b", "seq_len": "2048", "global_batch_tokens": "1048576", "cluster": "a100_nvlink_ib", "devices": "32", "tp": "1", "pp": "1", "zero": "0", "recompute": "full", "micro_batch": "1", "precision": "bf16", "metric": "megatron_tflops_per_gpu", "value": "137", "notes": "mixed precision on Selene (DGX A100); interleaved schedule; micro-batch not reported (1 assumed)"},
    {"source": "Narayanan et al. 2021 Table 1", "url": "https://arxiv.org/abs/2104.04473", "model": "megatron-gpt-3.6b", "seq_len": "2048", "global_batch_tokens": "1048576", "cluster": "a100_nvlink_ib", "devices": "64", "tp": "2", "pp": "1", "zero": "0", "recompute": "full", "micro_batch": "1", "precision": "bf16", "metric": "megatron_tflops_per_gpu", "value": "138", "notes": "mixed precision on Selene (DGX A100); interleaved schedule; micro-batch not reported (1 assumed)"},
    {"source": "Narayanan et al. 2021 Table 1", "url": "https://arxiv.org/abs/2104.04473", "model": "megatron-gpt-7.5b", "seq_len": "2048", "global_batch_tokens": "1048576", "cluster": "a100_nvlink_ib", "devices": "128", "tp": "4", "pp": "1", "zero": "0", "recompute": "full", "micro_batch": "1", "precision": "bf16", "metric": "megatron_tflops_per_gpu", "value": "142", "notes": "mixed precision on Selene (DGX A100); interleaved schedule; micro-batch not reported (1 assumed)"},
    {"source": "Narayanan et al. 2021 Table 1", "url": "https://arxiv.org/abs/2104.04473", "model": "megatron-gpt-18.4b", "seq_len": "2048", "global_batch_tokens": "2097152", "cluster": "a100_nvlink_ib", "devices": "256", "tp": "8", "pp": "1", "zero": "0", "recompute": "full", "micro_batch": "1", "precision": "bf16", "metric": "megatron_tflops_per_gpu", "value": "135", "notes": "mixed precision on Selene (DGX A100); interleaved schedule; micro-batch not reported (1 assumed)"},
    {"source": "Narayanan et al. 2021 Table 1", "url": "https://arxiv.org/abs/2104.04473", "model": "megatron-gpt-39.1b", "seq_len": "2048", "global_batch_tokens": "3145728", "cluster": "a100_nvlink_ib", "devices": "512", "tp": "8", "pp": "2", "zero": "0", "recompute": "full", "micro_batch": "1", "precision": "bf16", "metric": "megatron_tflops_per_gpu", "value": "138", "notes": "mixed precision on Selene (DGX A100); interleaved schedule; micro-batch not reported (1 assumed)"},
    {"source": "Narayanan et al. 2021 Table 1", "url": "https://arxiv.org/abs/2104.04473", "model": "megatron-gpt-76.1b", "seq_len": "2048", "global_batch_tokens": "3670016", "cluster": "a100_nvlink_ib", "devices": "1024", "tp": "8", "pp": "4", "zero": "0", "recompute": "full", "micro_batch": "1", "precision": "bf16", "metric": "megatron_tflops_per_gpu", "value": "140", "notes": "mixed precision on Selene (DGX A100); interleaved schedule; micro-batch not reported (1 assumed)"},
    {"source": "Narayanan et al. 2021 Table 1", "url": "https://arxiv.org/abs/2104.04473", "model": "megatron-gpt-145.6b", "seq_len": "2048", "global_batch_tokens": "4718592", "cluster": "a100_nvlink_ib", "devices": "1536", "tp": "8", "pp": "8", "zero": "0", "recompute": "full", "micro_batch": "1", "precision": "bf16", "metric": "megatron_tflops_per_gpu", "value": "148", "notes": "mixed precision on Selene (DGX A100); interleaved schedule; micro-batch not reported (1 assumed)"},
    {"source": "Narayanan et al. 2021 Table 1", "url": "https://arxiv.org/abs/2104.04473", "model": "megatron-gpt-310.1b", "seq_len": "2048", "global_batch_tokens": "4423680", "cluster": "a100_nvlink_ib", "devices": "1920", "tp": "8", "pp": "16", "zero": "0", "recompute": "full", "micro_batch": "1", "precision": "bf16", "metric": "megatron_tflops_per_gpu", "value": "155", "notes": "mixed precision on Selene (DGX A100); interleaved schedule; micro-batch not reported (1 assumed)"},
    {"source": "Narayanan et al. 2021 Table 1", "url": "https://arxiv.org/abs/2104.04473", "model": "megatron-gpt-529.6b", "seq_len": "2048", "global_batch_tokens": "5160960", "cluster": "a100_nvlink_ib", "devices": "2520", "tp": "8", "pp": "35", "zero": "0", "recompute": "full", "micro_batch": "1", "precision": "bf16", "metric": "megatron_tflops_per_gpu", "value": "163", "notes": "mixed precision on Selene (DGX A100); interleaved schedule; micro-batch not reported (1 assumed)"},
    {"source": "Narayanan et al. 2021 Table 1", "url": "https://arxiv.org/abs/2104.04473", "model": "megatron-gpt-1008b", "seq_len": "2048", "global_batch_tokens": "6291456", "cluster": "a100_nvlink_ib", "devices": "3072", "tp": "8", "pp": "64", "zero": "0", "recompute": "full", "micro_batch": "1", "precision": "bf16", "metric": "megatron_tflops_per_gpu", "value": "163", "notes": "mixed precision on Selene (DGX A100); interleaved schedule; micro-batch not reported (1 assumed)"},
    {"source": "Narayanan et al. 2021 Table 2", "url": "https://arxiv.org/abs/2104.04473", "model": "megatron-gpt-174.6b", "seq_len": "2048", "global_batch_tokens": "3145728", "cluster": "a100_nvlink_ib", "devices": "384", "tp": "8", "pp": "12", "zero": "0", "recompute": "full", "micro_batch": "1", "precision": "bf16", "metric": "megatron_tflops_per_gpu", "value": "153", "notes": "PTD-P; model-parallel size reported as 96 / 280"},
    {"source": "Narayanan et al. 2021 Table 2", "url": "https://arxiv.org/abs/2104.04473", "model": "megatron-gpt-174.6b", "seq_len": "2048", "global_batch_tokens": "3145728", "cluster": "a100_nvlink_ib", "devices": "768", "tp": "8", "pp": "12", "zero": "0", "recompute": "full", "micro_batch": "1", "precision": "bf16", "metric": "megatron_tflops_per_gpu", "value": "149", "notes": "PTD-P; model-parallel size reported as 96 / 280"},
    {"source": "Narayanan et al. 2021 Table 2", "url": "https://arxiv.org/abs/2104.04473", "model": "megatron-gpt-174.6b", "seq_len": "2048", "global_batch_tokens": "3145728", "cluster": "a100_nvlink_ib", "devices": "1536", "tp": "8", "pp": "12", "zero": "0", "recompute": "full", "micro_batch": "1", "precision": "bf16", "metric": "megatron_tflops_per_gpu", "value": "141", "notes": "PTD-P; model-parallel size reported as 96 / 280"},
    {"source": "Narayanan et al. 2021 Table 2", "url": "https://arxiv.org/abs/2104.04473", "model": "megatron-gpt-529.6b", "seq_len": "2048", "global_batch_tokens": "4587520", "cluster": "a100_nvlink_ib", "devices": "560", "tp": "8", "pp": "35", "zero": "0", "recompute": "full", "micro_batch": "1", "precision": "bf16", "metric": "megatron_tflops_per_gpu", "value": "171", "notes": "PTD-P; model-parallel size reported as 96 / 280"},
    {"source": "Narayanan et al. 2021 Table 2", "url": "https://arxiv.org/abs/2104.04473", "model": "megatron-gpt-529.6b", "seq_len": "2048", "global_batch_tokens": "4587520", "cluster": "a100_nvlink_ib", "devices": "1120", "tp": "8", "pp": "35", "zero": "0", "recompute": "full", "micro_batch": "1", "precision": "bf16", "metric": "megatron_tflops_per_gpu", "value": "167", "notes": "PTD-P; model-parallel size reported as 96 / 280"},
    {"source": "Narayanan et al. 2021 Table 2", "url": "https://arxiv.org/abs/2104.04473", "model": "megatron-gpt-529.6b", "seq_len": "2048", "global_batch_tokens": "4587520", "cluster": "a100_nvlink_ib", "devices": "2240", "tp": "8", "pp": "35", "zero": "0", "recompute": "full", "micro_batch": "1", "precision": "bf16", "metric": "megatron_tflops_per_gpu", "value": "159", "notes": "PTD-P; model-parallel size reported as 96 / 280"},
    {"source": "Narayanan et al. 2021 Table 2", "url": "https://arxiv.org/abs/2104.04473", "model": "megatron-gpt-174.6b", "seq_len": "2048", "global_batch_tokens": "3145728", "cluster": "a100_nvlink_ib", "devices": "384", "tp": "1", "pp": "1", "zero": "3", "recompute": "full", "micro_batch": "4", "precision": "bf16", "metric": "megatron_tflops_per_gpu", "value": "144", "notes": "ZeRO-3 via DeepSpeed without model parallelism"},
    {"source": "Narayanan et al. 2021 Table 2", "url": "https://arxiv.org/abs/2104.04473", "model": "megatron-gpt-174.6b", "seq_len": "2048", "global_batch_tokens": "3145728", "cluster": "a100_nvlink_ib", "devices": "768", "tp": "1", "pp": "1", "zero": "3", "recompute": "full", "micro_batch": "2", "precision": "bf16", "metric": "megatron_tflops_per_gpu", "value": "88", "notes": "ZeRO-3 via DeepSpeed without model parallelism"},
    {"source": "Narayanan et al. 2021 Table 2", "url": "https://arxiv.org/abs/2104.04473", "model": "megatron-gpt-174.6b", "seq_len": "2048", "global_batch_tokens": "3145728", "cluster": "a100_nvlink_ib", "devices": "1536", "tp": "1", "pp": "1", "zero": "3", "recompute": "full", "micro_batch": "1", "precision": "bf16", "metric": "megatron_tflops_per_gpu", "value": "44", "notes": "ZeRO-3 via DeepSpeed without model parallelism"},
    {"source": "Narayanan et al. 2021 Table 2", "url": "https://arxiv.org/abs/2104.04473", "model": "megatron-gpt-529.6b", "seq_len": "2048", "global_batch_tokens": "5242880", "cluster": "a100_nvlink_ib", "devices": "640", "tp": "1", "pp": "1", "zero": "3", "recompute": "full", "micro_batch": "4", "precision": "bf16", "metric": "megatron_tflops_per_gpu", "value": "138", "notes": "ZeRO-3 via DeepSpeed without model parallelism; 640-GPU row uses batch 2560 as marked in the paper"},
    {"source": "Narayanan et al. 2021 Table 2", "url": "https://arxiv.org/abs/2104.04473", "model": "megatron-gpt-529.6b", "seq_len": "2048", "global_batch_tokens": "4587520", "cluster": "a100_nvlink_ib", "devices": "1120", "tp": "1", "pp": "1", "zero": "3", "recompute": "full", "micro_batch": "2", "precision": "bf16", "metric": "megatron_tflops_per_gpu", "value": "98", "notes": "ZeRO-3 via DeepSpeed without model parallelism"},
    {"source": "Narayanan et al. 2021 Table 2", "url": "https://arxiv.org/abs/2104.04473", "model": "megatron-gpt-529.6b", "seq_len": "2048", "global_batch_tokens": "4587520", "cluster": "a100_nvlink_ib", "devices": "2240", "tp": "1", "pp": "1", "zero": "3", "recompute": "full", "micro_batch": "1", "precision": "bf16", "metric": "megatron_tflops_per_gpu", "value": "48", "notes": "ZeRO-3 via DeepSpeed without model parallelism"},
]

CALIBRATED: Dict[str, dict] = {
    "a100_nvlink_ib": {
        "params": {"zero3_overlap": 0.0, "e_matmul": 0.6400824774976149, "bubble_scale": 0.41036560876264244, "gemm_h_half": 320.5480034537571, "dp_overlap": 0.9999999999999999, "zero3_latency_us": 78.40485850566189},
        "observations": 22, "loo_mape_pct": 6.38,
        "source": "Narayanan et al. 2021 (arXiv:2104.04473), Tables 1-2",
    },
}

CALIBRATION_RESULTS = {"held_out_ptd_mape": 5.9, "held_out_kendall": 0.84, "held_out_spearman": 0.94, "loo_mape": 6.38, "zero3_transfer_mape": 10.3, "uncalibrated_mape": 10.9}


def calibration_points() -> List[dict]:
    """Measured step time from published per-GPU throughput with the paper's eq. (3), then predicted by the planner."""
    points: List[dict] = []
    for r in OBSERVATIONS:
        model = MODELS[r["model"]]
        devices = int(r["devices"])
        s = int(r["seq_len"])
        B = math.floor(int(r["global_batch_tokens"]) / s)
        F = 96 * B * s * model.layers * model.hidden ** 2 * (1 + s / (6 * model.hidden) + model.vocab / (16 * model.layers * model.hidden))
        measured = F / (devices * float(r["value"]) * 1e12)
        prob = {
            "model": model, "cluster": CLUSTERS[r["cluster"]], "seq_len": s,
            "global_batch_tokens": int(r["global_batch_tokens"]), "total_tokens": 1e12,
            "device_counts": [devices], "usable_memory": 0.92,
        }
        plan = {"devices": devices, "tp": int(r["tp"]), "pp": int(r["pp"]), "zero": int(r["zero"]),
                "recompute": r["recompute"], "micro_batch": int(r["micro_batch"]), "matmul_dtype": r["precision"]}
        predicted = TrainingPack(TrainingProblem(**prob), CALIBRATED[r["cluster"]]["params"]).evaluate(plan)["breakdown"]["step_s"]
        uncalibrated = TrainingPack(TrainingProblem(**prob)).evaluate(plan)["breakdown"]["step_s"]
        points.append({
            "label": f"{model.name}, {devices} GPUs, TP{r['tp']} PP{r['pp']}" + (f" ZeRO-{r['zero']}" if r["zero"] != "0" else ""),
            "source": r["source"], "plan": plan, "measured": measured, "predicted": predicted,
            "uncalibrated": uncalibrated, "strategy": "ZeRO-3" if r["zero"] == "3" else "Tensor + pipeline",
        })
    return points
