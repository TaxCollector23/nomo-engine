"""Regenerate scripts/product-golden.json from the Python product_models reference.

Restores the lost generator for the third parity golden. Fixture inputs (chip
spec, workloads, reliability problem, traffic, fine-tune problem, TCO option,
RL plan) are the same ones the browser engine has always been verified against;
expected values are recomputed from nomo_planner.product_models.

Usage (from nomo-planner/):  python3 studies/export_product_golden.py --out ../scripts/product-golden.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nomo_planner.product_models import (ChipSpec, FineTuneProblem, ReliabilityProblem, RLProblem, TCOOption,  # noqa: E402
                                         TrafficHour, Workload, evaluate_chip, evaluate_reliability, evaluate_rl_plan,
                                         evaluate_tco, fine_tune_options, size_fleet)


def chip_spec(d: dict) -> ChipSpec:
    return ChipSpec(
        name=d["name"], memory_bandwidth_gbps=d["memoryBandwidthGbps"], sram_gb=d["sramGb"],
        bf16_tops=d["bf16Tops"], fp8_tops=d["fp8Tops"], interconnect_gbps=d["interconnectGbps"],
        neuromorphic_cores=d["neuromorphicCores"], neuromorphic_memory_gb=d["neuromorphicMemoryGb"],
        cost_usd=d["costUsd"], power_w=d["powerW"], hourly_cost_usd=d["hourlyCostUsd"])


def workload(d: dict) -> Workload:
    return Workload(name=d["name"], dense_flops=d["denseFlops"], memory_bytes=d["memoryBytes"],
                    spiking_ops=d.get("spikingOps", 0.0), requires_neuromorphic=d.get("requiresNeuromorphic", False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "..", "scripts", "product-golden.json"))
    ap.add_argument("--from", dest="src", default=os.path.join(here, "..", "scripts", "product-golden.json"),
                    help="existing golden file supplying the fixture inputs")
    args = ap.parse_args()

    with open(args.src) as f:
        old = json.load(f)

    # chip ------------------------------------------------------------------
    chip = old["chip"]
    spec = chip_spec(chip["spec"])
    wl = workload(chip["workload"])
    result = evaluate_chip(spec, wl, precision=chip["precision"])
    chip_out = {
        "spec": chip["spec"], "workload": chip["workload"], "precision": chip["precision"],
        "expected": {
            "seconds": result["seconds"], "energy_j": result["energy_j"], "cost_usd": result["cost_usd"],
            "bottleneck": result["mapping"]["bottleneck"], "dense_precision": result["mapping"]["dense_precision"],
            "neuromorphic_cores": result["mapping"]["neuromorphic_cores"],
        },
    }

    # reliability ------------------------------------------------------------
    rel = old["reliability"]
    p = rel["problem"]
    rel_result = evaluate_reliability(
        ReliabilityProblem(failure_rate_per_hour=p["failureRatePerHour"], checkpoint_write_hours=p["checkpointWriteHours"],
                           restart_hours=p["restartHours"], gpu_count=p["gpuCount"], gpu_cost_per_hour=p["gpuCostPerHour"]),
        rel["interval"])
    rel_out = {"problem": p, "interval": rel["interval"],
               "expected": {"goodput": rel_result["goodput"], "cost_usd_per_useful_hour": rel_result["cost_usd_per_useful_hour"]}}

    # fleet ------------------------------------------------------------------
    traffic = [TrafficHour(hour=row["hour"], arrival_rps=row["arrivalRps"], service_rps_per_replica=row["serviceRpsPerReplica"],
                           p99_target_ms=row["p99TargetMs"]) for row in old["fleet"]["traffic"]]
    fleet_rows = size_fleet(traffic)
    fleet_out = {"traffic": old["fleet"]["traffic"],
                 "expected": [{"hour": r["hour"], "replicas": r["replicas"], "approx_p99_ms": r["approx_p99_ms"],
                               "hourly_cost_usd": r["hourly_cost_usd"], "within_target": r["within_target"]} for r in fleet_rows]}

    # fine-tune ----------------------------------------------------------------
    fp = old["finetune"]["problem"]
    ft_rows = fine_tune_options(FineTuneProblem(
        parameter_count=fp["parameterCount"], training_tokens=fp["trainingTokens"], memory_bytes=fp["memoryBytes"],
        peak_tflops=fp["peakTflops"], gpu_memory_bytes=fp["gpuMemoryBytes"], gpu_cost_per_hour=fp["gpuCostPerHour"],
        quality_loss=fp["qualityLoss"], adapter_param_fraction=fp.get("adapterParamFraction", 0.005)))
    ft_out = {"problem": fp,
              "expected": [{"mode": r["mode"], "memory_bytes": r["memory_bytes"], "time_hours": r["time_hours"],
                            "cost_usd": r["cost_usd"], "quality_loss": r["quality_loss"], "feasible": r["feasible"]} for r in ft_rows]}

    # TCO ----------------------------------------------------------------------
    to = old["tco"]["option"]
    tco_result = evaluate_tco(TCOOption(
        name=to["name"], purchase_usd=to["purchaseUsd"], lease_usd_per_hour=to["leaseUsdPerHour"],
        utilization=to["utilization"], years=to["years"], hours_per_year=to["hoursPerYear"],
        energy_usd_per_hour=to["energyUsdPerHour"], maintenance_pct=to["maintenancePct"],
        price_low_usd=to.get("priceLowUsd"), price_high_usd=to.get("priceHighUsd")))
    tco_out = {"option": to,
               "expected": {"buy_usd_per_useful_hour": tco_result["buy_usd_per_useful_hour"],
                            "rent_usd_per_useful_hour": tco_result["rent_usd_per_useful_hour"],
                            "range_low": tco_result["range_low"], "range_high": tco_result["range_high"],
                            "recommended": tco_result["recommended"]}}

    # RL plan ------------------------------------------------------------------
    rp = old["rl"]["problem"]
    ro = old["rl"]["options"]
    rl_result = evaluate_rl_plan(
        RLProblem(answer_length_p50=rp["answerLengthP50"], answer_length_p95=rp["answerLengthP95"],
                  rollout_tokens_per_s_per_gpu=rp["rolloutTokensPerSPerGpu"], train_tokens_per_s_per_gpu=rp["trainTokensPerSPerGpu"],
                  reward_tokens_per_s_per_gpu=rp["rewardTokensPerSPerGpu"], rollout_gpus=(ro["rolloutGpus"],),
                  train_gpus=(ro["trainGpus"],), reward_gpus=(ro["rewardGpus"],), gpu_cost_per_hour=rp["gpuCostPerHour"],
                  training_tokens_per_update=rp["trainingTokensPerUpdate"], target_samples_per_hour=rp["targetSamplesPerHour"]),
        rollout_gpus=ro["rolloutGpus"], train_gpus=ro["trainGpus"], reward_gpus=ro["rewardGpus"],
        colocated=ro["colocated"], asynchronous=ro["asynchronous"], batch_size=ro["batchSize"], kv_precision=ro["kvPrecision"])
    rl_out = {"problem": rp, "options": ro,
              "expected": {"samples_per_hour": rl_result["samples_per_hour"], "cost_usd_per_update": rl_result["cost_usd_per_update"],
                           "step_seconds": rl_result["step_seconds"], "schedule": rl_result["schedule"],
                           "colocation": rl_result["colocation"], "feasible": rl_result["feasible"]}}

    out_path = os.path.abspath(args.out)
    with open(out_path, "w") as f:
        json.dump({"chip": chip_out, "reliability": rel_out, "fleet": fleet_out, "finetune": ft_out, "tco": tco_out, "rl": rl_out},
                  f, indent=1)
        f.write("\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
