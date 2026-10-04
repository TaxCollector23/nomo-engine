"""Export fixed-seed layer-planner cases consumed by the browser parity check."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from nomo_planner.layers import TrainingProblem, build_graph, search
from nomo_planner.model_configs import BUILTIN_MODEL_CONFIGS


def main(output: str) -> None:
    cases = []
    for key in ("llama3_8b", "llama3_70b", "mixtral_8x7b"):
        config = BUILTIN_MODEL_CONFIGS[key]
        graph = build_graph(config, seq_len=128, batch_size=1, source="golden")
        problem = TrainingProblem(graph, pipeline_stages=(1, 2, 4, 8, 16, 32), max_candidates=600, total_steps=1000, seed=20261003)
        result = search(problem)
        assert result.best is not None and result.best_metrics is not None
        cases.append({
            "key": key,
            "config": config,
            "seq_len": 128,
            "batch_size": 1,
            "pipeline_stages": [1, 2, 4, 8, 16, 32],
            "max_candidates": 600,
            "total_steps": 1000,
            "seed": 20261003,
            "best": {
                "stages": list(result.best.stages), "precision": list(result.best.precision),
                "recompute": list(result.best.recompute), "offload": list(result.best.offload),
            },
            "best_objectives": result.best_metrics.objectives,
            "global_bf16": {
                "stages": list(result.global_bf16[0].stages), "precision": list(result.global_bf16[0].precision),
                "recompute": list(result.global_bf16[0].recompute), "offload": list(result.global_bf16[0].offload),
            },
            "global_best": {
                "stages": list(result.global_best[0].stages), "precision": list(result.global_best[0].precision),
                "recompute": list(result.global_best[0].recompute), "offload": list(result.global_best[0].offload),
            },
            "global_objectives": result.global_best[1].objectives,
            "precision_gain_pct": result.precision_gain_pct,
            "per_layer_gain_pct": result.per_layer_gain_pct,
        })
    Path(output).write_text(json.dumps({"cases": cases}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scripts/layer-golden.json")
