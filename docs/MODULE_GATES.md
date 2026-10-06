# Module gate ledger

Every module is judged against the five serious criteria in `docs/EXECUTION_AUDIT.md`:

1. real artifacts;
2. operator/time simulation with timelines and distributions;
3. large constrained search using repair/NSGA-II-compatible validity rules;
4. production-usable artifacts and tested exports;
5. published/measured validation shown with held-out error.

Missing any criterion means the UI must say `Preview` and identify the missing evidence.

| Module | Artifact inputs | Time/distribution output | Search | Production output | Validation | Gate |
|---|---|---|---|---|---|---|
| M1 Train | model graph, topology, batch/tokens | GPU Gantt, memory, intervals | parallelism/schedule/recompute/precision | Megatron/DeepSpeed/torchtitan | Study 1 + customer held-out | IMPLEMENTED / PREVIEW |
| M2 Serve | model, hardware, traffic/trace, SLO | TTFT/ITL/KV/goodput | precision/TP/batching/cache/speculation | vLLM/SGLang/TensorRT-LLM | public serving set + customer held-out | IMPLEMENTED / PREVIEW |
| M3 Audit | framework configs/logs/metrics | current and candidate timelines | constrained repair/search | same-format diff + HTML/PDF | customer's held-out logs | IMPLEMENTED / PREVIEW |
| M4 Design model | architecture/search bounds/laws | train + serve cost distributions | architecture search | architecture/train/serve configs | multiple cited laws | IMPLEMENTED / PREVIEW |
| M5 Chip | chip ranges/workload suite | candidate workload timelines | joint hardware/software search | design report | sourced/measured PPA | IMPLEMENTED / PREVIEW |
| M6 RL | rollout/reward/train artifacts | phase utilization/staleness | placement/schedule search | runnable plan | measured loop data | IMPLEMENTED / PREVIEW |
| M7 Reliability | failure/checkpoint/restart inputs | whole-run goodput Monte Carlo | interval/spare policy search | checkpoint policy | Young/Daly + held-out | IMPLEMENTED / PREVIEW |
| M8 Fleets | traces, models, GPU pool | days-long utilization/cost/SLO | autoscaling policy search | policy/config report | replay or customer traces | IMPLEMENTED / PREVIEW |
| M9 Fine-tuning | model, adapter, quantization config | memory/throughput distributions | rank/target/precision search | framework configs | customer quality evals | IMPLEMENTED / PREVIEW |
| M10 TCO | contracts, prices, demand curves | cash-flow/demand Monte Carlo | purchase/lease/cloud search | procurement report | sourced prices + scenarios | IMPLEMENTED / PREVIEW |
| M11 Cost tracker | open model/token sources, serving inputs | training/serving cost ranges | model/hardware/provider comparison | cited cost report | published sources | IMPLEMENTED / PREVIEW |

For each gate, the final report records hand-checked cases, physics sanity tests, golden parity, export
parsing, validation rows, intervals/coverage, runtime, and 1440px/390px browser evidence.
