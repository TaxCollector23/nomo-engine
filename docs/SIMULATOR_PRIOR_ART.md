# Simulator calibration and prior art

This note documents the evidence boundary for
`nomo-planner/nomo_planner/simcalibration.py`. The module is an ingestion,
calibration, and validation layer for measured simulator inputs. It is not a
replacement for a discrete-event simulator and it does not add benchmark
measurements to the repository.

## Evidence contract

The accepted evidence kinds are deliberately separate:

- `operator`: measured kernel or operator microbenchmarks, such as latency,
  duration, or throughput at a stated shape and precision;
- `topology`: measured directed-link or collective-link observations, such as
  bandwidth or latency with source and destination features;
- `training`: measured training observations, such as step time or throughput
  for an identified parallelism strategy; and
- `serving`: measured serving observations, such as TTFT, TPOT/TBT, request
  latency, or token/request throughput.

Every row must have a stable `row_id`, positive measured `value`, `metric`,
`unit`, `strategy`, and source provenance. Provenance has a source identifier,
citation, and an HTTP(S) URL or DOI. JSON can put sources in a `sources`
registry and refer to them with `source_id`; CSV can repeat the provenance
columns on each row. Rows marked as `predicted`, `proxy`, `estimate`, or
`synthetic` are rejected. The parser never fills a missing measurement from a
default, a paper headline, or a simulator estimate.

Both of these shapes are supported:

The numeric values in the examples below are illustrative schema fixtures,
not measurements or benchmark claims. Production rows must be copied from a
measurement export and retain its source provenance.

```csv
row_id,kind,strategy,metric,value,unit,features_json,source_id,source_citation,source_url,split
op-001,operator,fused-attention,latency_us,8.2,us,"{""seq_len"":2048,""batch"":4}",bench-01,Local operator log,https://example.invalid/bench-01,train
link-001,topology,nvlink,bandwidth_gbps,890,Gbps,"{""src"":""gpu0"",""dst"":""gpu1""}",bench-01,Local operator log,https://example.invalid/bench-01,test
```

```json
{
  "sources": [
    {"id": "paper-01", "citation": "Measured source", "url": "https://example.invalid/paper-01"}
  ],
  "training_observations": [
    {
      "row_id": "train-001",
      "strategy": "PTD-P",
      "metric": "step_time",
      "value": 1.25,
      "unit": "s",
      "features": {"devices": 64, "tp": 8, "pp": 4, "dp": 2},
      "source_id": "paper-01"
    }
  ]
}
```

Grouped JSON arrays are named `operator_microbenchmarks`, `topology_links`,
`training_observations`, and `serving_observations`. A row may instead carry a
`metrics` object; each scalar becomes a separate row with a suffixed ID, so a
topology export containing both link bandwidth and link latency does not lose
either measurement. The loader also understands the existing
`nomo-planner/data/training_observations.csv` shape without changing that file.

## Fit, sample, and validate

For one `(kind, metric, unit, strategy)` group, the reference fit is the
deterministic log-linear model

\[
  \log y = \beta_0 + \sum_j \beta_j \log(1+x_j) + \text{categorical terms}.
\]

Numeric features must be non-negative and categorical features use a stable
sorted baseline. Features missing from any training row are excluded unless
the caller explicitly requires them; an explicitly requested missing feature
is an error. The fit is solved with standard-library linear algebra and a
small configurable ridge term. Rows are sorted by `row_id` before fitting, so
the result does not depend on input order.

`bootstrap_parameter_samples` resamples only the supplied training rows with a
recorded seed and refits the same model. `parameter_intervals` reports central
coefficient intervals. `predictive_interval` additionally resamples the fit's
observed log residuals; it does not draw invented Gaussian noise. Intervals
are empirical bootstrap/predictive intervals, not Bayesian posteriors and not
guarantees for a different hardware or workload distribution.

`split_held_out` honors explicit `split=train` and `split=test` labels. When
labels are absent, it uses a stable SHA-256 ordering of the seed, group, and
row ID, and splits within each evidence group. `evaluate_held_out` fits only
the training side and returns a `by_strategy` report plus row-level
predictions and interval bounds. A group with no held-out rows is reported as
`no_held_out_rows`; undefined statistics are `null`/`None`, not fabricated.

The report fields are:

- `MAE`: mean absolute error in the measured unit;
- `MAPE`: mean absolute percentage error, in percent;
- `PTD-P`: the derived absolute aggregate prediction-vs-data deviation,
  `abs(sum(predicted)-sum(actual))/sum(actual) * 100`. This is a compatibility
  report field, not a claim that the literature uses this exact name;
- `rank_correlation`: tie-aware Spearman correlation, or `None` when there are
  too few or non-varying held-out values; and
- `coverage`: the fraction of held-out values inside the requested empirical
  interval, with the numerator and denominator included.

`PTD-P` is also retained as the exact strategy label used for the
pipeline/tensor/data-parallel training family. That label is independent of
the derived `PTD-P` report field. The implementation does not infer a local
performance value from the cited PTD-P paper.

Minimal use:

```python
from nomo_planner.simcalibration import evaluate_held_out, load_evidence

evidence = load_evidence("measured-evidence.json")
report = evaluate_held_out(evidence, confidence=0.90, bootstrap_replicates=256)
print(report.as_dict()["by_strategy"])
```

## Prior-art map and explicit overlap

The entries below are primary papers or project pages. The overlap column says
what this evidence layer borrows conceptually; it does not imply that Nomo
reproduces the cited implementation or imports its measurements.

| Prior art | Primary citation | Explicit overlap with this layer | Boundary / non-overlap |
|---|---|---|---|
| Calculon | Isaev et al., *Calculon: a Methodology and Tool for High-Level Codesign of Systems and Large Language Models*, SC 2023. [DOI](https://doi.org/10.1145/3581784.3607102), [paper](https://www.nicm.dev/pubs/mikhailisaev_calculon_sc_2023.pdf), [tool](https://github.com/calculon-ai/calculon) | Parameterized LLM training/inference performance, software strategy, and hardware/topology inputs motivate separate strategy and topology evidence. | Calculon is an analytical co-design model and optimizer. This module does not reimplement its equations, optimizer, or reported results; it only calibrates supplied scalar observations. |
| ASTRA-sim | Rashidi et al., *ASTRA-SIM: Enabling SW/HW Co-Design Exploration for Distributed DL Training Platforms*, ISPASS 2020, [DOI](https://doi.org/10.1109/ISPASS48437.2020.00018), [project](https://astra-sim.github.io/). The topology/disaggregation extension is [ASTRA-sim2.0, DOI](https://doi.org/10.1109/ISPASS57527.2023.00035). | Network topology, collective communication, distributed-training strategy, and simulator validation are why link observations and provenance are first-class rows. | ASTRA-sim provides network/compute/memory simulator backends and detailed workload execution. This layer does not claim cycle-level fidelity or provide ASTRA-sim input generation. |
| Vidur | Agrawal et al., *Vidur: A Large-Scale Simulation Framework for LLM Inference*, MLSys 2024, [paper record](https://proceedings.mlsys.org/paper_files/paper/2024/hash/b74a8de47d2b3c928360e0a011f48351-Abstract-Conference.html), [arXiv](https://arxiv.org/abs/2405.05465). | Vidur’s combination of operator profiling and predictive modeling directly overlaps with operator-microbenchmark ingestion, serving observations, and held-out error/coverage reporting. | Vidur models request-level inference workloads and schedulers. This implementation is a general evidence/fitting boundary and does not reproduce Vidur-Bench, its workload traces, or its simulator. |
| Megatron-LM | Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*, [arXiv](https://arxiv.org/abs/1909.08053); Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*, [DOI](https://doi.org/10.1145/3458817.3476209), [arXiv](https://arxiv.org/abs/2104.04473). | Tensor/pipeline/data strategy labels, measured training throughput/step observations, and topology-aware feature columns are represented without collapsing them into an untraceable scalar. | The papers and code define training execution and parallelism. This layer does not claim to run Megatron-LM or to promote published values into local measurements. |
| Korthikanti et al. | Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models*, [arXiv](https://arxiv.org/abs/2205.05198), [arXiv DOI](https://doi.org/10.48550/arXiv.2205.05198). | Sequence parallelism, activation recomputation, tensor parallelism, and memory/throughput trade-offs motivate operator and training features plus strategy-specific validation. | No activation-memory or throughput number is copied into the calibration data by this implementation; such a result must arrive as a measured, sourced row. |
| Zero Bubble | Qi et al., *Zero Bubble Pipeline Parallelism*, [arXiv](https://arxiv.org/abs/2401.10241), [OpenReview](https://openreview.net/forum?id=tuzTN0eIO5). | Pipeline scheduling is preserved as an explicit strategy value, so a Zero-Bubble run can be fit and compared against other strategies without mixing schedules. | The module does not simulate backward-pass decomposition, schedule search, or the paper’s pipeline execution semantics. |
| FlashAttention | Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, [arXiv](https://arxiv.org/abs/2205.14135), [OpenReview](https://openreview.net/forum?id=H4DqfPSibmx). | IO-aware attention makes shape, sequence length, precision, and kernel variant meaningful operator-microbenchmark features. | A cited FlashAttention result is provenance, not a local calibration point. Kernel correctness and attention IO complexity are outside this module. |
| vLLM / PagedAttention | Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, [DOI](https://doi.org/10.1145/3600006.3613165), [arXiv](https://arxiv.org/abs/2309.06180). | Serving rows can carry KV-cache, prompt/decode length, batching, strategy, TTFT/TPOT, and throughput observations; source metadata keeps a vLLM measurement distinct from a proxy. | This layer does not implement PagedAttention, block allocation, prefix sharing, or vLLM scheduling. |
| Sarathi-Serve | Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, [arXiv](https://arxiv.org/abs/2403.02310), [code](https://github.com/microsoft/sarathi-serve). | Chunked-prefill and scheduler variants fit the strategy field; phase-level latency/throughput observations fit the serving evidence schema. | The parser does not emulate chunked-prefill scheduling or assert Sarathi-Serve capacity. Any such result must be uploaded as measured evidence with its source. |
| DistServe | Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*, [USENIX paper page](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin), [arXiv](https://arxiv.org/abs/2401.09670). | Prefill/decode phase labels, bandwidth/topology links, TTFT/TPOT, and goodput-related observations can be ingested and validated separately. | Disaggregation placement, resource optimization, queueing, and SLO attainment are not implemented by this calibration layer. |
| Splitwise | Patel et al., *Splitwise: Efficient Generative LLM Inference Using Phase Splitting*, [DOI](https://doi.org/10.1109/ISCA59077.2024.00019), [arXiv](https://arxiv.org/abs/2311.18677). | Phase-specific operator profiles, KV-cache transfer links, serving traces, and workload provenance map directly to the four evidence kinds. | The module does not run SplitwiseSim, model request arrivals, or infer cluster cost/power from a paper’s reported configurations. |
| Speculative decoding | Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, [PMLR](https://proceedings.mlr.press/v202/leviathan23a.html), [arXiv](https://arxiv.org/abs/2211.17192). Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, [arXiv](https://arxiv.org/abs/2302.01318). | Draft length, acceptance, verification, decode latency, and throughput can be recorded as serving/operator features and fit only when measured rows supply them. | No acceptance rate, speedup, or quality claim is generated without an observed value; the algorithm and exact-sampling proof are outside scope. |
| Data-constrained scaling | Muennighoff et al., *Scaling Data-Constrained Language Models*, [JMLR](https://www.jmlr.org/papers/volume26/24-1000/24-1000.html), [arXiv](https://arxiv.org/abs/2305.16264). | The work is a reminder that extrapolation must preserve data regime and provenance. The same evidence boundary, deterministic log-space fitting, and held-out reporting apply to training-scaling observations. | This module calibrates runtime/system measurements, not validation loss or compute-optimal data/model scaling laws; no scaling-law measurement is included here. |

The citations establish what prior systems measure or model. They do not
authorize replacing missing local measurements with published headline values.
For a production calibration, retain the raw CSV/JSON, source URL/DOI,
hardware/software identity, workload features, explicit train/held-out labels,
and the generated `ValidationReport` together.
