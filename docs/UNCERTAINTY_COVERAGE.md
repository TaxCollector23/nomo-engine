# A2 held-out uncertainty coverage

The dependency-free Python reference in `nomo-planner/` fits 128 bootstrap
refits per leave-one-out split, stratified by tensor/pipeline versus ZeRO-3
strategy, and evaluates a central predictive 90% interval on the omitted
published run.

- Overall: 18/22 covered (**81.8%**); nominal target: 90%.
- Tensor/pipeline: 14/16 (**87.5%**).
- ZeRO-3: 4/6 (**66.7%**).
- 95% Wilson interval for overall coverage: **61.5%–92.7%**.

The interval is useful validation evidence but does not meet the nominal target
on this small sample. The Lab reports the shortfall and keeps serving and
co-design intervals unavailable because they have no published calibration
data. Full row-level output is in `nomo-planner/docs/UNCERTAINTY_COVERAGE.json`.
