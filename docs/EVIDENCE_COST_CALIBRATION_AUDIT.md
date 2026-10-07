# Evidence, cost, and calibration boundary audit

Audit date: 2026-10-06.

This pass only uses values already checked into the repository and the source URLs already attached to those values.
It adds no price, throughput, latency, or hardware measurement.

## Implemented checks

- `nomo-planner/nomo_planner/source_freshness.py` validates date-only ISO metadata and reports `fresh`, `stale`,
  `undated`, or `expired` relative to an explicit reference date. It never makes a network request.
- `cost_tracker.py` validates non-negative prices/ranges and date metadata, reports open-model coverage separately
  from provider pricing, and keeps undated model-card rows visibly undated.
- The browser Cost Tracker mirrors the checked-in rows, displays the coverage boundary and source status, and keeps
  physical serving cost dependent on user-supplied GPU rate and measured throughput.
- `simcalibration.py` rejects malformed `accessed_at` metadata, duplicate source IDs, and evidence rows whose source
  ID is not declared in the bundle; it exposes per-source row counts plus metadata-only freshness through
  `EvidenceBundle.source_report(...)`.
- Customer calibration now rejects missing run/cluster identity, handles quoted CSV fields, and labels its empirical
  central 90% interval as in-sample rather than predictive.
- The Evidence module displays the checked-in 18/22 nominal-90% held-out coverage, while the Simulation Evidence
  registry distinguishes cited/replayed measurements from unsupplied customer telemetry and records source-date
  status.

## Verification

- Python focused evidence/cost/calibration tests: 17 passed.
- `npm run build`: passed.
- `npm run verify`: passed, 14,337 checks; this includes quoted customer CSV parsing, interval labels, cost coverage,
  and deterministic source freshness checks.

## Precise remaining data blocker

The checked-in cost data contains two open-model Llama 3 training-compute rows and three closed-provider token-price
rows. It contains zero cited open-model serving/token-price rows. The Meta model-card rows have no checked-on date,
so the freshness checker correctly labels them `undated`. Closing those gaps requires a user-supplied or newly cited
open-model serving/token-price source and an explicit source-review date; neither can be inferred from existing data.

Calibration has the same boundary: the public Sarathi rows are cited measurements, but the raw request trace is not
checked in, and customer serving/co-design observations are absent. No new interval or cost claim is promoted beyond
the evidence currently present.
