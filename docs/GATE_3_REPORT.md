# Gate 3 report — uncertainty and experiment design

Status: Partial foundation.

Built:

- Deterministic bootstrap uncertainty for the calibrated A100 training pack.
- Central 90% predictive intervals, uncertainty whiskers, and probability-of-best display.
- Conservative interval-regret "Safest plan" selection in Rigor mode for calibrated training front plans.
- Existing leave-one-out check: 18 of 22 published rows covered, 81.8%, against a nominal 90% target.
- Browser-local recommendation-sensitive experiment ranking in the run auditor.

Open:

- Serving and co-design remain point estimates because no published calibration rows are available.
- The 22-row coverage result is evidence, not a future guarantee.
- The safest-plan surface is implemented for calibrated training; customer-held-out validation for serving/co-design
  and server-side/published experiment-result ingestion remain to be built. Completed experiment-result CSVs can be
  loaded into the browser-local calibration fit with candidate-name validation.
