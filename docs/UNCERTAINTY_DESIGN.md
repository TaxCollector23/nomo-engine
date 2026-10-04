# A1–A3 design note: calibrated uncertainty for the Nomo Lab

Status: design gate for the A1–A3 implementation, 2026-09-29.  This note is
written before implementation.  The current public checkout contains the
browser TypeScript planner and generated calibration observations, but not the
Python reference package named by the project brief; the first implementation
must therefore recreate a small, auditable Python reference for the uncertainty
layer and compare its deterministic outputs with the browser port.

## Method

Use a non-parametric bootstrap over the 22 published Narayanan et al. (2021)
runs, stratified by the observed strategy (tensor/pipeline parallel and
ZeRO-3).  Each replicate refits the existing positive hardware parameters in
log space using the same objective and bounds as the calibrated point model;
the fit is performed only on the resampled rows.  A fixed seed, explicit row
IDs, and a stored fit status make the posterior reproducible.  The result is a
finite empirical posterior per hardware cluster: parameter vectors plus the
fit residual scale.  We will not call this a Bayesian posterior or add a prior
that is not supported by the published data.

For a plan, evaluate every posterior sample and add a residual draw on the
log-time scale.  Report the median and central 90% interval (5th–95th
percentiles) for each objective.  The residual is sampled independently only
for interval construction; ranking/probability calculations use the parameter
draws and preserve correlations between fitted parameters.  Invalid or
non-finite draws are excluded with a counted diagnostic, never silently
replaced.  The existing point estimate remains available as a clearly labeled
calibrated point prediction for backward-compatible exports.

## Validation plan

The new `studies/uncertainty_coverage.py` study will use leave-one-out rows as
the primary held-out check: fit bootstrap samples without row *i*, predict row
*i*, and record whether its published measurement falls inside the nominal
90% interval.  Coverage, interval width, median absolute percentage error, and
the number of usable held-out rows will be reported overall and by strategy.
The study will also report the less favourable cluster/strategy coverage, not
only the aggregate.  With 22 rows, “about 90%” is an acceptance target, not a
claim that exact nominal coverage is statistically established; the report
will include the binomial uncertainty and call out any shortfall.

The implementation is accepted only if: (1) Python unit tests cover bootstrap
reproducibility, quantiles, invalid-draw handling, interval containment, and
probability normalization; (2) TypeScript golden verification agrees with the
Python reference at the existing `1e-10` relative tolerance; (3) the study
emits held-out coverage and caveats; and (4) the existing calibration headline
metrics do not change.  No new data or fabricated measurements may enter
`data/`; provenance stays attached to every observation and posterior manifest.

## UI and export changes

The answer card will show a median followed by a compact “90% interval” for
each objective, with its existing reliability badge unchanged unless the
interval source changes.  The trade-off chart will render a low-opacity
uncertainty band/whisker for front plans; keyboard focus and the accessible SVG
label will include the interval.  Each front plan will show its posterior
probability of being best for the selected objective, e.g. “72% chance this is
the fastest plan,” with “probability from bootstrap samples” in Rigor mode.
Guided mode stays plain-language; Explore exposes sample count and toggles;
Rigor exposes the bootstrap method, row IDs, source link, held-out coverage,
interval definition, and limitations.

CSV/JSON exports will add `median`, `interval_90_low`, `interval_90_high`,
`probability_best`, `uncertainty_method`, and `posterior_sample_count` without
removing current point fields.  The export folder remains
`1-summary` through `5-paper-materials`; methods text and the new coverage
report are included in `5-paper-materials`.  Until the Python reference and
held-out study are present, the UI must say uncertainty is unavailable rather
than displaying estimates as measured results.

References: Narayanan et al. 2021, arXiv:2104.04473, Tables 1–2; the existing
Nomo calibration report and generated observation manifest in
`src/planner/calibration.ts`.
