# Gate 2 report — run auditor and customer evidence

Status: Partial, preview slice shipped.

Built:

- Megatron command, DeepSpeed JSON, vLLM command, and log metric parsing into one canonical record.
- Loss-aware same-format export for the fields the parser observed.
- Browser-local customer CSV scale fit with central 90% range, empirical coverage, and leave-one-out held-out error.
- Recommendation-sensitive experiment ranking by estimated probability of changing a decision per benchmark hour.
- Known published model names connect to the shared graph's bounded recommendation search.
- Python reference tests, auditor fixtures, and browser parity/build coverage.

Integrity boundary:

- Parsing is not calibration. Observed logs are evidence attached to a run, not proof that a model has been refit.
- The local fit is a multiplicative preview; leave-one-out error is reported but is not a published hardware study.
- Unsupported framework fields are not invented or silently preserved.

Open:

- Full source-format round-trip fixtures for every framework option, beyond the supported core fields.
- A six-parameter customer refit with held-out validation and hardware/model identity checks.
- Connecting auditor output directly to a recommendation engine for arbitrary customer configurations.
