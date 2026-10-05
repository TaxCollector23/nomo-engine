# Gate 2 report — run auditor and customer evidence

Status: Partial, preview slice shipped.

Built:

- Megatron command, DeepSpeed JSON, vLLM command, and log metric parsing into one canonical record.
- Loss-aware same-format export for the fields the parser observed.
- Browser-local customer CSV scale fit with central 90% range and empirical coverage.
- Recommendation-sensitive experiment ranking by estimated probability of changing a decision per benchmark hour.
- Python reference tests and browser build coverage.

Integrity boundary:

- Parsing is not calibration. Observed logs are evidence attached to a run, not proof that a model has been refit.
- The local fit is a multiplicative preview and is not a held-out coverage study.
- Unsupported framework fields are not invented or silently preserved.

Open:

- Full source-format round-trip fixtures for every supported DeepSpeed and vLLM option.
- A six-parameter customer refit with held-out validation and hardware/model identity checks.
- Connecting auditor output directly to a recommendation engine for arbitrary customer configurations.
