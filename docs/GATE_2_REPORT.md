# Gate 2 report — run auditor and customer evidence

Status: Partial, preview slice shipped.

Built:

- Megatron command, DeepSpeed JSON, TorchTitan v0.2.2 TOML, vLLM command, and log metric parsing into one canonical record.
- Versioned bounded validation fixtures for common documented fields, with explicit invalid/partial/validated reports.
- Loss-aware same-format export for the fields the parser observed.
- Browser-local customer CSV scale fit with a labelled empirical 90% central range, in-sample coverage, and
  leave-one-out held-out error.
- Recommendation-sensitive experiment ranking by estimated probability of changing a decision per benchmark hour.
- Known published model names connect to the shared graph's bounded recommendation search.
- Python reference tests, auditor fixtures, and browser parity/build coverage.

Integrity boundary:

- Parsing is not calibration. Observed logs are evidence attached to a run, not proof that a model has been refit.
- The local fit is a multiplicative preview; leave-one-out error is reported but is not a published hardware study.
- The interval is computed from customer-supplied residuals and is not a predictive guarantee; missing run or cluster
  identity is rejected before fitting.
- Unsupported framework fields are not assigned invented semantics; they are preserved in same-format exports and surfaced as unknown.

Open:

- Full source-format round-trip fixtures for every framework option, beyond the bounded versioned fields in `docs/FRAMEWORK_AUDITOR.md`.
- Browser TypeScript consumption of the Python fixture reports and target-framework dry-run validation.
- A six-parameter customer refit with held-out validation and hardware/model identity checks.
- Connecting auditor output directly to a recommendation engine for arbitrary customer configurations.
