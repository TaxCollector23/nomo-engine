# Enterprise design profiles

Enterprise profiles make a Nomo run a team-owned deployment contract instead
of a loose collection of knobs. A profile records objectives, constraints,
hardware assumptions, safety policies, allowed precision, required evidence,
organization/project metadata, and optional `config` overrides for a base
`nomo.yaml`.

```python
from nomo.enterprise import (
    capability_evidence_report,
    load_profile,
    merge_with_base_config,
)

profile = load_profile("nomo/enterprise/examples/hard_realtime_robotics.yaml")
config = merge_with_base_config("nomo.yaml", profile)
report = capability_evidence_report(
    profile,
    evidence={
        "task_accuracy": {"evidence_level": "measured", "source": "holdout-v3"},
        "end_to_end_latency": {"evidence_level": "measured", "source": "target-board"},
        "energy_per_inference": {"evidence_level": "proxy", "source": "analytic-profile"},
        "memory_footprint": {"evidence_level": "simulated", "source": "static-analyzer"},
    },
)
```

The merge contract is deterministic:

* `base nomo.yaml < profile.config`.
* Mappings merge recursively.
* Lists and scalar values are replaced by the profile value.
* The validated profile is recorded at `enterprise.profile`.
* `canonical_json(...)` and `profile.fingerprint` are stable across mapping key order.

Required claims are not inferred from profile assumptions. The evidence report
marks each claim as `missing`, `proxy`, `simulated`, or `measured`, and only
marks a profile `release_ready` when every required claim meets its minimum
level.
