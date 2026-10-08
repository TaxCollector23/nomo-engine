# Gate 4 report — infrastructure product packs

Status: Partial, explicitly labelled Preview.

Built:

- Reference estimators and browser ports for chip bottlenecks/Pareto filtering, RL scheduling, checkpoint/restart goodput, fleet sizing, fine-tuning options, and buy/rent TCO.
- Physics-sanity tests for positive inputs, bottleneck selection, Young checkpoint behavior, fleet constraints, fine-tuning feasibility, and TCO selection.
- Formula references for Young (1961), Daly (1990), and Erlang (1917) in the source and Methods surface.
- Product Studio UI with editable inputs, visible user-input/assumption labels, and per-pack JSON/CSV preview exports.

Open:

- Customer calibration and joint chip/software co-optimization are still open; the current forms/exports are reference previews.
- Queueing, chip, RL, and fine-tuning quality effects are simplified estimates; they are not silicon, benchmark, or production guarantees.
- No market price is silently embedded in the product estimators.
