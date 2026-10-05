from nomo_planner.experiments import ExperimentCandidate, csv_template, rank_experiments


def test_experiment_designer_prioritizes_uncertain_short_benchmarks():
    ranked = rank_experiments([
        ExperimentCandidate("long-uncertain", {"tp": 4}, 1.1, 0.9, 2),
        ExperimentCandidate("short-clear", {"tp": 2}, 1.2, 0.1, 1),
    ], current_step_s=1.0, alternate_step_s=1.2)
    assert ranked[0].name == "long-uncertain"
    assert "config_json" in csv_template(ranked)
