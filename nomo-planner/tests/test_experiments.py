from nomo_planner.experiments import ExperimentCandidate, csv_template, parse_experiment_results_csv, rank_experiments


def test_experiment_designer_prioritizes_uncertain_short_benchmarks():
    ranked = rank_experiments([
        ExperimentCandidate("long-uncertain", {"tp": 4}, 1.1, 0.9, 2),
        ExperimentCandidate("short-clear", {"tp": 2}, 1.2, 0.1, 1),
    ], current_step_s=1.0, alternate_step_s=1.2)
    assert ranked[0].name == "long-uncertain"
    assert "config_json" in csv_template(ranked)


def test_completed_experiment_rows_feed_local_calibration_without_guessing():
    ranked = rank_experiments([
        ExperimentCandidate("tp-2", {"tp": 2}, 1.2, 0.1, 1),
    ], current_step_s=1.0, alternate_step_s=1.2)
    rows = parse_experiment_results_csv(
        "experiment,config_json,observed_step_s,observed_tokens_per_s,notes\n"
        "tp-2,,1.5,,measured locally\n",
        ranked,
    )
    assert rows[0].predicted_step_s == 1.2
    assert rows[0].observed_step_s == 1.5
