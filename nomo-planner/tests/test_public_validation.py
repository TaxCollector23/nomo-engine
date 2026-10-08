from nomo_planner.public_validation import (
    DEFAULT_SARATHI_PATH,
    compare_serving_predictions,
    load_sarathi_table4,
    run_sarathi_table4_replay,
)


def test_sarathi_table4_fixture_is_measured_and_traceable():
    rows = load_sarathi_table4(DEFAULT_SARATHI_PATH)
    assert len(rows) == 12
    assert {row.workload for row in rows} == {"openchat_sharegpt4", "arxiv_summarization"}
    assert {row.scheduler for row in rows} == {
        "hybrid-batching-only",
        "chunked-prefills-only",
        "Sarathi-Serve (combined)",
    }
    assert all(row.source_url.startswith("https://arxiv.org/") for row in rows)
    assert all(row.request_count == 128 and row.token_budget == 1024 for row in rows)


def test_public_serving_comparison_reports_missing_rows_without_filling_them():
    rows = load_sarathi_table4()
    predictions = {rows[0].row_id: rows[0].value_s}
    report = compare_serving_predictions(rows, predictions)
    assert report.status == "incomplete"
    assert report.matched_count == 1
    assert len(report.missing_row_ids) == 11
    assert report.mean_absolute_percentage_error_pct == 0.0


def test_sarathi_replay_runs_the_serving_simulator_against_all_published_rows():
    report = run_sarathi_table4_replay(seed=20261005)
    assert report.status == "validated"
    assert report.observation_count == 12
    assert report.matched_count == 12
    assert report.missing_row_ids == ()
    assert report.mean_absolute_percentage_error_pct is not None
    assert report.mean_absolute_percentage_error_pct >= 0
    assert any("raw Sarathi request traces" in note for note in report.notes)
