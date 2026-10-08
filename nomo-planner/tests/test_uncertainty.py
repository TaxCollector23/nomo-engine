from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nomo_planner.uncertainty import (  # noqa: E402
    CALIBRATED_PARAMS,
    fit_bootstrap,
    fit_parameters,
    load_observations,
    measured_step,
    predictive_interval,
    predict_step,
    safest_plan,
)


ROWS = load_observations(Path(__file__).parents[1] / "data" / "training_observations.csv")


def test_source_has_22_published_rows_and_valid_measurements():
    assert len(ROWS) == 22
    assert all(row.source.startswith("Narayanan et al. 2021") for row in ROWS)
    assert all(measured_step(row) > 0 for row in ROWS)


def test_point_fit_is_finite_and_refit_does_not_make_error_worse():
    before = sum((predict_step(row, CALIBRATED_PARAMS) / measured_step(row) - 1) ** 2 for row in ROWS)
    params, sigma = fit_parameters(ROWS)
    after = sum((predict_step(row, params) / measured_step(row) - 1) ** 2 for row in ROWS)
    assert sigma >= 0
    assert after <= before + 1e-9


def test_bootstrap_is_reproducible_and_stratified():
    a = fit_bootstrap(ROWS, replicates=6, seed=7)
    b = fit_bootstrap(ROWS, replicates=6, seed=7)
    assert [sample.as_dict() for sample in a] == [sample.as_dict() for sample in b]
    assert all(sample.source_rows == 22 for sample in a)


def test_predictive_interval_is_ordered_and_contains_its_median():
    samples = fit_bootstrap(ROWS, replicates=12, seed=4)
    interval = predictive_interval(ROWS[0], samples, seed=8)
    assert interval["low"] <= interval["median"] <= interval["high"]


def test_safest_plan_uses_interval_regret_bounds():
    plans = [{"key": "a"}, {"key": "b"}]
    intervals = {
        "a": {"time": {"median": 10, "low": 8, "high": 14}},
        "b": {"time": {"median": 11, "low": 9, "high": 12}},
    }
    assert safest_plan(plans, intervals, [("time", False)]) == plans[1]
