from nomo_planner.customer_calibration import apply_customer_fit, fit_customer_scale, parse_customer_csv


def test_customer_csv_refit_recovers_known_scale_and_interval():
    rows = parse_customer_csv("run_id,cluster,predicted_step_s,observed_step_s\n"
                             "a,h100,1,1.2\n"
                             "b,h100,2,2.4\n"
                             "c,h100,4,4.8\n")
    fit = fit_customer_scale(rows, cluster="h100")
    assert fit.observations == 3
    assert fit.scale == 1.2
    interval = apply_customer_fit(10, fit)
    assert interval["median"] == 12
    assert interval["low"] <= interval["median"] <= interval["high"]
    assert fit.coverage == 1
    assert fit.held_out_observations == 3
    assert fit.held_out_mape_pct is not None and fit.held_out_mape_pct < 1e-12


def test_customer_fit_rejects_missing_or_invalid_values():
    try:
        parse_customer_csv("run_id,cluster,predicted_step_s\na,h100,1\n")
    except ValueError as error:
        assert "missing columns" in str(error)
    else:
        raise AssertionError("missing columns should fail loudly")


def test_customer_fit_reports_unfitted_error_for_noisy_rows():
    rows = parse_customer_csv("run_id,cluster,predicted_step_s,observed_step_s\n"
                             "a,h100,1,1\n"
                             "b,h100,1,2\n"
                             "c,h100,1,1\n")
    fit = fit_customer_scale(rows, cluster="h100")
    assert fit.held_out_observations == 3
    assert fit.held_out_mape_pct is not None and fit.held_out_mape_pct > 0
    assert fit.interval_level == 0.90
    assert "not a predictive guarantee" in fit.interval_scope


def test_customer_fit_requires_run_and_cluster_identity():
    try:
        parse_customer_csv("run_id,cluster,predicted_step_s,observed_step_s\n,h100,1,1\n")
    except ValueError as error:
        assert "run_id" in str(error)
    else:
        raise AssertionError("missing run identity should fail loudly")
