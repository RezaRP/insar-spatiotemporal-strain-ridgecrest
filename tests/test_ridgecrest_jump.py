import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ridgecrest_jump import (
    bayesian_change_point,
    frequentist_breakpoint_bootstrap,
    pelt_change_detection,
    scan_change_points,
)


def test_recovers_synthetic_hinge_change():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2017-01-01", periods=120, freq="6D")
    t = ((dates - dates[0]).days / 365.2425).to_numpy(float)
    split = 82
    tau = t[split]
    values = 2.0 + 1.5 * t + 0.8 * np.sin(2 * np.pi * t)
    values += -80.0 * np.maximum(0.0, t - tau)
    values += rng.normal(0.0, 0.35, size=t.size)

    result = scan_change_points(dates, values, min_before=20, min_after=5).iloc[0]

    assert result["model"] == "hinge"
    assert abs((pd.Timestamp(result["change_date"]) - dates[split]).days) <= 12
    assert result["delta_bic"] > 10
    assert result["amplitude"] < 0


def test_recovers_synthetic_step_change():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2017-01-01", periods=120, freq="6D")
    t = ((dates - dates[0]).days / 365.2425).to_numpy(float)
    split = 75
    values = -1.0 + 0.4 * t + 0.5 * np.cos(2 * np.pi * t)
    values += 12.0 * (np.arange(t.size) >= split)
    values += rng.normal(0.0, 0.3, size=t.size)

    result = scan_change_points(dates, values, min_before=20, min_after=5).iloc[0]

    assert result["model"] == "step"
    assert abs((pd.Timestamp(result["change_date"]) - dates[split]).days) <= 6
    assert result["delta_bic"] > 10
    assert result["amplitude"] > 0


def test_bayesian_method_prefers_synthetic_step():
    rng = np.random.default_rng(2026)
    dates = pd.date_range("2017-01-01", periods=100, freq="12D")
    t = ((dates - dates[0]).days / 365.2425).to_numpy(float)
    split = 70
    values = 2.0 + 0.8 * t + 0.5 * np.sin(2 * np.pi * t)
    values += 8.0 * (np.arange(t.size) >= split)
    values += rng.normal(0.0, 0.5, size=t.size)

    summary, _ = bayesian_change_point(
        dates, values, min_before=20, min_after=5
    )

    assert summary["p_step"] > summary["p_baseline"]
    assert summary["best_model"] == "step"
    assert abs((summary["best_change_date"] - dates[split]).days) <= 12
    assert summary["best_amplitude"] > 0


def test_frequentist_methods_recover_synthetic_shift():
    rng = np.random.default_rng(91)
    dates = pd.date_range("2017-01-01", periods=100, freq="12D")
    split = 65
    values = rng.normal(0.0, 0.4, size=dates.size)
    values[split:] += 5.0

    stability, _ = pelt_change_detection(
        dates,
        values,
        penalty_multipliers=(1.0, 2.0, 3.0),
        min_size=5,
    )
    detected = [
        date
        for dates_tuple in stability["breakpoint_dates"]
        for date in dates_tuple
    ]
    assert any(abs((date - dates[split]).days) <= 12 for date in detected)

    result = frequentist_breakpoint_bootstrap(
        dates,
        values,
        min_before=20,
        min_after=5,
        n_boot=99,
        seed=13,
    )
    assert abs((result["change_date"] - dates[split]).days) <= 12
    assert result["pvalue"] <= 0.05
