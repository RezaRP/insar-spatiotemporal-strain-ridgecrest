import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ridgecrest_transient import (
    PatchSeries,
    bayesian_transient_analysis,
    spatial_transient_matched_filter,
)


def test_bayesian_acceleration_recovers_synthetic_onset():
    rng = np.random.default_rng(11)
    dates = pd.date_range("2016-01-01", periods=100, freq="12D")
    t = ((dates - dates[0]).days / 365.2425).to_numpy(float)
    split = 72
    hinge = np.maximum(0.0, t - t[split])
    values = 2.0 + 0.5 * t + 0.8 * np.sin(2 * np.pi * t)
    values += 30.0 * hinge + 110.0 * 0.5 * hinge**2
    noise = rng.normal(0.0, 0.8, len(t))
    for i in range(1, len(noise)):
        noise[i] += 0.35 * noise[i - 1]
    values += noise

    summary, _, _ = bayesian_transient_analysis(
        dates,
        values,
        min_before=20,
        min_after=5,
        n_samples=500,
        seed=4,
    )

    assert summary["p_transient"] > 0.95
    assert summary["best_model"] == "acceleration"
    assert abs((summary["best_onset_date"] - dates[split]).days) <= 24
    assert summary["p_primary_coefficient_positive"] > 0.95


def test_spatial_matched_filter_recovers_one_synthetic_patch():
    rng = np.random.default_rng(8)
    dates = pd.date_range("2017-01-01", periods=80, freq="12D")
    t = ((dates - dates[0]).days / 365.2425).to_numpy(float)
    values = rng.normal(0.0, 0.4, size=(len(t), 5))
    split = 55
    values[:, 2] += 12.0 * np.maximum(0.0, t - t[split])
    metadata = pd.DataFrame(
        {
            "patch_id": np.arange(5),
            "latitude": np.linspace(35.6, 35.8, 5),
            "longitude": np.linspace(-117.7, -117.4, 5),
            "east_km": np.arange(5),
            "north_km": np.arange(5),
            "distance_km": np.arange(5),
            "along_strike_km": np.arange(5),
            "cross_strike_km": np.zeros(5),
            "pixel_count": np.full(5, 100),
        }
    )
    patches = PatchSeries(
        dates=dates,
        values=values,
        metadata=metadata,
        reference=np.zeros(len(t)),
        quality_mask=np.ones((1, 1), dtype=bool),
        target_mask=np.ones((1, 1), dtype=bool),
    )

    result, _, _ = spatial_transient_matched_filter(
        patches,
        min_before=20,
        min_after=5,
        n_boot=99,
        block_length=4,
        seed=5,
    )

    assert int(result.loc[result["max_statistic"].idxmax(), "patch_id"]) == 2
    assert result.loc[2, "corrected_pvalue"] <= 0.05
    assert abs((result.loc[2, "best_onset_date"] - dates[split]).days) <= 24
