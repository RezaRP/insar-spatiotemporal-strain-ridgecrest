import numpy as np
import pytest

from ridgecrest_local_vertical import (
    LocalVerticalConfig,
    LocalVerticalModel,
    adaptive_local_support,
    build_local_support_topology,
    calibrate_uncertainty_scale,
    estimate_interval_sill_mm2,
    predict_local_vertical,
    predict_local_vertical_from_topology,
)


def _network() -> np.ndarray:
    x, y = np.meshgrid(np.array([-10.0, 0.0, 10.0]), np.array([-10.0, 0.0, 10.0]))
    return np.column_stack([x.ravel(), y.ravel()])


def _model() -> LocalVerticalModel:
    config = LocalVerticalConfig(radii_km=(16.0,), min_stations=5, min_occupied_sectors=3)
    return LocalVerticalModel(
        family="ok_exponential",
        length_scale_km=12.0,
        nugget_mm=1.0,
        sill_mm2=100.0,
        config=config,
        validation_rmse_mm=1.0,
        validation_nlpd=1.0,
        validation_coverage90=0.9,
        baseline_rmse_mm=2.0,
        baseline_nlpd=2.0,
        bootstrap_delta_nlpd_lower95=0.1,
    )


def test_adaptive_support_uses_every_station_inside_first_adequate_radius():
    stations = _network()
    support = adaptive_local_support(np.array([0.0, 0.0]), stations, _model().config)
    assert support is not None
    assert len(support.indices) == len(stations)
    assert support.radius_km == 16.0
    assert support.occupied_sector_count >= 3
    assert support.inside_local_hull


def test_local_vertical_prediction_masks_extrapolation_outside_local_hull():
    stations = _network()
    values = 0.2 * stations[:, 0] - 0.1 * stations[:, 1]
    prediction = predict_local_vertical(
        _model(), stations, values, np.ones(len(stations)),
        np.array([[0.0, 0.0], [30.0, 30.0]]),
    )
    assert prediction.valid[0]
    assert prediction.support_count[0] == len(stations)
    assert not prediction.valid[1]
    assert np.isnan(prediction.mean_mm[1])


def test_cached_support_prediction_matches_direct_local_prediction():
    stations = _network()
    values = 0.2 * stations[:, 0] - 0.1 * stations[:, 1]
    targets = np.array([[0.0, 0.0], [1.0, 1.0]])
    direct = predict_local_vertical(_model(), stations, values, np.ones(len(stations)), targets)
    topology = build_local_support_topology(stations, targets, _model().config)
    cached = predict_local_vertical_from_topology(_model(), stations, values, np.ones(len(stations)), topology)
    assert np.allclose(cached.mean_mm, direct.mean_mm, equal_nan=True)
    assert np.allclose(cached.sigma_mm, direct.sigma_mm, equal_nan=True)
    assert np.array_equal(cached.support_count, direct.support_count)


@pytest.mark.parametrize(
    "family",
    [
        "ok_exponential",
        "ok_matern32",
        "ok_rbf",
        "uk_matern32",
        "uk_rbf",
        "gp_matern32",
        "gp_rbf",
    ],
)
def test_all_covariance_and_mean_candidates_match_cached_prediction(family):
    stations = _network()
    values = 0.2 * stations[:, 0] - 0.1 * stations[:, 1]
    targets = np.array([[0.0, 0.0], [1.0, 1.0]])
    model = LocalVerticalModel(
        **{**_model().__dict__, "family": family}
    )
    direct = predict_local_vertical(
        model, stations, values, np.ones(len(stations)), targets
    )
    topology = build_local_support_topology(stations, targets, model.config)
    cached = predict_local_vertical_from_topology(
        model, stations, values, np.ones(len(stations)), topology
    )
    assert np.all(direct.valid)
    assert np.allclose(cached.mean_mm, direct.mean_mm, atol=1.0e-9)
    assert np.allclose(cached.sigma_mm, direct.sigma_mm, atol=1.0e-9)


def test_universal_kriging_recovers_local_linear_drift():
    stations = _network()
    values = 4.0 + 0.2 * stations[:, 0] - 0.1 * stations[:, 1]
    target = np.array([[2.0, -3.0]])
    model = LocalVerticalModel(
        **{**_model().__dict__, "family": "uk_matern32"}
    )
    prediction = predict_local_vertical(
        model,
        stations,
        values,
        np.full(len(stations), 1.0e-3),
        target,
    )
    assert prediction.valid[0]
    assert np.isclose(prediction.mean_mm[0], 4.7, atol=1.0e-3)


def test_interval_sill_uses_only_supplied_values_and_removes_noise_variance():
    values = np.array([-10.0, 0.0, 10.0, 20.0])
    sigma = np.full(4, 2.0)
    sill = estimate_interval_sill_mm2(values, sigma)
    assert sill > 1.0
    changed = estimate_interval_sill_mm2(np.r_[values, 1000.0], np.r_[sigma, 2.0])
    assert changed > sill


def test_uncertainty_scale_uses_calibration_residual_quantile():
    observed = np.linspace(-2.0, 2.0, 100)
    predicted = observed + np.linspace(-1.0, 1.0, 100)
    table = {
        "observed_mm": observed,
        "predicted_mm": predicted,
        "predictive_sigma_mm": np.ones(100),
    }
    scale = calibrate_uncertainty_scale(
        __import__("pandas").DataFrame(table)
    )
    assert 0.25 <= scale < 1.0
