import numpy as np

from ridgecrest_vertical_los import (
    LocalKrigingModel,
    predict_local_ordinary_kriging_field,
    select_local_ordinary_kriging_model,
)


def _model() -> LocalKrigingModel:
    return LocalKrigingModel(
        length_scale_km=8.0,
        nugget_mm=1.0,
        search_radius_km=20.0,
        min_stations=4,
        sill_mm2=100.0,
        loo_rmse_mm=0.0,
        loo_mae_mm=0.0,
        loo_nlpd=0.0,
        loo_standardized_rms=1.0,
        loo_coverage=1.0,
    )


def test_local_kriging_uses_all_nearby_stations_not_fixed_nearest_count():
    xy = np.array([
        [-5.0, -5.0], [5.0, -5.0], [-5.0, 5.0], [5.0, 5.0],
        [0.0, -7.0], [0.0, 7.0], [-7.0, 0.0], [7.0, 0.0],
    ])
    values = 2.0 * xy[:, 0] - xy[:, 1]
    mean, sigma, count = predict_local_ordinary_kriging_field(
        _model(), xy, values, np.ones(len(values)), np.array([[0.0, 0.0]])
    )
    assert count[0] == len(xy)
    assert np.isfinite(mean[0])
    assert np.isfinite(sigma[0])


def test_local_kriging_masks_target_without_enough_local_stations():
    xy = np.array([
        [-5.0, -5.0], [5.0, -5.0], [-5.0, 5.0], [5.0, 5.0],
        [0.0, -7.0], [0.0, 7.0], [-7.0, 0.0], [7.0, 0.0],
    ])
    mean, sigma, count = predict_local_ordinary_kriging_field(
        _model(), xy, np.arange(len(xy), dtype=float), np.ones(len(xy)),
        np.array([[100.0, 100.0]]),
    )
    assert count[0] == 0
    assert np.isnan(mean[0])
    assert np.isnan(sigma[0])


def test_local_kriging_cv_requires_full_holdout_coverage():
    xy = np.array([
        [-10.0, -10.0], [10.0, -10.0], [-10.0, 10.0], [10.0, 10.0],
        [0.0, -12.0], [0.0, 12.0], [-12.0, 0.0], [12.0, 0.0],
    ])
    values = 0.5 * xy[:, 0] - 0.25 * xy[:, 1]
    model, score, predictions = select_local_ordinary_kriging_model(
        xy, values, np.ones(len(values)),
        search_radii_km=(40.0,), length_scales_km=(10.0,), nuggets_mm=(1.0,),
        min_stations=4,
    )
    assert model.loo_coverage == 1.0
    assert score.loc[score["selected"], "loo_coverage"].iloc[0] == 1.0
    assert predictions["predicted"].all()
