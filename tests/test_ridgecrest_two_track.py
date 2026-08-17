import numpy as np

from ridgecrest_two_track import (
    correct_vertical_los_on_grid,
    masked_bilinear_resample,
    rmls_incremental_strain,
    solve_two_track_horizontal,
)


def test_vertical_correction_accepts_native_reference_summary():
    corrected, contribution, sigma, reference, reference_sigma = correct_vertical_los_on_grid(
        np.array([[12.0]]), np.array([[0.8]]),
        np.array([[10.0]]), np.array([[2.0]]),
        reference_value_mm=6.0, reference_sigma_mm=1.0,
    )
    assert np.isclose(reference, 6.0)
    assert np.isclose(reference_sigma, 1.0)
    assert np.isclose(contribution[0, 0], 2.0)
    assert np.isclose(corrected[0, 0], 10.0)
    assert np.isclose(sigma[0, 0], np.hypot(1.6, 1.0))


def test_masked_bilinear_resample_requires_complete_support():
    latitude = np.array([0.0, 1.0])
    longitude = np.array([0.0, 1.0])
    values = np.array([[0.0, 1.0], [1.0, 2.0]])
    target_latitude = np.array([[0.5]])
    target_longitude = np.array([[0.5]])
    sampled, support = masked_bilinear_resample(
        latitude, longitude, values, np.ones_like(values, dtype=bool),
        target_latitude, target_longitude,
    )
    assert np.isclose(sampled[0, 0], 1.0)
    assert np.isclose(support[0, 0], 1.0)

    masked, _ = masked_bilinear_resample(
        latitude, longitude, values, np.array([[True, False], [True, True]]),
        target_latitude, target_longitude,
    )
    assert np.isnan(masked[0, 0])


def test_two_track_horizontal_solution_recovers_enu():
    east_true = 10.0
    north_true = 20.0
    ae, an = -0.6, -0.1
    de, dn = 0.6, -0.1
    ascending = ae * east_true + an * north_true
    descending = de * east_true + dn * north_true
    solution = solve_two_track_horizontal(
        np.array([[ascending]]), np.array([[descending]]),
        np.array([[ae]]), np.array([[an]]),
        np.array([[de]]), np.array([[dn]]),
        np.array([[2.0]]), np.array([[2.0]]),
        vertical_los_sigma_ascending_mm=np.zeros((1, 1)),
        vertical_los_sigma_descending_mm=np.zeros((1, 1)),
    )
    assert solution.valid[0, 0]
    assert np.isclose(solution.east_mm[0, 0], east_true)
    assert np.isclose(solution.north_mm[0, 0], north_true)


def test_two_track_horizontal_adds_vertical_variance_and_covariance():
    solution = solve_two_track_horizontal(
        np.array([[3.0]]), np.array([[4.0]]),
        np.array([[1.0]]), np.array([[0.0]]),
        np.array([[0.0]]), np.array([[1.0]]),
        np.array([[2.0]]), np.array([[3.0]]),
        vertical_los_sigma_ascending_mm=np.array([[4.0]]),
        vertical_los_sigma_descending_mm=np.array([[5.0]]),
        vertical_correlation=0.5,
    )
    assert solution.valid[0, 0]
    assert np.isclose(solution.sigma_east_mm[0, 0], np.sqrt(20.0))
    assert np.isclose(solution.sigma_north_mm[0, 0], np.sqrt(34.0))
    assert np.isclose(
        solution.covariance_east_north_mm2[0, 0],
        10.0,
    )


def test_two_track_invalid_geometry_has_nan_condition_number():
    solution = solve_two_track_horizontal(
        np.array([[1.0]]), np.array([[1.0]]),
        np.array([[0.6]]), np.array([[0.1]]),
        np.array([[0.6]]), np.array([[0.1]]),
        np.array([[2.0]]), np.array([[2.0]]),
    )
    assert not solution.valid[0, 0]
    assert np.isnan(solution.condition_number[0, 0])


def test_rmls_recovers_known_affine_incremental_strain():
    east_axis = np.arange(-4.0, 5.0)
    north_axis = np.arange(-4.0, 5.0)
    x, y = np.meshgrid(east_axis, north_axis, indexing="xy")
    xy = np.column_stack([x.ravel(), y.ravel()])
    # Derivatives in mm/km: dE/dx=2, dE/dy=3, dN/dx=-1, dN/dy=4.
    east = 2.0 * xy[:, 0] + 3.0 * xy[:, 1]
    north = -xy[:, 0] + 4.0 * xy[:, 1]
    output = rmls_incremental_strain(
        xy, east, north, np.ones(len(xy)), np.ones(len(xy)),
        np.array([[0.0, 0.0]]),
        support_radius_km=4.0, bandwidth_km=2.0, min_samples=16,
    )
    assert output.loc[0, "valid"]
    assert np.isclose(output.loc[0, "epsilon_xx_nstrain"], 2000.0)
    assert np.isclose(output.loc[0, "epsilon_yy_nstrain"], 4000.0)
    assert np.isclose(output.loc[0, "epsilon_xy_nstrain"], 1000.0)
    assert np.isclose(output.loc[0, "dilatation_nstrain"], 6000.0)
