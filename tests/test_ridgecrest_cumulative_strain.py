import numpy as np

from ridgecrest_cumulative_strain import (
    build_fixed_joint_mls,
    evaluate_fixed_joint_mls,
    fixed_joint_mls_component_sigma,
    target_values_to_grid,
)


def test_fixed_joint_mls_recovers_cumulative_affine_strain():
    axis = np.arange(-5.0, 6.0)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    sample_xy = np.column_stack([x.ravel(), y.ravel()])
    factors = np.array([0.0, 0.5, 1.0])
    east = np.asarray(
        [factor * (10.0 + 2.0 * sample_xy[:, 0] + 3.0 * sample_xy[:, 1])
         for factor in factors]
    )
    north = np.asarray(
        [factor * (-4.0 - sample_xy[:, 0] + 4.0 * sample_xy[:, 1])
         for factor in factors]
    )
    covariance = np.repeat(np.eye(2)[None, :, :], len(sample_xy), axis=0)
    model = build_fixed_joint_mls(
        sample_xy,
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        covariance_en_mm2=covariance,
        support_radius_km=5.0,
        bandwidth_km=2.5,
        min_samples=16,
    )
    output = evaluate_fixed_joint_mls(model, east, north)
    assert np.all(model.valid)
    assert np.allclose(output["epsilon_EE_microstrain"][:, 0], 2.0 * factors)
    assert np.allclose(output["epsilon_NN_microstrain"][:, 0], 4.0 * factors)
    assert np.allclose(output["gamma_EN_microstrain"][:, 0], 2.0 * factors)
    assert np.allclose(output["dilatation_microstrain"][:, 0], 6.0 * factors)
    assert np.allclose(output["rotation_microradian"][:, 0], -2.0 * factors)
    sigma = fixed_joint_mls_component_sigma(model)
    assert np.isfinite(sigma["dilatation_microstrain"]).all()
    assert np.all(sigma["dilatation_microstrain"] > 0.0)


def test_target_values_to_grid_leaves_unsupported_cells_nan():
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    grid = target_values_to_grid(
        values,
        np.array([0, 1]),
        np.array([1, 2]),
        (2, 3),
    )
    assert grid.shape == (2, 2, 3)
    assert np.isnan(grid[:, 0, 0]).all()
    assert np.array_equal(grid[:, 0, 1], np.array([1.0, 3.0]))
    assert np.array_equal(grid[:, 1, 2], np.array([2.0, 4.0]))


def test_fixed_joint_mls_does_not_use_samples_across_finite_fault():
    sample_xy = np.array(
        [
            [-3.0, -2.0],
            [-3.0, 0.0],
            [-3.0, 2.0],
            [-2.0, -2.0],
            [-2.0, 0.0],
            [-2.0, 2.0],
            [2.0, -2.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [3.0, -2.0],
            [3.0, 0.0],
            [3.0, 2.0],
        ]
    )
    barrier = np.array([[[0.0, -4.0], [0.0, 4.0]]])
    model = build_fixed_joint_mls(
        sample_xy,
        np.array([[-1.0, 0.0]]),
        support_radius_km=6.0,
        bandwidth_km=3.0,
        min_samples=3,
        fault_segments_xy_km=barrier,
    )
    neighbour_x = sample_xy[model.neighbour_indices[0], 0]
    assert model.valid[0]
    assert np.all(neighbour_x < 0.0)
    assert model.barrier_excluded_count[0] == 6
