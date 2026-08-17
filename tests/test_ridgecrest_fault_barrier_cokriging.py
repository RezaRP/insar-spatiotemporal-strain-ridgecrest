import numpy as np

from ridgecrest_fault_barrier_cokriging import (
    build_fixed_fault_barrier_cokriging,
    cokriging_diagnostics,
    evaluate_fixed_fault_barrier_cokriging,
    finite_segment_crossing_mask,
    matern32_correlation,
)

ASCENDING_LOOK = np.array([-0.55, -0.10])
DESCENDING_LOOK = np.array([0.55, -0.12])


def _affine_displacement(xy, factor):
    east = factor * (5.0 + 0.7 * xy[:, 0] - 0.2 * xy[:, 1])
    north = factor * (-2.0 + 0.3 * xy[:, 0] + 0.4 * xy[:, 1])
    return east, north


def _project(east, north, look):
    return east * look[0] + north * look[1]


def test_matern32_correlation_has_unit_origin_and_monotonic_decay():
    distance = np.array([0.0, 1.0, 4.0, 12.0])
    correlation = matern32_correlation(distance, length_scale_km=4.0)
    assert correlation[0] == 1.0
    assert np.all(np.diff(correlation) < 0.0)
    assert np.all((correlation > 0.0) & (correlation <= 1.0))


def test_finite_segment_barrier_blocks_crossings_but_respects_endpoints():
    fault = np.array([[[0.0, -1.0], [0.0, 1.0]]])
    start = np.array([-2.0, 0.0])
    end = np.array(
        [
            [2.0, 0.0],   # Crosses the segment interior.
            [2.0, 2.0],   # Touches its upper endpoint.
            [2.0, 3.0],   # Passes beyond the finite upper endpoint.
            [-3.0, 1.0],  # Remains on the target side.
        ]
    )
    crossing = finite_segment_crossing_mask(start, end, fault)
    assert np.array_equal(crossing, np.array([True, True, False, False]))


def test_universal_cokriging_recovers_affine_en_and_desc_for_80_epochs():
    axis = np.arange(-8.0, 8.1, 4.0)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    sample_xy = np.column_stack([x.ravel(), y.ravel()])
    target_xy = np.array([[0.5, -0.7], [2.2, 1.4]])

    model = build_fixed_fault_barrier_cokriging(
        sample_xy,
        target_xy,
        sample_ascending_look_en=ASCENDING_LOOK,
        sample_descending_look_en=DESCENDING_LOOK,
        target_ascending_look_en=ASCENDING_LOOK,
        target_descending_look_en=DESCENDING_LOOK,
        latent_covariance_mm2=np.array([[225.0, 30.0], [30.0, 144.0]]),
        length_scale_km=6.0,
        support_radius_km=20.0,
        sample_ascending_noise_variance_mm2=0.25,
        sample_descending_noise_variance_mm2=0.36,
        target_ascending_noise_variance_mm2=0.16,
        minimum_paired_samples=8,
        maximum_paired_samples=None,
    )
    assert np.all(model.valid)
    assert all(weight.shape[0] == 3 for weight in model.weights)
    assert np.nanmax(model.unbiasedness_error) < 1.0e-10

    factors = np.linspace(0.0, 1.8, 80)
    sample_east = []
    sample_north = []
    target_east = []
    target_north = []
    for factor in factors:
        east, north = _affine_displacement(sample_xy, factor)
        sample_east.append(east)
        sample_north.append(north)
        east, north = _affine_displacement(target_xy, factor)
        target_east.append(east)
        target_north.append(north)
    sample_east = np.asarray(sample_east)
    sample_north = np.asarray(sample_north)
    target_east = np.asarray(target_east)
    target_north = np.asarray(target_north)

    output = evaluate_fixed_fault_barrier_cokriging(
        model,
        _project(sample_east, sample_north, ASCENDING_LOOK),
        _project(sample_east, sample_north, DESCENDING_LOOK),
        _project(target_east, target_north, ASCENDING_LOOK),
    )
    expected_descending = _project(
        target_east,
        target_north,
        DESCENDING_LOOK,
    )
    assert output["east_mm"].shape == (80, 2)
    assert np.allclose(output["east_mm"], target_east, atol=1.0e-9)
    assert np.allclose(output["north_mm"], target_north, atol=1.0e-9)
    assert np.allclose(
        output["descending_los_mm"],
        expected_descending,
        atol=1.0e-9,
    )
    assert np.allclose(
        output["ascending_conditioning_residual_mm"],
        0.0,
        atol=1.0e-9,
    )

    for target_index, descending_look in enumerate(
        model.target_descending_look_en
    ):
        covariance = model.posterior_covariance[target_index]
        assert np.min(np.linalg.eigvalsh(covariance)) >= -1.0e-10
        assert np.isclose(
            covariance[2, 2],
            descending_look
            @ covariance[:2, :2]
            @ descending_look,
            atol=1.0e-10,
        )


def test_builder_excludes_cross_fault_samples_and_reports_diagnostics():
    x, y = np.meshgrid(
        np.array([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0]),
        np.array([-4.0, -2.0, 0.0, 2.0, 4.0]),
        indexing="xy",
    )
    sample_xy = np.column_stack([x.ravel(), y.ravel()])
    target_xy = np.array([[-2.0, 0.0]])
    fault = np.array([[[0.0, -10.0], [0.0, 10.0]]])

    model = build_fixed_fault_barrier_cokriging(
        sample_xy,
        target_xy,
        sample_ascending_look_en=ASCENDING_LOOK,
        sample_descending_look_en=DESCENDING_LOOK,
        target_ascending_look_en=ASCENDING_LOOK,
        target_descending_look_en=DESCENDING_LOOK,
        fault_segments_xy_km=fault,
        length_scale_km=4.0,
        support_radius_km=8.0,
        minimum_paired_samples=6,
        maximum_paired_samples=None,
    )
    assert model.valid[0]
    assert model.blocked_count[0] > 0
    assert model.eligible_count[0] + model.blocked_count[0] == (
        model.candidate_count[0]
    )
    used_xy = sample_xy[model.neighbour_indices[0]]
    assert np.all(used_xy[:, 0] < 0.0)

    diagnostic = cokriging_diagnostics(model)
    assert diagnostic["valid"][0]
    assert diagnostic["drift_rank"][0] == 6
    assert diagnostic["unbiasedness_error"][0] < 1.0e-10
    assert np.isfinite(diagnostic["posterior_sigma_east_mm"][0])
    assert diagnostic["posterior_sigma_east_mm"][0] > 0.0


def test_evaluation_keeps_fixed_weights_and_only_masks_missing_epoch_target():
    axis = np.arange(-6.0, 6.1, 3.0)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    sample_xy = np.column_stack([x.ravel(), y.ravel()])
    target_xy = np.array([[0.0, 0.0], [1.0, 1.0]])
    model = build_fixed_fault_barrier_cokriging(
        sample_xy,
        target_xy,
        sample_ascending_look_en=ASCENDING_LOOK,
        sample_descending_look_en=DESCENDING_LOOK,
        target_ascending_look_en=ASCENDING_LOOK,
        target_descending_look_en=DESCENDING_LOOK,
        support_radius_km=12.0,
        minimum_paired_samples=8,
        maximum_paired_samples=None,
    )
    weights_before = tuple(weight.copy() for weight in model.weights)

    east, north = _affine_displacement(sample_xy, factor=1.0)
    target_east, target_north = _affine_displacement(target_xy, factor=1.0)
    sample_ascending = np.repeat(
        _project(east, north, ASCENDING_LOOK)[None, :],
        3,
        axis=0,
    )
    sample_descending = np.repeat(
        _project(east, north, DESCENDING_LOOK)[None, :],
        3,
        axis=0,
    )
    target_ascending = np.repeat(
        _project(target_east, target_north, ASCENDING_LOOK)[None, :],
        3,
        axis=0,
    )
    target_ascending[1, 0] = np.nan

    output = evaluate_fixed_fault_barrier_cokriging(
        model,
        sample_ascending,
        sample_descending,
        target_ascending,
    )
    assert np.isnan(output["descending_los_mm"][1, 0])
    assert np.isfinite(output["descending_los_mm"][1, 1])
    assert np.isfinite(output["descending_los_mm"][[0, 2], 0]).all()
    for previous, current in zip(weights_before, model.weights):
        assert np.array_equal(previous, current)


def test_unconditioned_buffered_baseline_records_exclusions_and_needs_no_target_asc():
    axis = np.arange(-9.0, 9.1, 3.0)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    sample_xy = np.column_stack([x.ravel(), y.ravel()])
    target_xy = np.array([[0.0, 0.0]])
    model = build_fixed_fault_barrier_cokriging(
        sample_xy,
        target_xy,
        sample_ascending_look_en=ASCENDING_LOOK,
        sample_descending_look_en=DESCENDING_LOOK,
        target_ascending_look_en=ASCENDING_LOOK,
        target_descending_look_en=DESCENDING_LOOK,
        condition_on_target_ascending=False,
        target_exclusion_radius_km=3.1,
        support_radius_km=15.0,
        minimum_paired_samples=8,
        maximum_paired_samples=None,
    )
    assert model.valid[0]
    assert not model.condition_on_target_ascending
    assert model.buffer_excluded_count[0] == 5
    assert np.array_equal(
        model.excluded_sample_indices[0],
        np.sort(model.buffer_excluded_indices[0]),
    )
    excluded_distance = np.linalg.norm(
        sample_xy[model.buffer_excluded_indices[0]] - target_xy[0],
        axis=1,
    )
    used_distance = np.linalg.norm(
        sample_xy[model.neighbour_indices[0]] - target_xy[0],
        axis=1,
    )
    assert np.all(excluded_distance <= 3.1)
    assert np.all(used_distance > 3.1)
    assert model.weights[0].shape == (3, 2 * model.used_count[0])

    east, north = _affine_displacement(sample_xy, factor=1.3)
    target_east, target_north = _affine_displacement(target_xy, factor=1.3)
    output = evaluate_fixed_fault_barrier_cokriging(
        model,
        _project(east, north, ASCENDING_LOOK),
        _project(east, north, DESCENDING_LOOK),
    )
    assert np.allclose(output["east_mm"][0], target_east, atol=1.0e-9)
    assert np.allclose(output["north_mm"][0], target_north, atol=1.0e-9)
    assert np.isnan(output["ascending_conditioning_residual_mm"]).all()
