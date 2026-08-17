import numpy as np

from ridgecrest_strain_change import (
    duration_normalize,
    empirical_upper_tail_pvalue,
    fit_robust_baseline,
    holm_adjust,
    positive_page_cusum,
    signed_spatial_clusters,
    sliding_block_cusum_maxima,
    sliding_block_maximum,
    standardized_innovation,
    strain_energy,
)


def test_duration_normalization_scales_value_and_sigma():
    value = np.array([[[12.0]], [[6.0]]])
    sigma = np.array([[[2.4]], [[1.2]]])
    rate, sigma_rate = duration_normalize(value, sigma, np.array([12.0, 6.0]))
    assert np.allclose(rate[:, 0, 0], 1.0)
    assert np.allclose(sigma_rate[:, 0, 0], 0.2)


def test_frozen_baseline_does_not_leak_surveillance_value():
    value = np.array([0.0, 0.2, -0.2, 0.1, 10.0])[:, None, None]
    sigma = np.ones_like(value)
    mask = np.array([True, True, True, True, False])
    baseline = fit_robust_baseline(value, sigma, mask, min_observations=4)
    first = standardized_innovation(value, sigma, baseline)[-1, 0, 0]
    altered = value.copy()
    altered[-1] = 1000.0
    second_baseline = fit_robust_baseline(altered, sigma, mask, min_observations=4)
    assert np.allclose(second_baseline.center, baseline.center)
    assert np.allclose(second_baseline.excess_scale, baseline.excess_scale)
    assert first > 9.0


def test_signed_clusters_do_not_join_opposite_signs():
    east = np.array([0.0, 1.0, 0.0, 1.0])
    north = np.array([0.0, 0.0, 1.0, 1.0])
    z = np.array([[3.0, 2.5, -3.0, -2.5]])
    clusters = signed_spatial_clusters(
        z,
        east,
        north,
        ["dilatation"],
        threshold=1.96,
        min_cells=2,
    )
    assert len(clusters) == 2
    assert {cluster.sign for cluster in clusters} == {-1, 1}
    assert all(cluster.cell_count == 2 for cluster in clusters)


def test_block_max_pvalue_holm_and_energy():
    null = sliding_block_maximum(np.array([1.0, 3.0, 2.0, 4.0]), 2)
    assert np.array_equal(null, np.array([3.0, 3.0, 4.0]))
    assert empirical_upper_tail_pvalue(null, 4.0) == 0.5
    adjusted = holm_adjust(np.array([0.01, 0.04, 0.03]))
    assert np.allclose(adjusted, np.array([0.03, 0.06, 0.06]))
    energy = strain_energy(np.array([[[1.0, -3.0]], [[2.0, -4.0]]]), 0.5)
    assert np.allclose(energy, np.array([2.0, 3.0]))


def test_page_cusum_and_serial_block_null():
    values = np.array([0.0, 1.0, 1.0, -2.0])
    assert np.allclose(
        positive_page_cusum(values, reference=0.5),
        np.array([0.0, 0.5, 1.0, 0.0]),
    )
    null = sliding_block_cusum_maxima(values, 2, reference=0.5)
    assert np.allclose(null, np.array([0.5, 1.0, 0.5]))
