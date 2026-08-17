import numpy as np
import pandas as pd

from ridgecrest_los_projection import (
    heading_incidence_from_look_vector,
    look_vector_from_heading_incidence,
    project_enu_mm,
    remove_baseline_median,
)


def test_portal_geometry_reproduces_expected_ascending_and_descending_signs():
    ascending = look_vector_from_heading_incidence(-10.146887, 39.6181)
    descending = look_vector_from_heading_incidence(-169.84503, 33.7677)

    assert np.allclose(
        ascending, [-0.62769383, -0.11233934, 0.77031184], atol=1.0e-7
    )
    assert np.allclose(
        descending, [0.54711975, -0.09799853, 0.83129794], atol=1.0e-7
    )
    assert np.isclose(np.linalg.norm(ascending), 1.0)
    assert np.isclose(np.linalg.norm(descending), 1.0)


def test_heading_incidence_round_trip():
    for heading, incidence in ((-10.146887, 39.6181), (-169.84503, 33.7677)):
        vector = look_vector_from_heading_incidence(heading, incidence)
        recovered_heading, recovered_incidence = heading_incidence_from_look_vector(
            vector
        )
        assert np.isclose(recovered_heading, heading)
        assert np.isclose(recovered_incidence, incidence)


def test_projection_is_full_enu_dot_product():
    vector = np.array([-0.6, -0.1, 0.793725393])
    vector /= np.linalg.norm(vector)
    result = project_enu_mm(100.0, -20.0, 10.0, vector)
    expected = vector @ np.array([100.0, -20.0, 10.0])
    assert np.isclose(result, expected)


def test_baseline_removal_fits_only_one_constant():
    dates = pd.date_range("2018-01-01", periods=6, freq="D")
    values = np.array([4.0, 5.0, 6.0, 100.0, 101.0, 102.0])
    centred, baseline, count = remove_baseline_median(
        values, dates, start="2018-01-01", end="2018-01-03"
    )
    assert baseline == 5.0
    assert count == 3
    assert np.allclose(centred, values - 5.0)
