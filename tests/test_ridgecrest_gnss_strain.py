import numpy as np
import pandas as pd

from ridgecrest_gnss_strain import finite_element_triangle_strain
from ridgecrest_vertical_los import sample_gnss_endpoint


def test_finite_element_recovers_known_affine_strain():
    # Coordinates are deliberately in UTM-km units.  Both Delaunay triangles
    # sample the same imposed affine displacement field exactly.
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    east = 10.0 + 2.0 * xy[:, 0] + 3.0 * xy[:, 1]
    north = -4.0 - 1.0 * xy[:, 0] + 4.0 * xy[:, 1]
    output = finite_element_triangle_strain(
        xy,
        east,
        north,
        np.full(4, 0.5),
        np.full(4, 0.5),
        max_edge_km=2.0,
        min_angle_deg=10.0,
        max_condition_number=10.0,
    )
    assert output["valid"].all()
    assert np.allclose(output["epsilon_xx_nstrain"], 2000.0)
    assert np.allclose(output["epsilon_yy_nstrain"], 4000.0)
    assert np.allclose(output["epsilon_xy_nstrain"], 1000.0)
    assert np.allclose(output["gamma_xy_nstrain"], 2000.0)
    assert np.allclose(output["dilatation_nstrain"], 6000.0)
    assert np.allclose(output["rotation_nrad"], -2000.0)


def test_mapped_rupture_masks_crossing_triangles():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    # A rupture trace through the square must mask both Delaunay triangles.
    rupture = np.array([[[0.5, -0.5], [0.5, 1.5]]])
    output = finite_element_triangle_strain(
        xy,
        np.zeros(4),
        np.zeros(4),
        np.ones(4),
        np.ones(4),
        rupture_segments_xy_km=rupture,
        rupture_buffer_km=0.05,
        rupture_mask_sample_spacing_km=0.05,
        max_edge_km=2.0,
        min_angle_deg=10.0,
        max_condition_number=10.0,
    )
    assert output["within_rupture_buffer"].all()
    assert not output["valid"].any()


def test_historical_daily_endpoint_interpolates_without_datetime_unit_mismatch():
    dates = pd.date_range("2019-06-10", periods=3, freq="D")
    history = pd.DataFrame(
        {
            "date": dates,
            "__east(m)": [0.0, 1.0, 2.0],
            "_north(m)": [0.0, 1.0, 2.0],
            "____up(m)": [0.0, 1.0, 2.0],
            "sig_e(m)": [0.001, 0.001, 0.001],
            "sig_n(m)": [0.001, 0.001, 0.001],
            "sig_u(m)": [0.002, 0.002, 0.002],
        }
    )
    value, sigma, method = sample_gnss_endpoint(
        history,
        "east",
        target=pd.Timestamp("2019-06-10T12:00:00"),
        event_times=(),
    )
    assert method == "within_regime_daily_linear"
    assert np.isclose(value, 0.5)
    assert sigma > 0.0
