"""Robust GNSS horizontal-strain utilities for the Ridgecrest network.

The functions in this module estimate a local affine EN displacement field with
Gaussian-distance and formal-uncertainty weights.  They are intended for
network-scale, off-fault strain diagnostics, not pixel-scale deformation or
strain directly on a discontinuous rupture.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence
import math

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import Delaunay, cKDTree


NANOSTRAIN_PER_MM_PER_KM = 1000.0


@dataclass(frozen=True)
class AffineVectorFit:
    """Local robust affine EN displacement fit evaluated at one target."""

    predicted_east_mm: float
    predicted_north_mm: float
    sigma_predicted_east_mm: float
    sigma_predicted_north_mm: float
    d_east_dx_mm_per_km: float
    d_east_dy_mm_per_km: float
    d_north_dx_mm_per_km: float
    d_north_dy_mm_per_km: float
    sigma_d_east_dx_mm_per_km: float
    sigma_d_east_dy_mm_per_km: float
    sigma_d_north_dx_mm_per_km: float
    sigma_d_north_dy_mm_per_km: float
    effective_station_count: float
    stations_with_weight: int
    condition_number: float


def to_utm11_km(
    longitude: np.ndarray | float,
    latitude: np.ndarray | float,
) -> np.ndarray:
    """Project WGS84 coordinates to UTM zone 11 in kilometres."""
    transformer = Transformer.from_crs(4326, 32611, always_xy=True)
    east_m, north_m = transformer.transform(longitude, latitude)
    return np.column_stack(
        [
            np.asarray(east_m, dtype=float).ravel() / 1000.0,
            np.asarray(north_m, dtype=float).ravel() / 1000.0,
        ]
    )


def from_utm11_km(east_km: np.ndarray, north_km: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return longitude, latitude from UTM zone 11 coordinates in kilometres."""
    transformer = Transformer.from_crs(32611, 4326, always_xy=True)
    longitude, latitude = transformer.transform(
        np.asarray(east_km, dtype=float) * 1000.0,
        np.asarray(north_km, dtype=float) * 1000.0,
    )
    return np.asarray(longitude, dtype=float), np.asarray(latitude, dtype=float)


def _robust_affine_component(
    station_xy_km: np.ndarray,
    values_mm: np.ndarray,
    sigma_mm: np.ndarray,
    target_xy_km: np.ndarray,
    *,
    bandwidth_km: float,
    huber_k: float = 1.345,
    iterations: int = 8,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """Fit one EN component using uncertainty- and distance-weighted RMLS."""
    xy = np.asarray(station_xy_km, dtype=float)
    values = np.asarray(values_mm, dtype=float)
    sigmas = np.maximum(np.asarray(sigma_mm, dtype=float), 0.25)
    target = np.asarray(target_xy_km, dtype=float)
    distance = np.sqrt(np.sum(np.square(xy - target[None, :]), axis=1))
    kernel = np.exp(-0.5 * np.square(distance / bandwidth_km))
    keep = kernel >= np.exp(-4.5)  # three bandwidths
    if int(keep.sum()) < 6:
        raise ValueError("Fewer than six materially weighted GNSS stations")
    dx = (xy[keep, 0] - target[0]) / bandwidth_km
    dy = (xy[keep, 1] - target[1]) / bandwidth_km
    design = np.column_stack([np.ones(len(dx)), dx, dy])
    y = values[keep]
    base_weight = kernel[keep] / np.square(sigmas[keep])
    robust_weight = np.ones_like(base_weight)
    beta = np.zeros(3, dtype=float)
    covariance = np.full((3, 3), np.nan)
    for _ in range(iterations):
        weight = base_weight * robust_weight
        normal = design.T @ (weight[:, None] * design)
        rhs = design.T @ (weight * y)
        covariance = np.linalg.pinv(normal)
        beta = covariance @ rhs
        residual = y - design @ beta
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        scale = max(float(scale), float(np.median(sigmas[keep])), 0.25)
        robust_weight = np.minimum(1.0, huber_k * scale / np.maximum(np.abs(residual), 1.0e-12))
    final_weight = base_weight * robust_weight
    normal = design.T @ (final_weight[:, None] * design)
    covariance = np.linalg.pinv(normal)
    residual = y - design @ beta
    dof = max(1, len(y) - design.shape[1])
    chi2_scale = float(np.sum(final_weight * np.square(residual)) / dof)
    covariance *= max(1.0, chi2_scale)
    condition = float(np.linalg.cond(normal))
    kernel_effective = float(
        np.square(np.sum(kernel[keep])) / np.sum(np.square(kernel[keep]))
    )
    return beta, covariance, kernel_effective, condition, int(keep.sum())


def fit_local_vector_affine(
    station_xy_km: np.ndarray,
    east_mm: np.ndarray,
    north_mm: np.ndarray,
    sigma_east_mm: np.ndarray,
    sigma_north_mm: np.ndarray,
    target_xy_km: np.ndarray,
    *,
    bandwidth_km: float,
) -> AffineVectorFit:
    """Fit robust local EN displacement and derive its spatial derivatives."""
    beta_e, cov_e, n_eff_e, cond_e, n_e = _robust_affine_component(
        station_xy_km,
        east_mm,
        sigma_east_mm,
        target_xy_km,
        bandwidth_km=bandwidth_km,
    )
    beta_n, cov_n, n_eff_n, cond_n, n_n = _robust_affine_component(
        station_xy_km,
        north_mm,
        sigma_north_mm,
        target_xy_km,
        bandwidth_km=bandwidth_km,
    )
    scale = bandwidth_km
    return AffineVectorFit(
        predicted_east_mm=float(beta_e[0]),
        predicted_north_mm=float(beta_n[0]),
        sigma_predicted_east_mm=float(np.sqrt(max(cov_e[0, 0], 0.0))),
        sigma_predicted_north_mm=float(np.sqrt(max(cov_n[0, 0], 0.0))),
        d_east_dx_mm_per_km=float(beta_e[1] / scale),
        d_east_dy_mm_per_km=float(beta_e[2] / scale),
        d_north_dx_mm_per_km=float(beta_n[1] / scale),
        d_north_dy_mm_per_km=float(beta_n[2] / scale),
        sigma_d_east_dx_mm_per_km=float(np.sqrt(max(cov_e[1, 1], 0.0)) / scale),
        sigma_d_east_dy_mm_per_km=float(np.sqrt(max(cov_e[2, 2], 0.0)) / scale),
        sigma_d_north_dx_mm_per_km=float(np.sqrt(max(cov_n[1, 1], 0.0)) / scale),
        sigma_d_north_dy_mm_per_km=float(np.sqrt(max(cov_n[2, 2], 0.0)) / scale),
        effective_station_count=min(n_eff_e, n_eff_n),
        stations_with_weight=min(n_e, n_n),
        condition_number=max(cond_e, cond_n),
    )


def leave_one_out_bandwidths(
    station_xy_km: np.ndarray,
    east_mm: np.ndarray,
    north_mm: np.ndarray,
    sigma_east_mm: np.ndarray,
    sigma_north_mm: np.ndarray,
    *,
    bandwidths_km: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate candidate RMLS bandwidths by leaving each GNSS station out."""
    xy = np.asarray(station_xy_km, dtype=float)
    east = np.asarray(east_mm, dtype=float)
    north = np.asarray(north_mm, dtype=float)
    sigma_e = np.asarray(sigma_east_mm, dtype=float)
    sigma_n = np.asarray(sigma_north_mm, dtype=float)
    summary_rows: list[dict[str, float | int]] = []
    prediction_rows: list[dict[str, float | int]] = []
    for bandwidth in bandwidths_km:
        east_residuals: list[float] = []
        north_residuals: list[float] = []
        horizontal_residuals: list[float] = []
        for holdout in range(len(xy)):
            keep = np.arange(len(xy)) != holdout
            try:
                fit = fit_local_vector_affine(
                    xy[keep],
                    east[keep],
                    north[keep],
                    sigma_e[keep],
                    sigma_n[keep],
                    xy[holdout],
                    bandwidth_km=float(bandwidth),
                )
            except ValueError:
                continue
            residual_e = east[holdout] - fit.predicted_east_mm
            residual_n = north[holdout] - fit.predicted_north_mm
            horizontal = float(np.hypot(residual_e, residual_n))
            east_residuals.append(float(residual_e))
            north_residuals.append(float(residual_n))
            horizontal_residuals.append(horizontal)
            prediction_rows.append(
                {
                    "bandwidth_km": float(bandwidth),
                    "holdout_index": holdout,
                    "observed_east_mm": float(east[holdout]),
                    "predicted_east_mm": fit.predicted_east_mm,
                    "observed_north_mm": float(north[holdout]),
                    "predicted_north_mm": fit.predicted_north_mm,
                    "residual_east_mm": float(residual_e),
                    "residual_north_mm": float(residual_n),
                    "horizontal_residual_mm": horizontal,
                    "condition_number": fit.condition_number,
                    "effective_station_count": fit.effective_station_count,
                }
            )
        if not horizontal_residuals:
            continue
        horizontal = np.asarray(horizontal_residuals)
        summary_rows.append(
            {
                "bandwidth_km": float(bandwidth),
                "holdout_count": len(horizontal),
                "loo_east_rmse_mm": float(np.sqrt(np.mean(np.square(east_residuals)))),
                "loo_north_rmse_mm": float(np.sqrt(np.mean(np.square(north_residuals)))),
                "loo_horizontal_rmse_mm": float(np.sqrt(np.mean(np.square(horizontal)))),
                "loo_horizontal_mean_mm": float(np.mean(horizontal)),
                "loo_horizontal_rmse_se_mm": float(
                    np.std(horizontal, ddof=1) / np.sqrt(len(horizontal))
                ) if len(horizontal) > 1 else float("inf"),
            }
        )
    scores = pd.DataFrame(summary_rows).sort_values("bandwidth_km").reset_index(drop=True)
    predictions = pd.DataFrame(prediction_rows)
    if scores.empty:
        raise ValueError("No candidate bandwidth produced leave-one-out predictions")
    return scores, predictions


def select_bandwidth_one_standard_error(scores: pd.DataFrame) -> pd.Series:
    """Select the smoothest bandwidth within one standard error of best RMSE."""
    best = scores.loc[scores["loo_horizontal_rmse_mm"].idxmin()]
    threshold = float(best["loo_horizontal_rmse_mm"] + best["loo_horizontal_rmse_se_mm"])
    eligible = scores.loc[scores["loo_horizontal_rmse_mm"] <= threshold]
    selected = eligible.sort_values("bandwidth_km", ascending=False).iloc[0].copy()
    return selected


def evaluate_strain_grid(
    station_xy_km: np.ndarray,
    east_mm: np.ndarray,
    north_mm: np.ndarray,
    sigma_east_mm: np.ndarray,
    sigma_north_mm: np.ndarray,
    grid_xy_km: np.ndarray,
    *,
    bandwidth_km: float,
    min_effective_stations: float = 5.0,
    max_condition_number: float = 100.0,
) -> pd.DataFrame:
    """Evaluate local affine strain components on a user-specified grid."""
    xy = np.asarray(grid_xy_km, dtype=float)
    rows: list[dict[str, float | bool]] = []
    for target in xy:
        try:
            fit = fit_local_vector_affine(
                station_xy_km,
                east_mm,
                north_mm,
                sigma_east_mm,
                sigma_north_mm,
                target,
                bandwidth_km=bandwidth_km,
            )
        except ValueError:
            rows.append(
                {
                    "east_km": float(target[0]),
                    "north_km": float(target[1]),
                    "valid_geometry": False,
                }
            )
            continue
        factor = NANOSTRAIN_PER_MM_PER_KM
        exx = fit.d_east_dx_mm_per_km * factor
        eyy = fit.d_north_dy_mm_per_km * factor
        exy = 0.5 * (
            fit.d_east_dy_mm_per_km + fit.d_north_dx_mm_per_km
        ) * factor
        gamma_xy = 2.0 * exy
        dilation = exx + eyy
        rotation = 0.5 * (
            fit.d_north_dx_mm_per_km - fit.d_east_dy_mm_per_km
        ) * factor
        mean_strain = 0.5 * dilation
        radius = math.hypot(0.5 * (exx - eyy), exy)
        principal_max = mean_strain + radius
        principal_min = mean_strain - radius
        principal_azimuth_deg = float(
            np.rad2deg(
                0.5 * np.arctan2(2.0 * exy, exx - eyy)
            )
        )
        sigma_exx = fit.sigma_d_east_dx_mm_per_km * factor
        sigma_eyy = fit.sigma_d_north_dy_mm_per_km * factor
        sigma_exy = 0.5 * math.hypot(
            fit.sigma_d_east_dy_mm_per_km,
            fit.sigma_d_north_dx_mm_per_km,
        ) * factor
        sigma_dilation = math.hypot(sigma_exx, sigma_eyy)
        sigma_rotation = sigma_exy
        valid_geometry = (
            fit.effective_station_count >= min_effective_stations
            and fit.condition_number <= max_condition_number
        )
        rows.append(
            {
                "east_km": float(target[0]),
                "north_km": float(target[1]),
                "valid_geometry": bool(valid_geometry),
                "effective_station_count": fit.effective_station_count,
                "stations_with_weight": fit.stations_with_weight,
                "condition_number": fit.condition_number,
                "predicted_east_mm": fit.predicted_east_mm,
                "predicted_north_mm": fit.predicted_north_mm,
                "sigma_predicted_east_mm": fit.sigma_predicted_east_mm,
                "sigma_predicted_north_mm": fit.sigma_predicted_north_mm,
                "epsilon_xx_nstrain": exx,
                "epsilon_yy_nstrain": eyy,
                "epsilon_xy_nstrain": exy,
                "gamma_xy_nstrain": gamma_xy,
                "dilatation_nstrain": dilation,
                "rotation_nrad": rotation,
                "principal_max_nstrain": principal_max,
                "principal_min_nstrain": principal_min,
                "principal_azimuth_deg": principal_azimuth_deg,
                "sigma_epsilon_xx_nstrain": sigma_exx,
                "sigma_epsilon_yy_nstrain": sigma_eyy,
                "sigma_epsilon_xy_nstrain": sigma_exy,
                "sigma_dilatation_nstrain": sigma_dilation,
                "sigma_rotation_nrad": sigma_rotation,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fault-aware finite-element strain
# ---------------------------------------------------------------------------
#
# A moving least-squares field is useful for a smooth, densely sampled
# velocity field.  It is not appropriate across the July 2019 surface rupture:
# its bandwidth necessarily mixes points on opposite sides of a displacement
# discontinuity.  The routines below therefore make a deliberately more
# conservative product.  A linear EN field is estimated *only inside each
# Delaunay triangle*, and triangles that intersect or approach the mapped
# rupture are withheld rather than interpolated through.  This resolves strain
# at GNSS-network scale; it is not a pixel-scale InSAR strain map.


def _geometry_lines(geometry: dict) -> list[np.ndarray]:
    """Return coordinate lines from a GeoJSON LineString/MultiLineString."""
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "LineString":
        lines = [coordinates]
    elif geometry_type == "MultiLineString":
        lines = coordinates
    else:
        return []
    output: list[np.ndarray] = []
    for line in lines:
        array = np.asarray(line, dtype=float)
        if array.ndim == 2 and array.shape[1] >= 2 and len(array) >= 2:
            output.append(array[:, :2])
    return output


def load_rupture_segments_utm(
    geojson_file: str | Path,
    *,
    certain_only: bool = True,
    bounds_km: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Read mapped rupture segments into UTM 11 coordinates in kilometres.

    Parameters
    ----------
    geojson_file
        CGS mapped-surface-rupture GeoJSON.
    certain_only
        Retain only features whose ``IdentityConfidence`` is ``"certain"``.
    bounds_km
        Optional ``(xmin, xmax, ymin, ymax)`` screen in UTM km.  It reduces
        later triangle-buffer calculations but does not otherwise alter lines.
    """
    payload = json.loads(Path(geojson_file).read_text(encoding="utf-8"))
    pieces: list[np.ndarray] = []
    for feature in payload.get("features", []):
        properties = feature.get("properties") or {}
        if certain_only and properties.get("IdentityConfidence") != "certain":
            continue
        for line in _geometry_lines(feature.get("geometry") or {}):
            xy = to_utm11_km(line[:, 0], line[:, 1])
            if len(xy) >= 2:
                pieces.append(np.stack([xy[:-1], xy[1:]], axis=1))
    if not pieces:
        return np.empty((0, 2, 2), dtype=float)
    segments = np.concatenate(pieces, axis=0).astype(float, copy=False)
    if bounds_km is None:
        return segments
    xmin, xmax, ymin, ymax = (float(value) for value in bounds_km)
    segment_xmin = np.minimum(segments[:, 0, 0], segments[:, 1, 0])
    segment_xmax = np.maximum(segments[:, 0, 0], segments[:, 1, 0])
    segment_ymin = np.minimum(segments[:, 0, 1], segments[:, 1, 1])
    segment_ymax = np.maximum(segments[:, 0, 1], segments[:, 1, 1])
    keep = (
        (segment_xmax >= xmin)
        & (segment_xmin <= xmax)
        & (segment_ymax >= ymin)
        & (segment_ymin <= ymax)
    )
    return segments[keep]


def _cross2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return _cross2d(b - a, c - a)


def _point_on_segment(a: np.ndarray, b: np.ndarray, point: np.ndarray) -> bool:
    return bool(
        min(a[0], b[0]) - 1.0e-10 <= point[0] <= max(a[0], b[0]) + 1.0e-10
        and min(a[1], b[1]) - 1.0e-10 <= point[1] <= max(a[1], b[1]) + 1.0e-10
    )


def _segments_intersect(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> bool:
    """Return whether two closed two-dimensional line segments intersect."""
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    if (o1 > 0.0) != (o2 > 0.0) and (o3 > 0.0) != (o4 > 0.0):
        return True
    return bool(
        (abs(o1) <= 1.0e-10 and _point_on_segment(a, b, c))
        or (abs(o2) <= 1.0e-10 and _point_on_segment(a, b, d))
        or (abs(o3) <= 1.0e-10 and _point_on_segment(c, d, a))
        or (abs(o4) <= 1.0e-10 and _point_on_segment(c, d, b))
    )


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    direction = b - a
    denominator = float(direction @ direction)
    if denominator <= 0.0:
        return float(np.linalg.norm(point - a))
    fraction = float(np.clip(((point - a) @ direction) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + fraction * direction)))


def _segment_distance(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> float:
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(
        _point_segment_distance(a, c, d),
        _point_segment_distance(b, c, d),
        _point_segment_distance(c, a, b),
        _point_segment_distance(d, a, b),
    )


def _point_in_triangle(point: np.ndarray, triangle: np.ndarray) -> bool:
    orientation = [
        _orientation(triangle[index], triangle[(index + 1) % 3], point)
        for index in range(3)
    ]
    return bool(
        all(value >= -1.0e-10 for value in orientation)
        or all(value <= 1.0e-10 for value in orientation)
    )


def _segment_triangle_distance(
    segment: np.ndarray,
    triangle: np.ndarray,
) -> float:
    a, b = segment
    if _point_in_triangle(a, triangle) or _point_in_triangle(b, triangle):
        return 0.0
    return min(
        _segment_distance(
            triangle[index], triangle[(index + 1) % 3], a, b
        )
        for index in range(3)
    )


def _triangle_metrics(points: np.ndarray) -> dict[str, float]:
    """Return finite-element geometry metrics for three UTM-km vertices."""
    vertices = np.asarray(points, dtype=float)
    if vertices.shape != (3, 2):
        raise ValueError("Triangle vertices must have shape (3, 2)")
    edge_vectors = np.roll(vertices, -1, axis=0) - vertices
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    twice_area = abs(_cross2d(vertices[1] - vertices[0], vertices[2] - vertices[0]))
    area = 0.5 * twice_area
    angles: list[float] = []
    for index in range(3):
        vector_a = vertices[(index + 1) % 3] - vertices[index]
        vector_b = vertices[(index + 2) % 3] - vertices[index]
        denominator = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
        cosine = 1.0 if denominator <= 0.0 else float((vector_a @ vector_b) / denominator)
        angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
    jacobian = np.vstack([vertices[1] - vertices[0], vertices[2] - vertices[0]])
    return {
        "area_km2": float(area),
        "longest_edge_km": float(np.max(edge_lengths)),
        "shortest_edge_km": float(np.min(edge_lengths)),
        "minimum_angle_deg": float(np.min(angles)),
        "geometry_condition_number": float(np.linalg.cond(jacobian)),
    }


def _candidate_rupture_segments(
    rupture_segments_xy_km: np.ndarray,
    triangle: np.ndarray,
    buffer_km: float,
) -> np.ndarray:
    """Screen line segments by a triangle's axis-aligned buffered extent."""
    if len(rupture_segments_xy_km) == 0:
        return rupture_segments_xy_km
    xmin, ymin = np.min(triangle, axis=0) - float(buffer_km)
    xmax, ymax = np.max(triangle, axis=0) + float(buffer_km)
    segments = np.asarray(rupture_segments_xy_km, dtype=float)
    segment_xmin = np.minimum(segments[:, 0, 0], segments[:, 1, 0])
    segment_xmax = np.maximum(segments[:, 0, 0], segments[:, 1, 0])
    segment_ymin = np.minimum(segments[:, 0, 1], segments[:, 1, 1])
    segment_ymax = np.maximum(segments[:, 0, 1], segments[:, 1, 1])
    overlap = (
        (segment_xmax >= xmin)
        & (segment_xmin <= xmax)
        & (segment_ymax >= ymin)
        & (segment_ymin <= ymax)
    )
    return segments[overlap]


def _rupture_distance_to_triangle(
    rupture_segments_xy_km: np.ndarray,
    triangle: np.ndarray,
    buffer_km: float,
) -> float:
    """Return closest mapped-rupture distance after a fast bounding-box screen."""
    candidates = _candidate_rupture_segments(
        rupture_segments_xy_km, triangle, buffer_km
    )
    if len(candidates) == 0:
        return float("inf")
    return float(min(_segment_triangle_distance(segment, triangle) for segment in candidates))


def _rupture_cell_index(
    rupture_segments_xy_km: np.ndarray,
    *,
    cell_size_km: float = 2.0,
) -> dict[tuple[int, int], list[int]]:
    """Index short mapped-rupture segments in a fixed UTM-km cell grid."""
    cells: dict[tuple[int, int], list[int]] = {}
    midpoints = np.mean(rupture_segments_xy_km, axis=1)
    for index, midpoint in enumerate(midpoints):
        key = tuple(np.floor(midpoint / float(cell_size_km)).astype(int))
        cells.setdefault(key, []).append(index)
    return cells


def _cell_index_candidates(
    rupture_segments_xy_km: np.ndarray,
    cell_index: dict[tuple[int, int], list[int]],
    triangle: np.ndarray,
    *,
    buffer_km: float,
    maximum_segment_half_length_km: float,
    cell_size_km: float = 2.0,
) -> np.ndarray:
    """Return all segments that could meet a triangle's buffered extent."""
    extra = float(buffer_km) + float(maximum_segment_half_length_km)
    lower = np.min(triangle, axis=0) - extra
    upper = np.max(triangle, axis=0) + extra
    lower_cell = np.floor(lower / float(cell_size_km)).astype(int) - 1
    upper_cell = np.floor(upper / float(cell_size_km)).astype(int) + 1
    candidate_indices: list[int] = []
    for east_cell in range(int(lower_cell[0]), int(upper_cell[0]) + 1):
        for north_cell in range(int(lower_cell[1]), int(upper_cell[1]) + 1):
            candidate_indices.extend(cell_index.get((east_cell, north_cell), ()))
    if not candidate_indices:
        return np.empty((0, 2, 2), dtype=float)
    return rupture_segments_xy_km[np.asarray(candidate_indices, dtype=int)]


def _triangle_sample_points(
    triangle: np.ndarray,
    *,
    spacing_km: float,
) -> np.ndarray:
    """Cover a triangle with interior-grid and edge samples for a safety mask."""
    vertices = np.asarray(triangle, dtype=float)
    if spacing_km <= 0.0:
        raise ValueError("spacing_km must be positive")
    lower = np.min(vertices, axis=0)
    upper = np.max(vertices, axis=0)
    east = np.arange(lower[0], upper[0] + spacing_km, spacing_km)
    north = np.arange(lower[1], upper[1] + spacing_km, spacing_km)
    grid_east, grid_north = np.meshgrid(east, north, indexing="xy")
    candidates = np.column_stack([grid_east.ravel(), grid_north.ravel()])
    inside = np.asarray([_point_in_triangle(point, vertices) for point in candidates])
    points = [candidates[inside], vertices]
    for index in range(3):
        start = vertices[index]
        stop = vertices[(index + 1) % 3]
        count = max(2, int(np.ceil(np.linalg.norm(stop - start) / spacing_km)) + 1)
        points.append(np.linspace(start, stop, count))
    return np.vstack(points)


def _conservative_rupture_distance_bounds(
    rupture_midpoint_tree: cKDTree | None,
    triangle: np.ndarray,
    *,
    max_segment_half_length_km: float,
    sample_spacing_km: float,
) -> tuple[float, float]:
    """Bound rupture distance using a dense triangle sample safety mask.

    The lower bound is intentionally conservative: a triangle is accepted only
    if this bound exceeds the requested rupture buffer.  Therefore any
    ambiguity caused by the sample spacing is masked, never presented as a
    resolved off-fault strain estimate.
    """
    if rupture_midpoint_tree is None:
        return float("inf"), float("inf")
    samples = _triangle_sample_points(triangle, spacing_km=sample_spacing_km)
    nearest_midpoint_distance = rupture_midpoint_tree.query(samples, k=1)[0]
    upper_bound = float(np.min(nearest_midpoint_distance))
    # A square grid has worst-case in-plane cover sqrt(2)*spacing.  The mapped
    # segments are short but we also subtract their maximum half-length.
    lower_bound = max(
        0.0,
        upper_bound
        - math.sqrt(2.0) * float(sample_spacing_km)
        - float(max_segment_half_length_km),
    )
    return float(lower_bound), upper_bound


def finite_element_triangle_strain(
    station_xy_km: np.ndarray,
    east_mm: np.ndarray,
    north_mm: np.ndarray,
    sigma_east_mm: np.ndarray,
    sigma_north_mm: np.ndarray,
    *,
    station_ids: Sequence[str] | None = None,
    rupture_segments_xy_km: np.ndarray | None = None,
    rupture_buffer_km: float = 5.0,
    rupture_mask_sample_spacing_km: float = 0.25,
    max_edge_km: float = 60.0,
    min_angle_deg: float = 15.0,
    max_condition_number: float = 10.0,
) -> pd.DataFrame:
    """Estimate strain in quality-screened GNSS Delaunay triangles.

    A triangle is considered usable only when it has adequate geometry and its
    complete area lies farther than ``rupture_buffer_km`` from a mapped rupture
    segment.  Uncertainties are propagated from the supplied GNSS endpoint
    uncertainties; because each linear triangle exactly fits three stations,
    they do *not* represent unresolved within-triangle model error.

    The returned components are in nanostrain (and rotation in nanoradians).
    Invalid triangles are retained in the table with ``valid=False`` so that a
    plotting notebook can show the masked information rather than silently
    filling it.
    """
    xy = np.asarray(station_xy_km, dtype=float)
    east = np.asarray(east_mm, dtype=float)
    north = np.asarray(north_mm, dtype=float)
    sigma_east = np.maximum(np.asarray(sigma_east_mm, dtype=float), 0.25)
    sigma_north = np.maximum(np.asarray(sigma_north_mm, dtype=float), 0.25)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("station_xy_km must have shape (station, 2)")
    if len(xy) < 3:
        raise ValueError("At least three GNSS stations are required")
    if not all(len(array) == len(xy) for array in (east, north, sigma_east, sigma_north)):
        raise ValueError("GNSS coordinate/value arrays must have matching lengths")
    if not np.all(np.isfinite(xy)):
        raise ValueError("GNSS station coordinates contain non-finite values")
    if station_ids is None:
        labels = np.asarray([f"station_{index}" for index in range(len(xy))], dtype=object)
    else:
        labels = np.asarray(station_ids, dtype=object)
        if len(labels) != len(xy):
            raise ValueError("station_ids must match station_xy_km")
    rupture_segments = (
        np.empty((0, 2, 2), dtype=float)
        if rupture_segments_xy_km is None
        else np.asarray(rupture_segments_xy_km, dtype=float)
    )
    if rupture_segments.ndim != 3 or rupture_segments.shape[1:] != (2, 2):
        raise ValueError("rupture_segments_xy_km must have shape (segment, 2, 2)")

    # The CGS trace consists of many metre-scale segments.  A KD-tree on their
    # midpoints plus a dense triangle sampling mask is substantially faster
    # than computing every segment-to-triangle distance.  Crucially, its lower
    # distance bound is conservative, so a borderline triangle is withheld.
    if len(rupture_segments):
        segment_half_lengths = 0.5 * np.linalg.norm(
            rupture_segments[:, 1] - rupture_segments[:, 0], axis=1
        )
        max_segment_half_length = float(np.max(segment_half_lengths))
        rupture_midpoint_tree: cKDTree | None = cKDTree(
            np.mean(rupture_segments, axis=1)
        )
    else:
        max_segment_half_length = 0.0
        rupture_midpoint_tree = None

    simplices = Delaunay(xy).simplices
    rows: list[dict[str, float | int | bool | str]] = []
    for triangle_index, indices in enumerate(simplices):
        vertices = xy[indices]
        metrics = _triangle_metrics(vertices)
        centroid = np.mean(vertices, axis=0)
        rupture_lower_bound, rupture_upper_bound = _conservative_rupture_distance_bounds(
            rupture_midpoint_tree,
            vertices,
            max_segment_half_length_km=max_segment_half_length,
            sample_spacing_km=rupture_mask_sample_spacing_km,
        )
        geometry_ok = bool(
            metrics["longest_edge_km"] <= float(max_edge_km)
            and metrics["minimum_angle_deg"] >= float(min_angle_deg)
            and metrics["geometry_condition_number"] <= float(max_condition_number)
        )
        outside_rupture_buffer = bool(
            rupture_lower_bound > float(rupture_buffer_km)
        )
        valid = bool(geometry_ok and outside_rupture_buffer)
        row: dict[str, float | int | bool | str] = {
            "triangle_index": int(triangle_index),
            "station_index_1": int(indices[0]),
            "station_index_2": int(indices[1]),
            "station_index_3": int(indices[2]),
            "station_1": str(labels[indices[0]]),
            "station_2": str(labels[indices[1]]),
            "station_3": str(labels[indices[2]]),
            "centroid_east_km": float(centroid[0]),
            "centroid_north_km": float(centroid[1]),
            "rupture_distance_lower_bound_km": float(rupture_lower_bound),
            "rupture_distance_upper_bound_km": float(rupture_upper_bound),
            "rupture_mask_sample_spacing_km": float(rupture_mask_sample_spacing_km),
            "within_rupture_buffer": bool(not outside_rupture_buffer),
            "passes_geometry": geometry_ok,
            "valid": valid,
            **metrics,
        }
        if metrics["area_km2"] <= 0.0:
            rows.append(row)
            continue
        # Coordinates centred on the triangle centroid give an intercept equal
        # to the centroid displacement and preserve derivatives in mm/km.
        design = np.column_stack([np.ones(3), vertices - centroid])
        inverse_design = np.linalg.inv(design)
        beta_east = inverse_design @ east[indices]
        beta_north = inverse_design @ north[indices]
        covariance_east = inverse_design @ np.diag(np.square(sigma_east[indices])) @ inverse_design.T
        covariance_north = inverse_design @ np.diag(np.square(sigma_north[indices])) @ inverse_design.T
        factor = NANOSTRAIN_PER_MM_PER_KM
        epsilon_xx = beta_east[1] * factor
        epsilon_yy = beta_north[2] * factor
        epsilon_xy = 0.5 * (beta_east[2] + beta_north[1]) * factor
        gamma_xy = 2.0 * epsilon_xy
        dilatation = epsilon_xx + epsilon_yy
        rotation = 0.5 * (beta_north[1] - beta_east[2]) * factor
        mean_strain = 0.5 * dilatation
        radius = math.hypot(0.5 * (epsilon_xx - epsilon_yy), epsilon_xy)
        principal_max = mean_strain + radius
        principal_min = mean_strain - radius
        principal_azimuth = float(
            np.rad2deg(0.5 * np.arctan2(2.0 * epsilon_xy, epsilon_xx - epsilon_yy))
        )
        sigma_epsilon_xx = math.sqrt(max(covariance_east[1, 1], 0.0)) * factor
        sigma_epsilon_yy = math.sqrt(max(covariance_north[2, 2], 0.0)) * factor
        sigma_epsilon_xy = 0.5 * math.sqrt(
            max(covariance_east[2, 2], 0.0) + max(covariance_north[1, 1], 0.0)
        ) * factor
        sigma_dilatation = math.hypot(sigma_epsilon_xx, sigma_epsilon_yy)
        sigma_rotation = sigma_epsilon_xy
        row.update(
            {
                "predicted_east_mm": float(beta_east[0]),
                "predicted_north_mm": float(beta_north[0]),
                "sigma_predicted_east_mm": float(math.sqrt(max(covariance_east[0, 0], 0.0))),
                "sigma_predicted_north_mm": float(math.sqrt(max(covariance_north[0, 0], 0.0))),
                "epsilon_xx_nstrain": float(epsilon_xx),
                "epsilon_yy_nstrain": float(epsilon_yy),
                "epsilon_xy_nstrain": float(epsilon_xy),
                "gamma_xy_nstrain": float(gamma_xy),
                "dilatation_nstrain": float(dilatation),
                "rotation_nrad": float(rotation),
                "principal_max_nstrain": float(principal_max),
                "principal_min_nstrain": float(principal_min),
                "principal_azimuth_deg": principal_azimuth,
                "sigma_epsilon_xx_nstrain": float(sigma_epsilon_xx),
                "sigma_epsilon_yy_nstrain": float(sigma_epsilon_yy),
                "sigma_epsilon_xy_nstrain": float(sigma_epsilon_xy),
                "sigma_dilatation_nstrain": float(sigma_dilatation),
                "sigma_rotation_nrad": float(sigma_rotation),
                "formal_uncertainty_only": True,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("triangle_index").reset_index(drop=True)
