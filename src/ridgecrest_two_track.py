"""Two-track horizontal-displacement and strain sensitivity utilities.

This module supports an explicitly labelled sensitivity workflow:

``U_hat -> l_U U_hat -> corrected LOS -> two-track E/N -> off-fault strain``.

It does not establish that a GNSS-interpolated vertical field is spatially
resolved.  Callers must retain that validation status and must not relabel a
forced spatial vertical predictor as an independently observed field.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

NANOSTRAIN_PER_MM_PER_KM = 1000.0


@dataclass(frozen=True)
class HorizontalSolution:
    """Per-pixel two-track horizontal solution and propagated covariance."""

    east_mm: np.ndarray
    north_mm: np.ndarray
    sigma_east_mm: np.ndarray
    sigma_north_mm: np.ndarray
    covariance_east_north_mm2: np.ndarray
    condition_number: np.ndarray
    valid: np.ndarray


def from_utm11_km(
    east_km: np.ndarray | float,
    north_km: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform UTM zone 11 coordinates in km to longitude and latitude."""
    transformer = Transformer.from_crs(32611, 4326, always_xy=True)
    longitude, latitude = transformer.transform(
        np.asarray(east_km, dtype=float) * 1000.0,
        np.asarray(north_km, dtype=float) * 1000.0,
    )
    return np.asarray(longitude, dtype=float), np.asarray(latitude, dtype=float)


def to_utm11_km(
    longitude: np.ndarray | float,
    latitude: np.ndarray | float,
) -> np.ndarray:
    """Transform WGS84 longitude/latitude to UTM zone 11 coordinates in km."""
    transformer = Transformer.from_crs(4326, 32611, always_xy=True)
    east_m, north_m = transformer.transform(longitude, latitude)
    return np.column_stack(
        [
            np.asarray(east_m, dtype=float).ravel() / 1000.0,
            np.asarray(north_m, dtype=float).ravel() / 1000.0,
        ]
    )


def common_utm11_grid(
    latitude_axes: Sequence[np.ndarray],
    longitude_axes: Sequence[np.ndarray],
    *,
    spacing_km: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a common UTM-km grid over the geographic intersection of tracks."""
    if len(latitude_axes) != len(longitude_axes) or not latitude_axes:
        raise ValueError("At least one matched latitude/longitude axis is required")
    latitude_min = max(float(np.nanmin(axis)) for axis in latitude_axes)
    latitude_max = min(float(np.nanmax(axis)) for axis in latitude_axes)
    longitude_min = max(float(np.nanmin(axis)) for axis in longitude_axes)
    longitude_max = min(float(np.nanmax(axis)) for axis in longitude_axes)
    if latitude_min >= latitude_max or longitude_min >= longitude_max:
        raise ValueError("Input grids have no geographic overlap")
    corners_lon = np.array([longitude_min, longitude_min, longitude_max, longitude_max])
    corners_lat = np.array([latitude_min, latitude_max, latitude_min, latitude_max])
    corners = to_utm11_km(corners_lon, corners_lat)
    east_axis = np.arange(
        math.ceil(float(np.min(corners[:, 0])) / spacing_km) * spacing_km,
        math.floor(float(np.max(corners[:, 0])) / spacing_km) * spacing_km + 0.5 * spacing_km,
        spacing_km,
    )
    north_axis = np.arange(
        math.ceil(float(np.min(corners[:, 1])) / spacing_km) * spacing_km,
        math.floor(float(np.max(corners[:, 1])) / spacing_km) * spacing_km + 0.5 * spacing_km,
        spacing_km,
    )
    east_grid, north_grid = np.meshgrid(east_axis, north_axis, indexing="xy")
    longitude_grid, latitude_grid = from_utm11_km(east_grid, north_grid)
    inside_intersection = (
        (latitude_grid >= latitude_min)
        & (latitude_grid <= latitude_max)
        & (longitude_grid >= longitude_min)
        & (longitude_grid <= longitude_max)
    )
    return east_grid, north_grid, latitude_grid, longitude_grid, inside_intersection


def _ascending_axis_and_array(axis: np.ndarray, array: np.ndarray, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a monotonically ascending axis and corresponding data order."""
    values = np.asarray(axis, dtype=float)
    output = np.asarray(array, dtype=float)
    if len(values) < 2:
        raise ValueError("A rectilinear source axis needs at least two coordinates")
    if values[0] < values[-1]:
        return values, output
    return values[::-1], np.flip(output, axis=dimension)


def masked_bilinear_resample(
    latitude: np.ndarray,
    longitude: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
    *,
    minimum_weight: float = 0.999,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a masked rectilinear field without silently filling invalid data.

    The returned support weight is bilinear support from the source mask.  A
    target value is retained only when all material source support is valid;
    this deliberately avoids blending a coherent pixel with a masked pixel.
    """
    source_values = np.asarray(values, dtype=float)
    source_valid = np.asarray(valid, dtype=bool) & np.isfinite(source_values)
    if source_values.shape != source_valid.shape:
        raise ValueError("values and valid must share a grid shape")
    lat_axis, source_values = _ascending_axis_and_array(latitude, source_values, 0)
    _, source_valid_float = _ascending_axis_and_array(
        latitude, source_valid.astype(float), 0
    )
    lon_axis, source_values = _ascending_axis_and_array(longitude, source_values, 1)
    _, source_valid_float = _ascending_axis_and_array(
        longitude, source_valid_float, 1
    )
    numerator = np.where(source_valid_float > 0.5, source_values, 0.0)
    points = np.column_stack(
        [np.asarray(target_latitude, dtype=float).ravel(), np.asarray(target_longitude, dtype=float).ravel()]
    )
    num_interpolator = RegularGridInterpolator(
        (lat_axis, lon_axis), numerator, bounds_error=False, fill_value=np.nan
    )
    support_interpolator = RegularGridInterpolator(
        (lat_axis, lon_axis), source_valid_float, bounds_error=False, fill_value=0.0
    )
    sampled_numerator = num_interpolator(points)
    support = support_interpolator(points)
    sampled = np.divide(
        sampled_numerator,
        support,
        out=np.full_like(sampled_numerator, np.nan, dtype=float),
        where=support >= float(minimum_weight),
    )
    return sampled.reshape(np.shape(target_latitude)), support.reshape(np.shape(target_latitude))


def normalize_look_vectors(
    look_east: np.ndarray,
    look_north: np.ndarray,
    look_up: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unit-normalized sampled LOS vectors with invalid cells as NaN."""
    e = np.asarray(look_east, dtype=float)
    n = np.asarray(look_north, dtype=float)
    u = np.asarray(look_up, dtype=float)
    norm = np.sqrt(np.square(e) + np.square(n) + np.square(u))
    valid = np.isfinite(norm) & (norm > 0.5)
    return tuple(
        np.divide(component, norm, out=np.full_like(component, np.nan), where=valid)
        for component in (e, n, u)
    )


def correct_vertical_los_on_grid(
    referenced_los_mm: np.ndarray,
    look_up: np.ndarray,
    vertical_mm: np.ndarray,
    vertical_sigma_mm: np.ndarray,
    reference_mask: np.ndarray | None = None,
    *,
    reference_value_mm: float | None = None,
    reference_sigma_mm: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Project predicted vertical displacement to LOS and reference it once.

    The reference can be evaluated on the same analysis grid via
    ``reference_mask`` or supplied from the native-resolution reference disk.
    The latter avoids weakening a valid native spatial reference merely because
    a coarse common grid contains too few cells in that disk.
    """
    los = np.asarray(referenced_los_mm, dtype=float)
    up = np.asarray(look_up, dtype=float)
    vertical = np.asarray(vertical_mm, dtype=float)
    vertical_sigma = np.asarray(vertical_sigma_mm, dtype=float)
    raw_vertical_los = up * vertical
    raw_sigma = np.abs(up) * vertical_sigma
    supplied = (reference_value_mm is not None) or (reference_sigma_mm is not None)
    if supplied:
        if reference_value_mm is None or reference_sigma_mm is None:
            raise ValueError("Supply both native-reference value and uncertainty, or neither")
        reference_value = float(reference_value_mm)
        reference_sigma = float(reference_sigma_mm)
    else:
        if reference_mask is None:
            raise ValueError("Provide a reference mask or native-reference summary")
        use_reference = (
            np.asarray(reference_mask, dtype=bool)
            & np.isfinite(raw_vertical_los)
            & np.isfinite(raw_sigma)
        )
        if int(use_reference.sum()) < 4:
            raise ValueError("Too few common-grid samples in the vertical reference disk")
        reference_value = float(np.nanmedian(raw_vertical_los[use_reference]))
        reference_sigma = float(np.nanmedian(raw_sigma[use_reference]))
    vertical_los = raw_vertical_los - reference_value
    vertical_los_sigma = np.sqrt(np.square(raw_sigma) + reference_sigma**2)
    corrected = los - vertical_los
    invalid = ~(
        np.isfinite(los)
        & np.isfinite(vertical_los)
        & np.isfinite(vertical_los_sigma)
    )
    corrected[invalid] = np.nan
    vertical_los[invalid] = np.nan
    vertical_los_sigma[invalid] = np.nan
    return corrected, vertical_los, vertical_los_sigma, reference_value, reference_sigma


def solve_two_track_horizontal(
    ascending_los_mm: np.ndarray,
    descending_los_mm: np.ndarray,
    ascending_east: np.ndarray,
    ascending_north: np.ndarray,
    descending_east: np.ndarray,
    descending_north: np.ndarray,
    ascending_sigma_mm: np.ndarray,
    descending_sigma_mm: np.ndarray,
    *,
    vertical_los_sigma_ascending_mm: np.ndarray | None = None,
    vertical_los_sigma_descending_mm: np.ndarray | None = None,
    vertical_correlation: float = 1.0,
    max_condition_number: float = 8.0,
) -> HorizontalSolution:
    """Solve E/N from two corrected LOS observations with covariance propagation.

    ``vertical_correlation=1`` is intentionally conservative for a shared
    GNSS-derived vertical field.  It avoids treating the two vertical
    corrections as independent merely because they are applied to separate
    tracks.  ``ascending_sigma_mm`` and ``descending_sigma_mm`` describe the
    InSAR terms before GNSS-vertical removal.  When vertical uncertainties are
    supplied, their variances are added to the corresponding LOS diagonal and
    their shared covariance is added off diagonal.
    """
    da = np.asarray(ascending_los_mm, dtype=float)
    dd = np.asarray(descending_los_mm, dtype=float)
    ae = np.asarray(ascending_east, dtype=float)
    an = np.asarray(ascending_north, dtype=float)
    de = np.asarray(descending_east, dtype=float)
    dn = np.asarray(descending_north, dtype=float)
    sa = np.asarray(ascending_sigma_mm, dtype=float)
    sd = np.asarray(descending_sigma_mm, dtype=float)
    arrays = (da, dd, ae, an, de, dn, sa, sd)
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("All two-track arrays must have the same shape")
    determinant = ae * dn - an * de
    trace = np.square(ae) + np.square(an) + np.square(de) + np.square(dn)
    discriminant = np.maximum(np.square(trace) - 4.0 * np.square(determinant), 0.0)
    singular_max = 0.5 * (trace + np.sqrt(discriminant))
    singular_min = 0.5 * (trace - np.sqrt(discriminant))
    condition = np.sqrt(
        np.divide(
            singular_max,
            singular_min,
            out=np.full_like(singular_max, np.inf),
            where=singular_min > 0.0,
        )
    )
    valid = (
        np.isfinite(da)
        & np.isfinite(dd)
        & np.isfinite(ae)
        & np.isfinite(an)
        & np.isfinite(de)
        & np.isfinite(dn)
        & np.isfinite(sa)
        & np.isfinite(sd)
        & (np.abs(determinant) > 1.0e-6)
        & (condition <= float(max_condition_number))
    )
    east = np.full_like(da, np.nan, dtype=float)
    north = np.full_like(da, np.nan, dtype=float)
    east[valid] = (dn[valid] * da[valid] - an[valid] * dd[valid]) / determinant[valid]
    north[valid] = (-de[valid] * da[valid] + ae[valid] * dd[valid]) / determinant[valid]

    vertical_pair_supplied = (
        vertical_los_sigma_ascending_mm is not None,
        vertical_los_sigma_descending_mm is not None,
    )
    if any(vertical_pair_supplied) and not all(vertical_pair_supplied):
        raise ValueError(
            "Supply both ascending and descending vertical uncertainties, "
            "or neither"
        )
    variance_ascending_los = np.square(sa)
    variance_descending_los = np.square(sd)
    covariance_los = np.zeros_like(da, dtype=float)
    if all(vertical_pair_supplied):
        sva = np.asarray(vertical_los_sigma_ascending_mm, dtype=float)
        svd = np.asarray(vertical_los_sigma_descending_mm, dtype=float)
        if sva.shape != da.shape or svd.shape != da.shape:
            raise ValueError("Vertical uncertainty arrays must match LOS arrays")
        correlation = float(vertical_correlation)
        if not -1.0 <= correlation <= 1.0:
            raise ValueError("vertical_correlation must lie in [-1, 1]")
        valid &= np.isfinite(sva) & np.isfinite(svd)
        variance_ascending_los = variance_ascending_los + np.square(sva)
        variance_descending_los = variance_descending_los + np.square(svd)
        covariance_los = correlation * sva * svd
    inverse_ae = np.full_like(da, np.nan, dtype=float)
    inverse_ad = np.full_like(da, np.nan, dtype=float)
    inverse_ne = np.full_like(da, np.nan, dtype=float)
    inverse_nd = np.full_like(da, np.nan, dtype=float)
    inverse_ae[valid] = dn[valid] / determinant[valid]
    inverse_ad[valid] = -an[valid] / determinant[valid]
    inverse_ne[valid] = -de[valid] / determinant[valid]
    inverse_nd[valid] = ae[valid] / determinant[valid]
    variance_east = (
        np.square(inverse_ae) * variance_ascending_los
        + np.square(inverse_ad) * variance_descending_los
        + 2.0 * inverse_ae * inverse_ad * covariance_los
    )
    variance_north = (
        np.square(inverse_ne) * variance_ascending_los
        + np.square(inverse_nd) * variance_descending_los
        + 2.0 * inverse_ne * inverse_nd * covariance_los
    )
    covariance_east_north = (
        inverse_ae * inverse_ne * variance_ascending_los
        + inverse_ad * inverse_nd * variance_descending_los
        + (inverse_ae * inverse_nd + inverse_ad * inverse_ne) * covariance_los
    )
    sigma_east = np.sqrt(np.maximum(variance_east, 0.0))
    sigma_north = np.sqrt(np.maximum(variance_north, 0.0))
    for array in (east, north, sigma_east, sigma_north, covariance_east_north):
        array[~valid] = np.nan
    # Preserve the distinction between a poorly conditioned usable solution and
    # an invalid cell with no paired LOS support.  Returning infinity for the
    # latter would corrupt map/global condition-number summaries.
    condition = condition.astype(float, copy=True)
    condition[~valid] = np.nan
    return HorizontalSolution(
        east_mm=east,
        north_mm=north,
        sigma_east_mm=sigma_east,
        sigma_north_mm=sigma_north,
        covariance_east_north_mm2=covariance_east_north,
        condition_number=condition,
        valid=valid,
    )


def rupture_point_distance_lower_bound_km(
    target_xy_km: np.ndarray,
    rupture_segments_xy_km: np.ndarray,
) -> np.ndarray:
    """Return conservative lower bounds on distance from targets to rupture."""
    targets = np.asarray(target_xy_km, dtype=float)
    segments = np.asarray(rupture_segments_xy_km, dtype=float)
    if segments.ndim != 3 or segments.shape[1:] != (2, 2):
        raise ValueError("rupture_segments_xy_km must have shape (segment, 2, 2)")
    if len(segments) == 0:
        return np.full(len(targets), np.inf, dtype=float)
    midpoint = np.mean(segments, axis=1)
    max_half_length = float(np.max(0.5 * np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)))
    distance = cKDTree(midpoint).query(targets, k=1)[0]
    return np.maximum(distance - max_half_length, 0.0)


def _robust_component_affine(
    sample_xy_km: np.ndarray,
    values_mm: np.ndarray,
    sigma_mm: np.ndarray,
    target_xy_km: np.ndarray,
    *,
    support_radius_km: float,
    bandwidth_km: float,
    min_samples: int,
    huber_k: float = 1.345,
    iterations: int = 6,
) -> tuple[np.ndarray, np.ndarray, float, int, float] | None:
    distance = np.linalg.norm(sample_xy_km - target_xy_km, axis=1)
    keep = distance <= float(support_radius_km)
    if int(keep.sum()) < int(min_samples):
        return None
    xy = sample_xy_km[keep]
    values = values_mm[keep]
    sigma = np.maximum(sigma_mm[keep], 1.0)
    local_distance = distance[keep]
    design = np.column_stack(
        [
            np.ones(len(xy)),
            (xy[:, 0] - target_xy_km[0]) / bandwidth_km,
            (xy[:, 1] - target_xy_km[1]) / bandwidth_km,
        ]
    )
    base_weight = np.exp(-0.5 * np.square(local_distance / bandwidth_km)) / np.square(sigma)
    robust_weight = np.ones_like(base_weight)
    coefficient = np.zeros(3, dtype=float)
    covariance = np.full((3, 3), np.nan)
    for _ in range(iterations):
        weight = base_weight * robust_weight
        normal = design.T @ (weight[:, None] * design)
        covariance = np.linalg.pinv(normal)
        coefficient = covariance @ (design.T @ (weight * values))
        residual = values - design @ coefficient
        scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))), np.median(sigma), 1.0)
        robust_weight = np.minimum(1.0, huber_k * scale / np.maximum(np.abs(residual), 1.0e-12))
    final_weight = base_weight * robust_weight
    normal = design.T @ (final_weight[:, None] * design)
    covariance = np.linalg.pinv(normal)
    residual = values - design @ coefficient
    dof = max(1, len(values) - 3)
    residual_scale = max(1.0, float(np.sum(final_weight * np.square(residual)) / dof))
    covariance *= residual_scale
    effective_samples = float(np.square(final_weight.sum()) / np.square(final_weight).sum())
    condition = float(np.linalg.cond(normal))
    return coefficient, covariance, effective_samples, int(keep.sum()), condition


def rmls_incremental_strain(
    sample_xy_km: np.ndarray,
    east_mm: np.ndarray,
    north_mm: np.ndarray,
    sigma_east_mm: np.ndarray,
    sigma_north_mm: np.ndarray,
    target_xy_km: np.ndarray,
    *,
    support_radius_km: float = 8.0,
    bandwidth_km: float = 4.0,
    min_samples: int = 16,
    max_condition_number: float = 100.0,
) -> pd.DataFrame:
    """Estimate off-fault incremental strain on a sparse target grid by RMLS.

    This method assumes local continuity.  Callers should enforce a fault-safe
    target mask at least ``support_radius_km`` beyond a mapped rupture buffer.
    The returned uncertainty is conditional on resampled-displacement errors;
    it does not cure spatial correlation in the input InSAR grid.
    """
    sample_xy = np.asarray(sample_xy_km, dtype=float)
    targets = np.asarray(target_xy_km, dtype=float)
    e = np.asarray(east_mm, dtype=float)
    n = np.asarray(north_mm, dtype=float)
    sigma_e = np.asarray(sigma_east_mm, dtype=float)
    sigma_n = np.asarray(sigma_north_mm, dtype=float)
    finite = (
        np.isfinite(sample_xy).all(axis=1)
        & np.isfinite(e)
        & np.isfinite(n)
        & np.isfinite(sigma_e)
        & np.isfinite(sigma_n)
    )
    sample_xy, e, n, sigma_e, sigma_n = (
        sample_xy[finite], e[finite], n[finite], sigma_e[finite], sigma_n[finite]
    )
    rows: list[dict[str, float | int | bool]] = []
    for target in targets:
        fit_e = _robust_component_affine(
            sample_xy, e, sigma_e, target,
            support_radius_km=support_radius_km, bandwidth_km=bandwidth_km,
            min_samples=min_samples,
        )
        fit_n = _robust_component_affine(
            sample_xy, n, sigma_n, target,
            support_radius_km=support_radius_km, bandwidth_km=bandwidth_km,
            min_samples=min_samples,
        )
        row: dict[str, float | int | bool] = {
            "east_km": float(target[0]),
            "north_km": float(target[1]),
            "valid": False,
        }
        if fit_e is None or fit_n is None:
            rows.append(row)
            continue
        beta_e, covariance_e, n_eff_e, n_samples_e, condition_e = fit_e
        beta_n, covariance_n, n_eff_n, n_samples_n, condition_n = fit_n
        derivative_scale = float(bandwidth_km)
        exx = beta_e[1] / derivative_scale * NANOSTRAIN_PER_MM_PER_KM
        eyy = beta_n[2] / derivative_scale * NANOSTRAIN_PER_MM_PER_KM
        exy = 0.5 * (beta_e[2] + beta_n[1]) / derivative_scale * NANOSTRAIN_PER_MM_PER_KM
        rotation = 0.5 * (beta_n[1] - beta_e[2]) / derivative_scale * NANOSTRAIN_PER_MM_PER_KM
        sigma_exx = math.sqrt(max(covariance_e[1, 1], 0.0)) / derivative_scale * NANOSTRAIN_PER_MM_PER_KM
        sigma_eyy = math.sqrt(max(covariance_n[2, 2], 0.0)) / derivative_scale * NANOSTRAIN_PER_MM_PER_KM
        sigma_exy = 0.5 * math.sqrt(
            max(covariance_e[2, 2], 0.0) + max(covariance_n[1, 1], 0.0)
        ) / derivative_scale * NANOSTRAIN_PER_MM_PER_KM
        dilatation = exx + eyy
        sigma_dilatation = math.hypot(sigma_exx, sigma_eyy)
        mean_strain = 0.5 * dilatation
        radius = math.hypot(0.5 * (exx - eyy), exy)
        condition = max(condition_e, condition_n)
        row.update(
            {
                "valid": bool(condition <= max_condition_number),
                "effective_samples": float(min(n_eff_e, n_eff_n)),
                "sample_count": int(min(n_samples_e, n_samples_n)),
                "condition_number": float(condition),
                "east_mm": float(beta_e[0]),
                "north_mm": float(beta_n[0]),
                "epsilon_xx_nstrain": float(exx),
                "epsilon_yy_nstrain": float(eyy),
                "epsilon_xy_nstrain": float(exy),
                "gamma_xy_nstrain": float(2.0 * exy),
                "dilatation_nstrain": float(dilatation),
                "rotation_nrad": float(rotation),
                "principal_max_nstrain": float(mean_strain + radius),
                "principal_min_nstrain": float(mean_strain - radius),
                "principal_azimuth_deg": float(np.rad2deg(0.5 * np.arctan2(2.0 * exy, exx - eyy)) % 180.0),
                "sigma_epsilon_xx_nstrain": float(sigma_exx),
                "sigma_epsilon_yy_nstrain": float(sigma_eyy),
                "sigma_epsilon_xy_nstrain": float(sigma_exy),
                "sigma_dilatation_nstrain": float(sigma_dilatation),
                "sigma_rotation_nrad": float(sigma_exy),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
