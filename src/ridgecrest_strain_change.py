"""Leakage-safe change detection for uncertain 2-D strain time series.

The utilities in this module operate on interval strain *rates*.  Baseline
location, excess scatter, and every detection threshold must be estimated from
pre-event intervals only.  Spatial candidates are evaluated with a signed
maximum-cluster statistic so that the observed spatial dependence and the
simultaneous search over strain components are retained in the empirical null.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label


@dataclass(frozen=True)
class RobustBaseline:
    """Frozen pre-event center and uncertainty model."""

    center: np.ndarray
    empirical_scale: np.ndarray
    median_reported_sigma: np.ndarray
    excess_scale: np.ndarray
    observation_count: np.ndarray
    supported: np.ndarray


@dataclass(frozen=True)
class SpatialCluster:
    """One sign-consistent connected component in a strain-score map."""

    component_index: int
    component: str
    sign: int
    cell_count: int
    area_km2: float
    cluster_mass: float
    maximum_abs_z: float
    centroid_east_km: float
    centroid_north_km: float
    target_indices: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "component_index": self.component_index,
            "component": self.component,
            "sign": self.sign,
            "cell_count": self.cell_count,
            "area_km2": self.area_km2,
            "cluster_mass": self.cluster_mass,
            "maximum_abs_z": self.maximum_abs_z,
            "centroid_east_km": self.centroid_east_km,
            "centroid_north_km": self.centroid_north_km,
            "target_indices": ";".join(str(value) for value in self.target_indices),
        }


def duration_normalize(
    values: np.ndarray,
    sigma: np.ndarray,
    duration_days: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert interval estimates and standard errors to per-day rates."""

    values = np.asarray(values, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    duration = np.asarray(duration_days, dtype=float)
    if values.shape != sigma.shape:
        raise ValueError("values and sigma must have identical shapes")
    if values.ndim < 1 or duration.shape != (values.shape[0],):
        raise ValueError("duration_days must have one entry per time interval")
    if np.any(~np.isfinite(duration)) or np.any(duration <= 0.0):
        raise ValueError("Every interval duration must be finite and positive")
    reshape = (duration.size,) + (1,) * (values.ndim - 1)
    return values / duration.reshape(reshape), sigma / duration.reshape(reshape)


def _nanmedian(values: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered")
        return np.nanmedian(values, axis=axis)


def fit_robust_baseline(
    values: np.ndarray,
    sigma: np.ndarray,
    baseline_mask: np.ndarray,
    *,
    min_observations: int,
) -> RobustBaseline:
    """Fit a robust location plus excess-scatter model on frozen intervals.

    The empirical MAD contains measurement scatter.  To avoid counting it
    twice, only variance in excess of the median reported variance is retained
    as ``excess_scale``.
    """

    values = np.asarray(values, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    baseline_mask = np.asarray(baseline_mask, dtype=bool)
    if values.shape != sigma.shape or values.ndim < 2:
        raise ValueError("values and sigma must share a time-first shape")
    if baseline_mask.shape != (values.shape[0],):
        raise ValueError("baseline_mask must have one entry per interval")
    if int(baseline_mask.sum()) < int(min_observations):
        raise ValueError("The requested baseline has too few intervals")

    paired = np.isfinite(values) & np.isfinite(sigma) & (sigma > 0.0)
    baseline_values = np.where(paired[baseline_mask], values[baseline_mask], np.nan)
    baseline_sigma = np.where(paired[baseline_mask], sigma[baseline_mask], np.nan)
    count = np.sum(np.isfinite(baseline_values), axis=0)
    center = _nanmedian(baseline_values, axis=0)
    mad = _nanmedian(np.abs(baseline_values - center[None, ...]), axis=0)
    empirical_scale = 1.4826 * mad
    median_sigma = _nanmedian(baseline_sigma, axis=0)
    excess_variance = np.maximum(empirical_scale**2 - median_sigma**2, 0.0)
    excess_scale = np.sqrt(excess_variance)
    supported = (
        (count >= int(min_observations))
        & np.isfinite(center)
        & np.isfinite(excess_scale)
        & np.isfinite(median_sigma)
        & (median_sigma > 0.0)
    )
    for array in (center, empirical_scale, median_sigma, excess_scale):
        array[~supported] = np.nan
    return RobustBaseline(
        center=center,
        empirical_scale=empirical_scale,
        median_reported_sigma=median_sigma,
        excess_scale=excess_scale,
        observation_count=count,
        supported=supported,
    )


def standardized_innovation(
    values: np.ndarray,
    sigma: np.ndarray,
    baseline: RobustBaseline,
) -> np.ndarray:
    """Apply a frozen robust baseline to all interval strain rates."""

    values = np.asarray(values, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if values.shape != sigma.shape:
        raise ValueError("values and sigma must have identical shapes")
    if values.shape[1:] != baseline.center.shape:
        raise ValueError("Baseline shape does not match the component/target axes")
    total_sigma = np.sqrt(sigma**2 + baseline.excess_scale[None, ...] ** 2)
    valid = (
        baseline.supported[None, ...]
        & np.isfinite(values)
        & np.isfinite(total_sigma)
        & (total_sigma > 0.0)
    )
    score = np.full(values.shape, np.nan, dtype=float)
    np.divide(
        values - baseline.center[None, ...],
        total_sigma,
        out=score,
        where=valid,
    )
    return score


def leave_one_out_baseline_innovations(
    values: np.ndarray,
    sigma: np.ndarray,
    baseline_mask: np.ndarray,
    *,
    min_observations: int,
) -> np.ndarray:
    """Score every baseline interval against the remaining baseline only."""

    values = np.asarray(values, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    baseline_mask = np.asarray(baseline_mask, dtype=bool)
    output = np.full(values.shape, np.nan, dtype=float)
    for index in np.flatnonzero(baseline_mask):
        training = baseline_mask.copy()
        training[index] = False
        model = fit_robust_baseline(
            values,
            sigma,
            training,
            min_observations=min_observations,
        )
        output[index] = standardized_innovation(
            values[index : index + 1],
            sigma[index : index + 1],
            model,
        )[0]
    return output


def _target_grid(
    east_km: np.ndarray,
    north_km: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    east = np.asarray(east_km, dtype=float)
    north = np.asarray(north_km, dtype=float)
    if east.ndim != 1 or north.shape != east.shape:
        raise ValueError("east_km and north_km must be paired one-dimensional arrays")
    if np.any(~np.isfinite(east)) or np.any(~np.isfinite(north)):
        raise ValueError("Target coordinates must be finite")
    east_axis = np.unique(east)
    north_axis = np.unique(north)
    row = np.searchsorted(north_axis, north)
    column = np.searchsorted(east_axis, east)
    if len(np.unique(np.column_stack([row, column]), axis=0)) != len(east):
        raise ValueError("Target coordinates contain duplicates")
    dx = float(np.median(np.diff(east_axis))) if east_axis.size > 1 else 1.0
    dy = float(np.median(np.diff(north_axis))) if north_axis.size > 1 else 1.0
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("Target coordinate axes are not increasing")
    return east_axis, north_axis, row, column, dx * dy


def signed_spatial_clusters(
    z_by_component_target: np.ndarray,
    east_km: np.ndarray,
    north_km: np.ndarray,
    component_names: list[str] | tuple[str, ...],
    *,
    threshold: float = 1.96,
    min_cells: int = 4,
) -> list[SpatialCluster]:
    """Find sign-consistent 8-neighbour clusters above ``|z|`` threshold."""

    z = np.asarray(z_by_component_target, dtype=float)
    if z.ndim != 2:
        raise ValueError("z_by_component_target must have component x target shape")
    if z.shape[0] != len(component_names):
        raise ValueError("component_names does not match the component axis")
    if z.shape[1] != len(east_km):
        raise ValueError("Coordinate count does not match the target axis")
    if threshold <= 0.0 or min_cells < 1:
        raise ValueError("threshold and min_cells must be positive")

    east_axis, north_axis, row, column, cell_area = _target_grid(east_km, north_km)
    index_grid = np.full((north_axis.size, east_axis.size), -1, dtype=int)
    index_grid[row, column] = np.arange(z.shape[1])
    connectivity = np.ones((3, 3), dtype=int)
    records: list[SpatialCluster] = []
    for component_index, component in enumerate(component_names):
        grid = np.full(index_grid.shape, np.nan, dtype=float)
        grid[row, column] = z[component_index]
        for sign in (-1, 1):
            component_labels, count = label(
                np.isfinite(grid) & (sign * grid >= float(threshold)),
                structure=connectivity,
            )
            for cluster_id in range(1, count + 1):
                use = component_labels == cluster_id
                cell_count = int(use.sum())
                if cell_count < int(min_cells):
                    continue
                target_indices = tuple(
                    int(value) for value in index_grid[use] if value >= 0
                )
                signed_score = sign * grid[use]
                records.append(
                    SpatialCluster(
                        component_index=component_index,
                        component=str(component),
                        sign=sign,
                        cell_count=cell_count,
                        area_km2=cell_count * cell_area,
                        cluster_mass=float(np.sum(signed_score - threshold)),
                        maximum_abs_z=float(np.max(signed_score)),
                        centroid_east_km=float(np.mean(east_km[list(target_indices)])),
                        centroid_north_km=float(np.mean(north_km[list(target_indices)])),
                        target_indices=target_indices,
                    )
                )
    return sorted(records, key=lambda item: item.cluster_mass, reverse=True)


def maximum_signed_cluster_mass(
    z_by_component_target: np.ndarray,
    east_km: np.ndarray,
    north_km: np.ndarray,
    component_names: list[str] | tuple[str, ...],
    *,
    threshold: float = 1.96,
    min_cells: int = 4,
) -> float:
    clusters = signed_spatial_clusters(
        z_by_component_target,
        east_km,
        north_km,
        component_names,
        threshold=threshold,
        min_cells=min_cells,
    )
    return 0.0 if not clusters else float(clusters[0].cluster_mass)


def sliding_block_maximum(values: np.ndarray, block_length: int) -> np.ndarray:
    """Maxima of consecutive blocks, preserving observed serial ordering."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("values must be a finite one-dimensional series")
    if block_length < 1 or block_length > values.size:
        raise ValueError("block_length must be between one and the series length")
    return np.asarray(
        [
            float(np.max(values[start : start + block_length]))
            for start in range(values.size - block_length + 1)
        ],
        dtype=float,
    )


def empirical_upper_tail_pvalue(null: np.ndarray, observed: float) -> float:
    """Finite-sample upper-tail p-value with the standard plus-one correction."""

    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0 or not math.isfinite(float(observed)):
        raise ValueError("A finite observed value and nonempty null are required")
    return float((1.0 + np.count_nonzero(null >= observed)) / (null.size + 1.0))


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Holm step-down family-wise adjustment."""

    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("pvalues must be a finite one-dimensional array")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("pvalues must lie in [0, 1]")
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        (values.size - np.arange(values.size)) * values[order]
    )
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def strain_energy(z: np.ndarray, quantile: float = 0.90) -> np.ndarray:
    """Robust map-level magnitude summary used for temporal change detection."""

    z = np.asarray(z, dtype=float)
    if z.ndim < 2:
        raise ValueError("z must have time plus at least one analysis axis")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between zero and one")
    flattened = np.abs(z).reshape(z.shape[0], -1)
    output = np.full(z.shape[0], np.nan, dtype=float)
    for index, row in enumerate(flattened):
        finite = row[np.isfinite(row)]
        if finite.size:
            output[index] = float(np.quantile(finite, quantile))
    return output


def positive_page_cusum(values: np.ndarray, *, reference: float = 0.5) -> np.ndarray:
    """One-sided Page CUSUM for a persistent positive standardized shift."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("values must be a finite one-dimensional series")
    if reference < 0.0:
        raise ValueError("reference must be non-negative")
    output = np.empty(values.size, dtype=float)
    running = 0.0
    for index, value in enumerate(values):
        running = max(0.0, running + float(value) - float(reference))
        output[index] = running
    return output


def sliding_block_cusum_maxima(
    baseline_values: np.ndarray,
    block_length: int,
    *,
    reference: float = 0.5,
) -> np.ndarray:
    """CUSUM maxima for every observed consecutive baseline block."""

    values = np.asarray(baseline_values, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("baseline_values must be a finite one-dimensional series")
    if block_length < 1 or block_length > values.size:
        raise ValueError("block_length must be between one and the series length")
    return np.asarray(
        [
            float(
                np.max(
                    positive_page_cusum(
                        values[start : start + block_length],
                        reference=reference,
                    )
                )
            )
            for start in range(values.size - block_length + 1)
        ],
        dtype=float,
    )
