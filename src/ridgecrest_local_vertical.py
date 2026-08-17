"""All-station adaptive local interpolation for GNSS vertical displacement.

The functions here deliberately do *not* fit a map-wide plane.  For every
target location, every GNSS station within the smallest adequate physical
radius is used.  The support is screened for station count, azimuthal coverage,
and local convex-hull containment before a local ordinary-Kriging or Matérn
Gaussian-process prediction is made.

This module is intended for the vertical-to-LOS correction sequence:

``GNSS U -> Uhat(x,y) -> lU(x,y) Uhat(x,y) -> corrected LOS``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, QhullError

Family = Literal[
    "ok_exponential",
    "ok_matern32",
    "ok_rbf",
    "uk_matern32",
    "uk_rbf",
    # Backward-compatible plug-in-mean GP specifications retained so older
    # manifests remain readable. New model comparisons use explicit kriging
    # terminology.
    "gp_matern32",
    "gp_rbf",
]


@dataclass(frozen=True)
class LocalVerticalConfig:
    """Predeclared geometry rules for a moving local interpolation window."""

    radii_km: tuple[float, ...] = (35.0, 45.0, 55.0, 65.0, 75.0, 90.0, 105.0, 120.0)
    min_stations: int = 5
    sector_count: int = 8
    min_occupied_sectors: int = 3
    require_local_hull: bool = True

    def __post_init__(self) -> None:
        if not self.radii_km or any(radius <= 0.0 for radius in self.radii_km):
            raise ValueError("Local-search radii must be positive")
        if tuple(sorted(set(self.radii_km))) != self.radii_km:
            raise ValueError("Local-search radii must be sorted and unique")
        if self.min_stations < 3:
            raise ValueError("At least three local stations are required")
        if not 1 <= self.min_occupied_sectors <= self.sector_count:
            raise ValueError("Invalid azimuth-sector requirement")


@dataclass(frozen=True)
class LocalVerticalModel:
    """Covariance family and parameters selected before event-time prediction."""

    family: Family
    length_scale_km: float
    nugget_mm: float
    sill_mm2: float
    config: LocalVerticalConfig
    validation_rmse_mm: float
    validation_nlpd: float
    validation_coverage90: float
    baseline_rmse_mm: float
    baseline_nlpd: float
    bootstrap_delta_nlpd_lower95: float


@dataclass(frozen=True)
class LocalSupport:
    """Geometry-qualified GNSS support for one target point."""

    indices: np.ndarray
    radius_km: float
    occupied_sector_count: int
    inside_local_hull: bool


@dataclass(frozen=True)
class LocalVerticalPrediction:
    """A target-wise local vertical interpolation result."""

    mean_mm: np.ndarray
    sigma_mm: np.ndarray
    support_count: np.ndarray
    support_radius_km: np.ndarray
    occupied_sector_count: np.ndarray
    inside_local_hull: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class LocalSupportTopology:
    """Precomputed local-support geometry for a fixed GNSS network and grid."""

    targets_xy_km: np.ndarray
    supports: tuple[LocalSupport | None, ...]


def _inside_hull(target_xy: np.ndarray, support_xy: np.ndarray) -> bool:
    """Return whether a target lies inside a non-degenerate local station hull."""
    if len(support_xy) < 3:
        return False
    try:
        hull = Delaunay(np.asarray(support_xy, dtype=float))
    except QhullError:
        return False
    return bool(hull.find_simplex(np.asarray(target_xy, dtype=float).reshape(1, 2))[0] >= 0)


def _occupied_sectors(target_xy: np.ndarray, station_xy: np.ndarray, sector_count: int) -> int:
    relative = np.asarray(station_xy, dtype=float) - np.asarray(target_xy, dtype=float)
    angle = np.mod(np.arctan2(relative[:, 1], relative[:, 0]), 2.0 * np.pi)
    sector = np.floor(angle / (2.0 * np.pi / sector_count)).astype(int)
    return int(np.unique(sector).size)


def adaptive_local_support(
    target_xy_km: np.ndarray,
    station_xy_km: np.ndarray,
    config: LocalVerticalConfig,
    *,
    station_allowed: np.ndarray | None = None,
    require_local_hull: bool | None = None,
) -> LocalSupport | None:
    """Select every station in the smallest geometry-adequate local radius.

    ``station_allowed`` is a per-target boolean visibility mask.  It enables a
    caller to apply a mapped-rupture barrier without altering the interpolation
    equations; disallowed station-to-target paths simply never enter support.
    """
    target = np.asarray(target_xy_km, dtype=float).reshape(2)
    stations = np.asarray(station_xy_km, dtype=float)
    if stations.ndim != 2 or stations.shape[1] != 2:
        raise ValueError("station_xy_km must have shape (station, 2)")
    allowed = np.ones(len(stations), dtype=bool) if station_allowed is None else np.asarray(station_allowed, dtype=bool)
    if allowed.shape != (len(stations),):
        raise ValueError("station_allowed must have one value per GNSS station")
    distance = np.linalg.norm(stations - target, axis=1)
    need_hull = config.require_local_hull if require_local_hull is None else bool(require_local_hull)
    for radius in config.radii_km:
        indices = np.flatnonzero(allowed & (distance <= radius))
        if len(indices) < config.min_stations:
            continue
        sectors = _occupied_sectors(target, stations[indices], config.sector_count)
        if sectors < config.min_occupied_sectors:
            continue
        inside = _inside_hull(target, stations[indices])
        if need_hull and not inside:
            continue
        return LocalSupport(indices, float(radius), sectors, inside)
    return None


def _local_constant_predict(values: np.ndarray, sigma_mm: np.ndarray) -> tuple[float, float]:
    """Same-neighbourhood uncertainty-weighted constant baseline."""
    values = np.asarray(values, dtype=float)
    sigma = np.maximum(np.asarray(sigma_mm, dtype=float), 1.0e-3)
    weights = 1.0 / np.square(sigma)
    mean = float(np.sum(weights * values) / np.sum(weights))
    effective_n = float(np.square(weights.sum()) / np.square(weights).sum())
    weighted_variance = float(np.sum(weights * np.square(values - mean)) / np.sum(weights))
    prediction_sigma = math.sqrt(max(weighted_variance, 0.0) + 1.0 / weights.sum())
    # The effective-N term prevents a single very precise station from creating
    # an implausibly precise local constant baseline.
    prediction_sigma = max(prediction_sigma, math.sqrt(max(weighted_variance, 0.0) / max(effective_n, 1.0)))
    return mean, prediction_sigma


def estimate_interval_sill_mm2(
    values_mm: np.ndarray,
    sigma_mm: np.ndarray,
    *,
    floor_mm2: float = 1.0,
) -> float:
    """Estimate one interval-wide latent vertical variance.

    The estimate is deliberately data-adaptive because coseismic vertical
    variability is much larger than quiet pre-event variability. In
    leave-one-station-out validation the caller must pass only the training
    stations. At application time all endpoint-valid stations are passed. This
    makes validation and application use the same no-leakage rule.
    """

    values = np.asarray(values_mm, dtype=float)
    sigma = np.asarray(sigma_mm, dtype=float)
    finite = np.isfinite(values) & np.isfinite(sigma)
    values = values[finite]
    sigma = sigma[finite]
    if len(values) < 3:
        return float(floor_mm2)
    centred = values - np.median(values)
    robust_scale = 1.4826 * np.median(np.abs(centred))
    sample_variance = float(np.var(values, ddof=1))
    robust_variance = float(robust_scale**2)
    observed_variance = max(sample_variance, robust_variance)
    measurement_variance = float(np.median(np.square(np.maximum(sigma, 0.0))))
    return float(max(floor_mm2, observed_variance - measurement_variance))


def _pairwise_distance_km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = np.asarray(a, dtype=float)[:, None, :] - np.asarray(
        b, dtype=float
    )[None, :, :]
    return np.linalg.norm(delta, axis=2)


def _covariance(
    a: np.ndarray,
    b: np.ndarray,
    *,
    kernel: Literal["exponential", "matern32", "rbf"],
    length_scale_km: float,
    sill_mm2: float,
) -> np.ndarray:
    """Evaluate a stationary isotropic covariance in projected kilometres."""

    distance = _pairwise_distance_km(a, b)
    length = float(length_scale_km)
    if length <= 0.0:
        raise ValueError("length_scale_km must be positive")
    sill = float(sill_mm2)
    if sill <= 0.0:
        raise ValueError("sill_mm2 must be positive")
    if kernel == "exponential":
        return sill * np.exp(-distance / length)
    if kernel == "matern32":
        scaled = math.sqrt(3.0) * distance / length
        return sill * (1.0 + scaled) * np.exp(-scaled)
    if kernel == "rbf":
        return sill * np.exp(-0.5 * np.square(distance / length))
    raise ValueError(f"Unsupported covariance kernel: {kernel}")


def _family_specification(
    family: Family,
) -> tuple[
    Literal["ordinary", "universal_linear", "plugin_constant"],
    Literal["exponential", "matern32", "rbf"],
]:
    if family == "ok_exponential":
        return "ordinary", "exponential"
    if family == "ok_matern32":
        return "ordinary", "matern32"
    if family == "ok_rbf":
        return "ordinary", "rbf"
    if family == "uk_matern32":
        return "universal_linear", "matern32"
    if family == "uk_rbf":
        return "universal_linear", "rbf"
    if family == "gp_matern32":
        return "plugin_constant", "matern32"
    if family == "gp_rbf":
        return "plugin_constant", "rbf"
    raise ValueError(f"Unsupported local vertical family: {family}")


def _ordinary_kriging_predict(
    train_xy: np.ndarray,
    train_values: np.ndarray,
    train_sigma: np.ndarray,
    target_xy: np.ndarray,
    *,
    kernel: Literal["exponential", "matern32", "rbf"],
    length_scale_km: float,
    sill_mm2: float,
    nugget_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary kriging with a chosen covariance and heteroscedastic errors."""

    stations = np.asarray(train_xy, dtype=float)
    targets = np.asarray(target_xy, dtype=float).reshape(-1, 2)
    values = np.asarray(train_values, dtype=float)
    sigma = np.asarray(train_sigma, dtype=float)
    covariance = _covariance(
        stations,
        stations,
        kernel=kernel,
        length_scale_km=length_scale_km,
        sill_mm2=sill_mm2,
    )
    covariance += np.diag(np.square(sigma) + float(nugget_mm) ** 2)
    count = len(values)
    system = np.empty((count + 1, count + 1), dtype=float)
    system[:count, :count] = covariance
    system[:count, count] = 1.0
    system[count, :count] = 1.0
    system[count, count] = 0.0
    cross = _covariance(
        stations,
        targets,
        kernel=kernel,
        length_scale_km=length_scale_km,
        sill_mm2=sill_mm2,
    )
    rhs = np.vstack([cross, np.ones((1, len(targets)))])
    solution = np.linalg.pinv(system, hermitian=True) @ rhs
    weights = solution[:count]
    lagrange = solution[count]
    mean = values @ weights
    variance = (
        float(sill_mm2)
        - np.sum(weights * cross, axis=0)
        - lagrange
    )
    return np.asarray(mean), np.sqrt(np.maximum(variance, 0.0))


def _plugin_constant_gp_predict(
    train_xy: np.ndarray,
    train_values: np.ndarray,
    train_sigma: np.ndarray,
    target_xy: np.ndarray,
    *,
    kernel: Literal["matern32", "rbf"],
    length_scale_km: float,
    sill_mm2: float,
    nugget_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Local GP with an uncertainty-weighted plug-in constant mean."""

    stations = np.asarray(train_xy, dtype=float)
    targets = np.asarray(target_xy, dtype=float).reshape(-1, 2)
    values = np.asarray(train_values, dtype=float)
    sigma = np.asarray(train_sigma, dtype=float)
    precision = 1.0 / np.square(np.maximum(sigma, 1.0e-3))
    local_mean = float(np.sum(precision * values) / precision.sum())
    covariance = _covariance(
        stations,
        stations,
        kernel=kernel,
        length_scale_km=length_scale_km,
        sill_mm2=sill_mm2,
    )
    covariance += np.diag(np.square(sigma) + float(nugget_mm) ** 2)
    cross = _covariance(
        stations,
        targets,
        kernel=kernel,
        length_scale_km=length_scale_km,
        sill_mm2=sill_mm2,
    )
    inverse_cross = np.linalg.solve(covariance, cross)
    alpha = np.linalg.solve(covariance, values - local_mean)
    mean = local_mean + cross.T @ alpha
    variance = float(sill_mm2) - np.sum(cross * inverse_cross, axis=0)
    return np.asarray(mean), np.sqrt(np.maximum(variance, 0.0))


def _universal_kriging_predict(
    train_xy: np.ndarray,
    train_values: np.ndarray,
    train_sigma: np.ndarray,
    target_xy: np.ndarray,
    *,
    kernel: Literal["matern32", "rbf"],
    length_scale_km: float,
    sill_mm2: float,
    nugget_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Local universal kriging with a first-order east/north drift.

    Coordinates are centred and scaled locally for numerical conditioning.
    The generalized least-squares trend uncertainty is included in the latent
    prediction variance. This is a local sensitivity model, not a map-wide
    displacement plane.
    """

    stations = np.asarray(train_xy, dtype=float)
    targets = np.asarray(target_xy, dtype=float).reshape(-1, 2)
    values = np.asarray(train_values, dtype=float)
    sigma = np.asarray(train_sigma, dtype=float)
    if len(stations) < 5:
        raise ValueError("Local universal kriging requires at least five stations")
    origin = np.mean(stations, axis=0)
    coordinate_scale = max(
        float(length_scale_km),
        float(np.max(np.ptp(stations, axis=0))),
        1.0,
    )
    station_scaled = (stations - origin) / coordinate_scale
    target_scaled = (targets - origin) / coordinate_scale
    design = np.column_stack(
        [np.ones(len(stations)), station_scaled[:, 0], station_scaled[:, 1]]
    )
    target_design = np.column_stack(
        [np.ones(len(targets)), target_scaled[:, 0], target_scaled[:, 1]]
    )
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise np.linalg.LinAlgError("Local universal-kriging drift is rank deficient")
    covariance = _covariance(
        stations,
        stations,
        kernel=kernel,
        length_scale_km=length_scale_km,
        sill_mm2=sill_mm2,
    )
    covariance += np.diag(np.square(sigma) + float(nugget_mm) ** 2)
    cross = _covariance(
        stations,
        targets,
        kernel=kernel,
        length_scale_km=length_scale_km,
        sill_mm2=sill_mm2,
    )
    inverse_design = np.linalg.solve(covariance, design)
    inverse_values = np.linalg.solve(covariance, values)
    normal = design.T @ inverse_design
    normal_inverse = np.linalg.pinv(normal, hermitian=True)
    beta = normal_inverse @ design.T @ inverse_values
    residual = values - design @ beta
    alpha = np.linalg.solve(covariance, residual)
    inverse_cross = np.linalg.solve(covariance, cross)
    mean = target_design @ beta + cross.T @ alpha
    drift_delta = target_design.T - design.T @ inverse_cross
    variance = (
        float(sill_mm2)
        - np.sum(cross * inverse_cross, axis=0)
        + np.sum(
            drift_delta * (normal_inverse @ drift_delta),
            axis=0,
        )
    )
    return np.asarray(mean), np.sqrt(np.maximum(variance, 0.0))


def _predict_group(
    family: Family,
    train_xy: np.ndarray,
    train_values: np.ndarray,
    train_sigma: np.ndarray,
    target_xy: np.ndarray,
    *,
    length_scale_km: float,
    nugget_mm: float,
    sill_mm2: float,
) -> tuple[np.ndarray, np.ndarray]:
    mean_kind, kernel = _family_specification(family)
    if mean_kind == "ordinary":
        return _ordinary_kriging_predict(
            train_xy,
            train_values,
            train_sigma,
            target_xy,
            kernel=kernel,
            length_scale_km=length_scale_km,
            sill_mm2=sill_mm2,
            nugget_mm=nugget_mm,
        )
    if mean_kind == "plugin_constant":
        return _plugin_constant_gp_predict(
            train_xy,
            train_values,
            train_sigma,
            target_xy,
            kernel=kernel,  # type: ignore[arg-type]
            length_scale_km=length_scale_km,
            sill_mm2=sill_mm2,
            nugget_mm=nugget_mm,
        )
    return _universal_kriging_predict(
        train_xy,
        train_values,
        train_sigma,
        target_xy,
        kernel=kernel,  # type: ignore[arg-type]
        length_scale_km=length_scale_km,
        sill_mm2=sill_mm2,
        nugget_mm=nugget_mm,
    )


def _predict_with_support(
    family: Family,
    support: LocalSupport,
    station_xy: np.ndarray,
    values: np.ndarray,
    sigma_mm: np.ndarray,
    target_xy: np.ndarray,
    *,
    length_scale_km: float,
    nugget_mm: float,
    sill_mm2: float,
) -> tuple[float, float]:
    indices = support.indices
    mean, uncertainty = _predict_group(
        family,
        station_xy[indices],
        values[indices],
        sigma_mm[indices],
        np.asarray(target_xy, dtype=float).reshape(1, 2),
        length_scale_km=length_scale_km,
        nugget_mm=nugget_mm,
        sill_mm2=sill_mm2,
    )
    return float(mean[0]), float(uncertainty[0])


def predict_local_vertical(
    model: LocalVerticalModel,
    station_xy_km: np.ndarray,
    values_mm: np.ndarray,
    sigma_mm: np.ndarray,
    targets_xy_km: np.ndarray,
    *,
    station_allowed: np.ndarray | None = None,
) -> LocalVerticalPrediction:
    """Predict vertical displacement at every target using adaptive all-station support."""
    stations = np.asarray(station_xy_km, dtype=float)
    values = np.asarray(values_mm, dtype=float)
    sigma = np.asarray(sigma_mm, dtype=float)
    targets = np.asarray(targets_xy_km, dtype=float)
    if stations.shape != (len(values), 2) or sigma.shape != values.shape:
        raise ValueError("Station geometry, vertical values, and uncertainties are inconsistent")
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("targets_xy_km must have shape (target, 2)")
    if station_allowed is not None and np.asarray(station_allowed).shape != (len(targets), len(stations)):
        raise ValueError("station_allowed must have shape (target, station)")

    mean = np.full(len(targets), np.nan, dtype=float)
    uncertainty = np.full(len(targets), np.nan, dtype=float)
    count = np.zeros(len(targets), dtype=np.int16)
    radius = np.full(len(targets), np.nan, dtype=float)
    sectors = np.zeros(len(targets), dtype=np.int8)
    inside = np.zeros(len(targets), dtype=bool)
    for index, target in enumerate(targets):
        allowed = None if station_allowed is None else station_allowed[index]
        support = adaptive_local_support(target, stations, model.config, station_allowed=allowed)
        if support is None:
            continue
        mean[index], uncertainty[index] = _predict_with_support(
            model.family, support, stations, values, sigma, target,
            length_scale_km=model.length_scale_km,
            nugget_mm=model.nugget_mm,
            sill_mm2=model.sill_mm2,
        )
        count[index] = len(support.indices)
        radius[index] = support.radius_km
        sectors[index] = support.occupied_sector_count
        inside[index] = support.inside_local_hull
    valid = np.isfinite(mean) & np.isfinite(uncertainty)
    return LocalVerticalPrediction(mean, uncertainty, count, radius, sectors, inside, valid)


def build_local_support_topology(
    station_xy_km: np.ndarray,
    targets_xy_km: np.ndarray,
    config: LocalVerticalConfig,
    *,
    station_allowed: np.ndarray | None = None,
) -> LocalSupportTopology:
    """Precompute all local neighbourhoods for repeated epoch prediction.

    The support topology depends on station geometry and target location, not on
    vertical displacement.  Caching it makes a date-by-date time-series run
    feasible without changing which stations contribute at any pixel.
    """
    stations = np.asarray(station_xy_km, dtype=float)
    targets = np.asarray(targets_xy_km, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("targets_xy_km must have shape (target, 2)")
    if station_allowed is not None and np.asarray(station_allowed).shape != (len(targets), len(stations)):
        raise ValueError("station_allowed must have shape (target, station)")
    supports: list[LocalSupport | None] = []
    for index, target in enumerate(targets):
        allowed = None if station_allowed is None else station_allowed[index]
        supports.append(adaptive_local_support(target, stations, config, station_allowed=allowed))
    return LocalSupportTopology(targets.copy(), tuple(supports))


def predict_local_vertical_from_topology(
    model: LocalVerticalModel,
    station_xy_km: np.ndarray,
    values_mm: np.ndarray,
    sigma_mm: np.ndarray,
    topology: LocalSupportTopology,
) -> LocalVerticalPrediction:
    """Fast repeated local prediction using a precomputed support topology.

    Targets sharing an identical contributing-station set are solved together.
    This preserves the same local all-station method as ``predict_local_vertical``
    while avoiding one Gaussian-process fit per pixel per date.
    """
    stations = np.asarray(station_xy_km, dtype=float)
    values = np.asarray(values_mm, dtype=float)
    sigma = np.asarray(sigma_mm, dtype=float)
    targets = np.asarray(topology.targets_xy_km, dtype=float)
    if stations.shape != (len(values), 2) or sigma.shape != values.shape:
        raise ValueError("Station geometry, vertical values, and uncertainties are inconsistent")
    mean = np.full(len(targets), np.nan, dtype=float)
    uncertainty = np.full(len(targets), np.nan, dtype=float)
    count = np.zeros(len(targets), dtype=np.int16)
    radius = np.full(len(targets), np.nan, dtype=float)
    sectors = np.zeros(len(targets), dtype=np.int8)
    inside = np.zeros(len(targets), dtype=bool)
    groups: dict[tuple[int, ...], list[int]] = {}
    metadata: dict[tuple[int, ...], LocalSupport] = {}
    for target_index, support in enumerate(topology.supports):
        if support is None:
            continue
        key = tuple(int(value) for value in support.indices)
        groups.setdefault(key, []).append(target_index)
        metadata[key] = support
    for key, target_indices in groups.items():
        support = metadata[key]
        indices = support.indices
        query = targets[target_indices]
        result_mean, result_sigma = _predict_group(
            model.family,
            stations[indices],
            values[indices],
            sigma[indices],
            query,
            length_scale_km=model.length_scale_km,
            nugget_mm=model.nugget_mm,
            sill_mm2=model.sill_mm2,
        )
        target_indices_array = np.asarray(target_indices, dtype=int)
        mean[target_indices_array] = result_mean
        uncertainty[target_indices_array] = result_sigma
        count[target_indices_array] = len(indices)
        radius[target_indices_array] = support.radius_km
        sectors[target_indices_array] = support.occupied_sector_count
        inside[target_indices_array] = support.inside_local_hull
    valid = np.isfinite(mean) & np.isfinite(uncertainty)
    return LocalVerticalPrediction(mean, uncertainty, count, radius, sectors, inside, valid)


def _interior_holdout_indices(station_xy_km: np.ndarray) -> np.ndarray:
    """Return LOO folds that remain inside the training network hull.

    Global-hull boundary stations are explicitly excluded before scoring instead
    of being silently dropped candidate-by-candidate.  This makes all model
    comparisons use the same non-extrapolative held-out station set.
    """
    stations = np.asarray(station_xy_km, dtype=float)
    keep: list[int] = []
    for holdout in range(len(stations)):
        train = np.delete(stations, holdout, axis=0)
        if _inside_hull(stations[holdout], train):
            keep.append(holdout)
    if not keep:
        raise RuntimeError("No non-extrapolative GNSS holdout stations are available")
    return np.asarray(keep, dtype=int)


def _bootstrap_lower95(values: np.ndarray, *, seed: int = 640004, n_bootstrap: int = 2000) -> float:
    """Bootstrap 95% lower bound for a mean, resampling pre-event intervals."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("-inf")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    return float(np.quantile(samples.mean(axis=1), 0.05))


def select_local_vertical_model(
    interval_tables: Iterable[pd.DataFrame],
    *,
    configs: Sequence[LocalVerticalConfig] = (LocalVerticalConfig(),),
    families: Sequence[Family] = (
        "ok_exponential",
        "ok_matern32",
        "ok_rbf",
        "uk_matern32",
        "uk_rbf",
    ),
    length_scales_km: Sequence[float] = (25.0, 40.0, 60.0, 80.0),
    nuggets_mm: Sequence[float] = (0.0, 2.0, 5.0, 10.0),
    rmse_relative_tolerance: float = 0.02,
    require_acceptance: bool = True,
) -> tuple[LocalVerticalModel, pd.DataFrame, pd.DataFrame]:
    """Select a local spatial predictor using pre-event station LOO controls.

    Tables must contain ``station, latitude, longitude, up_mm, sigma_up_mm``.
    Candidate skill is evaluated on exactly the same interior LOO stations at
    every pre-event interval.  A candidate is accepted only when its
    interval-bootstrap lower 95% bound for NLPD improvement over the same-
    neighbourhood local-constant baseline is positive, its 90% interval is
    calibrated, and its RMSE is no worse than a predeclared practical-
    equivalence tolerance (default 2%).  This avoids rejecting a better
    probabilistic predictor because of a numerically negligible RMSE change.
    Universal candidates use a first-order trend inside each local support;
    no candidate fits one plane across the full map.
    """
    tables = [table.copy().sort_values("station").reset_index(drop=True) for table in interval_tables]
    if not tables:
        raise ValueError("At least one pre-event GNSS interval table is required")
    required = {"station", "latitude", "longitude", "up_mm", "sigma_up_mm"}
    for table in tables:
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"Interval table is missing {sorted(missing)}")
    station_order = tuple(tables[0]["station"])
    if any(tuple(table["station"]) != station_order for table in tables):
        raise ValueError("All pre-event tables must retain the same station set and order")
    station_xy = np.column_stack([
        tables[0]["longitude"].to_numpy(float), tables[0]["latitude"].to_numpy(float)
    ])
    # The caller supplies UTM coordinates in optional columns to avoid an
    # accidental geographic-degree covariance.  Requiring them is intentional.
    if {"east_km", "north_km"}.issubset(tables[0].columns):
        station_xy = tables[0][["east_km", "north_km"]].to_numpy(float)
    else:
        raise ValueError("Pre-event tables require UTM east_km and north_km columns")
    holdouts = _interior_holdout_indices(station_xy)

    rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for config_index, config in enumerate(configs):
        for family in families:
            for length in length_scales_km:
                for nugget in nuggets_mm:
                    residuals: list[float] = []
                    nlpd: list[float] = []
                    baseline_residuals: list[float] = []
                    baseline_nlpd: list[float] = []
                    coverage90: list[bool] = []
                    interval_delta: dict[int, list[float]] = {}
                    complete = True
                    for interval_index, table in enumerate(tables):
                        values = table["up_mm"].to_numpy(float)
                        sigma = table["sigma_up_mm"].to_numpy(float)
                        for holdout in holdouts:
                            train = np.arange(len(values)) != holdout
                            support = adaptive_local_support(
                                station_xy[holdout], station_xy[train], config,
                            )
                            if support is None:
                                complete = False
                                continue
                            train_xy = station_xy[train]
                            train_values = values[train]
                            train_sigma = sigma[train]
                            # The held-out value and uncertainty never enter the
                            # covariance amplitude used to predict that station.
                            sill = estimate_interval_sill_mm2(
                                train_values, train_sigma
                            )
                            prediction, prediction_sigma = _predict_with_support(
                                family, support, train_xy, train_values, train_sigma,
                                station_xy[holdout], length_scale_km=float(length),
                                nugget_mm=float(nugget), sill_mm2=sill,
                            )
                            base_mean, base_sigma = _local_constant_predict(
                                train_values[support.indices], train_sigma[support.indices]
                            )
                            total_sigma = math.hypot(max(prediction_sigma, 1.0e-3), sigma[holdout])
                            total_base_sigma = math.hypot(max(base_sigma, 1.0e-3), sigma[holdout])
                            residual = float(values[holdout] - prediction)
                            base_residual = float(values[holdout] - base_mean)
                            point_nlpd = 0.5 * (math.log(2.0 * math.pi * total_sigma**2) + (residual / total_sigma) ** 2)
                            point_base_nlpd = 0.5 * (math.log(2.0 * math.pi * total_base_sigma**2) + (base_residual / total_base_sigma) ** 2)
                            residuals.append(residual)
                            baseline_residuals.append(base_residual)
                            nlpd.append(point_nlpd)
                            baseline_nlpd.append(point_base_nlpd)
                            coverage90.append(abs(residual) <= 1.6448536269514722 * total_sigma)
                            interval_delta.setdefault(interval_index, []).append(point_base_nlpd - point_nlpd)
                            prediction_rows.append({
                                "config_index": config_index,
                                "family": family,
                                "length_scale_km": float(length),
                                "nugget_mm": float(nugget),
                                "interval_index": interval_index,
                                "holdout_station": station_order[holdout],
                                "observed_mm": float(values[holdout]),
                                "predicted_mm": prediction,
                                "predictive_sigma_mm": total_sigma,
                                "baseline_predicted_mm": base_mean,
                                "baseline_predictive_sigma_mm": total_base_sigma,
                                "nearby_station_count": len(support.indices),
                                "support_radius_km": support.radius_km,
                                "occupied_sector_count": support.occupied_sector_count,
                                "train_only_sill_mm2": sill,
                            })
                    if not residuals:
                        continue
                    interval_mean_delta = np.asarray([np.mean(value) for value in interval_delta.values()])
                    rows.append({
                        "config_index": config_index,
                        "family": family,
                        "length_scale_km": float(length),
                        "nugget_mm": float(nugget),
                        "complete_fixed_holdout_coverage": bool(complete),
                        "n_predictions": len(residuals),
                        "n_interior_holdouts": len(holdouts),
                        "n_control_intervals": len(tables),
                        "rmse_mm": float(np.sqrt(np.mean(np.square(residuals)))),
                        "mean_nlpd": float(np.mean(nlpd)),
                        "coverage90": float(np.mean(coverage90)),
                        "baseline_rmse_mm": float(np.sqrt(np.mean(np.square(baseline_residuals)))),
                        "baseline_mean_nlpd": float(np.mean(baseline_nlpd)),
                        "mean_delta_nlpd": float(np.mean(baseline_nlpd) - np.mean(nlpd)),
                        "bootstrap_delta_nlpd_lower95": _bootstrap_lower95(interval_mean_delta),
                    })
    score = pd.DataFrame(rows)
    if score.empty:
        raise RuntimeError("No local vertical model produced pre-event validation predictions")
    if rmse_relative_tolerance < 0.0:
        raise ValueError("rmse_relative_tolerance must be non-negative")
    score["rmse_relative_tolerance"] = float(rmse_relative_tolerance)
    score["accepted"] = (
        score["complete_fixed_holdout_coverage"]
        & (score["rmse_mm"] <= score["baseline_rmse_mm"] * (1.0 + float(rmse_relative_tolerance)))
        & (score["bootstrap_delta_nlpd_lower95"] > 0.0)
        & score["coverage90"].between(0.70, 1.00)
    )
    accepted = score.loc[score["accepted"]].copy()
    if accepted.empty:
        if require_acceptance:
            raise RuntimeError("No local spatial vertical model passed the pre-event validation gates")
        # Diagnostic-only route: retain the best complete candidate so the
        # caller can inspect *why* it failed, but its ``accepted`` flag remains
        # false and it must not be used as a validated vertical field.
        accepted = score.loc[score["complete_fixed_holdout_coverage"]].copy()
        if accepted.empty:
            raise RuntimeError("No local spatial vertical model achieved complete fixed-fold coverage")
    selected_index = accepted.sort_values(
        ["mean_nlpd", "rmse_mm", "config_index", "length_scale_km", "nugget_mm"]
    ).index[0]
    score["selected"] = False
    score.loc[selected_index, "selected"] = True
    chosen = score.loc[selected_index]
    config = configs[int(chosen["config_index"])]
    model = LocalVerticalModel(
        family=str(chosen["family"]),  # type: ignore[arg-type]
        length_scale_km=float(chosen["length_scale_km"]),
        nugget_mm=float(chosen["nugget_mm"]),
        sill_mm2=float(
            np.median(
                [
                    estimate_interval_sill_mm2(
                        table["up_mm"].to_numpy(float),
                        table["sigma_up_mm"].to_numpy(float),
                    )
                    for table in tables
                ]
            )
        ),
        config=config,
        validation_rmse_mm=float(chosen["rmse_mm"]),
        validation_nlpd=float(chosen["mean_nlpd"]),
        validation_coverage90=float(chosen["coverage90"]),
        baseline_rmse_mm=float(chosen["baseline_rmse_mm"]),
        baseline_nlpd=float(chosen["baseline_mean_nlpd"]),
        bootstrap_delta_nlpd_lower95=float(chosen["bootstrap_delta_nlpd_lower95"]),
    )
    return model, score.sort_values(["accepted", "mean_nlpd", "rmse_mm"], ascending=[False, True, True]).reset_index(drop=True), pd.DataFrame(prediction_rows)


def evaluate_local_vertical_model(
    interval_tables: Iterable[pd.DataFrame],
    model: LocalVerticalModel,
    *,
    rmse_relative_tolerance: float = 0.02,
    coverage_bounds: tuple[float, float] = (0.75, 0.99),
    uncertainty_scale: float = 1.0,
) -> tuple[dict[str, float | int | bool], pd.DataFrame]:
    """Evaluate a fixed local specification on independent time intervals.

    The covariance family, length scale, nugget, and support configuration are
    fixed by ``model``. For every interval and held-out station, covariance
    amplitude is re-estimated from the training stations only using
    :func:`estimate_interval_sill_mm2`. The same interval-adaptive rule is used
    later when all stations are available for pixel prediction.
    """

    tables = [
        table.copy().sort_values("station").reset_index(drop=True)
        for table in interval_tables
    ]
    if not tables:
        raise ValueError("At least one independent interval table is required")
    station_order = tuple(tables[0]["station"])
    if any(tuple(table["station"]) != station_order for table in tables):
        raise ValueError("Independent intervals must retain one station order")
    required = {
        "station",
        "up_mm",
        "sigma_up_mm",
        "east_km",
        "north_km",
    }
    for table in tables:
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"Interval table is missing {sorted(missing)}")
    station_xy = tables[0][["east_km", "north_km"]].to_numpy(float)
    holdouts = _interior_holdout_indices(station_xy)
    rows: list[dict[str, object]] = []
    interval_delta: dict[int, list[float]] = {}
    complete = True
    for interval_index, table in enumerate(tables):
        values = table["up_mm"].to_numpy(float)
        sigma = table["sigma_up_mm"].to_numpy(float)
        for holdout in holdouts:
            train = np.arange(len(values)) != holdout
            train_xy = station_xy[train]
            train_values = values[train]
            train_sigma = sigma[train]
            support = adaptive_local_support(
                station_xy[holdout], train_xy, model.config
            )
            if support is None:
                complete = False
                continue
            sill = estimate_interval_sill_mm2(train_values, train_sigma)
            prediction, prediction_sigma = _predict_with_support(
                model.family,
                support,
                train_xy,
                train_values,
                train_sigma,
                station_xy[holdout],
                length_scale_km=model.length_scale_km,
                nugget_mm=model.nugget_mm,
                sill_mm2=sill,
            )
            base_mean, base_sigma = _local_constant_predict(
                train_values[support.indices], train_sigma[support.indices]
            )
            raw_total_sigma = math.hypot(
                max(prediction_sigma, 1.0e-3), sigma[holdout]
            )
            total_sigma = float(uncertainty_scale) * raw_total_sigma
            total_base_sigma = math.hypot(
                max(base_sigma, 1.0e-3), sigma[holdout]
            )
            residual = float(values[holdout] - prediction)
            baseline_residual = float(values[holdout] - base_mean)
            nlpd = 0.5 * (
                math.log(2.0 * math.pi * total_sigma**2)
                + (residual / total_sigma) ** 2
            )
            baseline_nlpd = 0.5 * (
                math.log(2.0 * math.pi * total_base_sigma**2)
                + (baseline_residual / total_base_sigma) ** 2
            )
            interval_delta.setdefault(interval_index, []).append(
                baseline_nlpd - nlpd
            )
            rows.append(
                {
                    "interval_index": interval_index,
                    "holdout_station": station_order[holdout],
                    "observed_mm": float(values[holdout]),
                    "predicted_mm": prediction,
                    "predictive_sigma_mm": total_sigma,
                    "raw_predictive_sigma_mm": raw_total_sigma,
                    "residual_mm": residual,
                    "baseline_predicted_mm": base_mean,
                    "baseline_predictive_sigma_mm": total_base_sigma,
                    "baseline_residual_mm": baseline_residual,
                    "nlpd": nlpd,
                    "baseline_nlpd": baseline_nlpd,
                    "covered_90": abs(residual)
                    <= 1.6448536269514722 * total_sigma,
                    "support_count": len(support.indices),
                    "support_radius_km": support.radius_km,
                    "occupied_sector_count": support.occupied_sector_count,
                    "train_only_sill_mm2": sill,
                }
            )
    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise RuntimeError("The fixed model produced no independent predictions")
    interval_mean_delta = np.asarray(
        [np.mean(value) for value in interval_delta.values()], dtype=float
    )
    rmse = float(
        np.sqrt(np.mean(np.square(predictions["residual_mm"].to_numpy(float))))
    )
    baseline_rmse = float(
        np.sqrt(
            np.mean(
                np.square(predictions["baseline_residual_mm"].to_numpy(float))
            )
        )
    )
    mean_nlpd = float(predictions["nlpd"].mean())
    baseline_mean_nlpd = float(predictions["baseline_nlpd"].mean())
    coverage = float(predictions["covered_90"].mean())
    lower95 = _bootstrap_lower95(interval_mean_delta, seed=640013)
    accepted = bool(
        complete
        and rmse <= baseline_rmse * (1.0 + float(rmse_relative_tolerance))
        and mean_nlpd < baseline_mean_nlpd
        and lower95 > 0.0
        and coverage_bounds[0] <= coverage <= coverage_bounds[1]
    )
    summary: dict[str, float | int | bool] = {
        "accepted": accepted,
        "complete_fixed_holdout_coverage": bool(complete),
        "n_predictions": int(len(predictions)),
        "n_interior_holdouts": int(len(holdouts)),
        "n_intervals": int(len(tables)),
        "rmse_mm": rmse,
        "baseline_rmse_mm": baseline_rmse,
        "mean_nlpd": mean_nlpd,
        "baseline_mean_nlpd": baseline_mean_nlpd,
        "mean_delta_nlpd": baseline_mean_nlpd - mean_nlpd,
        "bootstrap_delta_nlpd_lower95": lower95,
        "coverage90": coverage,
        "rmse_relative_tolerance": float(rmse_relative_tolerance),
        "coverage_lower_bound": float(coverage_bounds[0]),
        "coverage_upper_bound": float(coverage_bounds[1]),
        "uncertainty_scale": float(uncertainty_scale),
    }
    return summary, predictions


def calibrate_uncertainty_scale(
    predictions: pd.DataFrame,
    *,
    nominal_coverage: float = 0.90,
    minimum_scale: float = 0.25,
    maximum_scale: float = 4.0,
) -> float:
    """Calibrate one predictive-sigma multiplier from training controls.

    ``predictions`` must contain ``observed_mm``, ``predicted_mm``, and
    ``predictive_sigma_mm`` from the calibration period only. The multiplier
    makes the empirical central interval attain ``nominal_coverage`` and is
    then held fixed for independent temporal evaluation.
    """

    if not 0.5 < nominal_coverage < 1.0:
        raise ValueError("nominal_coverage must lie between 0.5 and 1")
    required = {"observed_mm", "predicted_mm", "predictive_sigma_mm"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing {sorted(missing)}")
    observed = predictions["observed_mm"].to_numpy(float)
    predicted = predictions["predicted_mm"].to_numpy(float)
    sigma = predictions["predictive_sigma_mm"].to_numpy(float)
    finite = np.isfinite(observed) & np.isfinite(predicted) & np.isfinite(sigma)
    standardized = np.abs(observed[finite] - predicted[finite]) / np.maximum(
        sigma[finite], 1.0e-6
    )
    if len(standardized) < 20:
        raise ValueError("At least 20 calibration predictions are required")
    # Central 90% Gaussian interval; fixed to avoid a scipy.stats dependency
    # in this low-level module.
    gaussian_half_width = 1.6448536269514722
    scale = float(
        np.quantile(standardized, nominal_coverage) / gaussian_half_width
    )
    return float(np.clip(scale, minimum_scale, maximum_scale))
