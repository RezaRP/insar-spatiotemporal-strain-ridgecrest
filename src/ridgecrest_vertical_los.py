"""GNSS vertical-field prediction and vertical-to-LOS correction utilities.

This module implements the first, deliberately limited phase of a Ridgecrest
GNSS--InSAR experiment:

1. extract GNSS ENU increments at actual SAR acquisition times;
2. estimate a continuous vertical-increment field from a fixed local GNSS
   neighbourhood using cross-validated spatial predictors; and
3. project that vertical field into each LiCSAR LOS geometry as a sensitivity
   correction, promoting it to an HLOS product only if the vertical field
   passes out-of-sample spatial-resolution validation.

It does *not* derive east/north displacement or strain.  Those require a
separate, conditioned two-track inversion after the diagnostics produced here
have been reviewed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from pyproj import Transformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.linear_model import HuberRegressor

WAVELENGTH_MM = 55.46576
PHASE_TO_LOS_MM = -WAVELENGTH_MM / (4.0 * np.pi)
EARTH_RADIUS_KM = 6371.0088

COMPONENT_COLUMNS: Mapping[str, tuple[str, str]] = {
    "east": ("__east(m)", "sig_e(m)"),
    "north": ("_north(m)", "sig_n(m)"),
    "up": ("____up(m)", "sig_u(m)"),
}


@dataclass(frozen=True)
class Raster:
    """A georeferenced, referenced LOS raster in millimetres."""

    values_mm: np.ndarray
    coherence: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    valid: np.ndarray
    ramp_coefficients: np.ndarray
    ramp_scale_mm: float
    reference_offset_mm: float
    label: str


@dataclass(frozen=True)
class SpatialModel:
    """A leave-one-station-out selected vertical-field model."""

    method: str
    length_scale_km: float | None
    nugget_mm: float
    sill_mm2: float
    loo_rmse_mm: float
    loo_mae_mm: float
    loo_nlpd: float
    loo_standardized_rms: float


@dataclass(frozen=True)
class LocalKrigingModel:
    """Cross-validated moving-neighbourhood ordinary-Kriging configuration.

    This model has no planar or polynomial trend.  Every prediction uses all
    GNSS stations that fall inside its fixed-distance local search radius, with
    ordinary-Kriging weights obtained from the covariance model.  Stations are
    therefore not reduced to an arbitrary fixed set around one reference site.
    """

    length_scale_km: float
    nugget_mm: float
    search_radius_km: float
    min_stations: int
    sill_mm2: float
    loo_rmse_mm: float
    loo_mae_mm: float
    loo_nlpd: float
    loo_standardized_rms: float
    loo_coverage: float


def haversine_km(
    latitude: np.ndarray | float,
    longitude: np.ndarray | float,
    latitude0: float,
    longitude0: float,
) -> np.ndarray:
    """Great-circle distance to one point, in kilometres."""
    lat = np.deg2rad(np.asarray(latitude, dtype=float))
    lon = np.deg2rad(np.asarray(longitude, dtype=float))
    lat0 = math.radians(latitude0)
    lon0 = math.radians(longitude0)
    a = (
        np.sin((lat - lat0) / 2.0) ** 2
        + np.cos(lat) * math.cos(lat0) * np.sin((lon - lon0) / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def geotiff_axes(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Return pixel-centre latitude and longitude axes from a north-up TIFF."""
    scale = image.tag_v2[33550]
    tie = image.tag_v2[33922]
    width, height = image.size
    longitude = tie[3] + (np.arange(width) + 0.5) * scale[0]
    latitude = tie[4] - (np.arange(height) + 0.5) * scale[1]
    return latitude.astype(float), longitude.astype(float)


def load_tenv3(path: Path) -> pd.DataFrame:
    """Read one NGL ``.tenv3`` time series and retain valid daily records.

    NGL position columns are stored in metres.  The returned ``date`` column
    is a nominal UTC day label; it is not an intraday GNSS epoch.
    """
    data = pd.read_csv(path, sep=r"\s+", engine="python")
    required = {
        "YYMMMDD",
        "_latitude(deg)",
        "_longitude(deg)",
        *(column for pair in COMPONENT_COLUMNS.values() for column in pair),
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"{path.name} is missing tenv3 columns: {missing}")
    data = data.copy()
    data["date"] = pd.to_datetime(
        data["YYMMMDD"].astype(str), format="%y%b%d", errors="coerce"
    )
    invalid_dates = int(data["date"].isna().sum())
    data = data.loc[data["date"].notna()].copy()
    for value, sigma in COMPONENT_COLUMNS.values():
        data[value] = pd.to_numeric(data[value], errors="coerce")
        data[sigma] = pd.to_numeric(data[sigma], errors="coerce")
    data["station"] = path.stem.upper()
    data.attrs["invalid_date_rows"] = invalid_dates
    return data.sort_values("date").reset_index(drop=True)


def load_gnss_network(
    root: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load all NGL files and return histories plus one-row station metadata."""
    paths = sorted(root.glob("*.tenv3"))
    if not paths:
        raise FileNotFoundError(f"No .tenv3 files found in {root}")
    histories: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    for path in paths:
        history = load_tenv3(path)
        station = path.stem.upper()
        histories[station] = history
        rows.append(
            {
                "station": station,
                "latitude": float(np.nanmedian(history["_latitude(deg)"])),
                "longitude": float(np.nanmedian(history["_longitude(deg)"])),
                "invalid_date_rows": int(history.attrs["invalid_date_rows"]),
                "record_count": len(history),
            }
        )
    metadata = pd.DataFrame(rows).sort_values("station").reset_index(drop=True)
    return histories, metadata


def select_station_circle(
    metadata: pd.DataFrame,
    *,
    event_latitude: float,
    event_longitude: float,
    station_count: int = 10,
    margin_km: float = 0.20,
) -> tuple[pd.Series, float, pd.DataFrame]:
    """Fix a station circle from geometry alone, before inspecting motion."""
    if station_count < 3:
        raise ValueError("At least three stations are required")
    if len(metadata) < station_count:
        raise ValueError("The GNSS network contains fewer requested stations")
    work = metadata.copy()
    work["distance_to_event_km"] = haversine_km(
        work["latitude"], work["longitude"], event_latitude, event_longitude
    )
    centre = work.loc[work["distance_to_event_km"].idxmin()].copy()
    work["distance_to_centre_km"] = haversine_km(
        work["latitude"], work["longitude"],
        float(centre["latitude"]), float(centre["longitude"]),
    )
    selected = (
        work.sort_values(["distance_to_centre_km", "station"])
        .iloc[:station_count]
        .copy()
        .reset_index(drop=True)
    )
    radius_km = float(selected["distance_to_centre_km"].max() + margin_km)
    if int((work["distance_to_centre_km"] <= radius_km).sum()) != station_count:
        raise RuntimeError("Station radius did not select exactly the requested count")
    return centre, radius_km, selected


def choose_reference_station(
    stations: pd.DataFrame,
    *,
    event_points: Sequence[tuple[float, float]],
) -> pd.Series:
    """Choose the geometrically most distant circle station as map reference.

    The criterion deliberately depends only on station and earthquake geometry,
    not measured displacement.  The selected station is subsequently checked
    for coherence in every interferogram.
    """
    work = stations.copy()
    clearances = [
        haversine_km(work["latitude"], work["longitude"], lat, lon)
        for lat, lon in event_points
    ]
    work["event_clearance_km"] = np.minimum.reduce(clearances)
    return (
        work.sort_values(
            ["event_clearance_km", "distance_to_centre_km", "station"],
            ascending=[False, False, True],
        )
        .iloc[0]
        .copy()
    )


def _robust_daily_noise_m(
    history: pd.DataFrame,
    value_column: str,
    *,
    end: pd.Timestamp,
    floor_m: float,
) -> float:
    """Estimate pre-event one-day scatter from first differences."""
    use = (history["date"] >= end - pd.Timedelta(days=45)) & (
        history["date"] < end.normalize()
    )
    values = history.loc[use, value_column].to_numpy(float)
    values = values[np.isfinite(values)]
    if len(values) < 6:
        return floor_m
    differences = np.diff(values)
    scale = 1.4826 * np.median(np.abs(differences - np.median(differences)))
    return float(max(floor_m, scale / np.sqrt(2.0)))


def _fit_pre_event_trend(
    history: pd.DataFrame,
    value_column: str,
    sigma_column: str,
    *,
    target: pd.Timestamp,
    event_time: pd.Timestamp,
    pre_event_days: int,
    floor_m: float,
) -> tuple[float, float, str]:
    """Predict a pre-event endpoint without crossing the coseismic step."""
    last_safe_day = event_time.normalize() - pd.Timedelta(days=1)
    start_day = last_safe_day - pd.Timedelta(days=pre_event_days)
    use = (history["date"] >= start_day) & (history["date"] <= last_safe_day)
    values = history.loc[use, value_column].to_numpy(float)
    sigmas = history.loc[use, sigma_column].to_numpy(float)
    dates = pd.DatetimeIndex(history.loc[use, "date"])
    finite = np.isfinite(values) & np.isfinite(sigmas)
    if int(finite.sum()) < 7:
        raise ValueError("Too few pre-event GNSS records to predict SAR endpoint")
    values = values[finite]
    sigmas = np.maximum(sigmas[finite], floor_m)
    offsets = (dates[finite] - target).total_seconds().to_numpy(float) / 86400.0
    design = np.column_stack([np.ones_like(offsets), offsets])
    weights = 1.0 / np.square(sigmas)
    normal = design.T @ (weights[:, None] * design)
    covariance = np.linalg.pinv(normal)
    coefficients = covariance @ (design.T @ (weights * values))
    residuals = values - design @ coefficients
    robust_scale = 1.4826 * np.median(
        np.abs(residuals - np.median(residuals))
    )
    prediction_sigma = float(
        np.sqrt(max(covariance[0, 0], 0.0) + max(robust_scale, floor_m) ** 2)
    )
    return float(coefficients[0]), prediction_sigma, "pre_event_weighted_trend"


def _fit_post_event_trend(
    history: pd.DataFrame,
    value_column: str,
    sigma_column: str,
    *,
    target: pd.Timestamp,
    last_event_time: pd.Timestamp,
    post_event_days: int,
    floor_m: float,
) -> tuple[float, float, str]:
    """Predict a post-event SAR endpoint from its local daily GNSS regime.

    This is used when a daily series ends on the SAR endpoint date, or when
    using the following day's daily coordinate would imply a poorly defined
    intraday interpolation.  The residual scale explicitly carries the
    unresolved daily/postseismic variability into the endpoint uncertainty.
    """
    first_safe_day = last_event_time.normalize() + pd.Timedelta(days=2)
    start_day = max(first_safe_day, target.normalize() - pd.Timedelta(days=post_event_days))
    end_day = min(target.normalize(), pd.Timestamp(history["date"].max()))
    use = (history["date"] >= start_day) & (history["date"] <= end_day)
    values = history.loc[use, value_column].to_numpy(float)
    sigmas = history.loc[use, sigma_column].to_numpy(float)
    dates = pd.DatetimeIndex(history.loc[use, "date"])
    finite = np.isfinite(values) & np.isfinite(sigmas)
    if int(finite.sum()) < 5:
        raise ValueError("Too few post-event GNSS records to predict SAR endpoint")
    values = values[finite]
    sigmas = np.maximum(sigmas[finite], floor_m)
    offsets = (dates[finite] - target).total_seconds().to_numpy(float) / 86400.0
    design = np.column_stack([np.ones_like(offsets), offsets])
    weights = 1.0 / np.square(sigmas)
    normal = design.T @ (weights[:, None] * design)
    covariance = np.linalg.pinv(normal)
    coefficients = covariance @ (design.T @ (weights * values))
    residuals = values - design @ coefficients
    robust_scale = 1.4826 * np.median(
        np.abs(residuals - np.median(residuals))
    )
    prediction_sigma = float(
        np.sqrt(max(covariance[0, 0], 0.0) + max(robust_scale, floor_m) ** 2)
    )
    return float(coefficients[0]), prediction_sigma, "post_event_weighted_trend"


def _interpolate_daily_endpoint(
    history: pd.DataFrame,
    value_column: str,
    sigma_column: str,
    *,
    target: pd.Timestamp,
    event_times: Sequence[pd.Timestamp],
    floor_m: float,
) -> tuple[float, float, str]:
    """Linearly interpolate within one uninterrupted daily GNSS regime."""
    times = pd.DatetimeIndex(history["date"])
    values = history[value_column].to_numpy(float)
    sigmas = history[sigma_column].to_numpy(float)
    finite = np.isfinite(values) & np.isfinite(sigmas)
    times = times[finite]
    values = values[finite]
    sigmas = np.maximum(sigmas[finite], floor_m)
    if target < times.min() or target > times.max():
        raise ValueError(f"GNSS data do not bracket {target.isoformat()}")
    # Compare Timestamp/Timedelta objects directly.  Pandas may store a
    # DatetimeIndex in a non-nanosecond native unit (for example microseconds),
    # while ``Timestamp.value`` is nanoseconds; mixing those integer arrays
    # makes a valid historical endpoint appear unbracketed.
    right = int(times.searchsorted(target, side="left"))
    if right < len(times) and times[right] == target:
        return float(values[right]), float(sigmas[right]), "daily_solution"
    if right == 0 or right == len(times):
        raise ValueError(f"GNSS data do not bracket {target.isoformat()}")
    left = right - 1
    for event_time in event_times:
        if times[left] < event_time < times[right]:
            raise ValueError(
                "Refusing to interpolate a daily GNSS coordinate across a "
                f"coseismic discontinuity at {event_time.isoformat()}"
            )
    fraction = float((target - times[left]) / (times[right] - times[left]))
    value = (1.0 - fraction) * values[left] + fraction * values[right]
    sigma = math.sqrt(
        (1.0 - fraction) ** 2 * sigmas[left] ** 2
        + fraction**2 * sigmas[right] ** 2
    )
    return float(value), float(max(sigma, floor_m)), "within_regime_daily_linear"


def sample_gnss_endpoint(
    history: pd.DataFrame,
    component: str,
    *,
    target: pd.Timestamp,
    event_times: Sequence[pd.Timestamp],
    pre_event_days: int = 30,
    post_event_days: int = 8,
) -> tuple[float, float, str]:
    """Sample a GNSS component at a SAR time without crossing event steps.

    Daily NGL positions cannot resolve the sub-day M6.4 offset.  Any target on
    the event day before the origin time is therefore extrapolated from the
    previous pre-event segment, rather than interpolated through the rupture.
    """
    if component not in COMPONENT_COLUMNS:
        raise KeyError(component)
    value_column, sigma_column = COMPONENT_COLUMNS[component]
    floor_m = 0.002 if component in {"east", "north"} else 0.005
    target = pd.Timestamp(target).tz_localize(None)
    events = [pd.Timestamp(value).tz_localize(None) for value in event_times]
    prior_events = [value for value in events if target < value]
    same_day_prior = [
        value
        for value in prior_events
        if target.normalize() == value.normalize()
    ]
    if same_day_prior:
        return _fit_pre_event_trend(
            history,
            value_column,
            sigma_column,
            target=target,
            event_time=min(same_day_prior),
            pre_event_days=pre_event_days,
            floor_m=floor_m,
        )
    past_events = [value for value in events if target > value]
    if past_events:
        return _fit_post_event_trend(
            history,
            value_column,
            sigma_column,
            target=target,
            last_event_time=max(past_events),
            post_event_days=post_event_days,
            floor_m=floor_m,
        )
    return _interpolate_daily_endpoint(
        history,
        value_column,
        sigma_column,
        target=target,
        event_times=events,
        floor_m=floor_m,
    )


def gnss_interval_table(
    histories: Mapping[str, pd.DataFrame],
    stations: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    event_times: Sequence[pd.Timestamp],
    strict: bool = True,
) -> pd.DataFrame:
    """Return exact-track-epoch GNSS ENU increments in millimetres.

    ``strict=True`` is required for the fixed interpolation circle: an absent
    endpoint makes that circle unusable.  The independent all-network sign
    audit may use ``strict=False``; unavailable stations are then listed in
    ``DataFrame.attrs['skipped_stations']`` rather than changing the selected
    ten-station field.
    """
    if end <= start:
        raise ValueError("The interval end must follow its start")
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    for station_row in stations.itertuples(index=False):
        station = str(station_row.station)
        history = histories[station]
        row: dict[str, object] = {
            "station": station,
            "latitude": float(station_row.latitude),
            "longitude": float(station_row.longitude),
            "start_utc": pd.Timestamp(start),
            "end_utc": pd.Timestamp(end),
        }
        try:
            for component in ("east", "north", "up"):
                start_value, start_sigma, start_method = sample_gnss_endpoint(
                    history, component, target=start, event_times=event_times
                )
                end_value, end_sigma, end_method = sample_gnss_endpoint(
                    history, component, target=end, event_times=event_times
                )
                row[f"{component}_mm"] = (end_value - start_value) * 1000.0
                row[f"sigma_{component}_mm"] = math.sqrt(
                    start_sigma**2 + end_sigma**2
                ) * 1000.0
                row[f"{component}_start_method"] = start_method
                row[f"{component}_end_method"] = end_method
        except ValueError as exc:
            if strict:
                raise
            skipped.append({"station": station, "reason": str(exc)})
            continue
        rows.append(row)
    if not rows:
        raise ValueError("No GNSS stations have complete endpoint coverage")
    table = pd.DataFrame(rows).sort_values("station").reset_index(drop=True)
    table.attrs["skipped_stations"] = skipped
    return table


def to_utm11_km(
    longitude: np.ndarray | float,
    latitude: np.ndarray | float,
) -> np.ndarray:
    """Project WGS84 longitude/latitude to UTM zone 11, in kilometres."""
    transformer = Transformer.from_crs(4326, 32611, always_xy=True)
    east_m, north_m = transformer.transform(longitude, latitude)
    return np.column_stack(
        [np.asarray(east_m, dtype=float).ravel() / 1000.0,
         np.asarray(north_m, dtype=float).ravel() / 1000.0]
    )


def _pairwise_distance_km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(
        np.square(a[:, None, 0] - b[None, :, 0])
        + np.square(a[:, None, 1] - b[None, :, 1])
    )


def _ordinary_kriging_predict(
    train_xy: np.ndarray,
    train_values: np.ndarray,
    train_sigma: np.ndarray,
    target_xy: np.ndarray,
    *,
    length_scale_km: float,
    sill_mm2: float,
    nugget_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary kriging with exponential covariance and measurement nugget."""
    train_xy = np.asarray(train_xy, dtype=float)
    targets = np.asarray(target_xy, dtype=float)
    distances = _pairwise_distance_km(train_xy, train_xy)
    covariance = sill_mm2 * np.exp(-distances / length_scale_km)
    covariance += np.diag(np.square(train_sigma) + nugget_mm**2)
    n = len(train_values)
    system = np.empty((n + 1, n + 1), dtype=float)
    system[:n, :n] = covariance
    system[:n, n] = 1.0
    system[n, :n] = 1.0
    system[n, n] = 0.0
    inverse = np.linalg.pinv(system)
    cross = sill_mm2 * np.exp(
        -_pairwise_distance_km(train_xy, targets) / length_scale_km
    )
    rhs = np.vstack([cross, np.ones((1, len(targets)))])
    solution = inverse @ rhs
    weights = solution[:n]
    lagrange = solution[n]
    mean = np.asarray(train_values, dtype=float) @ weights
    # This covariance-form expression is the latent-field prediction variance.
    variance = sill_mm2 - np.sum(weights * cross, axis=0) - lagrange
    return mean, np.sqrt(np.maximum(variance, 0.0))


def _local_ordinary_kriging_single(
    train_xy: np.ndarray,
    train_values: np.ndarray,
    train_sigma: np.ndarray,
    target_xy: np.ndarray,
    *,
    length_scale_km: float,
    sill_mm2: float,
    nugget_mm: float,
    search_radius_km: float,
    min_stations: int,
) -> tuple[float, float, int]:
    """Predict one target from every station in its local search neighbourhood.

    The method intentionally uses a radius rather than a fixed nearest-station
    count.  A target with insufficient nearby observations is returned as
    unresolved rather than extrapolated from a remote, project-wide trend.
    """
    xy = np.asarray(train_xy, dtype=float)
    point = np.asarray(target_xy, dtype=float).reshape(1, 2)
    distance = np.linalg.norm(xy - point, axis=1)
    nearby = np.flatnonzero(distance <= float(search_radius_km))
    if len(nearby) < int(min_stations):
        return float("nan"), float("nan"), int(len(nearby))
    mean, sigma = _ordinary_kriging_predict(
        xy[nearby],
        np.asarray(train_values, dtype=float)[nearby],
        np.asarray(train_sigma, dtype=float)[nearby],
        point,
        length_scale_km=float(length_scale_km),
        sill_mm2=float(sill_mm2),
        nugget_mm=float(nugget_mm),
    )
    return float(mean[0]), float(sigma[0]), int(len(nearby))


def select_local_ordinary_kriging_model(
    xy_km: np.ndarray,
    values_mm: np.ndarray,
    sigma_mm: np.ndarray,
    *,
    search_radii_km: Sequence[float] = (50.0, 70.0, 90.0, 120.0),
    length_scales_km: Sequence[float] = (15.0, 25.0, 40.0, 60.0),
    nuggets_mm: Sequence[float] = (0.0, 2.0, 5.0, 10.0),
    min_stations: int = 6,
) -> tuple[LocalKrigingModel, pd.DataFrame, pd.DataFrame]:
    """Select an all-station, local ordinary-Kriging configuration by LOO CV.

    Each held-out GNSS station is predicted only from *all other* stations in
    its trial local radius.  The score table retains neighbourhood coverage and
    station count, so an apparently low error obtained by silently dropping
    difficult stations cannot be selected.  No planar trend is fitted.
    """
    xy = np.asarray(xy_km, dtype=float)
    values = np.asarray(values_mm, dtype=float)
    sigmas = np.asarray(sigma_mm, dtype=float)
    if xy.shape != (len(values), 2) or len(values) < int(min_stations) + 1:
        raise ValueError("Need at least min_stations + 1 finite GNSS observations")
    if not (np.isfinite(xy).all() and np.isfinite(values).all() and np.isfinite(sigmas).all()):
        raise ValueError("Local-Kriging inputs contain non-finite values")
    if any(float(radius) <= 0.0 for radius in search_radii_km):
        raise ValueError("All local search radii must be positive")
    if any(float(length) <= 0.0 for length in length_scales_km):
        raise ValueError("All covariance length scales must be positive")

    sill = float(max(np.var(values, ddof=1), 1.0))
    score_rows: list[dict[str, float | int | bool]] = []
    prediction_rows: list[dict[str, float | int | bool]] = []
    for radius in search_radii_km:
        for length_scale in length_scales_km:
            for nugget in nuggets_mm:
                observed: list[float] = []
                predicted: list[float] = []
                total_sigma: list[float] = []
                local_counts: list[int] = []
                for holdout in range(len(values)):
                    keep = np.arange(len(values)) != holdout
                    mean, prediction_sigma, count = _local_ordinary_kriging_single(
                        xy[keep], values[keep], sigmas[keep], xy[holdout],
                        length_scale_km=float(length_scale),
                        sill_mm2=sill,
                        nugget_mm=float(nugget),
                        search_radius_km=float(radius),
                        min_stations=int(min_stations),
                    )
                    valid = bool(np.isfinite(mean) and np.isfinite(prediction_sigma))
                    prediction_rows.append(
                        {
                            "search_radius_km": float(radius),
                            "length_scale_km": float(length_scale),
                            "nugget_mm": float(nugget),
                            "holdout_index": int(holdout),
                            "observed_mm": float(values[holdout]),
                            "predicted_mm": float(mean),
                            "predictive_sigma_mm": float(prediction_sigma),
                            "nearby_station_count": int(count),
                            "predicted": valid,
                        }
                    )
                    if valid:
                        observed.append(float(values[holdout]))
                        predicted.append(float(mean))
                        total_sigma.append(
                            math.hypot(max(float(prediction_sigma), 1.0e-3), float(sigmas[holdout]))
                        )
                        local_counts.append(int(count))
                coverage = float(len(observed) / len(values))
                if observed:
                    residual = np.asarray(observed) - np.asarray(predicted)
                    total = np.asarray(total_sigma)
                    standardized = residual / total
                    rmse = float(np.sqrt(np.mean(np.square(residual))))
                    mae = float(np.mean(np.abs(residual)))
                    nlpd = float(np.mean(0.5 * (np.log(2.0 * np.pi * np.square(total)) + np.square(standardized))))
                    standardized_rms = float(np.sqrt(np.mean(np.square(standardized))))
                    median_count = float(np.median(local_counts))
                else:
                    rmse = mae = nlpd = standardized_rms = float("inf")
                    median_count = 0.0
                score_rows.append(
                    {
                        "search_radius_km": float(radius),
                        "length_scale_km": float(length_scale),
                        "nugget_mm": float(nugget),
                        "min_stations": int(min_stations),
                        "loo_coverage": coverage,
                        "loo_rmse_mm": rmse,
                        "loo_mae_mm": mae,
                        "loo_nlpd": nlpd,
                        "loo_standardized_rms": standardized_rms,
                        "median_nearby_station_count": median_count,
                    }
                )
    score_table = pd.DataFrame(score_rows)
    # Full held-out coverage is required: a model cannot improve its score by
    # declining to predict the difficult parts of the network.
    score_table["eligible"] = score_table["loo_coverage"].eq(1.0)
    eligible = score_table.loc[score_table["eligible"]].copy()
    if eligible.empty:
        raise RuntimeError("No local-Kriging candidate predicted every held-out GNSS station")
    selected_index = eligible.sort_values(
        ["loo_nlpd", "loo_rmse_mm", "search_radius_km", "length_scale_km", "nugget_mm"]
    ).index[0]
    score_table["selected"] = False
    score_table.loc[selected_index, "selected"] = True
    selected = score_table.loc[selected_index]
    model = LocalKrigingModel(
        length_scale_km=float(selected["length_scale_km"]),
        nugget_mm=float(selected["nugget_mm"]),
        search_radius_km=float(selected["search_radius_km"]),
        min_stations=int(selected["min_stations"]),
        sill_mm2=sill,
        loo_rmse_mm=float(selected["loo_rmse_mm"]),
        loo_mae_mm=float(selected["loo_mae_mm"]),
        loo_nlpd=float(selected["loo_nlpd"]),
        loo_standardized_rms=float(selected["loo_standardized_rms"]),
        loo_coverage=float(selected["loo_coverage"]),
    )
    return model, score_table.sort_values(
        ["eligible", "loo_nlpd", "loo_rmse_mm"], ascending=[False, True, True]
    ).reset_index(drop=True), pd.DataFrame(prediction_rows)


def predict_local_ordinary_kriging_field(
    model: LocalKrigingModel,
    xy_km: np.ndarray,
    values_mm: np.ndarray,
    sigma_mm: np.ndarray,
    target_xy_km: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict a local-Kriging field and retain contributing-station counts.

    ``xy_km`` contains the complete GNSS network.  At every target the method
    considers every station in that target's spatial neighbourhood; it never
    selects a fixed circle around a GNSS station or fits a map-wide plane.
    """
    targets = np.asarray(target_xy_km, dtype=float)
    mean = np.full(len(targets), np.nan, dtype=float)
    sigma = np.full(len(targets), np.nan, dtype=float)
    count = np.zeros(len(targets), dtype=int)
    for index, target in enumerate(targets):
        mean[index], sigma[index], count[index] = _local_ordinary_kriging_single(
            xy_km, values_mm, sigma_mm, target,
            length_scale_km=model.length_scale_km,
            sill_mm2=model.sill_mm2,
            nugget_mm=model.nugget_mm,
            search_radius_km=model.search_radius_km,
            min_stations=model.min_stations,
        )
    return mean, sigma, count


def _weighted_constant_predict(
    values: np.ndarray,
    sigmas: np.ndarray,
) -> tuple[float, float]:
    weights = 1.0 / np.square(np.maximum(sigmas, 1.0e-3))
    mean = float(np.sum(weights * values) / np.sum(weights))
    scatter = float(
        np.sqrt(np.sum(weights * np.square(values - mean)) / np.sum(weights))
    )
    sigma = math.sqrt(1.0 / np.sum(weights) + scatter**2)
    return mean, sigma


def _fit_gp(
    xy: np.ndarray,
    values: np.ndarray,
    sigmas: np.ndarray,
    *,
    length_scale_km: float,
    nugget_mm: float,
    sill_mm2: float,
) -> GaussianProcessRegressor:
    kernel = ConstantKernel(
        constant_value=max(sill_mm2, 1.0), constant_value_bounds="fixed"
    ) * Matern(length_scale=length_scale_km, length_scale_bounds="fixed", nu=1.5)
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=np.square(np.maximum(sigmas, 1.0e-3)) + nugget_mm**2,
        normalize_y=False,
        optimizer=None,
        random_state=640004,
    )
    model.fit(xy, values)
    return model


def select_vertical_model(
    xy_km: np.ndarray,
    values_mm: np.ndarray,
    sigma_mm: np.ndarray,
    *,
    length_scales_km: Sequence[float] = (10.0, 15.0, 20.0, 30.0, 40.0, 60.0),
    nuggets_mm: Sequence[float] = (0.0, 2.0, 5.0),
) -> tuple[SpatialModel, pd.DataFrame, pd.DataFrame]:
    """Compare constant, ordinary-kriging, and Matérn-GP vertical predictors.

    Model choice uses leave-one-station-out negative log predictive density,
    with RMSE and standardized residual diagnostics retained for reporting. A
    spatial model is selected only if its paired improvement over the constant
    field exceeds one standard error *and* lowers RMSE; this one-standard-error
    parsimony rule prevents a ten-station network from manufacturing a
    pixel-scale vertical pattern that it cannot predict out of sample.
    """
    xy = np.asarray(xy_km, dtype=float)
    values = np.asarray(values_mm, dtype=float)
    sigmas = np.asarray(sigma_mm, dtype=float)
    if xy.shape != (len(values), 2) or len(values) < 5:
        raise ValueError("At least five co-located station values are required")
    if not (np.isfinite(xy).all() and np.isfinite(values).all() and np.isfinite(sigmas).all()):
        raise ValueError("Spatial model inputs contain non-finite values")
    rows: list[dict[str, float | str | None]] = []
    prediction_rows: list[dict[str, float | int | str]] = []
    candidates: list[tuple[str, float | None, float]] = [("constant", None, 0.0)]
    candidates.extend(
        (method, length, nugget)
        for method in ("ordinary_kriging", "matern_gp")
        for length in length_scales_km
        for nugget in nuggets_mm
    )
    for method, length_scale, nugget in candidates:
        predictions: list[float] = []
        predictive_sigmas: list[float] = []
        observations: list[float] = []
        for holdout in range(len(values)):
            keep = np.arange(len(values)) != holdout
            train_xy = xy[keep]
            train_values = values[keep]
            train_sigmas = sigmas[keep]
            sill = float(max(np.var(train_values, ddof=1), 1.0))
            if method == "constant":
                mean, sigma = _weighted_constant_predict(train_values, train_sigmas)
                prediction = mean
                predictive_sigma = sigma
            elif method == "ordinary_kriging":
                predicted, predicted_sigma = _ordinary_kriging_predict(
                    train_xy,
                    train_values,
                    train_sigmas,
                    xy[holdout : holdout + 1],
                    length_scale_km=float(length_scale),
                    sill_mm2=sill,
                    nugget_mm=nugget,
                )
                prediction = float(predicted[0])
                predictive_sigma = float(predicted_sigma[0])
            else:
                model = _fit_gp(
                    train_xy,
                    train_values,
                    train_sigmas,
                    length_scale_km=float(length_scale),
                    nugget_mm=nugget,
                    sill_mm2=sill,
                )
                predicted, predicted_sigma = model.predict(
                    xy[holdout : holdout + 1], return_std=True
                )
                prediction = float(predicted[0])
                predictive_sigma = float(predicted_sigma[0])
            total_sigma = math.sqrt(
                max(predictive_sigma, 1.0e-3) ** 2 + sigmas[holdout] ** 2
            )
            predictions.append(prediction)
            predictive_sigmas.append(total_sigma)
            observations.append(float(values[holdout]))
            prediction_rows.append(
                {
                    "method": method,
                    "length_scale_km": length_scale,
                    "nugget_mm": nugget,
                    "holdout_index": holdout,
                    "observed_mm": float(values[holdout]),
                    "predicted_mm": prediction,
                    "predictive_sigma_mm": total_sigma,
                }
            )
        observed = np.asarray(observations)
        predicted = np.asarray(predictions)
        total_sigma = np.asarray(predictive_sigmas)
        residuals = observed - predicted
        standardized = residuals / total_sigma
        rows.append(
            {
                "method": method,
                "length_scale_km": length_scale,
                "nugget_mm": nugget,
                "loo_rmse_mm": float(np.sqrt(np.mean(np.square(residuals)))),
                "loo_mae_mm": float(np.mean(np.abs(residuals))),
                "loo_nlpd": float(
                    np.mean(
                        0.5
                        * (
                            np.log(2.0 * np.pi * np.square(total_sigma))
                            + np.square(standardized)
                        )
                    )
                ),
                "loo_standardized_rms": float(
                    np.sqrt(np.mean(np.square(standardized)))
                ),
            }
        )
    score_table = pd.DataFrame(rows).sort_values(
        ["loo_nlpd", "loo_rmse_mm", "method"]
    ).reset_index(drop=True)
    predictions = pd.DataFrame(prediction_rows)
    residual = predictions["observed_mm"] - predictions["predicted_mm"]
    predictions["pointwise_nlpd"] = 0.5 * (
        np.log(2.0 * np.pi * np.square(predictions["predictive_sigma_mm"]))
        + np.square(residual / predictions["predictive_sigma_mm"])
    )
    constant_score = score_table.loc[score_table["method"].eq("constant")].iloc[0]
    constant_prediction = predictions.loc[predictions["method"].eq("constant")].set_index(
        "holdout_index"
    )
    improvements: list[float] = []
    improvement_errors: list[float] = []
    rmse_gains: list[float] = []
    spatial_supported: list[bool] = []
    for candidate in score_table.itertuples(index=False):
        if candidate.method == "constant":
            improvements.append(0.0)
            improvement_errors.append(0.0)
            rmse_gains.append(0.0)
            spatial_supported.append(True)
            continue
        candidate_predictions = predictions.loc[
            predictions["method"].eq(candidate.method)
            & np.isclose(predictions["nugget_mm"], candidate.nugget_mm)
            & (
                predictions["length_scale_km"].isna()
                if pd.isna(candidate.length_scale_km)
                else np.isclose(
                    predictions["length_scale_km"], candidate.length_scale_km
                )
            )
        ].set_index("holdout_index")
        paired = constant_prediction["pointwise_nlpd"].subtract(
            candidate_predictions["pointwise_nlpd"], fill_value=np.nan
        ).dropna()
        improvement = float(paired.mean())
        improvement_error = float(
            paired.std(ddof=1) / np.sqrt(len(paired))
        ) if len(paired) > 1 else float("inf")
        rmse_gain = float(constant_score["loo_rmse_mm"] - candidate.loo_rmse_mm)
        improvements.append(improvement)
        improvement_errors.append(improvement_error)
        rmse_gains.append(rmse_gain)
        spatial_supported.append(
            improvement > improvement_error and rmse_gain > 0.0
        )
    score_table["nlpd_improvement_vs_constant"] = improvements
    score_table["nlpd_improvement_se"] = improvement_errors
    score_table["rmse_gain_vs_constant_mm"] = rmse_gains
    score_table["spatial_model_supported"] = spatial_supported
    eligible = score_table.loc[score_table["spatial_model_supported"]].copy()
    best = eligible.sort_values(["loo_nlpd", "loo_rmse_mm", "method"]).iloc[0]
    score_table["selected"] = False
    score_table.loc[best.name, "selected"] = True
    sill_all = float(max(np.var(values, ddof=1), 1.0))
    selected = SpatialModel(
        method=str(best["method"]),
        length_scale_km=(
            None
            if pd.isna(best["length_scale_km"])
            else float(best["length_scale_km"])
        ),
        nugget_mm=float(best["nugget_mm"]),
        sill_mm2=sill_all,
        loo_rmse_mm=float(best["loo_rmse_mm"]),
        loo_mae_mm=float(best["loo_mae_mm"]),
        loo_nlpd=float(best["loo_nlpd"]),
        loo_standardized_rms=float(best["loo_standardized_rms"]),
    )
    return selected, score_table, predictions


def predict_vertical_field(
    model: SpatialModel,
    xy_km: np.ndarray,
    values_mm: np.ndarray,
    sigma_mm: np.ndarray,
    target_xy_km: np.ndarray,
    *,
    chunk_size: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict a selected vertical field, including posterior standard deviation."""
    xy = np.asarray(xy_km, dtype=float)
    values = np.asarray(values_mm, dtype=float)
    sigmas = np.asarray(sigma_mm, dtype=float)
    targets = np.asarray(target_xy_km, dtype=float)
    output_mean = np.empty(len(targets), dtype=float)
    output_sigma = np.empty(len(targets), dtype=float)
    if model.method == "constant":
        mean, sigma = _weighted_constant_predict(values, sigmas)
        output_mean.fill(mean)
        output_sigma.fill(sigma)
        return output_mean, output_sigma
    if model.length_scale_km is None:
        raise ValueError("A spatial model requires a length scale")
    if model.method == "matern_gp":
        fitted = _fit_gp(
            xy,
            values,
            sigmas,
            length_scale_km=model.length_scale_km,
            nugget_mm=model.nugget_mm,
            sill_mm2=model.sill_mm2,
        )
        for begin in range(0, len(targets), chunk_size):
            end = min(begin + chunk_size, len(targets))
            output_mean[begin:end], output_sigma[begin:end] = fitted.predict(
                targets[begin:end], return_std=True
            )
        return output_mean, output_sigma
    if model.method == "ordinary_kriging":
        for begin in range(0, len(targets), chunk_size):
            end = min(begin + chunk_size, len(targets))
            mean, sigma = _ordinary_kriging_predict(
                xy,
                values,
                sigmas,
                targets[begin:end],
                length_scale_km=model.length_scale_km,
                sill_mm2=model.sill_mm2,
                nugget_mm=model.nugget_mm,
            )
            output_mean[begin:end] = mean
            output_sigma[begin:end] = sigma
        return output_mean, output_sigma
    raise ValueError(f"Unsupported vertical model: {model.method}")


def _robust_far_field_plane(
    values_mm: np.ndarray,
    valid: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    event_points: Sequence[tuple[float, float]],
    exclusion_km: float,
    stride: int = 10,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply a pre-specified robust far-field planar-ramp strategy."""
    rows = np.arange(0, len(latitude), stride)
    cols = np.arange(0, len(longitude), stride)
    sampled_valid = valid[np.ix_(rows, cols)].copy()
    lat_grid, lon_grid = np.meshgrid(latitude[rows], longitude[cols], indexing="ij")
    distances = [
        haversine_km(lat_grid, lon_grid, event_lat, event_lon)
        for event_lat, event_lon in event_points
    ]
    sampled_valid &= np.minimum.reduce(distances) >= exclusion_km
    local_rows, local_cols = np.where(sampled_valid)
    if len(local_rows) < 1000:
        raise ValueError("Insufficient fixed far-field pixels for ramp estimation")
    global_rows = rows[local_rows]
    global_cols = cols[local_cols]
    lat0 = float(np.nanmedian(latitude))
    lon0 = float(np.nanmedian(longitude))
    design = np.column_stack(
        [longitude[global_cols] - lon0, latitude[global_rows] - lat0]
    )
    model = HuberRegressor(
        epsilon=1.35, alpha=0.0, fit_intercept=True, max_iter=500
    )
    model.fit(design, values_mm[global_rows, global_cols])
    ramp = (
        float(model.intercept_)
        + float(model.coef_[0]) * (longitude[None, :] - lon0)
        + float(model.coef_[1]) * (latitude[:, None] - lat0)
    )
    residual = values_mm - ramp
    residual[~valid] = np.nan
    fit_residual = values_mm[global_rows, global_cols] - model.predict(design)
    scale = float(
        1.4826 * np.median(np.abs(fit_residual - np.median(fit_residual)))
    )
    coefficients = np.asarray(
        [model.intercept_, model.coef_[0], model.coef_[1]], dtype=float
    )
    return residual.astype(np.float32), coefficients, scale


def _disk_mask(
    latitude: np.ndarray,
    longitude: np.ndarray,
    centre_latitude: float,
    centre_longitude: float,
    radius_km: float,
) -> np.ndarray:
    lat_grid, lon_grid = np.meshgrid(latitude, longitude, indexing="ij")
    return haversine_km(
        lat_grid, lon_grid, centre_latitude, centre_longitude
    ) <= radius_km


def read_and_reference_pair(
    pair_directory: Path,
    *,
    coherence_min: float,
    event_points: Sequence[tuple[float, float]],
    ramp_exclusion_km: float,
    reference_latitude: float,
    reference_longitude: float,
    reference_radius_km: float,
) -> Raster:
    """Read one raw LiCSAR pair, remove a fixed far-field ramp, and reference it.

    Each pair is processed with exactly the same pre-declared far-field rule
    and the same reference disk.  This makes summing adjacent ascending pairs
    legitimate at the level of a relative, referenced LOS field.
    """
    pair = pair_directory.name
    unw_path = pair_directory / f"{pair}.geo.unw.tif"
    coh_path = pair_directory / f"{pair}.geo.cc.tif"
    if not unw_path.exists() or not coh_path.exists():
        raise FileNotFoundError(f"Missing raw phase or coherence file in {pair_directory}")
    image = Image.open(unw_path)
    phase = np.asarray(tifffile.imread(unw_path), dtype=np.float32)
    coherence = np.asarray(tifffile.imread(coh_path), dtype=np.float32) / 255.0
    latitude, longitude = geotiff_axes(image)
    if phase.shape != coherence.shape:
        raise ValueError(f"Phase/coherence shape mismatch in {pair_directory}")
    nodata = str(image.tag_v2.get(42113, "")).strip().lower()
    valid = np.isfinite(phase) & np.isfinite(coherence) & (coherence >= coherence_min)
    if nodata == "0":
        valid &= phase != 0.0
    residual, coefficients, scale = _robust_far_field_plane(
        phase * PHASE_TO_LOS_MM,
        valid,
        latitude,
        longitude,
        event_points=event_points,
        exclusion_km=ramp_exclusion_km,
    )
    reference_mask = _disk_mask(
        latitude,
        longitude,
        reference_latitude,
        reference_longitude,
        reference_radius_km,
    ) & valid & np.isfinite(residual)
    if int(reference_mask.sum()) < 25:
        raise ValueError(
            f"Reference disk has too few coherent pixels for {pair}; "
            "choose a valid common reference station/disk"
        )
    reference_offset = float(np.nanmedian(residual[reference_mask]))
    values = residual - reference_offset
    values[~valid] = np.nan
    return Raster(
        values_mm=values.astype(np.float32),
        coherence=coherence.astype(np.float32),
        latitude=latitude,
        longitude=longitude,
        valid=valid,
        ramp_coefficients=coefficients,
        ramp_scale_mm=scale,
        reference_offset_mm=reference_offset,
        label=pair,
    )


def sum_referenced_pairs(rasters: Sequence[Raster], *, label: str) -> Raster:
    """Sum same-grid, same-reference LOS intervals into one longer interval."""
    if not rasters:
        raise ValueError("At least one raster is required")
    first = rasters[0]
    for raster in rasters[1:]:
        if not (
            np.allclose(first.latitude, raster.latitude, rtol=0.0, atol=1.0e-10)
            and np.allclose(first.longitude, raster.longitude, rtol=0.0, atol=1.0e-10)
            and first.values_mm.shape == raster.values_mm.shape
        ):
            raise ValueError(
                "Only co-registered, common-reference rasters may be summed; "
                "resample an actually different grid before composition"
            )
    valid = np.logical_and.reduce([raster.valid for raster in rasters])
    values = np.sum([raster.values_mm for raster in rasters], axis=0, dtype=np.float64)
    values[~valid] = np.nan
    coherence = np.minimum.reduce([raster.coherence for raster in rasters])
    return Raster(
        values_mm=values.astype(np.float32),
        coherence=coherence.astype(np.float32),
        latitude=first.latitude,
        longitude=first.longitude,
        valid=valid,
        ramp_coefficients=np.vstack([raster.ramp_coefficients for raster in rasters]),
        ramp_scale_mm=float(
            np.sqrt(sum(raster.ramp_scale_mm**2 for raster in rasters))
        ),
        reference_offset_mm=float(sum(raster.reference_offset_mm for raster in rasters)),
        label=label,
    )


def load_los_vectors(
    root: Path,
    frame: str,
    *,
    expected_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and normalize LiCSAR pixel-wise E/N/U LOS unit-vector rasters."""
    arrays: list[np.ndarray] = []
    for component in ("E", "N", "U"):
        path = root / f"{frame}.geo.{component}.tif"
        if not path.exists():
            raise FileNotFoundError(path)
        arrays.append(np.asarray(tifffile.imread(path), dtype=np.float32))
    if expected_shape is not None and arrays[0].shape != expected_shape:
        raise ValueError(f"LOS-vector grid does not match LOS raster for {frame}")
    norm = np.sqrt(sum(np.square(array) for array in arrays))
    valid = np.isfinite(norm) & (norm > 0.5)
    normalized: list[np.ndarray] = []
    for array in arrays:
        normalized.append(
            np.divide(
                array,
                norm,
                out=np.full_like(array, np.nan, dtype=np.float32),
                where=valid,
            )
        )
    return normalized[0], normalized[1], normalized[2]


def crop_slices_for_circle(
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    centre_latitude: float,
    centre_longitude: float,
    radius_km: float,
    padding_km: float = 1.5,
) -> tuple[slice, slice]:
    """Return a compact grid bounding box around a geographic circle."""
    lat_half_width = (radius_km + padding_km) / 110.574
    lon_half_width = (radius_km + padding_km) / (
        111.320 * math.cos(math.radians(centre_latitude))
    )
    rows = np.flatnonzero(np.abs(latitude - centre_latitude) <= lat_half_width)
    cols = np.flatnonzero(np.abs(longitude - centre_longitude) <= lon_half_width)
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("The GNSS circle does not overlap the interferogram grid")
    return slice(int(rows[0]), int(rows[-1] + 1)), slice(int(cols[0]), int(cols[-1] + 1))


def local_raster_median(
    raster: Raster,
    arrays: Sequence[np.ndarray],
    *,
    latitude: float,
    longitude: float,
    radius_km: float,
) -> tuple[list[float], int]:
    """Return local medians of a LOS field and co-registered arrays."""
    mask = _disk_mask(
        raster.latitude, raster.longitude, latitude, longitude, radius_km
    ) & raster.valid & np.isfinite(raster.values_mm)
    for array in arrays:
        mask &= np.isfinite(array)
    count = int(mask.sum())
    if count == 0:
        return [float("nan")] * (1 + len(arrays)), 0
    values = [float(np.nanmedian(raster.values_mm[mask]))]
    values.extend(float(np.nanmedian(array[mask])) for array in arrays)
    return values, count


def gnss_los_sign_audit(
    raster: Raster,
    los_e: np.ndarray,
    los_n: np.ndarray,
    los_u: np.ndarray,
    gnss: pd.DataFrame,
    *,
    station_radius_km: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Forward-project GNSS ENU and infer one data-wide InSAR polarity per track.

    The residual evaluation is centred by one robust offset because the raw
    interferogram has an arbitrary reference.  No station-derived ramp is ever
    fitted or applied to the raster.
    """
    rows: list[dict[str, float | int | str]] = []
    for station in gnss.itertuples(index=False):
        local, count = local_raster_median(
            raster,
            [los_e, los_n, los_u],
            latitude=float(station.latitude),
            longitude=float(station.longitude),
            radius_km=station_radius_km,
        )
        observed, e, n, u = local
        if count == 0:
            continue
        predicted = (
            e * float(station.east_mm)
            + n * float(station.north_mm)
            + u * float(station.up_mm)
        )
        sigma = math.sqrt(
            (e * float(station.sigma_east_mm)) ** 2
            + (n * float(station.sigma_north_mm)) ** 2
            + (u * float(station.sigma_up_mm)) ** 2
        )
        rows.append(
            {
                "station": str(station.station),
                "latitude": float(station.latitude),
                "longitude": float(station.longitude),
                "insar_los_mm": observed,
                "gnss_projected_los_mm": predicted,
                "gnss_projected_sigma_mm": sigma,
                "local_los_e": e,
                "local_los_n": n,
                "local_los_u": u,
                "pixel_count": count,
            }
        )
    table = pd.DataFrame(rows)
    if len(table) < 5:
        raise ValueError("Fewer than five GNSS stations overlap the LOS raster")
    observed = table["insar_los_mm"].to_numpy(float)
    projected = table["gnss_projected_los_mm"].to_numpy(float)
    correlation = float(np.corrcoef(observed, projected)[0, 1])
    sign = 1 if correlation >= 0.0 else -1
    aligned = sign * observed
    offset = float(np.nanmedian(aligned - projected))
    residual = aligned - projected - offset
    table["insar_sign_aligned_mm"] = aligned
    table["reference_offset_mm"] = offset
    table["centred_residual_mm"] = residual
    table["standardized_residual"] = residual / np.maximum(
        table["gnss_projected_sigma_mm"].to_numpy(float), 1.0
    )
    summary: dict[str, float | int] = {
        "station_count": len(table),
        "raw_correlation": correlation,
        "selected_insar_sign": sign,
        "centred_offset_mm": offset,
        "centred_rmse_mm": float(np.sqrt(np.mean(np.square(residual)))),
        "centred_mae_mm": float(np.mean(np.abs(residual))),
        "aligned_correlation": float(np.corrcoef(aligned, projected)[0, 1]),
    }
    return table, summary


def vertical_los_correction(
    los_u: np.ndarray,
    vertical_mean_mm: np.ndarray,
    vertical_sigma_mm: np.ndarray,
    reference_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Project a vertical field into LOS and reference it like the InSAR map.

    The reference uncertainty uses an independence approximation, which is
    conservative for the positive Matérn/exponential covariance used here.
    """
    raw = np.asarray(los_u, float) * np.asarray(vertical_mean_mm, float)
    raw_sigma = np.abs(np.asarray(los_u, float)) * np.asarray(vertical_sigma_mm, float)
    reference = reference_mask & np.isfinite(raw) & np.isfinite(raw_sigma)
    if int(reference.sum()) < 25:
        raise ValueError("Too few valid vertical predictions in reference disk")
    reference_value = float(np.nanmedian(raw[reference]))
    reference_sigma = float(np.nanmedian(raw_sigma[reference]))
    corrected = raw - reference_value
    sigma = np.sqrt(np.square(raw_sigma) + reference_sigma**2)
    return corrected.astype(np.float32), sigma.astype(np.float32), reference_value, reference_sigma


def fault_segments_in_bounds(
    path: Path,
    *,
    latitude_min: float,
    latitude_max: float,
    longitude_min: float,
    longitude_max: float,
) -> list[np.ndarray]:
    """Read only mapped rupture line segments relevant to a plot extent."""
    import json

    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments: list[np.ndarray] = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "LineString":
            continue
        coordinates = np.asarray(geometry.get("coordinates", []), dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] < 2:
            continue
        lon = coordinates[:, 0]
        lat = coordinates[:, 1]
        if (
            lon.max() < longitude_min
            or lon.min() > longitude_max
            or lat.max() < latitude_min
            or lat.min() > latitude_max
        ):
            continue
        segments.append(coordinates[:, :2])
    return segments
