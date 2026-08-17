"""Independent near-fault sampling for the 2019 Ridgecrest rupture zones."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from ridgecrest_transient import PatchSeries, time_years, transient_design

from ridgecrest_jump import huber_irls


@dataclass(frozen=True)
class FaultPointSeries:
    dates: pd.DatetimeIndex
    values: np.ndarray  # (time, point)
    metadata: pd.DataFrame
    reference: np.ndarray
    quality_mask: np.ndarray
    reference_mask: np.ndarray


def _feature_lines(feature: dict) -> list[np.ndarray]:
    geometry = feature["geometry"]
    if geometry["type"] == "LineString":
        return [np.asarray(geometry["coordinates"], dtype=float)]
    if geometry["type"] == "MultiLineString":
        return [
            np.asarray(coordinates, dtype=float)
            for coordinates in geometry["coordinates"]
        ]
    raise ValueError(f"Unsupported fault geometry: {geometry['type']}")


def load_fault_lines(
    geojson_file: str | Path,
    *,
    certain_only: bool = True,
) -> dict[str, list[np.ndarray]]:
    data = json.loads(Path(geojson_file).read_text(encoding="utf-8"))
    grouped: dict[str, list[np.ndarray]] = {}
    for feature in data["features"]:
        properties = feature["properties"]
        if certain_only and properties.get("IdentityConfidence") != "certain":
            continue
        label = str(properties.get("Label") or "Unlabelled rupture")
        valid_lines = [
            line
            for line in _feature_lines(feature)
            if line.ndim == 2 and line.shape[1] >= 2 and len(line) >= 2
        ]
        grouped.setdefault(label, []).extend(valid_lines)
    return grouped


def _lonlat_to_xy(
    coordinates: np.ndarray,
    center: tuple[float, float],
) -> np.ndarray:
    lat0, lon0 = center
    longitude = coordinates[:, 0]
    latitude = coordinates[:, 1]
    east = (longitude - lon0) * 111.195 * np.cos(np.deg2rad(lat0))
    north = (latitude - lat0) * 111.195
    return np.column_stack([east, north])


def _xy_to_lonlat(xy: np.ndarray, center: tuple[float, float]) -> np.ndarray:
    lat0, lon0 = center
    longitude = lon0 + xy[:, 0] / (
        111.195 * np.cos(np.deg2rad(lat0))
    )
    latitude = lat0 + xy[:, 1] / 111.195
    return np.column_stack([longitude, latitude])


def _weighted_covariance(
    xy: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weights = weights / np.sum(weights)
    center = np.sum(xy * weights[:, None], axis=0)
    demeaned = xy - center
    covariance = (demeaned * weights[:, None]).T @ demeaned
    return center, covariance


def build_fault_sampling_points(
    geojson_file: str | Path,
    *,
    center: tuple[float, float] = (35.74, -117.55),
    node_spacing_km: float = 5.0,
    side_offset_km: float = 2.0,
    certain_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[np.ndarray]]]:
    """Create fixed along-rupture nodes and paired points on both sides."""
    grouped = load_fault_lines(geojson_file, certain_only=certain_only)
    node_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []

    for label, lines in sorted(grouped.items()):
        midpoints: list[np.ndarray] = []
        lengths: list[np.ndarray] = []
        for line in lines:
            if len(line) < 2:
                continue
            xy = _lonlat_to_xy(line, center)
            delta = np.diff(xy, axis=0)
            segment_length = np.linalg.norm(delta, axis=1)
            for start, vector, length in zip(xy[:-1], delta, segment_length):
                if length <= 1e-5:
                    continue
                subdivisions = max(1, int(np.ceil(length / 1.0)))
                fractions = (np.arange(subdivisions) + 0.5) / subdivisions
                midpoints.append(start[None, :] + fractions[:, None] * vector)
                lengths.append(np.full(subdivisions, length / subdivisions))
        xy_mid = np.vstack(midpoints)
        weights = np.concatenate(lengths)
        origin, covariance = _weighted_covariance(xy_mid, weights)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        global_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        if global_axis[1] < 0:
            global_axis *= -1.0
        along = (xy_mid - origin) @ global_axis
        lower, upper = np.quantile(along, [0.01, 0.99])
        targets = np.arange(
            np.ceil(lower / node_spacing_km) * node_spacing_km,
            np.floor(upper / node_spacing_km) * node_spacing_km + 0.01,
            node_spacing_km,
        )
        zone_code = "PR" if "Paxton" in label else "SW"
        bearing = float(
            (np.degrees(np.arctan2(global_axis[0], global_axis[1])) + 360.0)
            % 360.0
        )

        for sequence, target in enumerate(targets, start=1):
            nearby = np.abs(along - target) <= node_spacing_km / 2.0
            if np.count_nonzero(nearby) < 1:
                continue
            local_xy = xy_mid[nearby]
            local_weights = weights[nearby]
            if len(local_xy) == 1:
                local_origin = local_xy[0]
                tangent = global_axis.copy()
            else:
                local_origin, local_covariance = _weighted_covariance(
                    local_xy, local_weights
                )
                _, local_vectors = np.linalg.eigh(local_covariance)
                tangent = local_vectors[:, -1]
                if tangent @ global_axis < 0:
                    tangent *= -1.0
            normal = np.array([-tangent[1], tangent[0]])
            local_projection = (local_xy - local_origin) @ tangent
            node_xy = (
                local_origin
                + tangent
                * np.average(local_projection, weights=local_weights)
            )
            lonlat = _xy_to_lonlat(node_xy[None, :], center)[0]
            node_id = f"{zone_code}{sequence:02d}"
            node_rows.append(
                {
                    "node_id": node_id,
                    "fault_zone": label,
                    "longitude": lonlat[0],
                    "latitude": lonlat[1],
                    "east_km": node_xy[0],
                    "north_km": node_xy[1],
                    "bearing_deg": bearing,
                    "tangent_east": tangent[0],
                    "tangent_north": tangent[1],
                    "normal_east": normal[0],
                    "normal_north": normal[1],
                }
            )
            for side, multiplier in (("minus", -1.0), ("plus", 1.0)):
                point_xy = node_xy + multiplier * side_offset_km * normal
                point_lonlat = _xy_to_lonlat(point_xy[None, :], center)[0]
                point_rows.append(
                    {
                        "point_id": f"{node_id}_{side}",
                        "node_id": node_id,
                        "fault_zone": label,
                        "side": side,
                        "fault_offset_km": multiplier * side_offset_km,
                        "longitude": point_lonlat[0],
                        "latitude": point_lonlat[1],
                        "east_km": point_xy[0],
                        "north_km": point_xy[1],
                    }
                )

    nodes = pd.DataFrame(node_rows).sort_values(
        ["fault_zone", "node_id"]
    ).reset_index(drop=True)
    points = pd.DataFrame(point_rows).sort_values(
        ["fault_zone", "node_id", "side"]
    ).reset_index(drop=True)
    return nodes, points, grouped


def extract_fault_point_series(
    h5_file: str | Path,
    points: pd.DataFrame,
    *,
    sample_radius_km: float = 1.0,
    reference_box: tuple[int, int, int, int] = (497, 518, 50, 71),
    coherence_min: float = 0.25,
    residual_rms_max_mm: float = 5.0,
    max_gaps: float = 2.0,
    max_loop_errors: float = 10.0,
    min_pixels: int = 30,
) -> FaultPointSeries:
    """Extract quality-masked disk medians at independently fixed locations."""
    h5_file = Path(h5_file)
    with h5py.File(h5_file, "r") as h5:
        dates = pd.to_datetime(
            np.asarray(h5["imdates"][:], dtype=np.int64).astype(str),
            format="%Y%m%d",
        )
        _, ny, nx = h5["cum"].shape
        row, col = np.indices((ny, nx))
        latitude = float(h5["corner_lat"][()]) + row * float(h5["post_lat"][()])
        longitude = float(h5["corner_lon"][()]) + col * float(h5["post_lon"][()])
        quality = (
            np.isfinite(h5["coh_avg"][:])
            & (h5["coh_avg"][:] >= coherence_min)
            & np.isfinite(h5["resid_rms"][:])
            & (h5["resid_rms"][:] <= residual_rms_max_mm)
            & np.isfinite(h5["n_gap"][:])
            & (h5["n_gap"][:] <= max_gaps)
            & np.isfinite(h5["n_loop_err"][:])
            & (h5["n_loop_err"][:] <= max_loop_errors)
        )
        x1, x2, y1, y2 = reference_box
        reference_mask = np.zeros((ny, nx), dtype=bool)
        reference_mask[y1:y2, x1:x2] = quality[y1:y2, x1:x2]
        if np.count_nonzero(reference_mask) < 25:
            raise ValueError("Fewer than 25 valid reference pixels")

        point_indices: list[np.ndarray] = []
        pixel_counts: list[int] = []
        for point in points.itertuples(index=False):
            north = (latitude - point.latitude) * 111.195
            east = (
                (longitude - point.longitude)
                * 111.195
                * np.cos(np.deg2rad(point.latitude))
            )
            mask = quality & (np.hypot(east, north) <= sample_radius_km)
            indices = np.flatnonzero(mask)
            if indices.size < min_pixels:
                raise ValueError(
                    f"{point.point_id} has only {indices.size} valid pixels"
                )
            point_indices.append(indices)
            pixel_counts.append(int(indices.size))

        values = np.empty((len(dates), len(points)), dtype=float)
        reference = np.empty(len(dates), dtype=float)
        for i in range(len(dates)):
            epoch = np.asarray(h5["cum"][i], dtype=np.float32)
            flat = epoch.ravel()
            reference[i] = float(np.nanmedian(epoch[reference_mask]))
            for j, indices in enumerate(point_indices):
                values[i, j] = (
                    float(np.nanmedian(flat[indices])) - reference[i]
                )

    metadata = points.copy()
    metadata["pixel_count"] = pixel_counts
    return FaultPointSeries(
        dates=pd.DatetimeIndex(dates),
        values=values,
        metadata=metadata,
        reference=reference,
        quality_mask=quality,
        reference_mask=reference_mask,
    )


def paired_fault_differences(
    point_series: FaultPointSeries,
    nodes: pd.DataFrame,
) -> PatchSeries:
    """Return plus-side minus minus-side displacement at every rupture node."""
    differences: list[np.ndarray] = []
    metadata_rows: list[pd.Series] = []
    for node in nodes.itertuples(index=False):
        minus_index = point_series.metadata.index[
            (point_series.metadata["node_id"] == node.node_id)
            & (point_series.metadata["side"] == "minus")
        ]
        plus_index = point_series.metadata.index[
            (point_series.metadata["node_id"] == node.node_id)
            & (point_series.metadata["side"] == "plus")
        ]
        if len(minus_index) != 1 or len(plus_index) != 1:
            raise ValueError(f"Expected exactly two sides for {node.node_id}")
        differences.append(
            point_series.values[:, plus_index[0]]
            - point_series.values[:, minus_index[0]]
        )
        metadata_rows.append(pd.Series(node._asdict()))
    return PatchSeries(
        dates=point_series.dates,
        values=np.column_stack(differences),
        metadata=pd.DataFrame(metadata_rows).reset_index(drop=True),
        reference=np.zeros(len(point_series.dates), dtype=float),
        quality_mask=point_series.quality_mask,
        target_mask=point_series.quality_mask,
    )


def fit_full_event_response(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    *,
    first_post_date: pd.Timestamp | str = "2019-07-16",
    postseismic_scale_days: float = 10.0,
) -> dict[str, object]:
    """Fit background + known event step + logarithmic postseismic response."""
    dates = pd.DatetimeIndex(dates)
    values = np.asarray(values, dtype=float)
    t = time_years(dates)
    baseline = transient_design(t, "baseline")
    first_post = pd.Timestamp(first_post_date)
    after = (dates >= first_post).astype(float)
    elapsed_days = np.maximum(
        (dates - first_post).total_seconds().to_numpy(float) / 86400.0,
        0.0,
    )
    post_log = after * np.log1p(elapsed_days / postseismic_scale_days)
    X = np.column_stack([baseline, after, post_log])
    beta, residual = huber_irls(X, values)
    fitted = X @ beta
    robust_rms = float(
        1.4826 * np.median(np.abs(residual - np.median(residual)))
    )
    return {
        "step_mm": float(beta[-2]),
        "postseismic_log_coefficient_mm": float(beta[-1]),
        "robust_residual_scale_mm": robust_rms,
        "fitted": fitted,
        "residual": residual,
    }
