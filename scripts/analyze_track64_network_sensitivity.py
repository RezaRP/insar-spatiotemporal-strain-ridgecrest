"""Controlled Track 64 full-network versus pre-event-only comparison.

This script never modifies the source HDF5 products.  It:

1. verifies that both products share the same ascending LOS geometry;
2. selects a common, stable, multi-pixel far-field reference;
3. restricts comparisons to acquisition dates shared by both products;
4. compares cumulative fields after a common temporal and spatial baseline;
5. compares reference-independent paired samples across mapped rupture traces;
6. runs the existing block-bootstrap and Bayesian transient diagnostics.

The two input products were generated with different automatically selected
temporal filter widths.  That difference is retained as an explicit
sensitivity limitation in every output.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ridgecrest_transient import (  # noqa: E402
    PatchSeries,
    bayesian_transient_analysis,
    spatial_transient_matched_filter,
)

from ridgecrest_fault_points import (  # noqa: E402
    build_fault_sampling_points,
    extract_fault_point_series,
    paired_fault_differences,
)

DEFAULT_PRE = Path(
    r"E:\R64\064A_05410_131313\Curv\cum_filt.h5"
)
DEFAULT_FULL = Path(
    r"E:\R64\064A_05410_131313\Curv\Earthquake\cum_filt.h5"
)
FAULT_FILE = ROOT / "data" / "cgs_2019_ridgecrest_fault_ruptures.geojson"
EVENTS = {
    "M6.4": (35.705, -117.504),
    "M7.1": (35.770, -117.599),
}


@dataclass(frozen=True)
class Grid:
    ny: int
    nx: int
    corner_lat: float
    corner_lon: float
    post_lat: float
    post_lon: float
    lat: np.ndarray
    lon: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre", type=Path, default=DEFAULT_PRE)
    parser.add_argument("--full", type=Path, default=DEFAULT_FULL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "track64_network_sensitivity",
    )
    parser.add_argument("--box-pixels", type=int, default=21)
    parser.add_argument("--min-distance-km", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=640071)
    return parser.parse_args()


def read_grid(path: Path) -> Grid:
    with h5py.File(path, "r") as h5:
        _, ny, nx = h5["cum"].shape
        corner_lat = float(h5["corner_lat"][()])
        corner_lon = float(h5["corner_lon"][()])
        post_lat = float(h5["post_lat"][()])
        post_lon = float(h5["post_lon"][()])
    row, col = np.indices((ny, nx))
    lat = corner_lat + row * post_lat
    lon = corner_lon + col * post_lon
    return Grid(
        ny=ny,
        nx=nx,
        corner_lat=corner_lat,
        corner_lon=corner_lon,
        post_lat=post_lat,
        post_lon=post_lon,
        lat=lat,
        lon=lon,
    )


def dates_from_h5(h5: h5py.File) -> pd.DatetimeIndex:
    return pd.to_datetime(
        np.asarray(h5["imdates"][:], dtype=np.int64).astype(str),
        format="%Y%m%d",
    )


def distance_km(
    lat: np.ndarray,
    lon: np.ndarray,
    center_lat: float,
    center_lon: float,
) -> np.ndarray:
    north = (lat - center_lat) * 111.195
    east = (
        (lon - center_lon)
        * 111.195
        * np.cos(np.deg2rad(center_lat))
    )
    return np.hypot(east, north)


def robust01(
    values: np.ndarray,
    eligible: np.ndarray,
    *,
    high_is_good: bool = False,
) -> np.ndarray:
    valid = eligible & np.isfinite(values)
    low, high = np.nanpercentile(values[valid], [5.0, 95.0])
    if high <= low:
        result = np.zeros_like(values, dtype=np.float32)
    else:
        result = np.clip((values - low) / (high - low), 0.0, 1.0)
    if high_is_good:
        result = 1.0 - result
    result[~np.isfinite(values)] = np.nan
    return result.astype(np.float32)


def local_mean(
    values: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    numerator = uniform_filter(
        np.where(finite, values, 0.0).astype(np.float32),
        size=size,
        mode="constant",
        cval=0.0,
    )
    fraction = uniform_filter(
        finite.astype(np.float32),
        size=size,
        mode="constant",
        cval=0.0,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = numerator / fraction
    mean[fraction == 0] = np.nan
    return mean, fraction


def check_compatible(pre: Path, full: Path) -> tuple[Grid, dict[str, object]]:
    grid = read_grid(pre)
    other = read_grid(full)
    fields = (
        "ny",
        "nx",
        "corner_lat",
        "corner_lon",
        "post_lat",
        "post_lon",
    )
    for field in fields:
        if not np.isclose(
            float(getattr(grid, field)),
            float(getattr(other, field)),
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError(f"Grid mismatch for {field}")

    los_equal: dict[str, bool] = {}
    los_medians: dict[str, float] = {}
    with h5py.File(pre, "r") as a, h5py.File(full, "r") as b:
        for name in ("E.geo", "N.geo", "U.geo"):
            av = np.asarray(a[name][:], dtype=np.float32)
            bv = np.asarray(b[name][:], dtype=np.float32)
            los_equal[name] = bool(np.array_equal(av, bv, equal_nan=True))
            valid = np.isfinite(av) & (av != 0)
            los_medians[name] = float(np.nanmedian(av[valid]))
    if not all(los_equal.values()):
        raise ValueError("LOS grids are not identical")
    if los_medians["E.geo"] >= 0:
        raise ValueError("Expected ascending geometry with negative E LOS")
    return grid, {
        "los_arrays_identical": los_equal,
        "median_los": los_medians,
    }


def common_dates(pre: Path, full: Path) -> pd.DatetimeIndex:
    with h5py.File(pre, "r") as a, h5py.File(full, "r") as b:
        dates_a = dates_from_h5(a)
        dates_b = dates_from_h5(b)
    return dates_a.intersection(dates_b).sort_values()


def read_quality(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        result = {
            name: np.asarray(h5[name][:], dtype=np.float32)
            for name in (
                "coh_avg",
                "resid_rms",
                "n_gap",
                "n_loop_err",
            )
        }
        if "mask" in h5:
            result["mask"] = np.asarray(h5["mask"][:], dtype=np.float32)
    return result


def temporal_rms(
    path: Path,
    dates: pd.DatetimeIndex,
    far_field: np.ndarray,
) -> np.ndarray:
    """RMS cumulative variation after common-date and far-field centering."""
    with h5py.File(path, "r") as h5:
        all_dates = dates_from_h5(h5)
        indices = all_dates.get_indexer(dates)
        if np.any(indices < 0):
            raise ValueError(f"Missing common date in {path}")
        first = np.asarray(h5["cum"][indices[0]], dtype=np.float32)
        sum_square = np.zeros(first.shape, dtype=np.float64)
        count = np.zeros(first.shape, dtype=np.uint16)
        for index in indices:
            epoch = np.asarray(h5["cum"][index], dtype=np.float32) - first
            center_mask = far_field & np.isfinite(epoch)
            center = float(np.nanmedian(epoch[center_mask]))
            epoch -= center
            valid = np.isfinite(epoch)
            sum_square[valid] += np.square(epoch[valid], dtype=np.float64)
            count[valid] += 1
    result = np.full(sum_square.shape, np.nan, dtype=np.float32)
    valid = count == len(dates)
    result[valid] = np.sqrt(
        sum_square[valid] / count[valid]
    ).astype(np.float32)
    return result


def separated_candidates(
    score: np.ndarray,
    eligible: np.ndarray,
    grid: Grid,
    *,
    count: int = 10,
    separation_km: float = 8.0,
) -> list[tuple[int, int]]:
    flat = np.flatnonzero(eligible & np.isfinite(score))
    if flat.size == 0:
        raise RuntimeError("No common reference candidate survived")
    order = flat[np.argsort(score.ravel()[flat])]
    selected: list[tuple[int, int]] = []
    for index in order:
        row, col = np.unravel_index(index, score.shape)
        if all(
            distance_km(
                np.array(grid.lat[row, col]),
                np.array(grid.lon[row, col]),
                grid.lat[old_row, old_col],
                grid.lon[old_row, old_col],
            )
            >= separation_km
            for old_row, old_col in selected
        ):
            selected.append((row, col))
        if len(selected) == count:
            break
    return selected


def select_reference(
    pre: Path,
    full: Path,
    grid: Grid,
    dates: pd.DatetimeIndex,
    *,
    box_pixels: int,
    min_distance_km: float,
) -> tuple[tuple[int, int, int, int], pd.DataFrame, dict[str, np.ndarray]]:
    event_distance = np.minimum.reduce(
        [
            distance_km(grid.lat, grid.lon, lat, lon)
            for lat, lon in EVENTS.values()
        ]
    )
    far_field = event_distance >= min_distance_km
    qa = read_quality(pre)
    qb = read_quality(full)
    min_coherence = np.minimum(qa["coh_avg"], qb["coh_avg"])
    max_residual = np.maximum(qa["resid_rms"], qb["resid_rms"])
    max_gap = np.maximum(qa["n_gap"], qb["n_gap"])
    max_loop = np.maximum(qa["n_loop_err"], qb["n_loop_err"])
    common = (
        far_field
        & np.isfinite(min_coherence)
        & np.isfinite(max_residual)
        & np.isfinite(max_gap)
        & np.isfinite(max_loop)
    )
    if "mask" in qa:
        common &= qa["mask"] > 0
    if "mask" in qb:
        common &= qb["mask"] > 0

    print("Computing pre-only temporal stability...", flush=True)
    rms_pre = temporal_rms(pre, dates, far_field)
    print("Computing full-network temporal stability...", flush=True)
    rms_full = temporal_rms(full, dates, far_field)
    worst_rms = np.fmax(rms_pre, rms_full)
    score = (
        0.45 * robust01(worst_rms, common)
        + 0.25 * robust01(min_coherence, common, high_is_good=True)
        + 0.20 * robust01(max_residual, common)
        + 0.05 * robust01(max_gap, common)
        + 0.05 * robust01(max_loop, common)
    )
    score[~common] = np.nan

    score_box, coverage = local_mean(score, box_pixels)
    coherence_box, _ = local_mean(min_coherence, box_pixels)
    residual_box, _ = local_mean(max_residual, box_pixels)
    rms_box, _ = local_mean(worst_rms, box_pixels)
    gap_box, _ = local_mean(max_gap, box_pixels)
    loop_box, _ = local_mean(max_loop, box_pixels)
    half = box_pixels // 2
    inside = np.zeros_like(common)
    margin = half + 30
    inside[margin:-margin, margin:-margin] = True
    eligible = (
        inside
        & far_field
        & (coverage >= 0.98)
        & (coherence_box >= 0.50)
        & (residual_box <= 2.0)
        & (gap_box <= 0.10)
        & (loop_box <= 5.0)
    )
    selected = separated_candidates(score_box, eligible, grid)
    rows = []
    for rank, (row, col) in enumerate(selected, start=1):
        x1, x2 = col - half, col + half + 1
        y1, y2 = row - half, row + half + 1
        rows.append(
            {
                "rank": rank,
                "score": float(score_box[row, col]),
                "refarea_xy": f"{x1}:{x2}/{y1}:{y2}",
                "x1": x1,
                "x2": x2,
                "y1": y1,
                "y2": y2,
                "center_lon": float(grid.lon[row, col]),
                "center_lat": float(grid.lat[row, col]),
                "distance_to_nearest_event_km": float(
                    event_distance[row, col]
                ),
                "mean_min_coherence": float(coherence_box[row, col]),
                "mean_max_residual_mm": float(residual_box[row, col]),
                "mean_worst_temporal_rms_mm": float(rms_box[row, col]),
                "mean_max_gap": float(gap_box[row, col]),
                "mean_max_loop_errors": float(loop_box[row, col]),
            }
        )
    table = pd.DataFrame(rows)
    best = table.iloc[0]
    reference = (
        int(best["x1"]),
        int(best["x2"]),
        int(best["y1"]),
        int(best["y2"]),
    )
    layers = {
        "score_box": score_box,
        "coherence_box": coherence_box,
        "rms_box": rms_box,
        "eligible": eligible,
    }
    return reference, table, layers


def read_common_delta(
    path: Path,
    dates: pd.DatetimeIndex,
    reference: tuple[int, int, int, int],
) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        all_dates = dates_from_h5(h5)
        first_index = int(all_dates.get_loc(dates[0]))
        last_index = int(all_dates.get_loc(dates[-1]))
        result = (
            np.asarray(h5["cum"][last_index], dtype=np.float32)
            - np.asarray(h5["cum"][first_index], dtype=np.float32)
        )
    x1, x2, y1, y2 = reference
    result -= float(np.nanmedian(result[y1:y2, x1:x2]))
    return result


def truncate_patch(
    series: PatchSeries,
    dates: pd.DatetimeIndex,
) -> PatchSeries:
    indices = series.dates.get_indexer(dates)
    if np.any(indices < 0):
        raise ValueError("Patch series is missing a common date")
    values = series.values[indices].copy()
    values -= values[0]
    return PatchSeries(
        dates=dates,
        values=values,
        metadata=series.metadata.copy(),
        reference=np.zeros(len(dates)),
        quality_mask=series.quality_mask,
        target_mask=series.target_mask,
    )


def plot_reference(
    path: Path,
    grid: Grid,
    candidates: pd.DataFrame,
    layers: dict[str, np.ndarray],
) -> None:
    extent = [
        float(grid.lon.min()),
        float(grid.lon.max()),
        float(grid.lat.min()),
        float(grid.lat.max()),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    panels = [
        (layers["score_box"], "Common reference score", "viridis_r"),
        (layers["coherence_box"], "Minimum coherence", "viridis"),
        (layers["rms_box"], "Worst pre-event temporal RMS (mm)", "magma"),
    ]
    for axis, (values, title, cmap) in zip(axes, panels):
        image = axis.imshow(
            values,
            extent=extent,
            origin="upper",
            cmap=cmap,
            aspect="equal",
        )
        for row in candidates.itertuples(index=False):
            axis.plot(row.center_lon, row.center_lat, "wo", mec="black", ms=6)
            axis.text(
                row.center_lon,
                row.center_lat,
                f" {row.rank}",
                color="white",
                weight="bold",
            )
        for label, (lat, lon) in EVENTS.items():
            axis.plot(lon, lat, "r*", ms=10)
            axis.text(lon, lat, f" {label}", color="red")
        axis.set_title(title)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.suptitle("Track 64 common-reference selection")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_fields(
    path: Path,
    grid: Grid,
    pre_delta: np.ndarray,
    full_delta: np.ndarray,
    nodes: pd.DataFrame,
    fault_lines: dict[str, list[np.ndarray]],
) -> None:
    difference = full_delta - pre_delta
    roi = (
        (grid.lat >= 35.25)
        & (grid.lat <= 36.15)
        & (grid.lon >= -117.95)
        & (grid.lon <= -116.95)
    )
    common_abs = float(
        np.nanpercentile(
            np.abs(np.concatenate([pre_delta[roi], full_delta[roi]])),
            99.0,
        )
    )
    diff_abs = float(np.nanpercentile(np.abs(difference[roi]), 99.0))
    extent = [-117.95, -116.95, 35.25, 36.15]
    row_mask = (grid.lat[:, 0] >= 35.25) & (grid.lat[:, 0] <= 36.15)
    col_mask = (grid.lon[0] >= -117.95) & (grid.lon[0] <= -116.95)
    figure, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    panels = [
        (pre_delta[np.ix_(row_mask, col_mask)], "Pre-event-only network", common_abs),
        (full_delta[np.ix_(row_mask, col_mask)], "Full network", common_abs),
        (
            difference[np.ix_(row_mask, col_mask)],
            "Full minus pre-only",
            diff_abs,
        ),
    ]
    for axis, (values, title, limit) in zip(axes, panels):
        image = axis.imshow(
            values,
            extent=extent,
            origin="upper",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            aspect="equal",
        )
        for lines in fault_lines.values():
            for line in lines:
                axis.plot(line[:, 0], line[:, 1], "k-", lw=0.5, alpha=0.7)
        axis.scatter(
            nodes["longitude"],
            nodes["latitude"],
            c="yellow",
            edgecolor="black",
            s=18,
            zorder=4,
        )
        axis.set_xlim(extent[:2])
        axis.set_ylim(extent[2:])
        axis.set_title(title)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        figure.colorbar(image, ax=axis, shrink=0.8, label="LOS displacement (mm)")
    figure.suptitle(
        "Track 64 cumulative change on shared dates: "
        "2017-05-21 to 2019-07-04"
    )
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_pair_diagnostics(
    path: Path,
    pre: PatchSeries,
    full: PatchSeries,
    result_pre: pd.DataFrame,
    result_full: pd.DataFrame,
) -> None:
    difference = full.values - pre.values
    endpoint = difference[-1]
    strongest = int(np.nanargmax(np.abs(endpoint)))
    node_id = str(pre.metadata.iloc[strongest]["node_id"])
    figure, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    axes[0].plot(
        pre.dates,
        pre.values[:, strongest],
        "o-",
        ms=3,
        label="Pre-event-only network",
    )
    axes[0].plot(
        full.dates,
        full.values[:, strongest],
        "s-",
        ms=3,
        label="Full network",
    )
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("Across-fault paired LOS (mm)")
    axes[0].set_title(
        f"Strongest July 4 network discrepancy: {node_id}"
    )
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    x = np.arange(len(endpoint))
    axes[1].bar(x - 0.2, pre.values[-1], width=0.4, label="Pre-only")
    axes[1].bar(x + 0.2, full.values[-1], width=0.4, label="Full")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pre.metadata["node_id"], rotation=45)
    axes[1].set_ylabel("July 4 paired LOS change (mm)")
    axes[1].set_title(
        "Reference-independent comparison at mapped rupture nodes"
    )
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in (args.pre, args.full, FAULT_FILE):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.box_pixels < 3 or args.box_pixels % 2 != 1:
        raise ValueError("--box-pixels must be odd and at least 3")

    grid, geometry = check_compatible(args.pre, args.full)
    dates = common_dates(args.pre, args.full)
    if dates[-1] != pd.Timestamp("2019-07-04"):
        raise ValueError(f"Unexpected final shared date: {dates[-1]}")
    print(
        f"Verified common ascending grid; {len(dates)} shared dates "
        f"from {dates[0].date()} through {dates[-1].date()}",
        flush=True,
    )

    reference, candidates, layers = select_reference(
        args.pre,
        args.full,
        grid,
        dates,
        box_pixels=args.box_pixels,
        min_distance_km=args.min_distance_km,
    )
    candidates.to_csv(
        args.output_dir / "reference_candidates.csv",
        index=False,
        float_format="%.6f",
    )
    (args.output_dir / "best_reference_box.txt").write_text(
        f"{reference[0]}:{reference[1]}/"
        f"{reference[2]}:{reference[3]}\n",
        encoding="utf-8",
    )
    plot_reference(
        args.output_dir / "01_reference_search.png",
        grid,
        candidates,
        layers,
    )
    print(f"Selected common reference {reference}", flush=True)

    nodes, points, fault_lines = build_fault_sampling_points(FAULT_FILE)
    pre_delta = read_common_delta(args.pre, dates, reference)
    full_delta = read_common_delta(args.full, dates, reference)
    difference = full_delta - pre_delta
    plot_fields(
        args.output_dir / "02_july4_spatial_network_comparison.png",
        grid,
        pre_delta,
        full_delta,
        nodes,
        fault_lines,
    )

    print("Extracting Track 64 near-fault point series...", flush=True)
    pre_points = extract_fault_point_series(
        args.pre,
        points,
        sample_radius_km=1.25,
        reference_box=reference,
        min_pixels=20,
    )
    full_points = extract_fault_point_series(
        args.full,
        points,
        sample_radius_km=1.25,
        reference_box=reference,
        min_pixels=20,
    )
    pre_pairs = truncate_patch(
        paired_fault_differences(pre_points, nodes),
        dates,
    )
    full_pairs = truncate_patch(
        paired_fault_differences(full_points, nodes),
        dates,
    )

    result_pre, _, _ = spatial_transient_matched_filter(
        pre_pairs,
        min_before=20,
        min_after=4,
        candidate_start_date="2019-01-01",
        candidate_end_date="2019-05-29",
        n_boot=999,
        block_length=4,
        seed=args.seed,
    )
    result_full, _, _ = spatial_transient_matched_filter(
        full_pairs,
        min_before=20,
        min_after=4,
        candidate_start_date="2019-01-01",
        candidate_end_date="2019-05-29",
        n_boot=999,
        block_length=4,
        seed=args.seed + 1,
    )
    result_pre.to_csv(
        args.output_dir / "preonly_transient_nodes.csv", index=False
    )
    result_full.to_csv(
        args.output_dir / "full_transient_nodes.csv", index=False
    )

    pair_difference = full_pairs.values - pre_pairs.values
    diagnostic_rows = []
    bayesian_rows = []
    for column, row in pre_pairs.metadata.iterrows():
        node = str(row["node_id"])
        endpoint = float(pair_difference[-1, column])
        diagnostic_rows.append(
            {
                "node_id": node,
                "preonly_july4_change_mm": float(
                    pre_pairs.values[-1, column]
                ),
                "full_july4_change_mm": float(
                    full_pairs.values[-1, column]
                ),
                "full_minus_preonly_july4_mm": endpoint,
                "common_date_difference_rms_mm": float(
                    np.sqrt(np.mean(np.square(pair_difference[:, column])))
                ),
            }
        )
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(
        args.output_dir / "pair_network_diagnostics.csv", index=False
    )

    strongest_indices = sorted(
        set(
            [
                int(np.argmax(np.abs(pair_difference[-1]))),
                int(result_pre["max_statistic"].idxmax()),
                int(result_full["max_statistic"].idxmax()),
            ]
        )
    )
    for product, series in (
        ("preonly", pre_pairs),
        ("full", full_pairs),
    ):
        for column in strongest_indices:
            summary, table, _ = bayesian_transient_analysis(
                series.dates,
                series.values[:, column],
                min_before=20,
                min_after=4,
                candidate_start_date="2019-01-01",
                candidate_end_date="2019-05-29",
                n_samples=5000,
                seed=args.seed + column,
            )
            node = str(series.metadata.iloc[column]["node_id"])
            table.to_csv(
                args.output_dir
                / f"bayesian_{product}_{node}_models.csv",
                index=False,
            )
            bayesian_rows.append(
                {
                    "product": product,
                    "node_id": node,
                    **{
                        key: (
                            value.isoformat()
                            if isinstance(value, pd.Timestamp)
                            else value
                        )
                        for key, value in summary.items()
                    },
                }
            )
    pd.DataFrame(bayesian_rows).to_csv(
        args.output_dir / "bayesian_candidate_summary.csv",
        index=False,
    )
    plot_pair_diagnostics(
        args.output_dir / "03_paired_fault_network_comparison.png",
        pre_pairs,
        full_pairs,
        result_pre,
        result_full,
    )

    roi = (
        (grid.lat >= 35.25)
        & (grid.lat <= 36.15)
        & (grid.lon >= -117.95)
        & (grid.lon <= -116.95)
    )
    with h5py.File(args.pre, "r") as a, h5py.File(args.full, "r") as b:
        provenance = {
            "pre_file": str(args.pre),
            "full_file": str(args.full),
            "pre_dates": [
                str(d.date()) for d in dates_from_h5(a)
            ],
            "full_dates": [
                str(d.date()) for d in dates_from_h5(b)
            ],
            "common_dates": [str(d.date()) for d in dates],
            "pre_original_reference": a["refarea"][()].decode(),
            "full_original_reference": b["refarea"][()].decode(),
            "pre_temporal_filter_years": float(a["filtwidth_yr"][()]),
            "full_temporal_filter_years": float(b["filtwidth_yr"][()]),
            "common_reference_xy": (
                f"{reference[0]}:{reference[1]}/"
                f"{reference[2]}:{reference[3]}"
            ),
            "common_reference_center_lon": float(
                candidates.iloc[0]["center_lon"]
            ),
            "common_reference_center_lat": float(
                candidates.iloc[0]["center_lat"]
            ),
            "geometry": geometry,
            "comparison_limitations": [
                "Different original interferogram networks",
                "Different automatically selected temporal filter widths",
                "GACOS status is not recorded in either HDF5 product",
            ],
        }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )

    summary = {
        "shared_epoch_count": len(dates),
        "first_shared_epoch": str(dates[0].date()),
        "last_shared_epoch": str(dates[-1].date()),
        "reference_box": provenance["common_reference_xy"],
        "spatial_difference_roi_rms_mm": float(
            np.sqrt(np.nanmean(np.square(difference[roi])))
        ),
        "spatial_difference_roi_max_abs_mm": float(
            np.nanmax(np.abs(difference[roi]))
        ),
        "paired_difference_rms_mm": float(
            np.sqrt(np.mean(np.square(pair_difference)))
        ),
        "paired_difference_max_abs_mm": float(
            np.max(np.abs(pair_difference))
        ),
        "july4_paired_difference_max_abs_mm": float(
            np.max(np.abs(pair_difference[-1]))
        ),
        "july4_strongest_difference_node": str(
            pre_pairs.metadata.iloc[
                int(np.argmax(np.abs(pair_difference[-1])))
            ]["node_id"]
        ),
        "preonly_significant_nodes_fwer_0_05": int(
            result_pre["fwer_significant_0_05"].sum()
        ),
        "full_significant_nodes_fwer_0_05": int(
            result_full["fwer_significant_0_05"].sum()
        ),
        "preonly_min_corrected_p": float(
            result_pre["corrected_pvalue"].min()
        ),
        "full_min_corrected_p": float(
            result_full["corrected_pvalue"].min()
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Outputs: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
