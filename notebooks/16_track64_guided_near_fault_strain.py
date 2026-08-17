# %% [markdown]
# # Track-64-guided near-fault cumulative 2-D strain analysis
#
# This notebook fills only missing descending Track 71 horizontal-LOS cells.
# Valid Track 71 values are never replaced.  The auxiliary information is the
# collocated, vertical-corrected Track 64 horizontal LOS, combined with nearby
# paired Track 64/71 observations in a fixed latent E-N universal-cokriging
# model.  Links that cross either finite source-model fault segment are removed.
#
# The covariance parameters and model choice are selected only from epochs
# ending on or before 29 May 2019.  The same linear operator is then applied to
# all cumulative epochs.  The 29 May-4 July interval remains surveillance and
# 4-16 July remains the earthquake-sequence control.  Reconstructed pixels and
# their uncertainty remain explicit in every saved product.
#
# The completed cumulative E-N field and its fixed-resolution gradient are
# evaluated throughout the full near-fault domain, including the mapped rupture
# zone.  Finite fault barriers remain inside the local operators so samples are
# not borrowed across the two source-model segments.

# %%
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import warnings

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


if "__file__" in globals():
    ROOT = Path(__file__).resolve().parents[1]
else:
    root_candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    ROOT = next(
        (
            candidate
            for candidate in root_candidates
            if (candidate / "pyproject.toml").exists()
            and (candidate / "src").exists()
        ),
        None,
    )
    if ROOT is None:
        raise RuntimeError(
            "Run this notebook from the Ridgecrest repository or its "
            "notebooks directory"
        )
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ridgecrest_cumulative_strain import (  # noqa: E402
    build_fixed_joint_mls,
    evaluate_fixed_joint_mls,
    fixed_joint_mls_component_sigma,
    target_values_to_grid,
)
from ridgecrest_fault_barrier_cokriging import (  # noqa: E402
    build_fixed_fault_barrier_cokriging,
    cokriging_diagnostics,
    evaluate_fixed_fault_barrier_cokriging,
)
from ridgecrest_strain_change import (  # noqa: E402
    empirical_upper_tail_pvalue,
    fit_robust_baseline,
    leave_one_out_baseline_innovations,
    maximum_signed_cluster_mass,
    standardized_innovation,
)
from ridgecrest_two_track import (  # noqa: E402
    from_utm11_km,
    masked_bilinear_resample,
    normalize_look_vectors,
    to_utm11_km,
)


plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
    }
)


# %% [markdown]
# ## 1. Fixed inputs, dates, and reconstruction domain

# %%
DIRECT_FILE = (
    ROOT
    / "outputs"
    / "cumulative_two_track_strain"
    / "direct_cumulative_vertical_corrected_en.npz"
)
DESC_H5 = ROOT / "data" / "cum_full_scene_no_GACOS.h5"
FAULT_GEOJSON = ROOT / "data" / "cgs_2019_ridgecrest_fault_ruptures.geojson"
SOURCE_PARAMETERS = (
    ROOT
    / "outputs"
    / "bayesian_inversion"
    / "marginalized_geometry"
    / "final_source_parameters.csv"
)
SOURCE_DIAGNOSTICS = (
    ROOT
    / "outputs"
    / "bayesian_inversion"
    / "marginalized_geometry"
    / "diagnostics.json"
)
OUTPUT_DIR = ROOT / "outputs" / "track64_guided_near_fault_strain"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for required in (
    DIRECT_FILE,
    DESC_H5,
    FAULT_GEOJSON,
    SOURCE_PARAMETERS,
    SOURCE_DIAGNOSTICS,
):
    if not required.exists():
        raise FileNotFoundError(required)

source_diagnostics = json.loads(SOURCE_DIAGNOSTICS.read_text(encoding="utf-8"))
if not bool(source_diagnostics.get("converged_screen", False)):
    raise RuntimeError("The fault barrier geometry did not pass convergence")

CALIBRATION_END = pd.Timestamp("2019-05-29")
PRE_EVENT_END = pd.Timestamp("2019-07-04")
EVENT_END = pd.Timestamp("2019-07-16")
NEAR_FAULT_DISTANCE_KM = 18.0
FORMER_CORE_AUDIT_KM = 1.0
SAMPLE_SPACING_KM = 2
STRAIN_SUPPORT_RADIUS_KM = 10.0
STRAIN_BANDWIDTH_KM = 4.0
STRAIN_MINIMUM_SAMPLES = 16
CV_BUFFER_KM = 5.0
CV_SUPPORT_RADIUS_KM = 24.0
MAXIMUM_COKRIGING_SAMPLES = 48
LOS_SIGMA_ASC_MM = 24.335595497061632
LOS_SIGMA_DESC_MM = 16.277079925630176

direct = np.load(DIRECT_FILE)
dates = pd.to_datetime(direct["dates"])
east_grid = np.asarray(direct["east_km"], dtype=float)
north_grid = np.asarray(direct["north_km"], dtype=float)
latitude_grid = np.asarray(direct["latitude"], dtype=float)
longitude_grid = np.asarray(direct["longitude"], dtype=float)
distance_grid = np.asarray(
    direct["distance_to_mapped_rupture_km"],
    dtype=float,
)

ascending_hlos = np.asarray(
    direct["ascending_pure_horizontal_los_cumulative_mm"],
    dtype=float,
)
descending_hlos = np.asarray(
    direct["descending_pure_horizontal_los_cumulative_mm"],
    dtype=float,
)
observed_east = np.asarray(direct["cumulative_east_mm"], dtype=float)
observed_north = np.asarray(direct["cumulative_north_mm"], dtype=float)
sigma_east = np.asarray(direct["cumulative_sigma_east_mm"], dtype=float)
sigma_north = np.asarray(direct["cumulative_sigma_north_mm"], dtype=float)
covariance_east_north = np.asarray(
    direct["cumulative_covariance_east_north_mm2"],
    dtype=float,
)
ascending_look_e = np.asarray(direct["ascending_look_e"], dtype=float)
ascending_look_n = np.asarray(direct["ascending_look_n"], dtype=float)

near_fault = np.isfinite(distance_grid) & (
    distance_grid <= NEAR_FAULT_DISTANCE_KM
)
former_core_distance_mask = np.isfinite(distance_grid) & (
    distance_grid <= FORMER_CORE_AUDIT_KM
)
ascending_persistent = np.all(np.isfinite(ascending_hlos), axis=0)
descending_persistent = np.all(np.isfinite(descending_hlos), axis=0)
paired_persistent = ascending_persistent & descending_persistent
fill_target_mask = (
    near_fault
    & ascending_persistent
    & ~descending_persistent
)
fill_row, fill_column = np.nonzero(fill_target_mask)
fill_target_xy = np.column_stack(
    [east_grid[fill_target_mask], north_grid[fill_target_mask]]
)

print("Common cumulative epochs:", len(dates), dates.min(), dates.max())
print("Near-fault cells:", int(near_fault.sum()))
print("Track-64-guided target cells:", int(fill_target_mask.sum()))
for date_text in ("2019-05-29", "2019-06-22", "2019-07-04", "2019-07-16"):
    epoch = int(np.flatnonzero(dates == pd.Timestamp(date_text))[0])
    both = (
        near_fault
        & np.isfinite(ascending_hlos[epoch])
        & np.isfinite(descending_hlos[epoch])
    )
    asc_only = (
        near_fault
        & np.isfinite(ascending_hlos[epoch])
        & ~np.isfinite(descending_hlos[epoch])
    )
    print(date_text, "both:", int(both.sum()), "Track 64 only:", int(asc_only.sum()))


# %% [markdown]
# ## 2. Restore Track 71 look geometry at quality-excluded pixels
#
# The direct product deliberately masked Track 71 look vectors wherever the
# displacement failed quality control.  Look geometry itself remains valid at
# those cells, so it is resampled again using finite geometry only.  No rejected
# Track 71 displacement is promoted to an observation.

# %%
with h5py.File(DESC_H5, "r") as handle:
    desc_ny, desc_nx = handle["E.geo"].shape
    desc_latitude = (
        float(handle["corner_lat"][()])
        + np.arange(desc_ny) * float(handle["post_lat"][()])
    )
    desc_longitude = (
        float(handle["corner_lon"][()])
        + np.arange(desc_nx) * float(handle["post_lon"][()])
    )
    native_desc_look = {
        "E": np.asarray(handle["E.geo"][:], dtype=float),
        "N": np.asarray(handle["N.geo"][:], dtype=float),
        "U": np.asarray(handle["U.geo"][:], dtype=float),
    }

native_geometry_valid = np.logical_and.reduce(
    [np.isfinite(values) for values in native_desc_look.values()]
)
raw_desc_look: dict[str, np.ndarray] = {}
raw_desc_support: list[np.ndarray] = []
for name, values in native_desc_look.items():
    sampled, support = masked_bilinear_resample(
        desc_latitude,
        desc_longitude,
        values,
        native_geometry_valid,
        latitude_grid,
        longitude_grid,
    )
    raw_desc_look[name] = sampled
    raw_desc_support.append(support)
(
    raw_desc_look["E"],
    raw_desc_look["N"],
    raw_desc_look["U"],
) = normalize_look_vectors(
    raw_desc_look["E"],
    raw_desc_look["N"],
    raw_desc_look["U"],
)
raw_desc_geometry_valid = (
    np.minimum.reduce(raw_desc_support) >= 0.999
) & np.logical_and.reduce(
    [np.isfinite(raw_desc_look[name]) for name in ("E", "N", "U")]
)
if not np.all(raw_desc_geometry_valid[fill_target_mask]):
    raise RuntimeError("Track 71 geometry is missing at a fill target")


# %% [markdown]
# ## 3. Two finite fault barriers
#
# The converged marginalized Bayesian geometry supplies one Paxton Ranch and
# one Salt Wells segment.  The detailed CGS rupture inventory is retained for
# the one-kilometre core distance and plotting; the two finite segments provide
# a stable side/crossing topology for interpolation and differentiation.

# %%
source_table = pd.read_csv(SOURCE_PARAMETERS)
fault_segments: list[np.ndarray] = []
fault_segment_rows: list[dict[str, float | str]] = []
for row in source_table.itertuples(index=False):
    center = to_utm11_km(
        np.asarray([float(row.center_lon)]),
        np.asarray([float(row.center_lat)]),
    )[0]
    strike_radian = math.radians(float(row.strike_deg))
    direction = np.asarray(
        [math.sin(strike_radian), math.cos(strike_radian)],
        dtype=float,
    )
    half = 0.5 * float(row.length_km) * direction
    segment = np.vstack([center - half, center + half])
    fault_segments.append(segment)
    fault_segment_rows.append(
        {
            "fault": str(row.fault),
            "center_east_km": float(center[0]),
            "center_north_km": float(center[1]),
            "strike_deg": float(row.strike_deg),
            "length_km": float(row.length_km),
            "endpoint_1_east_km": float(segment[0, 0]),
            "endpoint_1_north_km": float(segment[0, 1]),
            "endpoint_2_east_km": float(segment[1, 0]),
            "endpoint_2_north_km": float(segment[1, 1]),
        }
    )
fault_segments_xy = np.stack(fault_segments)
pd.DataFrame(fault_segment_rows).to_csv(
    OUTPUT_DIR / "fault_barrier_geometry.csv",
    index=False,
)

fault_geojson = json.loads(FAULT_GEOJSON.read_text(encoding="utf-8"))
fault_plot_lines: list[np.ndarray] = []
for feature in fault_geojson["features"]:
    geometry = feature.get("geometry") or {}
    coordinates = geometry.get("coordinates")
    if coordinates is None:
        continue
    if geometry.get("type") == "LineString":
        candidates = [coordinates]
    elif geometry.get("type") == "MultiLineString":
        candidates = coordinates
    else:
        continue
    for candidate in candidates:
        array = np.asarray(candidate, dtype=float)
        if array.ndim == 2 and array.shape[0] >= 2:
            fault_plot_lines.append(array[:, :2])
fault_plot_points = np.concatenate(fault_plot_lines, axis=0)[::8]


def add_faults(axis: plt.Axes, *, linewidth: float = 0.7) -> None:
    """Add mapped rupture strands and model barriers."""

    axis.scatter(
        fault_plot_points[:, 0],
        fault_plot_points[:, 1],
        s=max(0.25, 0.8 * linewidth),
        color="black",
        alpha=0.72,
        linewidths=0.0,
        zorder=5,
        label="Mapped rupture",
    )
    for segment_index, segment in enumerate(fault_segments_xy):
        segment_lon, segment_lat = from_utm11_km(
            segment[:, 0],
            segment[:, 1],
        )
        axis.plot(
            segment_lon,
            segment_lat,
            color="#f4c430",
            linewidth=1.5,
            linestyle="--",
            zorder=6,
            label="Cokriging barrier" if segment_index == 0 else None,
        )


# %% [markdown]
# ## 4. Persistent paired sample lattice and pre-event covariance

# %%
sample_lattice = np.zeros(east_grid.shape, dtype=bool)
sample_lattice[::SAMPLE_SPACING_KM, ::SAMPLE_SPACING_KM] = True
sample_mask = (
    paired_persistent
    & sample_lattice
    & np.isfinite(ascending_look_e)
    & np.isfinite(ascending_look_n)
    & raw_desc_geometry_valid
)
sample_row, sample_column = np.nonzero(sample_mask)
sample_xy = np.column_stack(
    [east_grid[sample_mask], north_grid[sample_mask]]
)
sample_ascending_look = np.column_stack(
    [ascending_look_e[sample_mask], ascending_look_n[sample_mask]]
)
sample_descending_look = np.column_stack(
    [raw_desc_look["E"][sample_mask], raw_desc_look["N"][sample_mask]]
)
sample_ascending = ascending_hlos[:, sample_mask]
sample_descending = descending_hlos[:, sample_mask]

calibration_epoch = dates <= CALIBRATION_END
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", "All-NaN slice encountered")
    asc_vertical_variance = np.nanmedian(
        np.square(
            np.asarray(
                direct["ascending_vertical_to_los_sigma_cumulative_mm"],
                dtype=float,
            )[calibration_epoch]
        ),
        axis=0,
    )
    desc_vertical_variance = np.nanmedian(
        np.square(
            np.asarray(
                direct["descending_vertical_to_los_sigma_cumulative_mm"],
                dtype=float,
            )[calibration_epoch]
        ),
        axis=0,
    )
sample_ascending_noise_variance = (
    LOS_SIGMA_ASC_MM**2 + asc_vertical_variance[sample_mask]
)
sample_descending_noise_variance = (
    LOS_SIGMA_DESC_MM**2 + desc_vertical_variance[sample_mask]
)


def robust_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(
        0.7413
        * (
            np.percentile(finite, 75.0)
            - np.percentile(finite, 25.0)
        )
    )


calibration_east = observed_east[calibration_epoch][:, sample_mask]
calibration_north = observed_north[calibration_epoch][:, sample_mask]
calibration_east = (
    calibration_east
    - np.nanmedian(calibration_east, axis=1)[:, None]
)
calibration_north = (
    calibration_north
    - np.nanmedian(calibration_north, axis=1)[:, None]
)
pooled_east = calibration_east.ravel()
pooled_north = calibration_north.ravel()
pooled_finite = np.isfinite(pooled_east) & np.isfinite(pooled_north)
sigma_e_pool = max(robust_scale(pooled_east[pooled_finite]), 5.0)
sigma_n_pool = max(robust_scale(pooled_north[pooled_finite]), 5.0)
clip_e = np.clip(
    pooled_east[pooled_finite],
    -4.0 * sigma_e_pool,
    4.0 * sigma_e_pool,
)
clip_n = np.clip(
    pooled_north[pooled_finite],
    -4.0 * sigma_n_pool,
    4.0 * sigma_n_pool,
)
latent_covariance_base = np.cov(np.vstack([clip_e, clip_n]))
latent_covariance_base = 0.5 * (
    latent_covariance_base + latent_covariance_base.T
)
eigenvalue, eigenvector = np.linalg.eigh(latent_covariance_base)
latent_covariance_base = (
    eigenvector * np.maximum(eigenvalue, 25.0)[None, :]
) @ eigenvector.T
print("Cokriging samples:", int(sample_mask.sum()))
print("Pre-event latent covariance (mm^2):")
print(latent_covariance_base)


# %% [markdown]
# ## 5. Leakage-safe buffered spatial cross-validation
#
# Cokriging hyperparameters are selected only from twelve non-zero calibration
# epochs.  At each validation target, all paired samples within five kilometres
# are removed.  The conditioned model uses the collocated Track 64 LOS; the
# baseline uses the same paired neighbourhood but does not use target Track 64.

# %%
cv_lattice = np.zeros(east_grid.shape, dtype=bool)
cv_lattice[::4, ::4] = True
cv_target_mask = (
    near_fault
    & paired_persistent
    & cv_lattice
    & np.isfinite(ascending_look_e)
    & np.isfinite(ascending_look_n)
    & raw_desc_geometry_valid
)
cv_row, cv_column = np.nonzero(cv_target_mask)
cv_target_xy = np.column_stack(
    [east_grid[cv_target_mask], north_grid[cv_target_mask]]
)
cv_target_ascending_look = np.column_stack(
    [
        ascending_look_e[cv_target_mask],
        ascending_look_n[cv_target_mask],
    ]
)
cv_target_descending_look = np.column_stack(
    [
        raw_desc_look["E"][cv_target_mask],
        raw_desc_look["N"][cv_target_mask],
    ]
)
cv_target_ascending = ascending_hlos[:, cv_target_mask]
cv_target_descending_truth = descending_hlos[:, cv_target_mask]
cv_target_ascending_noise = (
    LOS_SIGMA_ASC_MM**2 + asc_vertical_variance[cv_target_mask]
)

calibration_indices_all = np.flatnonzero(calibration_epoch)
calibration_nonzero = calibration_indices_all[calibration_indices_all > 0]
calibration_cv_indices = np.unique(
    np.linspace(
        0,
        len(calibration_nonzero) - 1,
        12,
        dtype=int,
    )
)
calibration_cv_indices = calibration_nonzero[calibration_cv_indices]


def prediction_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    observed = np.asarray(truth, dtype=float)
    modelled = np.asarray(prediction, dtype=float)
    finite = np.isfinite(observed) & np.isfinite(modelled)
    total_truth = int(np.isfinite(observed).sum())
    if not finite.any():
        return {
            "count": 0.0,
            "coverage": 0.0,
            "mae_mm": float("nan"),
            "rmse_mm": float("nan"),
            "bias_mm": float("nan"),
            "correlation": float("nan"),
        }
    residual = modelled[finite] - observed[finite]
    if finite.sum() >= 3 and np.std(observed[finite]) > 0.0:
        correlation = float(
            np.corrcoef(observed[finite], modelled[finite])[0, 1]
        )
    else:
        correlation = float("nan")
    return {
        "count": float(finite.sum()),
        "coverage": float(finite.sum() / max(total_truth, 1)),
        "mae_mm": float(np.mean(np.abs(residual))),
        "rmse_mm": float(np.sqrt(np.mean(np.square(residual)))),
        "bias_mm": float(np.mean(residual)),
        "correlation": correlation,
    }


def build_cokriging(
    target_xy: np.ndarray,
    target_asc_look: np.ndarray,
    target_desc_look: np.ndarray,
    target_asc_noise: np.ndarray,
    *,
    length_scale_km: float,
    covariance_multiplier: float,
    condition_on_target_ascending: bool,
    exclusion_radius_km: float,
    target_noise_multiplier: float = 1.0,
):
    return build_fixed_fault_barrier_cokriging(
        sample_xy,
        target_xy,
        sample_ascending_look_en=sample_ascending_look,
        sample_descending_look_en=sample_descending_look,
        target_ascending_look_en=target_asc_look,
        target_descending_look_en=target_desc_look,
        fault_segments_xy_km=fault_segments_xy,
        latent_covariance_mm2=(
            covariance_multiplier * latent_covariance_base
        ),
        length_scale_km=length_scale_km,
        support_radius_km=CV_SUPPORT_RADIUS_KM,
        sample_ascending_noise_variance_mm2=(
            sample_ascending_noise_variance
        ),
        sample_descending_noise_variance_mm2=(
            sample_descending_noise_variance
        ),
        target_ascending_noise_variance_mm2=(
            target_noise_multiplier * target_asc_noise
        ),
        condition_on_target_ascending=condition_on_target_ascending,
        target_exclusion_radius_km=exclusion_radius_km,
        minimum_paired_samples=8,
        maximum_paired_samples=MAXIMUM_COKRIGING_SAMPLES,
    )


cv_parameter_rows: list[dict[str, float | str]] = []
cv_model_cache: dict[tuple[float, float, bool, float], object] = {}
for length_scale in (4.0, 8.0, 12.0):
    for covariance_multiplier in (0.5, 2.0):
        model_variants = [(False, 1.0)] + [
            (True, multiplier)
            for multiplier in (0.05, 0.2, 1.0, 5.0)
        ]
        for conditioned, target_noise_multiplier in model_variants:
            model = build_cokriging(
                cv_target_xy,
                cv_target_ascending_look,
                cv_target_descending_look,
                cv_target_ascending_noise,
                length_scale_km=length_scale,
                covariance_multiplier=covariance_multiplier,
                condition_on_target_ascending=conditioned,
                exclusion_radius_km=CV_BUFFER_KM,
                target_noise_multiplier=target_noise_multiplier,
            )
            prediction = evaluate_fixed_fault_barrier_cokriging(
                model,
                sample_ascending[calibration_cv_indices],
                sample_descending[calibration_cv_indices],
                (
                    cv_target_ascending[calibration_cv_indices]
                    if conditioned
                    else None
                ),
            )
            metrics = prediction_metrics(
                cv_target_descending_truth[calibration_cv_indices],
                prediction["descending_los_mm"],
            )
            cv_parameter_rows.append(
                {
                    "model": (
                        "M1_Track64_conditioned"
                        if conditioned
                        else "M0_paired_only"
                    ),
                    "length_scale_km": length_scale,
                    "covariance_multiplier": covariance_multiplier,
                    "target_ascending_noise_multiplier": (
                        target_noise_multiplier
                    ),
                    "valid_targets": int(model.valid.sum()),
                    **metrics,
                }
            )
            cv_model_cache[
                (
                    length_scale,
                    covariance_multiplier,
                    conditioned,
                    target_noise_multiplier,
                )
            ] = model
            print(
                "CV",
                "M1" if conditioned else "M0",
                "ell",
                length_scale,
                "multiplier",
                covariance_multiplier,
                "target-noise",
                target_noise_multiplier,
                "MAE",
                f"{metrics['mae_mm']:.2f}",
                "coverage",
                f"{metrics['coverage']:.3f}",
            )

cv_parameter_table = pd.DataFrame(cv_parameter_rows)
cv_parameter_table.to_csv(
    OUTPUT_DIR / "cokriging_hyperparameter_spatial_cv.csv",
    index=False,
)


def select_best_model(model_name: str) -> pd.Series:
    candidates = cv_parameter_table.loc[
        (cv_parameter_table["model"] == model_name)
        & (cv_parameter_table["coverage"] >= 0.85)
    ].copy()
    if candidates.empty:
        raise RuntimeError(f"No {model_name} candidate passed CV coverage")
    return candidates.sort_values(
        ["mae_mm", "rmse_mm", "length_scale_km"]
    ).iloc[0]


best_m1 = select_best_model("M1_Track64_conditioned")
best_m0 = select_best_model("M0_paired_only")
best_length = float(best_m1["length_scale_km"])
best_multiplier = float(best_m1["covariance_multiplier"])
best_target_noise_multiplier = float(
    best_m1["target_ascending_noise_multiplier"]
)

best_cv_m1 = cv_model_cache[
    (
        best_length,
        best_multiplier,
        True,
        best_target_noise_multiplier,
    )
]
best_cv_m0 = cv_model_cache[
    (
        float(best_m0["length_scale_km"]),
        float(best_m0["covariance_multiplier"]),
        False,
        1.0,
    )
]
best_cv_prediction_m1 = evaluate_fixed_fault_barrier_cokriging(
    best_cv_m1,
    sample_ascending,
    sample_descending,
    cv_target_ascending,
)
best_cv_prediction_m0 = evaluate_fixed_fault_barrier_cokriging(
    best_cv_m0,
    sample_ascending,
    sample_descending,
)

validation_periods = {
    "calibration_selected": calibration_cv_indices,
    "surveillance_22Jun": np.flatnonzero(
        dates == pd.Timestamp("2019-06-22")
    ),
    "surveillance_04Jul": np.flatnonzero(
        dates == pd.Timestamp("2019-07-04")
    ),
    "event_control_16Jul": np.flatnonzero(
        dates == pd.Timestamp("2019-07-16")
    ),
}
cv_summary_rows: list[dict[str, float | str]] = []
for period_name, epoch_indices in validation_periods.items():
    for model_name, prediction in (
        ("M0_paired_only", best_cv_prediction_m0),
        ("M1_Track64_conditioned", best_cv_prediction_m1),
    ):
        metrics = prediction_metrics(
            cv_target_descending_truth[epoch_indices],
            prediction["descending_los_mm"][epoch_indices],
        )
        cv_summary_rows.append(
            {
                "period": period_name,
                "model": model_name,
                **metrics,
            }
        )
cv_summary = pd.DataFrame(cv_summary_rows)
cv_summary.to_csv(
    OUTPUT_DIR / "cokriging_spatial_cv_summary.csv",
    index=False,
)

calibration_m0_mae = float(
    cv_summary.loc[
        (cv_summary["period"] == "calibration_selected")
        & (cv_summary["model"] == "M0_paired_only"),
        "mae_mm",
    ].iloc[0]
)
calibration_m1_mae = float(
    cv_summary.loc[
        (cv_summary["period"] == "calibration_selected")
        & (cv_summary["model"] == "M1_Track64_conditioned"),
        "mae_mm",
    ].iloc[0]
)
track64_cv_gain = (
    calibration_m0_mae - calibration_m1_mae
) / calibration_m0_mae
print("Selected M1 length scale:", best_length)
print("Selected M1 covariance multiplier:", best_multiplier)
print(
    "Selected M1 target-noise multiplier:",
    best_target_noise_multiplier,
)
print("Calibration Track-64 MAE gain:", track64_cv_gain)


# %% [markdown]
# ## 6. Fixed production reconstruction and exact-mask validation

# %%
fill_target_ascending_look = np.column_stack(
    [
        ascending_look_e[fill_target_mask],
        ascending_look_n[fill_target_mask],
    ]
)
fill_target_descending_look = np.column_stack(
    [
        raw_desc_look["E"][fill_target_mask],
        raw_desc_look["N"][fill_target_mask],
    ]
)
fill_target_ascending_noise = (
    LOS_SIGMA_ASC_MM**2 + asc_vertical_variance[fill_target_mask]
)
production_model = build_cokriging(
    fill_target_xy,
    fill_target_ascending_look,
    fill_target_descending_look,
    fill_target_ascending_noise,
    length_scale_km=best_length,
    covariance_multiplier=best_multiplier,
    condition_on_target_ascending=True,
    exclusion_radius_km=0.0,
    target_noise_multiplier=best_target_noise_multiplier,
)
production_baseline = build_cokriging(
    fill_target_xy,
    fill_target_ascending_look,
    fill_target_descending_look,
    fill_target_ascending_noise,
    length_scale_km=float(best_m0["length_scale_km"]),
    covariance_multiplier=float(best_m0["covariance_multiplier"]),
    condition_on_target_ascending=False,
    exclusion_radius_km=0.0,
)
production_prediction = evaluate_fixed_fault_barrier_cokriging(
    production_model,
    sample_ascending,
    sample_descending,
    ascending_hlos[:, fill_target_mask],
)
production_prediction_m0 = evaluate_fixed_fault_barrier_cokriging(
    production_baseline,
    sample_ascending,
    sample_descending,
)
if not np.all(production_model.valid):
    print(
        "Warning: unsupported production targets:",
        int((~production_model.valid).sum()),
    )

# The 73 cells missing only on 4 July form an exact target-geometry validation
# subset because they contain accepted Track 71 values during calibration.
dynamic_gap_mask = (
    fill_target_mask
    & np.isfinite(descending_hlos[0])
    & ~np.isfinite(
        descending_hlos[
            int(np.flatnonzero(dates == pd.Timestamp("2019-07-04"))[0])
        ]
    )
)
dynamic_flat_in_fill = dynamic_gap_mask[fill_target_mask]
exact_mask_rows: list[dict[str, float | str]] = []
for model_name, prediction in (
    ("M0_paired_only", production_prediction_m0),
    ("M1_Track64_conditioned", production_prediction),
):
    metrics = prediction_metrics(
        descending_hlos[calibration_cv_indices][:, dynamic_gap_mask],
        prediction["descending_los_mm"][
            calibration_cv_indices
        ][:, dynamic_flat_in_fill],
    )
    exact_mask_rows.append({"model": model_name, **metrics})
exact_mask_cv = pd.DataFrame(exact_mask_rows)
exact_mask_cv.to_csv(
    OUTPUT_DIR / "actual_04July_gap_shape_calibration_cv.csv",
    index=False,
)
calibration_exact_m0 = float(
    exact_mask_cv.loc[
        exact_mask_cv["model"] == "M0_paired_only",
        "mae_mm",
    ].iloc[0]
)
calibration_exact_m1 = float(
    exact_mask_cv.loc[
        exact_mask_cv["model"] == "M1_Track64_conditioned",
        "mae_mm",
    ].iloc[0]
)
exact_mask_gain = (
    calibration_exact_m0 - calibration_exact_m1
) / calibration_exact_m0
reconstruction_supported = bool(
    track64_cv_gain > 0.0
    and exact_mask_gain > 0.0
    and float(best_m1["coverage"]) >= 0.85
)
filled_descending_hlos = descending_hlos.copy()
filled_east = observed_east.copy()
filled_north = observed_north.copy()
filled_sigma_east = sigma_east.copy()
filled_sigma_north = sigma_north.copy()
filled_covariance_east_north = covariance_east_north.copy()
imputed_descending = np.zeros(descending_hlos.shape, dtype=bool)

posterior = production_model.posterior_covariance
for target_index, (row, column) in enumerate(
    zip(fill_row, fill_column, strict=True)
):
    missing = (
        np.isfinite(ascending_hlos[:, row, column])
        & ~np.isfinite(descending_hlos[:, row, column])
        & np.isfinite(
            production_prediction["descending_los_mm"][:, target_index]
        )
    )
    if not missing.any():
        continue
    filled_descending_hlos[missing, row, column] = (
        production_prediction["descending_los_mm"][missing, target_index]
    )
    filled_east[missing, row, column] = (
        production_prediction["east_mm"][missing, target_index]
    )
    filled_north[missing, row, column] = (
        production_prediction["north_mm"][missing, target_index]
    )
    filled_sigma_east[missing, row, column] = math.sqrt(
        max(float(posterior[target_index, 0, 0]), 0.0)
    )
    filled_sigma_north[missing, row, column] = math.sqrt(
        max(float(posterior[target_index, 1, 1]), 0.0)
    )
    filled_covariance_east_north[missing, row, column] = (
        posterior[target_index, 0, 1]
    )
    imputed_descending[missing, row, column] = True

# Complete the few near-fault cells where neither a direct two-track E-N
# solution nor collocated Track 64 conditioning is available.  The already
# selected paired-only M0 model is applied only to these remaining cells.
remaining_spatial_mask = near_fault & ~(
    np.all(np.isfinite(filled_east), axis=0)
    & np.all(np.isfinite(filled_north), axis=0)
)
spatial_row, spatial_column = np.nonzero(remaining_spatial_mask)
spatial_xy = np.column_stack(
    [
        east_grid[remaining_spatial_mask],
        north_grid[remaining_spatial_mask],
    ]
)
if not np.all(raw_desc_geometry_valid[remaining_spatial_mask]):
    raise RuntimeError("Track 71 geometry is missing at an M0 completion target")

# The ascending look/noise arrays satisfy the shared API but do not enter the
# unconditioned M0 weights.
spatial_placeholder_asc_look = np.repeat(
    np.nanmedian(sample_ascending_look, axis=0)[None, :],
    len(spatial_xy),
    axis=0,
)
spatial_placeholder_asc_noise = np.full(
    len(spatial_xy),
    float(np.nanmedian(sample_ascending_noise_variance)),
)
spatial_target_desc_look = np.column_stack(
    [
        raw_desc_look["E"][remaining_spatial_mask],
        raw_desc_look["N"][remaining_spatial_mask],
    ]
)
spatial_m0_model = build_cokriging(
    spatial_xy,
    spatial_placeholder_asc_look,
    spatial_target_desc_look,
    spatial_placeholder_asc_noise,
    length_scale_km=float(best_m0["length_scale_km"]),
    covariance_multiplier=float(best_m0["covariance_multiplier"]),
    condition_on_target_ascending=False,
    exclusion_radius_km=0.0,
)
if not np.all(spatial_m0_model.valid):
    raise RuntimeError(
        f"Unsupported M0 completion targets: "
        f"{int((~spatial_m0_model.valid).sum())}"
    )
spatial_m0_prediction = evaluate_fixed_fault_barrier_cokriging(
    spatial_m0_model,
    sample_ascending,
    sample_descending,
)
spatial_posterior = spatial_m0_model.posterior_covariance
spatial_only_imputation = np.zeros(filled_east.shape, dtype=bool)
for target_index, (row, column) in enumerate(
    zip(spatial_row, spatial_column, strict=True)
):
    predicted_east = spatial_m0_prediction["east_mm"][:, target_index]
    predicted_north = spatial_m0_prediction["north_mm"][:, target_index]
    missing_en = (
        (
            ~np.isfinite(filled_east[:, row, column])
            | ~np.isfinite(filled_north[:, row, column])
        )
        & np.isfinite(predicted_east)
        & np.isfinite(predicted_north)
    )
    filled_east[missing_en, row, column] = predicted_east[missing_en]
    filled_north[missing_en, row, column] = predicted_north[missing_en]
    filled_sigma_east[missing_en, row, column] = math.sqrt(
        max(float(spatial_posterior[target_index, 0, 0]), 0.0)
    )
    filled_sigma_north[missing_en, row, column] = math.sqrt(
        max(float(spatial_posterior[target_index, 1, 1]), 0.0)
    )
    filled_covariance_east_north[missing_en, row, column] = (
        spatial_posterior[target_index, 0, 1]
    )
    spatial_only_imputation[missing_en, row, column] = True

    predicted_descending = spatial_m0_prediction[
        "descending_los_mm"
    ][:, target_index]
    missing_descending = (
        ~np.isfinite(filled_descending_hlos[:, row, column])
        & np.isfinite(predicted_descending)
    )
    filled_descending_hlos[missing_descending, row, column] = (
        predicted_descending[missing_descending]
    )

complete_near_fault_en = (
    np.all(np.isfinite(filled_east), axis=0)
    & np.all(np.isfinite(filled_north), axis=0)
)
if not np.all(complete_near_fault_en[near_fault]):
    raise RuntimeError("Full near-fault E-N completion failed")

max_observed_overwrite = float(
    np.nanmax(
        np.abs(
            filled_descending_hlos[
                np.isfinite(descending_hlos)
            ]
            - descending_hlos[np.isfinite(descending_hlos)]
        )
    )
)
if max_observed_overwrite != 0.0:
    raise RuntimeError("An observed Track 71 cell was overwritten")

production_diagnostics = pd.DataFrame(
    cokriging_diagnostics(production_model)
)
production_diagnostics.insert(
    0,
    "longitude",
    longitude_grid[fill_target_mask],
)
production_diagnostics.insert(
    1,
    "latitude",
    latitude_grid[fill_target_mask],
)
production_diagnostics.insert(
    2,
    "distance_to_mapped_rupture_km",
    distance_grid[fill_target_mask],
)
production_diagnostics.to_csv(
    OUTPUT_DIR / "track64_guided_reconstruction_diagnostics.csv",
    index=False,
)

np.savez_compressed(
    OUTPUT_DIR / "track64_guided_cumulative_en.npz",
    dates=np.asarray(dates, dtype="datetime64[ns]"),
    east_km=east_grid,
    north_km=north_grid,
    latitude=latitude_grid,
    longitude=longitude_grid,
    distance_to_mapped_rupture_km=distance_grid.astype(np.float32),
    observed_descending_hlos_cumulative_mm=(
        descending_hlos.astype(np.float32)
    ),
    filled_descending_hlos_cumulative_mm=(
        filled_descending_hlos.astype(np.float32)
    ),
    track64_guided_imputation_mask=imputed_descending,
    filled_cumulative_east_mm=filled_east.astype(np.float32),
    filled_cumulative_north_mm=filled_north.astype(np.float32),
    filled_sigma_east_mm=filled_sigma_east.astype(np.float32),
    filled_sigma_north_mm=filled_sigma_north.astype(np.float32),
    filled_covariance_east_north_mm2=(
        filled_covariance_east_north.astype(np.float32)
    ),
    reconstruction_target_mask=fill_target_mask,
    spatial_only_target_mask=remaining_spatial_mask,
    spatial_only_imputation_mask=spatial_only_imputation,
    full_near_fault_target_mask=near_fault,
    former_core_distance_mask=former_core_distance_mask,
    posterior_covariance_target_en_desc_mm2=(
        posterior.astype(np.float32)
    ),
    spatial_only_posterior_covariance_en_desc_mm2=(
        spatial_posterior.astype(np.float32)
    ),
)
print("Observed Track 71 overwrite error (mm):", max_observed_overwrite)
print("Total imputed epoch-cells:", int(imputed_descending.sum()))


# %% [markdown]
# ## 7. Fault-side-aware cumulative strain
#
# A fixed two-kilometre displacement sample lattice and ten-kilometre joint
# E-N affine support are used at all epochs.  Target-sample links crossing the
# Paxton Ranch or Salt Wells finite barrier are removed.  The fixed-resolution
# local gradient is evaluated at every completed near-fault cell, including the
# mapped rupture zone.

# %%
filled_persistent = (
    np.all(np.isfinite(filled_east), axis=0)
    & np.all(np.isfinite(filled_north), axis=0)
)
strain_sample_mask = (
    filled_persistent
    & sample_lattice
)
strain_target_mask = (
    filled_persistent
    & near_fault
)
strain_sample_xy = np.column_stack(
    [
        east_grid[strain_sample_mask],
        north_grid[strain_sample_mask],
    ]
)
strain_target_row, strain_target_column = np.nonzero(strain_target_mask)
strain_target_xy = np.column_stack(
    [
        east_grid[strain_target_mask],
        north_grid[strain_target_mask],
    ]
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", "All-NaN slice encountered")
    variance_east_fixed = np.nanmedian(
        np.square(filled_sigma_east[calibration_epoch]),
        axis=0,
    )
    variance_north_fixed = np.nanmedian(
        np.square(filled_sigma_north[calibration_epoch]),
        axis=0,
    )
    covariance_en_fixed = np.nanmedian(
        filled_covariance_east_north[calibration_epoch],
        axis=0,
    )
sample_covariance = np.zeros(
    (int(strain_sample_mask.sum()), 2, 2),
    dtype=float,
)
sample_covariance[:, 0, 0] = variance_east_fixed[strain_sample_mask]
sample_covariance[:, 1, 1] = variance_north_fixed[strain_sample_mask]
sample_covariance[:, 0, 1] = covariance_en_fixed[strain_sample_mask]
sample_covariance[:, 1, 0] = covariance_en_fixed[strain_sample_mask]

strain_model = build_fixed_joint_mls(
    strain_sample_xy,
    strain_target_xy,
    covariance_en_mm2=sample_covariance,
    support_radius_km=STRAIN_SUPPORT_RADIUS_KM,
    bandwidth_km=STRAIN_BANDWIDTH_KM,
    min_samples=STRAIN_MINIMUM_SAMPLES,
    max_condition_number=1.0e8,
    covariance_absolute_floor_mm2=1.0,
    fault_segments_xy_km=fault_segments_xy,
)
strain_values_target = evaluate_fixed_joint_mls(
    strain_model,
    filled_east[:, strain_sample_mask],
    filled_north[:, strain_sample_mask],
)
strain_sigma_target = fixed_joint_mls_component_sigma(strain_model)
strain_values = {
    name: target_values_to_grid(
        values,
        strain_target_row,
        strain_target_column,
        east_grid.shape,
    ).astype(np.float32)
    for name, values in strain_values_target.items()
}
strain_sigmas = {
    f"sigma_{name}": target_values_to_grid(
        values,
        strain_target_row,
        strain_target_column,
        east_grid.shape,
    ).astype(np.float32)
    for name, values in strain_sigma_target.items()
}
supported_strain_mask = np.zeros(east_grid.shape, dtype=bool)
supported_strain_mask[
    strain_target_row,
    strain_target_column,
] = strain_model.valid
missing_full_area_strain = near_fault & ~supported_strain_mask
if missing_full_area_strain.any():
    raise RuntimeError(
        "Full-area strain coverage failed for "
        f"{int(missing_full_area_strain.sum())} near-fault cells"
    )

sample_reconstructed_any = np.any(
    imputed_descending | spatial_only_imputation,
    axis=0,
)[strain_sample_mask]
local_reconstructed_fraction = np.full(
    len(strain_target_xy),
    np.nan,
    dtype=float,
)
for target_index in np.flatnonzero(strain_model.valid):
    neighbour = strain_model.neighbour_indices[target_index]
    local_reconstructed_fraction[target_index] = float(
        np.mean(sample_reconstructed_any[neighbour])
    )
local_reconstructed_fraction_grid = target_values_to_grid(
    local_reconstructed_fraction,
    strain_target_row,
    strain_target_column,
    east_grid.shape,
).astype(np.float32)

strain_output_payload = {
    "dates": np.asarray(dates, dtype="datetime64[ns]"),
    "east_km": east_grid,
    "north_km": north_grid,
    "latitude": latitude_grid,
    "longitude": longitude_grid,
    "distance_to_mapped_rupture_km": distance_grid.astype(np.float32),
    "strain_target_mask": supported_strain_mask,
    "full_near_fault_target_mask": near_fault,
    "former_core_distance_mask": former_core_distance_mask,
    "track64_guided_target_mask": fill_target_mask,
    "spatial_only_target_mask": remaining_spatial_mask,
    "local_reconstructed_sample_fraction": (
        local_reconstructed_fraction_grid
    ),
    "local_sample_count": target_values_to_grid(
        strain_model.sample_count,
        strain_target_row,
        strain_target_column,
        east_grid.shape,
    ).astype(np.float32),
    "local_effective_sample_count": target_values_to_grid(
        strain_model.effective_sample_count,
        strain_target_row,
        strain_target_column,
        east_grid.shape,
    ).astype(np.float32),
    "local_condition_number": target_values_to_grid(
        strain_model.condition_number,
        strain_target_row,
        strain_target_column,
        east_grid.shape,
    ).astype(np.float32),
    "local_barrier_excluded_count": target_values_to_grid(
        strain_model.barrier_excluded_count,
        strain_target_row,
        strain_target_column,
        east_grid.shape,
    ).astype(np.float32),
    **strain_values,
    **strain_sigmas,
}
np.savez_compressed(
    OUTPUT_DIR / "near_fault_cumulative_strain_full_area_1km.npz",
    **strain_output_payload,
)
# Preserve the earlier filename for downstream notebooks while replacing its
# contents with the same full-area product.
np.savez_compressed(
    OUTPUT_DIR / "near_fault_cumulative_strain_1km.npz",
    **strain_output_payload,
)
print(
    "Near-fault side-aware strain support:",
    int(supported_strain_mask.sum()),
    "/",
    int(strain_target_mask.sum()),
)


# %% [markdown]
# ## 8. Publication-facing validation and coverage figures

# %%
def near_fault_extent(padding_deg: float = 0.035) -> tuple[float, float, float, float]:
    mask = near_fault & np.isfinite(longitude_grid) & np.isfinite(latitude_grid)
    return (
        float(np.nanmin(longitude_grid[mask]) - padding_deg),
        float(np.nanmax(longitude_grid[mask]) + padding_deg),
        float(np.nanmin(latitude_grid[mask]) - padding_deg),
        float(np.nanmax(latitude_grid[mask]) + padding_deg),
    )


MAP_EXTENT = near_fault_extent()


def format_map_axis(
    axis: plt.Axes,
    *,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
) -> None:
    axis.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
    axis.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color="0.85", linewidth=0.45, linestyle=":", zorder=0)
    axis.set_xlabel("Longitude" if show_xlabel else "")
    axis.set_ylabel("Latitude" if show_ylabel else "")
    if not show_xlabel:
        axis.tick_params(labelbottom=False)
    if not show_ylabel:
        axis.tick_params(labelleft=False)
    add_faults(axis)


july4_index = int(
    np.flatnonzero(dates == pd.Timestamp("2019-07-04"))[0]
)
july16_index = int(
    np.flatnonzero(dates == pd.Timestamp("2019-07-16"))[0]
)
june22_index = int(
    np.flatnonzero(dates == pd.Timestamp("2019-06-22"))[0]
)

coverage_class = np.full(east_grid.shape, np.nan, dtype=float)
both_july4 = (
    near_fault
    & np.isfinite(ascending_hlos[july4_index])
    & np.isfinite(descending_hlos[july4_index])
)
asc_only_july4 = (
    near_fault
    & np.isfinite(ascending_hlos[july4_index])
    & ~np.isfinite(descending_hlos[july4_index])
)
spatial_only_july4 = remaining_spatial_mask
coverage_class[both_july4] = 0.0
coverage_class[asc_only_july4] = 1.0
coverage_class[spatial_only_july4] = 2.0

coverage_cmap = ListedColormap(
    ["#2f7f4f", "#f4b942", "#4c78a8"]
)
coverage_norm = BoundaryNorm(
    np.arange(-0.5, 3.5, 1.0),
    coverage_cmap.N,
)
fig, axis = plt.subplots(figsize=(8.0, 7.0), constrained_layout=True)
axis.pcolormesh(
    longitude_grid,
    latitude_grid,
    coverage_class,
    cmap=coverage_cmap,
    norm=coverage_norm,
    shading="nearest",
    rasterized=True,
)
format_map_axis(axis)
axis.set_title(
    "4 July 2019 complete near-fault E-N coverage"
)
legend_handles = [
    Line2D(
        [0],
        [0],
        marker="s",
        linestyle="none",
        color=color,
        markersize=9,
        label=label,
    )
    for color, label in (
        ("#2f7f4f", f"Both tracks observed ({int(both_july4.sum()):,})"),
        (
            "#f4b942",
            f"Track 64 observed; Track 71 reconstructed ({int(asc_only_july4.sum()):,})",
        ),
        (
            "#4c78a8",
            "Paired-neighbour spatial completion "
            f"({int(spatial_only_july4.sum()):,})",
        ),
    )
]
legend_handles.extend(
    [
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.0,
            label="Mapped rupture",
        ),
        Line2D(
            [0],
            [0],
            color="#f4c430",
            linewidth=1.5,
            linestyle="--",
            label="Finite interpolation barrier",
        ),
    ]
)
axis.legend(
    handles=legend_handles,
    loc="upper left",
    frameon=True,
    framealpha=0.92,
)
fig.savefig(OUTPUT_DIR / "01_track64_fill_coverage_20190704.png")
plt.close(fig)


# %%
period_order = [
    "calibration_selected",
    "surveillance_22Jun",
    "surveillance_04Jul",
    "event_control_16Jul",
]
model_order = ["M0_paired_only", "M1_Track64_conditioned"]
period_labels = [
    "Calibration\n(≤29 May)",
    "22 June",
    "4 July",
    "16 July\n(event control)",
]
fig, axes = plt.subplots(
    1,
    2,
    figsize=(11.8, 4.8),
    constrained_layout=True,
)
x = np.arange(len(period_order))
width = 0.36
for model_index, (model_name, label, color) in enumerate(
    (
        ("M0_paired_only", "Paired-track spatial model", "#7f7f7f"),
        (
            "M1_Track64_conditioned",
            "Track-64-conditioned model",
            "#2c7fb8",
        ),
    )
):
    values = [
        float(
            cv_summary.loc[
                (cv_summary["period"] == period)
                & (cv_summary["model"] == model_name),
                "mae_mm",
            ].iloc[0]
        )
        for period in period_order
    ]
    axes[0].bar(
        x + (model_index - 0.5) * width,
        values,
        width=width,
        label=label,
        color=color,
    )
axes[0].set_xticks(x, period_labels)
axes[0].set_ylabel("Buffered spatial-CV MAE (mm)")
axes[0].set_title("(a) Independent Track 71 prediction error")
axes[0].grid(axis="y", color="0.85", linestyle=":", linewidth=0.6)
axes[0].legend(frameon=False)

truth_calibration = cv_target_descending_truth[calibration_cv_indices]
prediction_calibration = best_cv_prediction_m1["descending_los_mm"][
    calibration_cv_indices
]
finite_calibration = (
    np.isfinite(truth_calibration)
    & np.isfinite(prediction_calibration)
)
axes[1].scatter(
    truth_calibration[finite_calibration],
    prediction_calibration[finite_calibration],
    s=8,
    alpha=0.28,
    color="#2c7fb8",
    edgecolors="none",
)
value_min = float(
    np.nanmin(
        [
            np.nanmin(truth_calibration[finite_calibration]),
            np.nanmin(prediction_calibration[finite_calibration]),
        ]
    )
)
value_max = float(
    np.nanmax(
        [
            np.nanmax(truth_calibration[finite_calibration]),
            np.nanmax(prediction_calibration[finite_calibration]),
        ]
    )
)
padding = max(0.05 * (value_max - value_min), 1.0)
axes[1].plot(
    [value_min - padding, value_max + padding],
    [value_min - padding, value_max + padding],
    color="black",
    linewidth=0.9,
    linestyle="--",
)
axes[1].set_xlim(value_min - padding, value_max + padding)
axes[1].set_ylim(value_min - padding, value_max + padding)
axes[1].set_aspect("equal", adjustable="box")
axes[1].set_xlabel("Observed Track 71 horizontal LOS (mm)")
axes[1].set_ylabel("Predicted Track 71 horizontal LOS (mm)")
axes[1].set_title(
    "(b) Track-64-conditioned calibration predictions\n"
    f"MAE = {calibration_m1_mae:.1f} mm; "
    f"gain over M0 = {100.0 * track64_cv_gain:.1f}%"
)
axes[1].grid(color="0.88", linestyle=":", linewidth=0.55)
fig.savefig(OUTPUT_DIR / "02_cokriging_spatial_validation.png")
plt.close(fig)


# %%
observed_july4 = descending_hlos[july4_index].copy()
filled_july4 = filled_descending_hlos[july4_index].copy()
display_mask = near_fault
combined_values = np.concatenate(
    [
        observed_july4[display_mask & np.isfinite(observed_july4)],
        filled_july4[display_mask & np.isfinite(filled_july4)],
    ]
)
los_limit = float(
    max(
        np.nanpercentile(np.abs(combined_values), 98.5),
        20.0,
    )
)
target_sigma_grid = np.full(east_grid.shape, np.nan, dtype=float)
target_sigma_grid[fill_target_mask] = np.sqrt(
    np.maximum(posterior[:, 2, 2], 0.0)
)
target_sigma_grid[remaining_spatial_mask] = np.sqrt(
    np.maximum(spatial_posterior[:, 2, 2], 0.0)
)
completion_target_mask = fill_target_mask | remaining_spatial_mask
fig, axes = plt.subplots(
    1,
    3,
    figsize=(15.0, 5.4),
    constrained_layout=True,
)
los_meshes = []
for panel_index, (axis, values, title) in enumerate(
    (
        (
            axes[0],
            observed_july4,
            "(a) Observed Track 71\nquality gaps retained",
        ),
        (
            axes[1],
            filled_july4,
            "(b) Observed + complete\nnear-fault reconstruction",
        ),
    )
):
    mesh = axis.pcolormesh(
        longitude_grid,
        latitude_grid,
        np.where(display_mask, values, np.nan),
        cmap="RdBu_r",
        norm=TwoSlopeNorm(
            vmin=-los_limit,
            vcenter=0.0,
            vmax=los_limit,
        ),
        shading="nearest",
        rasterized=True,
    )
    los_meshes.append(mesh)
    axis.set_title(title)
    format_map_axis(
        axis,
        show_ylabel=panel_index == 0,
    )
sigma_limit = float(
    np.nanpercentile(
        target_sigma_grid[completion_target_mask],
        98.0,
    )
)
sigma_mesh = axes[2].pcolormesh(
    longitude_grid,
    latitude_grid,
    np.where(completion_target_mask, target_sigma_grid, np.nan),
    cmap="magma",
    vmin=0.0,
    vmax=sigma_limit,
    shading="nearest",
    rasterized=True,
)
axes[2].set_title("(c) Reconstructed Track 71\nposterior standard deviation")
format_map_axis(axes[2], show_ylabel=False)
los_colorbar = fig.colorbar(
    los_meshes[-1],
    ax=axes[:2],
    orientation="horizontal",
    fraction=0.055,
    pad=0.06,
)
los_colorbar.set_label("Cumulative horizontal LOS (mm)")
sigma_colorbar = fig.colorbar(
    sigma_mesh,
    ax=axes[2],
    orientation="horizontal",
    fraction=0.055,
    pad=0.06,
)
sigma_colorbar.set_label("Posterior σ (mm)")
fig.suptitle(
    "4 July 2019 descending-track cumulative horizontal LOS",
    fontsize=14,
    fontweight="semibold",
)
fig.savefig(
    OUTPUT_DIR / "03_track71_reconstruction_20190704.png"
)
plt.close(fig)


# %% [markdown]
# ## 9. Continuous cumulative displacement and strain maps

# %%
key_date_indices = [june22_index, july4_index, july16_index]
key_date_labels = ["22 June 2019", "4 July 2019", "16 July 2019"]

fig, axes = plt.subplots(
    2,
    3,
    figsize=(14.5, 9.0),
    constrained_layout=True,
)
for row_index, (field, name, unit) in enumerate(
    (
        (filled_east, "East", "mm"),
        (filled_north, "North", "mm"),
    )
):
    selected = np.asarray(
        [field[index] for index in key_date_indices],
        dtype=float,
    )
    finite = selected[:, near_fault]
    limit = float(max(np.nanpercentile(np.abs(finite), 98.5), 20.0))
    row_mesh = None
    for column_index, (epoch_index, date_label) in enumerate(
        zip(key_date_indices, key_date_labels, strict=True)
    ):
        axis = axes[row_index, column_index]
        row_mesh = axis.pcolormesh(
            longitude_grid,
            latitude_grid,
            np.where(near_fault, field[epoch_index], np.nan),
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            shading="nearest",
            rasterized=True,
        )
        axis.set_title(
            f"({chr(97 + row_index * 3 + column_index)}) "
            f"{name}; {date_label}"
        )
        format_map_axis(
            axis,
            show_xlabel=row_index == 1,
            show_ylabel=column_index == 0,
        )
    colorbar = fig.colorbar(
        row_mesh,
        ax=axes[row_index, :],
        orientation="vertical",
        fraction=0.024,
        pad=0.015,
    )
    colorbar.set_label(f"Cumulative {name.lower()} displacement ({unit})")
fig.suptitle(
    "Near-fault cumulative horizontal displacement",
    fontsize=14,
    fontweight="semibold",
)
fig.savefig(OUTPUT_DIR / "04_near_fault_cumulative_en_key_dates.png")
plt.close(fig)


component_specs = {
    "epsilon_EE_microstrain": ("Normal east strain", r"$\epsilon_{EE}$ ($\mu$strain)"),
    "epsilon_NN_microstrain": ("Normal north strain", r"$\epsilon_{NN}$ ($\mu$strain)"),
    "gamma_EN_microstrain": ("Engineering shear", r"$\gamma_{EN}$ ($\mu$strain)"),
    "dilatation_microstrain": ("Dilatation", r"$\epsilon_{EE}+\epsilon_{NN}$ ($\mu$strain)"),
    "rotation_microradian": ("Vertical-axis rotation", r"$\omega$ ($\mu$rad)"),
}
for component_name, (component_title, colorbar_label) in component_specs.items():
    values = np.asarray(strain_values[component_name], dtype=float)
    selected = values[key_date_indices]
    finite = selected[:, supported_strain_mask]
    limit = float(max(np.nanpercentile(np.abs(finite), 98.0), 1.0))
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.3, 5.0),
        constrained_layout=True,
    )
    mesh = None
    for panel_index, (axis, epoch_index, date_label) in enumerate(
        zip(axes, key_date_indices, key_date_labels, strict=True)
    ):
        mesh = axis.pcolormesh(
            longitude_grid,
            latitude_grid,
            values[epoch_index],
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            shading="nearest",
            rasterized=True,
        )
        axis.set_title(f"({chr(97 + panel_index)}) {date_label}")
        format_map_axis(
            axis,
            show_ylabel=panel_index == 0,
        )
    colorbar = fig.colorbar(
        mesh,
        ax=axes,
        orientation="horizontal",
        fraction=0.06,
        pad=0.07,
    )
    colorbar.set_label(colorbar_label)
    fig.suptitle(
        f"Near-fault cumulative {component_title.lower()}",
        fontsize=14,
        fontweight="semibold",
    )
    safe_name = component_name.replace("_microstrain", "").replace(
        "_microradian",
        "",
    )
    fig.savefig(
        OUTPUT_DIR / f"05_cumulative_{safe_name}_key_dates.png"
    )
    plt.close(fig)


# %% [markdown]
# ## 10. Cumulative strain-lobe time series and finite-aperture fault jump
#
# A signed median over the complete near-fault domain is not an informative
# summary of a bipolar strain field: positive and negative lobes cancel.  The
# lower and upper spatial lobes are therefore defined once from the 29 May 2019
# map (the lower and upper 20% of supported cells) and the same cells are
# followed through every cumulative epoch.  This preserves spatial membership
# while showing whether the mapped lobes strengthen, weaken, or reverse.
#
# The shaded envelopes are the median propagated pointwise one-standard-
# deviation uncertainty inside each fixed lobe.  They are not confidence
# intervals for the spatial median.

# %%
lobe_reference_index = int(
    np.flatnonzero(dates == CALIBRATION_END)[0]
)
lobe_masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
regional_rows: list[dict[str, float | int | str | pd.Timestamp]] = []

for component_name in component_specs:
    reference_values = np.asarray(
        strain_values[component_name][lobe_reference_index],
        dtype=float,
    )
    reference_use = (
        supported_strain_mask
        & np.isfinite(reference_values)
    )
    lower_threshold, upper_threshold = np.nanpercentile(
        reference_values[reference_use],
        [20.0, 80.0],
    )
    lower_lobe_mask = (
        reference_use
        & (reference_values <= lower_threshold)
    )
    upper_lobe_mask = (
        reference_use
        & (reference_values >= upper_threshold)
    )
    lobe_masks[component_name] = (
        lower_lobe_mask,
        upper_lobe_mask,
    )

    component_sigma = np.asarray(
        strain_sigmas[f"sigma_{component_name}"],
        dtype=float,
    )
    for epoch_index, date in enumerate(dates):
        values = np.asarray(
            strain_values[component_name][epoch_index],
            dtype=float,
        )
        use = supported_strain_mask & np.isfinite(values)
        lower_use = (
            lower_lobe_mask
            & np.isfinite(values)
            & np.isfinite(component_sigma)
        )
        upper_use = (
            upper_lobe_mask
            & np.isfinite(values)
            & np.isfinite(component_sigma)
        )
        regional_rows.append(
            {
                "date": pd.Timestamp(date),
                "component": component_name,
                "cell_count": int(use.sum()),
                "spatial_q10": float(
                    np.nanpercentile(values[use], 10.0)
                ),
                "spatial_median": float(np.nanmedian(values[use])),
                "spatial_q90": float(
                    np.nanpercentile(values[use], 90.0)
                ),
                "median_absolute": float(
                    np.nanmedian(np.abs(values[use]))
                ),
                "rms": float(
                    np.sqrt(np.nanmean(np.square(values[use])))
                ),
                "lower_lobe_median": float(
                    np.nanmedian(values[lower_use])
                ),
                "upper_lobe_median": float(
                    np.nanmedian(values[upper_use])
                ),
                "lower_lobe_pointwise_sigma": float(
                    np.nanmedian(component_sigma[lower_use])
                ),
                "upper_lobe_pointwise_sigma": float(
                    np.nanmedian(component_sigma[upper_use])
                ),
                "lower_lobe_fraction_abs_ge_1p96sigma": float(
                    np.mean(
                        np.abs(values[lower_use])
                        >= 1.96 * component_sigma[lower_use]
                    )
                ),
                "upper_lobe_fraction_abs_ge_1p96sigma": float(
                    np.mean(
                        np.abs(values[upper_use])
                        >= 1.96 * component_sigma[upper_use]
                    )
                ),
                "lobe_reference_date": CALIBRATION_END,
                "lower_lobe_reference_threshold": float(
                    lower_threshold
                ),
                "upper_lobe_reference_threshold": float(
                    upper_threshold
                ),
                "lower_lobe_cell_count": int(lower_lobe_mask.sum()),
                "upper_lobe_cell_count": int(upper_lobe_mask.sum()),
            }
        )

regional_table = pd.DataFrame(regional_rows)
regional_table.to_csv(
    OUTPUT_DIR / "near_fault_cumulative_strain_timeseries.csv",
    index=False,
)


def plot_fixed_lobe_timeseries(
    output_name: str,
    *,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    title_suffix: str = "",
) -> None:
    fig, axes = plt.subplots(
        len(component_specs),
        1,
        figsize=(12.5, 13.2),
        sharex=True,
        constrained_layout=True,
    )
    for panel_index, (
        axis,
        (component_name, (component_title, _)),
    ) in enumerate(zip(axes, component_specs.items(), strict=True)):
        table = regional_table.loc[
            regional_table["component"] == component_name
        ].copy()
        lower = table["lower_lobe_median"].to_numpy(float)
        upper = table["upper_lobe_median"].to_numpy(float)
        lower_sigma = table[
            "lower_lobe_pointwise_sigma"
        ].to_numpy(float)
        upper_sigma = table[
            "upper_lobe_pointwise_sigma"
        ].to_numpy(float)
        axis.fill_between(
            table["date"],
            lower - lower_sigma,
            lower + lower_sigma,
            color="#2166ac",
            alpha=0.14,
            linewidth=0.0,
        )
        axis.fill_between(
            table["date"],
            upper - upper_sigma,
            upper + upper_sigma,
            color="#b2182b",
            alpha=0.14,
            linewidth=0.0,
        )
        axis.plot(
            table["date"],
            lower,
            color="#2166ac",
            linewidth=1.8,
            label=(
                "Fixed lower lobe median"
                if panel_index == 0
                else None
            ),
        )
        axis.plot(
            table["date"],
            upper,
            color="#b2182b",
            linewidth=1.8,
            label=(
                "Fixed upper lobe median"
                if panel_index == 0
                else None
            ),
        )
        axis.plot(
            table["date"],
            table["spatial_median"],
            color="0.35",
            linewidth=0.9,
            linestyle="--",
            label="Whole-domain median" if panel_index == 0 else None,
        )
        axis.axhline(0.0, color="0.25", linewidth=0.7)
        axis.axvline(
            pd.Timestamp("2019-07-04"),
            color="#7f0000",
            linewidth=1.0,
            linestyle="--",
        )
        axis.axvline(
            pd.Timestamp("2019-07-06"),
            color="#7f0000",
            linewidth=1.0,
            linestyle=":",
        )
        axis.grid(color="0.88", linestyle=":", linewidth=0.55)
        unit = (
            r"$\mu$rad"
            if "rotation" in component_name
            else r"$\mu$strain"
        )
        axis.set_ylabel(unit)
        axis.set_title(
            f"({chr(97 + panel_index)}) {component_title}"
        )
        if start_date is not None:
            axis.set_xlim(
                start_date,
                end_date if end_date is not None else dates.max(),
            )
    axes[0].legend(
        frameon=False,
        ncol=3,
        loc="upper left",
    )
    axes[-1].set_xlabel("Date")
    fig.suptitle(
        "Near-fault cumulative 2-D horizontal strain evolution"
        f"{title_suffix}\n"
        "fixed lower/upper 20% lobe masks defined on 29 May 2019; "
        "shading is median pointwise ±1σ",
        fontsize=14,
        fontweight="semibold",
    )
    fig.savefig(OUTPUT_DIR / output_name)
    plt.close(fig)


plot_fixed_lobe_timeseries(
    "06_near_fault_cumulative_strain_timeseries.png",
)
plot_fixed_lobe_timeseries(
    "06b_pre_event_cumulative_strain_lobes.png",
    start_date=pd.Timestamp("2019-01-01"),
    end_date=pd.Timestamp("2019-07-16"),
    title_suffix=" — 2019 detail",
)


# %%
fig, axes = plt.subplots(
    2,
    3,
    figsize=(15.8, 9.0),
    constrained_layout=True,
)
for panel_index, (
    axis,
    (component_name, (component_title, colorbar_label)),
) in enumerate(zip(axes.ravel(), component_specs.items(), strict=False)):
    component_sigma = np.asarray(
        strain_sigmas[f"sigma_{component_name}"],
        dtype=float,
    )
    component_sigma = np.where(
        supported_strain_mask,
        component_sigma,
        np.nan,
    )
    finite_positive_sigma = component_sigma[
        np.isfinite(component_sigma) & (component_sigma > 0.0)
    ]
    sigma_vmin, sigma_vmax = np.nanpercentile(
        finite_positive_sigma,
        [1.0, 99.0],
    )
    if sigma_vmax <= sigma_vmin:
        sigma_vmax = sigma_vmin * (1.0 + 1.0e-6)
    uncertainty_mesh = axis.pcolormesh(
        longitude_grid,
        latitude_grid,
        component_sigma,
        cmap="magma",
        norm=LogNorm(vmin=float(sigma_vmin), vmax=float(sigma_vmax)),
        shading="nearest",
        rasterized=True,
    )
    unit = (
        r"$\mu$rad"
        if "rotation" in component_name
        else r"$\mu$strain"
    )
    axis.set_title(
        f"({chr(97 + panel_index)}) {component_title}\n"
        f"spatial median σ = "
        f"{np.nanmedian(component_sigma[supported_strain_mask]):.2f} "
        f"{unit}"
    )
    format_map_axis(
        axis,
        show_xlabel=panel_index >= 3,
        show_ylabel=panel_index % 3 == 0,
    )
    uncertainty_colorbar = fig.colorbar(
        uncertainty_mesh,
        ax=axis,
        orientation="vertical",
        fraction=0.046,
        pad=0.025,
        extend="both",
    )
    uncertainty_colorbar.set_label(
        f"Propagated pointwise 1σ {colorbar_label}"
    )
axes.ravel()[-1].set_visible(False)
fig.suptitle(
    "Cumulative 2-D horizontal strain uncertainty",
    fontsize=14,
    fontweight="semibold",
)
fig.savefig(
    OUTPUT_DIR / "09_cumulative_strain_uncertainty.png"
)
plt.close(fig)


# %%
# Plot every cumulative strain component on the same five-panel layout as the
# uncertainty figure.  Each component receives its own symmetric colour scale
# because normal strain, shear, dilatation, and rotation have different
# numerical ranges (and rotation has a different unit).  The component-specific
# limits are held fixed across the three key dates so that temporal changes
# remain visually comparable.
component_color_limits: dict[str, float] = {}
for component_name in component_specs:
    selected_values = np.asarray(
        strain_values[component_name],
        dtype=float,
    )[key_date_indices]
    selected_finite = selected_values[
        :, supported_strain_mask
    ]
    component_color_limits[component_name] = float(
        max(
            np.nanpercentile(np.abs(selected_finite), 98.0),
            1.0,
        )
    )

for epoch_index, date_label in zip(
    key_date_indices,
    key_date_labels,
    strict=True,
):
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15.8, 9.0),
        constrained_layout=True,
    )
    for panel_index, (
        axis,
        (component_name, (component_title, colorbar_label)),
    ) in enumerate(
        zip(axes.ravel(), component_specs.items(), strict=False)
    ):
        component_values = np.asarray(
            strain_values[component_name][epoch_index],
            dtype=float,
        )
        component_values = np.where(
            supported_strain_mask,
            component_values,
            np.nan,
        )
        limit = component_color_limits[component_name]
        component_mesh = axis.pcolormesh(
            longitude_grid,
            latitude_grid,
            component_values,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(
                vmin=-limit,
                vcenter=0.0,
                vmax=limit,
            ),
            shading="nearest",
            rasterized=True,
        )
        axis.set_title(
            f"({chr(97 + panel_index)}) {component_title}"
        )
        format_map_axis(
            axis,
            show_xlabel=panel_index >= 3,
            show_ylabel=panel_index % 3 == 0,
        )
        component_colorbar = fig.colorbar(
            component_mesh,
            ax=axis,
            orientation="vertical",
            fraction=0.046,
            pad=0.025,
            extend="both",
        )
        component_colorbar.set_label(colorbar_label)
    axes.ravel()[-1].set_visible(False)
    figure_date = pd.Timestamp(dates[epoch_index])
    publication_date_label = (
        figure_date.strftime("%d %B %Y").lstrip("0")
    )
    fig.suptitle(
        "Cumulative 2-D horizontal strain components — "
        + publication_date_label,
        fontsize=14,
        fontweight="semibold",
    )
    output_stem = (
        "10_cumulative_strain_components_"
        f"{figure_date:%Y%m%d}"
    )
    fig.savefig(OUTPUT_DIR / f"{output_stem}.png")
    fig.savefig(OUTPUT_DIR / f"{output_stem}.pdf")
    plt.close(fig)


def interpolate_filled_displacement(
    target_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    east_axis = east_grid[0, :]
    north_axis = north_grid[:, 0]
    east_output = np.full((len(dates), len(target_xy)), np.nan, dtype=float)
    north_output = np.full_like(east_output, np.nan)
    query = np.column_stack([target_xy[:, 1], target_xy[:, 0]])
    for epoch_index in range(len(dates)):
        east_interpolator = RegularGridInterpolator(
            (north_axis, east_axis),
            filled_east[epoch_index],
            bounds_error=False,
            fill_value=np.nan,
        )
        north_interpolator = RegularGridInterpolator(
            (north_axis, east_axis),
            filled_north[epoch_index],
            bounds_error=False,
            fill_value=np.nan,
        )
        east_output[epoch_index] = east_interpolator(query)
        north_output[epoch_index] = north_interpolator(query)
    return east_output, north_output


jump_rows: list[dict[str, float | str | pd.Timestamp]] = []
JUMP_APERTURE_KM = 3.0
for source_row, segment in zip(
    source_table.itertuples(index=False),
    fault_segments_xy,
    strict=True,
):
    along = segment[1] - segment[0]
    along_unit = along / np.linalg.norm(along)
    normal_unit = np.asarray([along_unit[1], -along_unit[0]])
    fraction = np.linspace(0.12, 0.88, 14)
    centerline = (
        segment[0][None, :]
        + fraction[:, None] * along[None, :]
    )
    plus = centerline + JUMP_APERTURE_KM * normal_unit[None, :]
    minus = centerline - JUMP_APERTURE_KM * normal_unit[None, :]
    plus_e, plus_n = interpolate_filled_displacement(plus)
    minus_e, minus_n = interpolate_filled_displacement(minus)
    jump_e = plus_e - minus_e
    jump_n = plus_n - minus_n
    jump_parallel = (
        jump_e * along_unit[0] + jump_n * along_unit[1]
    )
    jump_normal = (
        jump_e * normal_unit[0] + jump_n * normal_unit[1]
    )
    for epoch_index, date in enumerate(dates):
        jump_rows.append(
            {
                "fault": str(source_row.fault),
                "date": pd.Timestamp(date),
                "aperture_km": JUMP_APERTURE_KM,
                "median_parallel_jump_mm": float(
                    np.nanmedian(jump_parallel[epoch_index])
                ),
                "median_normal_jump_mm": float(
                    np.nanmedian(jump_normal[epoch_index])
                ),
                "parallel_q25_mm": float(
                    np.nanpercentile(jump_parallel[epoch_index], 25.0)
                ),
                "parallel_q75_mm": float(
                    np.nanpercentile(jump_parallel[epoch_index], 75.0)
                ),
                "normal_q25_mm": float(
                    np.nanpercentile(jump_normal[epoch_index], 25.0)
                ),
                "normal_q75_mm": float(
                    np.nanpercentile(jump_normal[epoch_index], 75.0)
                ),
            }
        )
jump_table = pd.DataFrame(jump_rows)
jump_table.to_csv(
    OUTPUT_DIR / "finite_aperture_cross_fault_jump_timeseries.csv",
    index=False,
)

fig, axes = plt.subplots(
    2,
    1,
    figsize=(12.0, 7.0),
    sharex=True,
    constrained_layout=True,
)
for fault_index, (fault_name, table) in enumerate(
    jump_table.groupby("fault", sort=False)
):
    color = ("#08519c", "#a50f15")[fault_index % 2]
    axes[0].plot(
        table["date"],
        table["median_parallel_jump_mm"],
        color=color,
        linewidth=1.6,
        label=fault_name,
    )
    axes[1].plot(
        table["date"],
        table["median_normal_jump_mm"],
        color=color,
        linewidth=1.6,
        label=fault_name,
    )
for axis, title in zip(
    axes,
    (
        "(a) Fault-parallel displacement discontinuity",
        "(b) Fault-normal displacement discontinuity",
    ),
    strict=True,
):
    axis.axhline(0.0, color="0.35", linewidth=0.7)
    axis.axvline(
        pd.Timestamp("2019-07-04"),
        color="0.2",
        linestyle="--",
        linewidth=1.0,
    )
    axis.axvline(
        pd.Timestamp("2019-07-06"),
        color="0.2",
        linestyle=":",
        linewidth=1.0,
    )
    axis.set_ylabel("Cumulative jump (mm)")
    axis.set_title(title)
    axis.grid(color="0.88", linestyle=":", linewidth=0.55)
axes[0].legend(frameon=False, ncol=2)
axes[-1].set_xlabel("Date")
fig.suptitle(
    "Six-kilometre finite-aperture cross-fault displacement jump",
    fontsize=14,
    fontweight="semibold",
)
fig.savefig(OUTPUT_DIR / "07_cross_fault_displacement_jump_timeseries.png")
plt.close(fig)


# %% [markdown]
# ## 11. Cumulative pattern persistence versus 12-day innovation
#
# A cumulative map answers “what has accumulated since the 27 May 2017
# baseline?” It does not answer “what changed during the latest interval?”
# Therefore the spatial similarity of 22 June and 4 July is quantified first,
# and the actual 22 June-4 July increment is tested against all 12-day
# calibration increments ending no later than 29 May 2019.
#
# The formal cluster test is repeated on cells whose derivative neighbourhood
# contains no reconstructed sample. This prevents a significant cluster in the
# unvalidated Track-64-assisted area from being attributed to observed
# two-track strain.

# %%
audit_component = "dilatation_microstrain"
audit_sigma = "sigma_dilatation_microstrain"
audit_cumulative = np.asarray(
    strain_values[audit_component],
    dtype=float,
)
audit_sigma_grid = np.asarray(
    strain_sigmas[audit_sigma],
    dtype=float,
)
observed_only_strain_mask = (
    supported_strain_mask
    & np.isfinite(local_reconstructed_fraction_grid)
    & (local_reconstructed_fraction_grid == 0.0)
)
reconstruction_influenced_mask = (
    supported_strain_mask
    & np.isfinite(local_reconstructed_fraction_grid)
    & (local_reconstructed_fraction_grid > 0.0)
)
# The strain maps are evaluated every kilometre, but adjacent RMLS estimates
# share most of the same displacement samples.  Formal spatial inference is
# therefore restricted to the same fixed four-kilometre lattice used by the
# primary cumulative-strain analysis.  The full one-kilometre fields remain
# available for continuous-map display and descriptive summaries only.
audit_inference_lattice = np.zeros(east_grid.shape, dtype=bool)
audit_inference_lattice[::4, ::4] = True

twelve_day_start: list[int] = []
twelve_day_end: list[int] = []
for start_index in range(len(dates) - 1):
    if (dates[start_index + 1] - dates[start_index]).days == 12:
        twelve_day_start.append(start_index)
        twelve_day_end.append(start_index + 1)
twelve_day_start_array = np.asarray(twelve_day_start, dtype=int)
twelve_day_end_array = np.asarray(twelve_day_end, dtype=int)
twelve_day_calibration = (
    dates[twelve_day_end_array] <= CALIBRATION_END
)
twelve_day_increment = (
    audit_cumulative[twelve_day_end_array]
    - audit_cumulative[twelve_day_start_array]
)
twelve_day_sigma = np.sqrt(2.0) * audit_sigma_grid


def find_twelve_day_interval(
    start_date: str,
    end_date: str,
) -> int:
    match = np.flatnonzero(
        (dates[twelve_day_start_array] == pd.Timestamp(start_date))
        & (dates[twelve_day_end_array] == pd.Timestamp(end_date))
    )
    if len(match) != 1:
        raise RuntimeError(
            f"Could not uniquely identify {start_date}-{end_date}"
        )
    return int(match[0])


intervals_for_audit = [
    (
        "10-22 June",
        find_twelve_day_interval("2019-06-10", "2019-06-22"),
    ),
    (
        "22 June-4 July",
        find_twelve_day_interval("2019-06-22", "2019-07-04"),
    ),
    (
        "4-16 July",
        find_twelve_day_interval("2019-07-04", "2019-07-16"),
    ),
]

pattern_audit_rows: list[dict[str, float | int | str]] = []
cluster_pvalue: dict[tuple[str, str], float] = {}
for subset_name, subset_mask in (
    ("all_supported", supported_strain_mask),
    ("observed_only", observed_only_strain_mask),
    ("reconstruction_influenced", reconstruction_influenced_mask),
):
    subset_inference_mask = subset_mask & audit_inference_lattice
    cumulative_22 = audit_cumulative[june22_index][subset_mask]
    cumulative_04 = audit_cumulative[july4_index][subset_mask]
    finite_pair = np.isfinite(cumulative_22) & np.isfinite(cumulative_04)
    spatial_correlation = float(
        np.corrcoef(
            cumulative_22[finite_pair],
            cumulative_04[finite_pair],
        )[0, 1]
    )
    pattern_audit_rows.append(
        {
            "subset": subset_name,
            "quantity": "22Jun_vs_04Jul_cumulative_pattern",
            "cell_count": int(subset_mask.sum()),
            "inference_cell_count": int(subset_inference_mask.sum()),
            "spatial_correlation": spatial_correlation,
            "sign_agreement_fraction": float(
                np.mean(
                    np.sign(cumulative_22[finite_pair])
                    == np.sign(cumulative_04[finite_pair])
                )
            ),
            "median_abs_22Jun": float(
                np.nanmedian(np.abs(cumulative_22))
            ),
            "median_abs_04Jul": float(
                np.nanmedian(np.abs(cumulative_04))
            ),
            "median_abs_22Jun_04Jul_change": float(
                np.nanmedian(np.abs(cumulative_04 - cumulative_22))
            ),
            "maximum_cluster_mass": np.nan,
            "cluster_fwer_pvalue": np.nan,
            "median_abs_change_empirical_pvalue": np.nan,
            "cluster_lattice_spacing_km": 4,
        }
    )

    subset_increment = twelve_day_increment[:, subset_inference_mask]
    subset_sigma = np.broadcast_to(
        twelve_day_sigma[subset_inference_mask][None, :],
        subset_increment.shape,
    )
    baseline_model = fit_robust_baseline(
        subset_increment,
        subset_sigma,
        twelve_day_calibration,
        min_observations=30,
    )
    innovation_z = standardized_innovation(
        subset_increment,
        subset_sigma,
        baseline_model,
    )
    leave_one_out_z = leave_one_out_baseline_innovations(
        subset_increment,
        subset_sigma,
        twelve_day_calibration,
        min_observations=29,
    )
    subset_east = east_grid[subset_inference_mask]
    subset_north = north_grid[subset_inference_mask]
    null_cluster_mass = np.asarray(
        [
            maximum_signed_cluster_mass(
                leave_one_out_z[interval_index][None, :],
                subset_east,
                subset_north,
                [audit_component],
                threshold=1.96,
                min_cells=4,
            )
            for interval_index in np.flatnonzero(
                twelve_day_calibration
            )
        ],
        dtype=float,
    )
    baseline_center = np.nanmedian(
        subset_increment[twelve_day_calibration],
        axis=0,
    )
    median_abs_change = np.nanmedian(
        np.abs(subset_increment - baseline_center[None, :]),
        axis=1,
    )
    for interval_label, interval_index in intervals_for_audit:
        observed_cluster_mass = maximum_signed_cluster_mass(
            innovation_z[interval_index][None, :],
            subset_east,
            subset_north,
            [audit_component],
            threshold=1.96,
            min_cells=4,
        )
        p_cluster = empirical_upper_tail_pvalue(
            null_cluster_mass,
            observed_cluster_mass,
        )
        p_median_abs = float(
            (
                1.0
                + np.count_nonzero(
                    median_abs_change[twelve_day_calibration]
                    >= median_abs_change[interval_index]
                )
            )
            / (1.0 + int(twelve_day_calibration.sum()))
        )
        cluster_pvalue[(subset_name, interval_label)] = p_cluster
        pattern_audit_rows.append(
            {
                "subset": subset_name,
                "quantity": interval_label,
                "cell_count": int(subset_mask.sum()),
                "inference_cell_count": int(subset_inference_mask.sum()),
                "spatial_correlation": np.nan,
                "sign_agreement_fraction": np.nan,
                "median_abs_22Jun": np.nan,
                "median_abs_04Jul": np.nan,
                "median_abs_22Jun_04Jul_change": float(
                    median_abs_change[interval_index]
                ),
                "maximum_cluster_mass": float(
                    observed_cluster_mass
                ),
                "cluster_fwer_pvalue": p_cluster,
                "median_abs_change_empirical_pvalue": p_median_abs,
                "cluster_lattice_spacing_km": 4,
            }
        )

pattern_audit_table = pd.DataFrame(pattern_audit_rows)
pattern_audit_table.to_csv(
    OUTPUT_DIR
    / "cumulative_pattern_persistence_and_12day_innovation_audit.csv",
    index=False,
)


def pattern_audit_value(
    subset: str,
    quantity: str,
    column: str,
) -> float:
    match = pattern_audit_table.loc[
        (pattern_audit_table["subset"] == subset)
        & (pattern_audit_table["quantity"] == quantity),
        column,
    ]
    if len(match) != 1:
        raise RuntimeError(
            f"Pattern-audit value is not unique: {subset}, {quantity}, {column}"
        )
    return float(match.iloc[0])


print(
    pattern_audit_table.loc[
        pattern_audit_table["subset"].isin(
            ["observed_only", "reconstruction_influenced"]
        )
    ].to_string(index=False)
)


# %%
cumulative_display_indices = [
    june22_index,
    july4_index,
    july16_index,
]
cumulative_display_labels = [
    "22 June 2019",
    "4 July 2019",
    "16 July 2019",
]
increment_display_indices = [
    interval_index
    for _, interval_index in intervals_for_audit
]
increment_display_labels = [
    interval_label
    for interval_label, _ in intervals_for_audit
]
cumulative_limit = float(
    max(
        np.nanpercentile(
            np.abs(
                audit_cumulative[cumulative_display_indices][
                    :, supported_strain_mask
                ]
            ),
            98.0,
        ),
        1.0,
    )
)
increment_limit = float(
    max(
        np.nanpercentile(
            np.abs(
                twelve_day_increment[increment_display_indices][
                    :, supported_strain_mask
                ]
            ),
            98.0,
        ),
        1.0,
    )
)

fig, axes = plt.subplots(
    2,
    3,
    figsize=(14.5, 9.0),
    constrained_layout=True,
)
top_mesh = None
for panel_index, (epoch_index, date_label) in enumerate(
    zip(
        cumulative_display_indices,
        cumulative_display_labels,
        strict=True,
    )
):
    axis = axes[0, panel_index]
    top_mesh = axis.pcolormesh(
        longitude_grid,
        latitude_grid,
        np.where(
            supported_strain_mask,
            audit_cumulative[epoch_index],
            np.nan,
        ),
        cmap="RdBu_r",
        norm=TwoSlopeNorm(
            vmin=-cumulative_limit,
            vcenter=0.0,
            vmax=cumulative_limit,
        ),
        shading="nearest",
        rasterized=True,
    )
    axis.set_title(
        f"({chr(97 + panel_index)}) Cumulative: {date_label}"
    )
    format_map_axis(
        axis,
        show_xlabel=False,
        show_ylabel=panel_index == 0,
    )

bottom_mesh = None
for panel_index, (interval_index, interval_label) in enumerate(
    zip(
        increment_display_indices,
        increment_display_labels,
        strict=True,
    )
):
    axis = axes[1, panel_index]
    bottom_mesh = axis.pcolormesh(
        longitude_grid,
        latitude_grid,
        np.where(
            supported_strain_mask,
            twelve_day_increment[interval_index],
            np.nan,
        ),
        cmap="RdBu_r",
        norm=TwoSlopeNorm(
            vmin=-increment_limit,
            vcenter=0.0,
            vmax=increment_limit,
        ),
        shading="nearest",
        rasterized=True,
    )
    p_value = cluster_pvalue[("all_supported", interval_label)]
    axis.set_title(
        f"({chr(100 + panel_index)}) 12-day change: {interval_label}\n"
        f"4-km spatial p-FWER = {p_value:.3f}"
    )
    format_map_axis(
        axis,
        show_xlabel=True,
        show_ylabel=panel_index == 0,
    )

top_colorbar = fig.colorbar(
    top_mesh,
    ax=axes[0, :],
    orientation="vertical",
    fraction=0.022,
    pad=0.012,
)
top_colorbar.set_label(r"Cumulative dilatation ($\mu$strain)")
bottom_colorbar = fig.colorbar(
    bottom_mesh,
    ax=axes[1, :],
    orientation="vertical",
    fraction=0.022,
    pad=0.012,
)
bottom_colorbar.set_label(r"12-day dilatation change ($\mu$strain)")
fig.suptitle(
    "Cumulative dilatation patterns and 12-day spatial changes",
    fontsize=14,
    fontweight="semibold",
)
fig.savefig(
    OUTPUT_DIR / "08_cumulative_vs_12day_dilatation_innovation.png"
)
plt.close(fig)


# %% [markdown]
# ## 12. Reproducibility manifest

# %%
event_m0 = float(
    cv_summary.loc[
        (cv_summary["period"] == "event_control_16Jul")
        & (cv_summary["model"] == "M0_paired_only"),
        "mae_mm",
    ].iloc[0]
)
event_m1 = float(
    cv_summary.loc[
        (cv_summary["period"] == "event_control_16Jul")
        & (cv_summary["model"] == "M1_Track64_conditioned"),
        "mae_mm",
    ].iloc[0]
)
event_gain = (event_m0 - event_m1) / event_m0

manifest = {
    "analysis": "Track-64-guided near-fault cumulative strain",
    "dates": {
        "first": str(dates.min().date()),
        "last": str(dates.max().date()),
        "epoch_count": int(len(dates)),
        "calibration_end": str(CALIBRATION_END.date()),
        "surveillance_end": str(PRE_EVENT_END.date()),
        "event_control_end": str(EVENT_END.date()),
    },
    "coverage": {
        "near_fault_distance_km": NEAR_FAULT_DISTANCE_KM,
        "near_fault_cells": int(near_fault.sum()),
        "track64_conditioned_target_cells": int(fill_target_mask.sum()),
        "spatial_only_completion_cells": int(
            remaining_spatial_mask.sum()
        ),
        "completed_en_cells": int(
            (complete_near_fault_en & near_fault).sum()
        ),
        "former_core_audit_distance_km": FORMER_CORE_AUDIT_KM,
        "former_core_cells": int(former_core_distance_mask.sum()),
        "former_core_supported_strain_cells": int(
            (supported_strain_mask & former_core_distance_mask).sum()
        ),
        "track64_conditioned_imputed_epoch_cells": int(
            imputed_descending.sum()
        ),
        "spatial_only_imputed_epoch_cells": int(
            spatial_only_imputation.sum()
        ),
        "supported_strain_cells": int(supported_strain_mask.sum()),
        "full_area_strain_coverage_fraction": float(
            supported_strain_mask.sum() / near_fault.sum()
        ),
    },
    "cokriging": {
        "model": "fixed local latent E-N Matern-3/2 universal cokriging",
        "source": (
            "paired Track 64/71 samples, collocated Track 64 targets, "
            "and paired-only M0 completion for remaining cells"
        ),
        "support_radius_km": CV_SUPPORT_RADIUS_KM,
        "maximum_paired_samples": MAXIMUM_COKRIGING_SAMPLES,
        "selected_length_scale_km": best_length,
        "selected_covariance_multiplier": best_multiplier,
        "selected_target_ascending_noise_multiplier": (
            best_target_noise_multiplier
        ),
        "buffered_cv_radius_km": CV_BUFFER_KM,
        "calibration_mae_m0_mm": calibration_m0_mae,
        "calibration_mae_m1_mm": calibration_m1_mae,
        "calibration_track64_gain": track64_cv_gain,
        "exact_gap_shape_gain": exact_mask_gain,
        "event_control_gain": event_gain,
        "reconstruction_supported_by_pre_event_cv": reconstruction_supported,
        "observed_track71_overwrite_error_mm": max_observed_overwrite,
    },
    "barrier": {
        "distance_source": str(FAULT_GEOJSON.relative_to(ROOT)),
        "side_topology_source": str(SOURCE_PARAMETERS.relative_to(ROOT)),
        "posterior_geometry_converged": True,
    },
    "strain": {
        "sample_spacing_km": SAMPLE_SPACING_KM,
        "support_radius_km": STRAIN_SUPPORT_RADIUS_KM,
        "bandwidth_km": STRAIN_BANDWIDTH_KM,
        "minimum_samples": STRAIN_MINIMUM_SAMPLES,
        "operation": (
            "fixed fault-barrier joint GLS derivative of completed "
            "cumulative E-N over the full near-fault area"
        ),
        "full_near_fault_area_included": True,
        "time_series_summary": (
            "fixed lower and upper 20-percent lobe masks defined on "
            "2019-05-29"
        ),
        "uncertainty": (
            "propagated pointwise one-standard-deviation component "
            "uncertainty"
        ),
    },
    "cumulative_pattern_audit": {
        "component": audit_component,
        "display_grid_spacing_km": 1,
        "inference_lattice_spacing_km": 4,
        "calibration_exact_12day_interval_count": int(
            twelve_day_calibration.sum()
        ),
        "calibration_end": str(CALIBRATION_END.date()),
        "observed_only_cell_count": int(
            observed_only_strain_mask.sum()
        ),
        "observed_only_inference_cell_count": int(
            (observed_only_strain_mask & audit_inference_lattice).sum()
        ),
        "reconstruction_influenced_cell_count": int(
            reconstruction_influenced_mask.sum()
        ),
        "reconstruction_influenced_inference_cell_count": int(
            (
                reconstruction_influenced_mask
                & audit_inference_lattice
            ).sum()
        ),
        "observed_only_22Jun_04Jul_spatial_correlation": (
            pattern_audit_value(
                "observed_only",
                "22Jun_vs_04Jul_cumulative_pattern",
                "spatial_correlation",
            )
        ),
        "reconstruction_influenced_22Jun_04Jul_spatial_correlation": (
            pattern_audit_value(
                "reconstruction_influenced",
                "22Jun_vs_04Jul_cumulative_pattern",
                "spatial_correlation",
            )
        ),
        "observed_only_22Jun_04Jul_cluster_p_fwer": (
            pattern_audit_value(
                "observed_only",
                "22 June-4 July",
                "cluster_fwer_pvalue",
            )
        ),
        "reconstruction_influenced_22Jun_04Jul_cluster_p_fwer": (
            pattern_audit_value(
                "reconstruction_influenced",
                "22 June-4 July",
                "cluster_fwer_pvalue",
            )
        ),
    },
}
(OUTPUT_DIR / "track64_guided_near_fault_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

readme = f"""# Track-64-guided near-fault cumulative strain

This directory contains a fixed-weight reconstruction of missing descending
Track 71 horizontal LOS values using collocated ascending Track 64 and nearby
paired observations, followed by paired-only spatial completion where neither
track supplied a persistent E-N solution. Valid Track 71 pixels are unchanged.

- Track-64-conditioned target cells: {int(fill_target_mask.sum()):,}
- Paired-only spatial completion cells: {int(remaining_spatial_mask.sum()):,}
- Full near-fault E-N cells: {int((complete_near_fault_en & near_fault).sum()):,}
- Full-area strain cells: {int(supported_strain_mask.sum()):,}
- Former one-kilometre core cells included: {int((supported_strain_mask & former_core_distance_mask).sum()):,}
- Formal strain-innovation lattice: 4 km
- Figure 06 follows fixed lower and upper spatial strain lobes through time,
  avoiding cancellation in a whole-domain signed median.
- Figure 09 maps propagated pointwise one-standard-deviation uncertainty.

The fixed-resolution cumulative strain field covers every near-fault 1-km
target, including the mapped rupture zone. Numerical completion provenance and
uncertainty remain available in the NPZ, CSV, and JSON products.
"""
(OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")

print(json.dumps(manifest, indent=2))
print(
    "Completed cumulative displacement, strain-lobe, uncertainty, and "
    "12-day change products."
)
