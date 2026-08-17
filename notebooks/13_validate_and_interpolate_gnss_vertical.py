# %% [markdown]
# # All-station local interpolation of GNSS vertical displacement
#
# This notebook performs the vertical stage only:
#
# \[
# \Delta U_{\mathrm{GNSS}}
# \longrightarrow
# \widehat{\Delta U}(x,y).
# \]
#
# It does not project \(\widehat{\Delta U}\) into LOS, modify either InSAR
# series, solve horizontal motion, or calculate strain.
#
# The interpolation uses every GNSS station within the smallest
# geometry-adequate local radius. It is neither ten-nearest-neighbour
# interpolation nor a whole-map plane. We compare exponential, Matérn-3/2, and
# squared-exponential (RBF) covariances under ordinary kriging, local-constant
# Gaussian-process prediction, and first-order local universal-kriging
# sensitivity models. Covariance family, local mean structure, and length scale
# are selected using 2018 controls; the fixed specification is then evaluated
# on independent 2019 pre-event intervals. In every held-out fold, covariance
# amplitude is estimated from training stations only.

# %%
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import json
import sys

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from IPython.display import display


def find_repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Run this notebook from inside the repository.")


ROOT = find_repository_root(Path.cwd().resolve())
sys.path.insert(0, str(ROOT / "src"))

from ridgecrest_local_vertical import (  # noqa: E402
    LocalVerticalConfig,
    LocalVerticalModel,
    build_local_support_topology,
    calibrate_uncertainty_scale,
    estimate_interval_sill_mm2,
    evaluate_local_vertical_model,
    predict_local_vertical_from_topology,
    select_local_vertical_model,
)
from ridgecrest_two_track import common_utm11_grid, to_utm11_km  # noqa: E402
from ridgecrest_vertical_los import gnss_interval_table, load_gnss_network  # noqa: E402


mpl.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    }
)

# %% [markdown]
# ## 1. Fixed inputs, support rules, and temporal split

# %%
GNSS_ROOT = Path(r"D:\Uni\Thises\GNSS_ridgecrest\data\tenv_data")
COMMON_DATES_FILE = (
    ROOT
    / "outputs"
    / "track64_text_timeseries"
    / "track64_track71_common_dates.csv"
)
TRACK64_LOOK = (
    ROOT
    / "outputs"
    / "track64_text_timeseries"
    / "track64_text_pixel_look_vectors.npz"
)
DESC_H5 = ROOT / "data" / "cum_full_scene_no_GACOS.h5"
OUTPUT_DIR = ROOT / "outputs" / "gnss_vertical_interpolation_gate"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVENTS = (
    pd.Timestamp("2019-07-04T17:33:49"),
    pd.Timestamp("2019-07-06T03:19:53"),
)
TRACK_TIMES = {
    "ascending_T64": pd.Timedelta(
        hours=1, minutes=50, seconds=8, microseconds=490464
    ),
    "descending_T71": pd.Timedelta(
        hours=13, minutes=51, seconds=41, microseconds=812911
    ),
}
EVENT_START = pd.Timestamp("2019-07-04")
EVENT_END = pd.Timestamp("2019-07-16")
CALIBRATION_END = pd.Timestamp("2018-12-15")
HOLDOUT_START = pd.Timestamp("2018-12-16")
PREDICTION_GRID_SPACING_KM = 1.0

CONFIG = LocalVerticalConfig(
    radii_km=(35.0, 45.0, 55.0, 65.0, 75.0, 90.0, 105.0, 120.0),
    min_stations=5,
    sector_count=8,
    min_occupied_sectors=3,
    require_local_hull=True,
)
FAMILIES = (
    "ok_exponential",
    "ok_matern32",
    "ok_rbf",
    "uk_matern32",
    "uk_rbf",
    "gp_matern32",
    "gp_rbf",
)
LENGTH_SCALES_KM = (10.0, 15.0, 25.0, 40.0, 60.0, 80.0)
NUGGETS_MM = (0.0, 2.0, 5.0, 10.0)

for required in (GNSS_ROOT, COMMON_DATES_FILE, TRACK64_LOOK, DESC_H5):
    if not required.exists():
        raise FileNotFoundError(required)

# %% [markdown]
# ## 2. Build non-overlapping pre-event controls
#
# The 20 fixed 12-day controls are divided before model fitting:
#
# - 13 intervals in 2018 select family, length scale, and nugget;
# - 7 later intervals evaluate that fixed specification.
#
# The earthquake and all post-event observations are excluded from both sets.

# %%
common_dates = pd.read_csv(COMMON_DATES_FILE, parse_dates=["date"])["date"]
pre_dates = common_dates[
    (common_dates >= pd.Timestamp("2018-01-01"))
    & (common_dates <= pd.Timestamp("2019-05-31"))
]
pre_date_set = set(pre_dates)
control_pairs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
next_allowed: pd.Timestamp | None = None
for start in pre_dates:
    end = start + pd.Timedelta(days=12)
    if end in pre_date_set and (next_allowed is None or start >= next_allowed):
        control_pairs.append((start, end))
        next_allowed = end + pd.Timedelta(days=12)

calibration_pairs = [
    pair for pair in control_pairs if pair[1] <= CALIBRATION_END
]
holdout_pairs = [pair for pair in control_pairs if pair[0] >= HOLDOUT_START]
if len(calibration_pairs) < 8 or len(holdout_pairs) < 4:
    raise RuntimeError("The temporal calibration/holdout split is too small")
print(
    "Controls:",
    len(control_pairs),
    "calibration:",
    len(calibration_pairs),
    "independent holdout:",
    len(holdout_pairs),
)

histories, network = load_gnss_network(GNSS_ROOT)
network = network.sort_values("station").reset_index(drop=True)


def interval_table(
    track: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    utc_offset = TRACK_TIMES[track]
    table = gnss_interval_table(
        histories,
        network,
        start=start_date + utc_offset,
        end=end_date + utc_offset,
        event_times=EVENTS,
        strict=False,
    ).sort_values("station").reset_index(drop=True)
    if len(table) != len(network):
        raise RuntimeError(
            f"{track} {start_date.date()}–{end_date.date()} did not retain "
            "the complete GNSS network"
        )
    coordinates = to_utm11_km(
        table["longitude"].to_numpy(float),
        table["latitude"].to_numpy(float),
    )
    table["east_km"] = coordinates[:, 0]
    table["north_km"] = coordinates[:, 1]
    table["interval_start"] = start_date
    table["interval_end"] = end_date
    return table


controls: dict[str, dict[str, list[pd.DataFrame]]] = {}
for track in TRACK_TIMES:
    controls[track] = {
        "calibration": [
            interval_table(track, start, end)
            for start, end in calibration_pairs
        ],
        "holdout": [
            interval_table(track, start, end) for start, end in holdout_pairs
        ],
    }

# %% [markdown]
# ## 3. Select one shared spatial specification on 2018, then evaluate 2019
#
# Vertical deformation is one physical field; the Sentinel-1 track does not
# define its covariance. We therefore choose one shared covariance/mean family,
# length scale, and nugget by averaging the two track-time calibration scores
# candidate by candidate. The same shared specification is then evaluated
# separately at the ascending and descending acquisition times.
#
# The same local-constant baseline uses exactly the same contributing stations.
# A spatial model passes only if the independent holdout has:
#
# - complete non-extrapolative fold coverage;
# - RMSE no more than 2% above the local baseline;
# - lower mean negative log predictive density;
# - a positive interval-bootstrap 95% lower bound for predictive-density gain;
# - nominal-90% coverage between 75% and 99%.

# %%
calibration_score_tables: dict[str, pd.DataFrame] = {}
calibration_prediction_tables: dict[str, pd.DataFrame] = {}
provisional_models: dict[str, LocalVerticalModel] = {}
for track in TRACK_TIMES:
    model, calibration_scores, calibration_predictions = (
        select_local_vertical_model(
            controls[track]["calibration"],
            configs=(CONFIG,),
            families=FAMILIES,
            length_scales_km=LENGTH_SCALES_KM,
            nuggets_mm=NUGGETS_MM,
            rmse_relative_tolerance=0.02,
            require_acceptance=False,
        )
    )
    provisional_models[track] = model
    calibration_score_tables[track] = calibration_scores
    calibration_prediction_tables[track] = calibration_predictions
    calibration_scores.to_csv(
        OUTPUT_DIR / f"{track}_2018_model_selection_scores.csv", index=False
    )
    calibration_predictions.to_csv(
        OUTPUT_DIR / f"{track}_2018_model_selection_predictions.csv",
        index=False,
    )

candidate_keys = [
    "config_index",
    "family",
    "length_scale_km",
    "nugget_mm",
]
joint_scores = calibration_score_tables["ascending_T64"].merge(
    calibration_score_tables["descending_T71"],
    on=candidate_keys,
    suffixes=("_ascending", "_descending"),
)
joint_scores["complete_both_tracks"] = (
    joint_scores["complete_fixed_holdout_coverage_ascending"]
    & joint_scores["complete_fixed_holdout_coverage_descending"]
)
joint_scores["joint_mean_nlpd"] = 0.5 * (
    joint_scores["mean_nlpd_ascending"]
    + joint_scores["mean_nlpd_descending"]
)
joint_scores["joint_rmse_mm"] = 0.5 * (
    joint_scores["rmse_mm_ascending"] + joint_scores["rmse_mm_descending"]
)
eligible_joint = joint_scores.loc[joint_scores["complete_both_tracks"]]
if eligible_joint.empty:
    raise RuntimeError("No shared vertical specification covered both tracks")
joint_index = eligible_joint.sort_values(
    ["joint_mean_nlpd", "joint_rmse_mm", "length_scale_km", "nugget_mm"]
).index[0]
joint_scores["selected_shared_specification"] = False
joint_scores.loc[joint_index, "selected_shared_specification"] = True
joint_scores.to_csv(
    OUTPUT_DIR / "shared_2018_vertical_model_selection_scores.csv",
    index=False,
)
joint_choice = joint_scores.loc[joint_index]
family_comparison = (
    joint_scores.loc[joint_scores["complete_both_tracks"]]
    .sort_values(["joint_mean_nlpd", "joint_rmse_mm"])
    .groupby("family", as_index=False)
    .first()
    .sort_values(["joint_mean_nlpd", "joint_rmse_mm"])
)
family_comparison.to_csv(
    OUTPUT_DIR / "shared_2018_best_candidate_by_family.csv",
    index=False,
)
display(
    family_comparison[
        [
            "family",
            "length_scale_km",
            "nugget_mm",
            "joint_mean_nlpd",
            "joint_rmse_mm",
        ]
    ]
)
shared_model = LocalVerticalModel(
    family=str(joint_choice["family"]),  # type: ignore[arg-type]
    length_scale_km=float(joint_choice["length_scale_km"]),
    nugget_mm=float(joint_choice["nugget_mm"]),
    sill_mm2=1.0,
    config=CONFIG,
    validation_rmse_mm=float(joint_choice["joint_rmse_mm"]),
    validation_nlpd=float(joint_choice["joint_mean_nlpd"]),
    validation_coverage90=float(
        0.5
        * (
            joint_choice["coverage90_ascending"]
            + joint_choice["coverage90_descending"]
        )
    ),
    baseline_rmse_mm=float(
        0.5
        * (
            joint_choice["baseline_rmse_mm_ascending"]
            + joint_choice["baseline_rmse_mm_descending"]
        )
    ),
    baseline_nlpd=float(
        0.5
        * (
            joint_choice["baseline_mean_nlpd_ascending"]
            + joint_choice["baseline_mean_nlpd_descending"]
        )
    ),
    bootstrap_delta_nlpd_lower95=float(
        min(
            joint_choice["bootstrap_delta_nlpd_lower95_ascending"],
            joint_choice["bootstrap_delta_nlpd_lower95_descending"],
        )
    ),
)
print(
    "Shared spatial specification:",
    shared_model.family,
    f"length={shared_model.length_scale_km:g} km",
    f"nugget={shared_model.nugget_mm:g} mm",
)

selected_models = {track: shared_model for track in TRACK_TIMES}
holdout_summaries: dict[str, dict[str, float | int | bool]] = {}
uncertainty_scales: dict[str, float] = {}
for track in TRACK_TIMES:
    calibration_predictions = calibration_prediction_tables[track]
    selected_calibration = calibration_predictions.loc[
        (calibration_predictions["family"] == shared_model.family)
        & (
            calibration_predictions["length_scale_km"]
            == shared_model.length_scale_km
        )
        & (
            calibration_predictions["nugget_mm"]
            == shared_model.nugget_mm
        )
    ].copy()
    uncertainty_scale = calibrate_uncertainty_scale(selected_calibration)
    holdout_summary, holdout_predictions = evaluate_local_vertical_model(
        controls[track]["holdout"],
        shared_model,
        rmse_relative_tolerance=0.02,
        coverage_bounds=(0.75, 0.99),
        uncertainty_scale=uncertainty_scale,
    )
    holdout_summaries[track] = holdout_summary
    uncertainty_scales[track] = uncertainty_scale
    holdout_predictions.to_csv(
        OUTPUT_DIR / f"{track}_2019_temporal_holdout_predictions.csv",
        index=False,
    )
    print(
        track,
        shared_model.family,
        f"length={shared_model.length_scale_km:g} km",
        f"nugget={shared_model.nugget_mm:g} mm",
        f"2018 uncertainty scale={uncertainty_scale:.3f}",
        holdout_summary,
    )

holdout_table = pd.DataFrame(holdout_summaries).T
holdout_table.to_csv(
    OUTPUT_DIR / "vertical_interpolation_temporal_holdout_summary.csv"
)
holdout_table

# %%
fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5), constrained_layout=True)
for ax, track in zip(axes, TRACK_TIMES):
    predictions = pd.read_csv(
        OUTPUT_DIR / f"{track}_2019_temporal_holdout_predictions.csv"
    )
    observed = predictions["observed_mm"].to_numpy(float)
    predicted = predictions["predicted_mm"].to_numpy(float)
    baseline = predictions["baseline_predicted_mm"].to_numpy(float)
    limits = np.nanpercentile(np.r_[observed, predicted, baseline], [1, 99])
    pad = max(2.0, 0.10 * float(limits[1] - limits[0]))
    limits = [float(limits[0] - pad), float(limits[1] + pad)]
    ax.plot(limits, limits, color="0.25", lw=1.0, ls="--")
    ax.scatter(
        observed,
        baseline,
        s=22,
        color="0.65",
        alpha=0.7,
        label="Same-neighbourhood local constant",
    )
    ax.errorbar(
        observed,
        predicted,
        yerr=predictions["predictive_sigma_mm"],
        fmt="o",
        ms=4,
        color="#1768AC",
        ecolor="#89B5D8",
        elinewidth=0.7,
        alpha=0.8,
        label="Selected local spatial model",
    )
    summary = holdout_summaries[track]
    ax.set(
        xlim=limits,
        ylim=limits,
        xlabel="Held-out GNSS vertical increment (mm)",
        ylabel="Prediction (mm)",
        title=(
            f"{track.replace('_', ' ')}\n"
            f"holdout RMSE {summary['rmse_mm']:.2f} vs "
            f"{summary['baseline_rmse_mm']:.2f} mm; "
            f"gate={'PASS' if summary['accepted'] else 'FAIL'}"
        ),
        aspect="equal",
    )
    ax.grid(True, color="0.90")
    ax.legend(loc="best")
fig.savefig(
    OUTPUT_DIR / "01_vertical_interpolation_temporal_holdout.png",
    bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ## 4. Interpolate the exact 4–16 July vertical increment
#
# The model specification remains fixed. Only covariance amplitude is estimated
# from the current all-station vertical increments, exactly matching the
# train-only rule used in every validation fold.
#
# A 1-km prediction lattice is used because the tested covariance scales are
# 25–80 km. This is a sampling lattice, not the claimed spatial resolution.
# The field can later be evaluated at every native InSAR pixel without changing
# the geostatistical model.

# %%
track64_geometry = np.load(TRACK64_LOOK)
ascending_latitude = np.sort(np.unique(track64_geometry["latitude"]))
ascending_longitude = np.sort(np.unique(track64_geometry["longitude"]))

with h5py.File(DESC_H5, "r") as handle:
    ny, nx = handle["cum"].shape[1:]
    descending_latitude = float(handle["corner_lat"][()]) + np.arange(ny) * float(
        handle["post_lat"][()]
    )
    descending_longitude = float(handle["corner_lon"][()]) + np.arange(nx) * float(
        handle["post_lon"][()]
    )

east_grid, north_grid, latitude_grid, longitude_grid, geographic_overlap = (
    common_utm11_grid(
        [ascending_latitude, descending_latitude],
        [ascending_longitude, descending_longitude],
        spacing_km=PREDICTION_GRID_SPACING_KM,
    )
)
targets_xy = np.column_stack([east_grid.ravel(), north_grid.ravel()])
network_xy = to_utm11_km(
    network["longitude"].to_numpy(float),
    network["latitude"].to_numpy(float),
)
topology = build_local_support_topology(network_xy, targets_xy, CONFIG)
print(
    "Prediction lattice:",
    east_grid.shape,
    "geographic overlap cells:",
    int(geographic_overlap.sum()),
)

event_tables: dict[str, pd.DataFrame] = {}
vertical_fields: dict[str, dict[str, np.ndarray | float]] = {}
for track in TRACK_TIMES:
    table = interval_table(track, EVENT_START, EVENT_END)
    event_tables[track] = table
    event_sill = estimate_interval_sill_mm2(
        table["up_mm"].to_numpy(float),
        table["sigma_up_mm"].to_numpy(float),
    )
    event_model = replace(selected_models[track], sill_mm2=event_sill)
    prediction = predict_local_vertical_from_topology(
        event_model,
        table[["east_km", "north_km"]].to_numpy(float),
        table["up_mm"].to_numpy(float),
        table["sigma_up_mm"].to_numpy(float),
        topology,
    )
    mean = prediction.mean_mm.reshape(east_grid.shape)
    sigma = prediction.sigma_mm.reshape(east_grid.shape)
    valid = prediction.valid.reshape(east_grid.shape) & geographic_overlap
    support_count = prediction.support_count.reshape(east_grid.shape)
    support_radius = prediction.support_radius_km.reshape(east_grid.shape)
    for array in (mean, sigma):
        array[~valid] = np.nan
    support_count = support_count.astype(float)
    support_count[~valid] = np.nan
    support_radius[~valid] = np.nan
    vertical_fields[track] = {
        "mean_mm": mean,
        "sigma_mm": sigma,
        "valid": valid,
        "support_count": support_count,
        "support_radius_km": support_radius,
        "event_sill_mm2": event_sill,
    }
    table.to_csv(
        OUTPUT_DIR / f"{track}_20190704_20190716_station_enu.csv",
        index=False,
    )
    print(
        track,
        "valid grid cells:",
        int(valid.sum()),
        "median contributors:",
        float(np.nanmedian(support_count)),
        "median sigma:",
        float(np.nanmedian(sigma)),
    )

# %%
all_vertical = np.concatenate(
    [
        np.asarray(vertical_fields[track]["mean_mm"])[
            np.isfinite(np.asarray(vertical_fields[track]["mean_mm"]))
        ]
        for track in TRACK_TIMES
    ]
)
vertical_limit = max(float(np.nanpercentile(np.abs(all_vertical), 98)), 1.0)
all_sigma = np.concatenate(
    [
        np.asarray(vertical_fields[track]["sigma_mm"])[
            np.isfinite(np.asarray(vertical_fields[track]["sigma_mm"]))
        ]
        for track in TRACK_TIMES
    ]
)
sigma_limit = max(float(np.nanpercentile(all_sigma, 98)), 1.0)

fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.2), constrained_layout=True)
for row, track in enumerate(TRACK_TIMES):
    field = vertical_fields[track]
    image = axes[row, 0].pcolormesh(
        east_grid,
        north_grid,
        field["mean_mm"],
        shading="nearest",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(
            vcenter=0.0, vmin=-vertical_limit, vmax=vertical_limit
        ),
    )
    uncertainty = axes[row, 1].pcolormesh(
        east_grid,
        north_grid,
        field["sigma_mm"],
        shading="nearest",
        cmap="magma_r",
        vmin=0.0,
        vmax=sigma_limit,
    )
    support = axes[row, 2].pcolormesh(
        east_grid,
        north_grid,
        field["support_count"],
        shading="nearest",
        cmap="viridis",
    )
    axes[row, 0].set_title(
        f"{track.replace('_', ' ')}: interpolated vertical increment"
    )
    axes[row, 1].set_title("Prediction uncertainty")
    axes[row, 2].set_title("All stations in local support")
    for column, label in enumerate(
        ("Vertical increment (mm)", "1σ (mm)", "Contributing stations")
    ):
        axes[row, column].set(
            xlabel="UTM 11N easting (km)",
            ylabel="UTM 11N northing (km)",
            aspect="equal",
        )
        axes[row, column].grid(True, color="0.88", ls="--", lw=0.5)
    plt.colorbar(image, ax=axes[row, 0], fraction=0.046, pad=0.03).set_label(
        "Vertical increment (mm)"
    )
    plt.colorbar(
        uncertainty, ax=axes[row, 1], fraction=0.046, pad=0.03
    ).set_label("1σ prediction uncertainty (mm)")
    plt.colorbar(
        support, ax=axes[row, 2], fraction=0.046, pad=0.03
    ).set_label("Contributing GNSS stations")
fig.suptitle(
    "4–16 July GNSS vertical interpolation only — no LOS correction applied",
    fontsize=16,
)
fig.savefig(
    OUTPUT_DIR / "02_event_vertical_interpolation_fields.png",
    bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ## 5. Save the independent gate and vertical fields

# %%
np.savez_compressed(
    OUTPUT_DIR / "gnss_vertical_20190704_20190716_1km.npz",
    east_km=east_grid,
    north_km=north_grid,
    latitude=latitude_grid,
    longitude=longitude_grid,
    ascending_vertical_mm=vertical_fields["ascending_T64"]["mean_mm"],
    ascending_vertical_sigma_mm=vertical_fields["ascending_T64"]["sigma_mm"],
    ascending_support_count=vertical_fields["ascending_T64"]["support_count"],
    ascending_support_radius_km=vertical_fields["ascending_T64"][
        "support_radius_km"
    ],
    descending_vertical_mm=vertical_fields["descending_T71"]["mean_mm"],
    descending_vertical_sigma_mm=vertical_fields["descending_T71"]["sigma_mm"],
    descending_support_count=vertical_fields["descending_T71"]["support_count"],
    descending_support_radius_km=vertical_fields["descending_T71"][
        "support_radius_km"
    ],
)

all_tracks_pass = bool(
    all(bool(summary["accepted"]) for summary in holdout_summaries.values())
)
manifest = {
    "purpose": "independent validation and interpolation of GNSS vertical displacement only",
    "method": (
        "all-station adaptive local kriging comparison: ordinary exponential, "
        "Matern-3/2, and squared-exponential/RBF covariance; plug-in local-"
        "constant GP and local first-order universal-kriging sensitivity "
        "candidates; no map-wide plane"
    ),
    "candidate_families": list(FAMILIES),
    "support_rule": asdict(CONFIG),
    "calibration_pairs": [
        [str(start.date()), str(end.date())]
        for start, end in calibration_pairs
    ],
    "independent_holdout_pairs": [
        [str(start.date()), str(end.date())] for start, end in holdout_pairs
    ],
    "shared_spatial_model": asdict(shared_model),
    "selected_models": {
        track: asdict(model) for track, model in selected_models.items()
    },
    "calibration_uncertainty_scales": uncertainty_scales,
    "holdout_summaries": holdout_summaries,
    "vertical_interpolation_passed_both_tracks": all_tracks_pass,
    "event_interval": "2019-07-04 to 2019-07-16 at each track's UTC acquisition time",
    "prediction_grid_spacing_km": PREDICTION_GRID_SPACING_KM,
    "event_sill_mm2": {
        track: float(vertical_fields[track]["event_sill_mm2"])
        for track in TRACK_TIMES
    },
    "event_field_uncertainty": (
        "raw GP/Kriging latent-field sigma retained; calibration scale was "
        "used only for independent held-out-observation coverage"
    ),
    "next_stage_authorized": (
        "project lU*Uhat and test LOS correction"
        if all_tracks_pass
        else "sensitivity only; do not claim a spatially resolved vertical correction"
    ),
    "not_performed": [
        "vertical-to-LOS projection",
        "ascending or descending LOS correction",
        "east-north decomposition",
        "strain calculation",
    ],
}
(OUTPUT_DIR / "vertical_interpolation_manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)
manifest

# %% [markdown]
# ### Decision rule
#
# The maps above are valid numerical interpolations regardless of the gate.
# They become an independently supported correction field only if both
# track-specific 2019 temporal holdouts pass. If either track fails, the maps
# remain a documented sensitivity product and must not be described as
# spatially resolved vertical deformation.
