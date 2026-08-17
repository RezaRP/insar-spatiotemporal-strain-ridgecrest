# %% [markdown]
# # Leave-one-station-out validation of vertical-corrected horizontal LOS
#
# This notebook validates the final cumulative horizontal-only LOS products
# against independent GNSS horizontal motion.  For every validation station:
#
# 1. the station is removed from the GNSS vertical interpolator;
# 2. cumulative vertical displacement is predicted at the surrounding InSAR
#    pixels and at the common reference disk;
# 3. the referenced vertical LOS contribution is removed from observed InSAR;
# 4. the result is compared with the station's GNSS horizontal projection
#    \(l_E E + l_N N\), relative to the same P463 reference.
#
# No offset, scale, ramp, or polarity is fitted in this validation.  The
# covariance parameters, station network, acquisition times, LOS signs, and
# reference definition are inherited from Notebook 15.

# %%
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import math
import sys

import h5py
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


def find_repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Run from inside the ridgecrest-insar repository.")


ROOT = find_repository_root(Path.cwd().resolve())
sys.path.insert(0, str(ROOT / "src"))

from ridgecrest_local_vertical import (  # noqa: E402
    LocalVerticalConfig,
    LocalVerticalModel,
    build_local_support_topology,
    predict_local_vertical_from_topology,
)
from ridgecrest_two_track import normalize_look_vectors  # noqa: E402
from ridgecrest_vertical_los import (  # noqa: E402
    gnss_interval_table,
    haversine_km,
    load_gnss_network,
    to_utm11_km,
)

mpl.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }
)

# %% [markdown]
# ## 1. Frozen inputs and validation rules

# %%
OUTPUT_DIR = ROOT / "outputs" / "cumulative_two_track_strain"
DIRECT_NPZ = OUTPUT_DIR / "direct_cumulative_vertical_corrected_en.npz"
VERTICAL_MODEL_MANIFEST = (
    ROOT
    / "outputs"
    / "gnss_vertical_interpolation_gate"
    / "vertical_interpolation_manifest.json"
)
FROZEN_MODEL_CSV = OUTPUT_DIR / "frozen_cumulative_vertical_interpolator.csv"
PHASE1_MANIFEST = (
    ROOT / "outputs" / "gnss_vertical_los_phase1" / "phase1_manifest.json"
)
TRACK64_LOOK = (
    ROOT
    / "outputs"
    / "track64_text_timeseries"
    / "track64_text_pixel_look_vectors.npz"
)
DESC_H5 = ROOT / "data" / "cum_full_scene_no_GACOS.h5"
GNSS_ROOT = Path(r"D:\Uni\Thises\GNSS_ridgecrest\data\tenv_data")

for required in (
    DIRECT_NPZ,
    VERTICAL_MODEL_MANIFEST,
    FROZEN_MODEL_CSV,
    PHASE1_MANIFEST,
    TRACK64_LOOK,
    DESC_H5,
    GNSS_ROOT,
):
    if not required.exists():
        raise FileNotFoundError(required)

TRACKS = ("ascending_T64", "descending_T71")
TRACK_PREFIX = {
    "ascending_T64": "ascending",
    "descending_T71": "descending",
}
TRACK_LABEL = {
    "ascending_T64": "Track 64 ascending",
    "descending_T71": "Track 71 descending",
}
TRACK_TIME = {
    "ascending_T64": pd.Timedelta(
        hours=1, minutes=50, seconds=8, microseconds=490464
    ),
    "descending_T71": pd.Timedelta(
        hours=13, minutes=51, seconds=41, microseconds=812911
    ),
}
EVENTS = (
    pd.Timestamp("2019-07-04T17:33:49"),
    pd.Timestamp("2019-07-06T03:19:53"),
)
CALIBRATION_END = pd.Timestamp("2019-05-29")
REFERENCE_STATION = "P463"
REFERENCE_RADIUS_KM = 1.5
STATION_SAMPLE_RADIUS_KM = 1.5
MIN_STATION_PIXELS = 2
MIN_REFERENCE_PIXELS = 10
MIN_VALID_EPOCH_FRACTION = 0.60
MIN_VALIDATION_EPOCHS = 20

direct = np.load(DIRECT_NPZ)
dates = pd.DatetimeIndex(pd.to_datetime(direct["dates"]))
reference_epoch = dates[0]
latitude_grid = np.asarray(direct["latitude"], dtype=float)
longitude_grid = np.asarray(direct["longitude"], dtype=float)
east_grid = np.asarray(direct["east_km"], dtype=float)
north_grid = np.asarray(direct["north_km"], dtype=float)
if not (
    latitude_grid.shape
    == longitude_grid.shape
    == east_grid.shape
    == north_grid.shape
):
    raise RuntimeError("The cumulative common-grid coordinates are inconsistent.")

persistent_stations = [
    value.decode("utf-8") if isinstance(value, bytes) else str(value)
    for value in direct["persistent_gnss_stations"]
]
if REFERENCE_STATION not in persistent_stations:
    raise RuntimeError("The frozen persistent network does not contain P463.")

vertical_manifest = json.loads(
    VERTICAL_MODEL_MANIFEST.read_text(encoding="utf-8")
)
phase1_manifest = json.loads(PHASE1_MANIFEST.read_text(encoding="utf-8"))
frozen_model_table = pd.read_csv(FROZEN_MODEL_CSV).set_index("track")


def restore_model(track: str) -> LocalVerticalModel:
    raw = dict(vertical_manifest["selected_models"][track])
    config_raw = dict(raw.pop("config"))
    config = LocalVerticalConfig(
        radii_km=tuple(float(value) for value in config_raw["radii_km"]),
        min_stations=int(config_raw["min_stations"]),
        sector_count=int(config_raw["sector_count"]),
        min_occupied_sectors=int(config_raw["min_occupied_sectors"]),
        require_local_hull=bool(config_raw["require_local_hull"]),
    )
    model = LocalVerticalModel(config=config, **raw)
    return replace(
        model,
        sill_mm2=float(
            frozen_model_table.loc[track, "frozen_pre_event_sill_mm2"]
        ),
    )


models = {track: restore_model(track) for track in TRACKS}
insar_sigma_mm = {
    track: float(phase1_manifest["tracks"][track]["insar_ramp_scale_mm"])
    for track in TRACKS
}

# %% [markdown]
# ## 2. Rebuild exact-time cumulative GNSS ENU tables
#
# The same event-aware endpoint sampler used by Notebook 15 is applied.  In
# particular, both 4 July acquisitions are evaluated from the pre-event GNSS
# segment because both precede the Mw 6.4 origin time.

# %%
histories, network = load_gnss_network(GNSS_ROOT)
network = (
    network.set_index("station")
    .loc[persistent_stations]
    .reset_index()
)
network_xy = to_utm11_km(
    network["longitude"].to_numpy(float),
    network["latitude"].to_numpy(float),
)

table_by_track: dict[str, list[pd.DataFrame | None]] = {
    track: [None] for track in TRACKS
}
for track in TRACKS:
    for date in dates[1:]:
        table = gnss_interval_table(
            histories,
            network,
            start=reference_epoch + TRACK_TIME[track],
            end=pd.Timestamp(date) + TRACK_TIME[track],
            event_times=EVENTS,
            strict=False,
        )
        available = set(table["station"].astype(str))
        missing = sorted(set(persistent_stations).difference(available))
        if missing:
            raise RuntimeError(
                f"{track} {date.date()} lost persistent stations: {missing}"
            )
        table_by_track[track].append(
            table.set_index("station")
            .loc[persistent_stations]
            .reset_index()
        )

frozen_sigma_up: dict[str, np.ndarray] = {}
for track in TRACKS:
    pre_event_tables = [
        table
        for date, table in zip(dates[1:], table_by_track[track][1:])
        if date <= CALIBRATION_END and table is not None
    ]
    frozen_sigma_up[track] = np.nanmedian(
        np.stack(
            [
                table["sigma_up_mm"].to_numpy(float)
                for table in pre_event_tables
            ]
        ),
        axis=0,
    )

for track in TRACKS:
    zero = network.copy()
    for component in ("east", "north", "up"):
        zero[f"{component}_mm"] = 0.0
        zero[f"sigma_{component}_mm"] = 0.0
    table_by_track[track][0] = zero

# %% [markdown]
# ## 3. Build native reference geometries
#
# Reference vertical LOS is recomputed for every held-out station using the
# native P463 disks used by Notebook 15, rather than approximating the reference
# with one common-grid cell.

# %%
track64_look = np.load(TRACK64_LOOK)
asc_reference_mask = (
    haversine_km(
        np.asarray(track64_look["latitude"], dtype=float),
        np.asarray(track64_look["longitude"], dtype=float),
        float(network.set_index("station").loc[REFERENCE_STATION, "latitude"]),
        float(network.set_index("station").loc[REFERENCE_STATION, "longitude"]),
    )
    <= REFERENCE_RADIUS_KM
)
asc_reference_mask &= (
    np.isfinite(track64_look["los_e"])
    & np.isfinite(track64_look["los_n"])
    & np.isfinite(track64_look["los_u"])
)
asc_reference_xy = to_utm11_km(
    np.asarray(track64_look["longitude"], dtype=float)[asc_reference_mask],
    np.asarray(track64_look["latitude"], dtype=float)[asc_reference_mask],
)
asc_reference_look = np.column_stack(
    [
        np.asarray(track64_look[f"los_{component}"], dtype=float)[
            asc_reference_mask
        ]
        for component in ("e", "n", "u")
    ]
)

with h5py.File(DESC_H5, "r") as handle:
    _, desc_ny, desc_nx = handle["cum"].shape
    desc_latitude = (
        float(handle["corner_lat"][()])
        + np.arange(desc_ny) * float(handle["post_lat"][()])
    )
    desc_longitude = (
        float(handle["corner_lon"][()])
        + np.arange(desc_nx) * float(handle["post_lon"][()])
    )
    desc_e, desc_n, desc_u = normalize_look_vectors(
        np.asarray(handle["E.geo"][:], dtype=float),
        np.asarray(handle["N.geo"][:], dtype=float),
        np.asarray(handle["U.geo"][:], dtype=float),
    )
    desc_quality = (
        np.isfinite(handle["coh_avg"][:])
        & (handle["coh_avg"][:] >= 0.30)
        & np.isfinite(handle["resid_rms"][:])
        & (handle["resid_rms"][:] <= 5.0)
        & np.isfinite(handle["n_gap"][:])
        & (handle["n_gap"][:] <= 2)
        & np.isfinite(handle["n_loop_err"][:])
        & (handle["n_loop_err"][:] <= 10)
    )

desc_lat_grid, desc_lon_grid = np.meshgrid(
    desc_latitude, desc_longitude, indexing="ij"
)
reference_row = network.set_index("station").loc[REFERENCE_STATION]
desc_reference_mask = desc_quality & (
    haversine_km(
        desc_lat_grid,
        desc_lon_grid,
        float(reference_row["latitude"]),
        float(reference_row["longitude"]),
    )
    <= REFERENCE_RADIUS_KM
)
desc_reference_mask &= (
    np.isfinite(desc_e) & np.isfinite(desc_n) & np.isfinite(desc_u)
)
desc_reference_xy = to_utm11_km(
    desc_lon_grid[desc_reference_mask],
    desc_lat_grid[desc_reference_mask],
)
desc_reference_look = np.column_stack(
    [
        component[desc_reference_mask]
        for component in (desc_e, desc_n, desc_u)
    ]
)

reference_xy = {
    "ascending_T64": asc_reference_xy,
    "descending_T71": desc_reference_xy,
}
reference_look = {
    "ascending_T64": asc_reference_look,
    "descending_T71": desc_reference_look,
}
for track in TRACKS:
    if len(reference_xy[track]) < MIN_REFERENCE_PIXELS:
        raise RuntimeError(f"{track} has too few native P463 reference pixels.")

# %% [markdown]
# ## 4. Leave-one-station-out HLOS reconstruction and GNSS comparison

# %%
validation_rows: list[dict[str, object]] = []
exclusion_rows: list[dict[str, object]] = []
station_metadata = network.set_index("station")
station_index = {station: index for index, station in enumerate(persistent_stations)}

for track in TRACKS:
    prefix = TRACK_PREFIX[track]
    observed_cube = np.asarray(
        direct[f"{prefix}_observed_los_cumulative_mm"], dtype=float
    )
    look_e = np.asarray(direct[f"{prefix}_look_e"], dtype=float)
    look_n = np.asarray(direct[f"{prefix}_look_n"], dtype=float)
    look_u = np.asarray(direct[f"{prefix}_look_u"], dtype=float)
    geometry_valid = (
        np.isfinite(look_e) & np.isfinite(look_n) & np.isfinite(look_u)
    )
    epoch_fraction = np.mean(np.isfinite(observed_cube), axis=0)
    reference_e = float(np.nanmedian(reference_look[track][:, 0]))
    reference_n = float(np.nanmedian(reference_look[track][:, 1]))

    for station in persistent_stations:
        if station == REFERENCE_STATION:
            exclusion_rows.append(
                {
                    "track": track,
                    "station": station,
                    "reason": "common spatial reference station",
                }
            )
            continue

        station_row = station_metadata.loc[station]
        station_mask = (
            haversine_km(
                latitude_grid,
                longitude_grid,
                float(station_row["latitude"]),
                float(station_row["longitude"]),
            )
            <= STATION_SAMPLE_RADIUS_KM
        )
        station_mask &= (
            geometry_valid
            & (epoch_fraction >= MIN_VALID_EPOCH_FRACTION)
        )
        if int(station_mask.sum()) < MIN_STATION_PIXELS:
            exclusion_rows.append(
                {
                    "track": track,
                    "station": station,
                    "reason": (
                        f"fewer than {MIN_STATION_PIXELS} persistent "
                        "common-grid pixels"
                    ),
                }
            )
            continue

        held_out_index = station_index[station]
        train_mask = np.ones(len(persistent_stations), dtype=bool)
        train_mask[held_out_index] = False
        train_stations = [
            name
            for index, name in enumerate(persistent_stations)
            if train_mask[index]
        ]
        train_xy = network_xy[train_mask]

        target_xy = np.column_stack(
            [east_grid[station_mask], north_grid[station_mask]]
        )
        target_look = np.column_stack(
            [
                look_e[station_mask],
                look_n[station_mask],
                look_u[station_mask],
            ]
        )
        target_topology = build_local_support_topology(
            train_xy, target_xy, models[track].config
        )
        ref_topology = build_local_support_topology(
            train_xy, reference_xy[track], models[track].config
        )
        if not any(support is not None for support in target_topology.supports):
            exclusion_rows.append(
                {
                    "track": track,
                    "station": station,
                    "reason": "held-out target lies outside qualified local support",
                }
            )
            continue
        if sum(support is not None for support in ref_topology.supports) < MIN_REFERENCE_PIXELS:
            exclusion_rows.append(
                {
                    "track": track,
                    "station": station,
                    "reason": "held-out P463 reference support is insufficient",
                }
            )
            continue

        station_row_count = 0
        for epoch_index, date in enumerate(dates):
            table = table_by_track[track][epoch_index]
            if table is None:
                continue
            ordered = table.set_index("station").loc[persistent_stations]
            train_values = ordered.loc[train_stations, "up_mm"].to_numpy(float)
            train_sigma = frozen_sigma_up[track][train_mask]
            target_prediction = predict_local_vertical_from_topology(
                models[track],
                train_xy,
                train_values,
                train_sigma,
                target_topology,
            )
            reference_prediction = predict_local_vertical_from_topology(
                models[track],
                train_xy,
                train_values,
                train_sigma,
                ref_topology,
            )
            valid_reference = (
                reference_prediction.valid
                & np.all(np.isfinite(reference_look[track]), axis=1)
            )
            if int(valid_reference.sum()) < MIN_REFERENCE_PIXELS:
                continue

            reference_raw_vlos = (
                reference_look[track][valid_reference, 2]
                * reference_prediction.mean_mm[valid_reference]
            )
            reference_vlos = float(np.nanmedian(reference_raw_vlos))
            reference_vlos_sigma = float(
                np.nanmedian(
                    np.abs(reference_look[track][valid_reference, 2])
                    * reference_prediction.sigma_mm[valid_reference]
                )
            )

            observed = observed_cube[epoch_index, station_mask]
            valid_target = (
                target_prediction.valid
                & np.isfinite(observed)
                & np.all(np.isfinite(target_look), axis=1)
            )
            if int(valid_target.sum()) < MIN_STATION_PIXELS:
                continue

            raw_vertical_los = (
                target_look[valid_target, 2]
                * target_prediction.mean_mm[valid_target]
            )
            referenced_vertical_los = raw_vertical_los - reference_vlos
            hlos_pixels = observed[valid_target] - referenced_vertical_los
            hlos_mm = float(np.nanmedian(hlos_pixels))
            observed_los_mm = float(np.nanmedian(observed[valid_target]))
            vertical_los_mm = float(np.nanmedian(referenced_vertical_los))
            local_e = float(np.nanmedian(target_look[valid_target, 0]))
            local_n = float(np.nanmedian(target_look[valid_target, 1]))

            target_vertical_sigma = float(
                np.nanmedian(
                    np.abs(target_look[valid_target, 2])
                    * target_prediction.sigma_mm[valid_target]
                )
            )
            hlos_sigma = math.sqrt(
                insar_sigma_mm[track] ** 2
                + target_vertical_sigma**2
                + reference_vlos_sigma**2
            )

            target_gnss = ordered.loc[station]
            reference_gnss = ordered.loc[REFERENCE_STATION]
            gnss_horizontal_target = (
                local_e * float(target_gnss["east_mm"])
                + local_n * float(target_gnss["north_mm"])
            )
            gnss_horizontal_reference = (
                reference_e * float(reference_gnss["east_mm"])
                + reference_n * float(reference_gnss["north_mm"])
            )
            gnss_horizontal = (
                gnss_horizontal_target - gnss_horizontal_reference
            )
            target_horizontal_sigma = math.sqrt(
                (
                    local_e
                    * float(target_gnss["sigma_east_mm"])
                )
                ** 2
                + (
                    local_n
                    * float(target_gnss["sigma_north_mm"])
                )
                ** 2
            )
            reference_horizontal_sigma = math.sqrt(
                (
                    reference_e
                    * float(reference_gnss["sigma_east_mm"])
                )
                ** 2
                + (
                    reference_n
                    * float(reference_gnss["sigma_north_mm"])
                )
                ** 2
            )
            gnss_horizontal_sigma = math.sqrt(
                target_horizontal_sigma**2 + reference_horizontal_sigma**2
            )
            residual = hlos_mm - gnss_horizontal
            total_sigma = math.sqrt(
                hlos_sigma**2 + gnss_horizontal_sigma**2
            )
            station_row_count += 1
            validation_rows.append(
                {
                    "track": track,
                    "track_label": TRACK_LABEL[track],
                    "date": pd.Timestamp(date),
                    "acquisition_utc": pd.Timestamp(date) + TRACK_TIME[track],
                    "station": station,
                    "latitude": float(station_row["latitude"]),
                    "longitude": float(station_row["longitude"]),
                    "held_out_from_vertical_interpolator": True,
                    "pixel_count": int(valid_target.sum()),
                    "observed_los_mm": observed_los_mm,
                    "loo_vertical_to_los_mm": vertical_los_mm,
                    "insar_horizontal_los_mm": hlos_mm,
                    "insar_horizontal_los_sigma_mm": hlos_sigma,
                    "gnss_horizontal_los_mm": gnss_horizontal,
                    "gnss_horizontal_los_sigma_mm": gnss_horizontal_sigma,
                    "residual_insar_minus_gnss_mm": residual,
                    "total_sigma_mm": total_sigma,
                    "standardized_residual": (
                        residual / total_sigma if total_sigma > 0.0 else np.nan
                    ),
                    "gnss_up_observed_mm": float(target_gnss["up_mm"]),
                    "gnss_up_loo_predicted_mm": float(
                        np.nanmedian(
                            target_prediction.mean_mm[valid_target]
                        )
                    ),
                    "gnss_up_loo_residual_mm": float(
                        np.nanmedian(
                            target_prediction.mean_mm[valid_target]
                        )
                        - float(target_gnss["up_mm"])
                    ),
                    "target_support_count_min": int(
                        np.min(
                            target_prediction.support_count[valid_target]
                        )
                    ),
                    "target_support_radius_km_max": float(
                        np.max(
                            target_prediction.support_radius_km[valid_target]
                        )
                    ),
                    "reference_pixel_count": int(valid_reference.sum()),
                }
            )

        if station_row_count < MIN_VALIDATION_EPOCHS:
            exclusion_rows.append(
                {
                    "track": track,
                    "station": station,
                    "reason": (
                        f"only {station_row_count} valid epochs "
                        f"(<{MIN_VALIDATION_EPOCHS})"
                    ),
                }
            )

validation = pd.DataFrame(validation_rows)
if validation.empty:
    raise RuntimeError("No station-level HLOS validation rows were produced.")

# Remove any station-track group that did not meet the epoch-count requirement.
group_count = validation.groupby(["track", "station"])["date"].transform("size")
validation = validation.loc[group_count >= MIN_VALIDATION_EPOCHS].copy()
validation.sort_values(["track", "station", "date"], inplace=True)
validation.to_csv(
    OUTPUT_DIR / "post_correction_station_validation_rows.csv",
    index=False,
)
pd.DataFrame(exclusion_rows).drop_duplicates().to_csv(
    OUTPUT_DIR / "post_correction_station_validation_exclusions.csv",
    index=False,
)

print(
    validation.groupby("track").agg(
        stations=("station", "nunique"),
        station_epochs=("date", "size"),
        first_date=("date", "min"),
        last_date=("date", "max"),
    )
)

# %% [markdown]
# ## 5. Descriptive validation metrics
#
# Metrics are reported without station-epoch independence assumptions.  No
# p-values are calculated because cumulative epochs and neighbouring station
# samples are temporally and spatially correlated.

# %%
def metric_record(
    data: pd.DataFrame,
    *,
    track: str,
    period: str,
    station: str,
) -> dict[str, object]:
    observed = data["insar_horizontal_los_mm"].to_numpy(float)
    expected = data["gnss_horizontal_los_mm"].to_numpy(float)
    residual = data["residual_insar_minus_gnss_mm"].to_numpy(float)
    standardized = data["standardized_residual"].to_numpy(float)
    finite = (
        np.isfinite(observed)
        & np.isfinite(expected)
        & np.isfinite(residual)
    )
    observed = observed[finite]
    expected = expected[finite]
    residual = residual[finite]
    standardized = standardized[finite]
    correlation = (
        float(np.corrcoef(observed, expected)[0, 1])
        if len(observed) >= 3
        and np.nanstd(observed) > 0.0
        and np.nanstd(expected) > 0.0
        else np.nan
    )
    return {
        "track": track,
        "period": period,
        "station": station,
        "station_count": int(data["station"].nunique()),
        "observation_count": int(len(observed)),
        "pearson_r": correlation,
        "median_bias_mm": float(np.nanmedian(residual)),
        "mae_mm": float(np.nanmean(np.abs(residual))),
        "rmse_mm": float(np.sqrt(np.nanmean(np.square(residual)))),
        "standardized_rms": float(
            np.sqrt(np.nanmean(np.square(standardized)))
        ),
        "coverage90_fraction": float(
            np.nanmean(np.abs(standardized) <= 1.6448536269514722)
        ),
        "median_abs_gnss_horizontal_mm": float(
            np.nanmedian(np.abs(expected))
        ),
        "median_abs_insar_horizontal_mm": float(
            np.nanmedian(np.abs(observed))
        ),
    }


metric_rows: list[dict[str, object]] = []
period_masks = {
    "pre_event_through_2019-07-04": (
        (validation["date"] > reference_epoch)
        & (validation["date"] <= pd.Timestamp("2019-07-04"))
    ),
    "earthquake_sequence_epoch_2019-07-16": (
        validation["date"] == pd.Timestamp("2019-07-16")
    ),
    "post_sequence_after_2019-07-16": (
        validation["date"] > pd.Timestamp("2019-07-16")
    ),
    "all_non_reference_epochs": (
        validation["date"] > reference_epoch
    ),
}
for track in TRACKS:
    track_data = validation.loc[validation["track"] == track]
    for period, global_mask in period_masks.items():
        data = track_data.loc[global_mask.loc[track_data.index]]
        if len(data) >= 3:
            metric_rows.append(
                metric_record(
                    data,
                    track=track,
                    period=period,
                    station="ALL",
                )
            )
    for station, station_data in track_data.groupby("station"):
        data = station_data.loc[station_data["date"] > reference_epoch]
        if len(data) >= 3:
            metric_rows.append(
                metric_record(
                    data,
                    track=track,
                    period="all_non_reference_epochs",
                    station=str(station),
                )
            )

metrics = pd.DataFrame(metric_rows)
metrics.to_csv(
    OUTPUT_DIR / "post_correction_station_validation_metrics.csv",
    index=False,
)
print(
    metrics.loc[metrics["station"] == "ALL"].round(
        {
            "pearson_r": 3,
            "median_bias_mm": 2,
            "mae_mm": 2,
            "rmse_mm": 2,
            "standardized_rms": 2,
            "coverage90_fraction": 3,
        }
    )
)

# %% [markdown]
# ## 6. Validation figure

# %%
figure_data = validation.loc[validation["date"] > reference_epoch].copy()
date_numbers = mdates.date2num(figure_data["date"])
date_norm = Normalize(vmin=float(date_numbers.min()), vmax=float(date_numbers.max()))
cmap = mpl.colormaps["viridis"]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14.2, 9.2),
    constrained_layout=True,
)
for column, track in enumerate(TRACKS):
    data = figure_data.loc[figure_data["track"] == track]
    aggregate = metrics.loc[
        (metrics["track"] == track)
        & (metrics["station"] == "ALL")
        & (metrics["period"] == "all_non_reference_epochs")
    ].iloc[0]

    ax = axes[0, column]
    ax.scatter(
        data["gnss_horizontal_los_mm"],
        data["insar_horizontal_los_mm"],
        c=mdates.date2num(data["date"]),
        cmap=cmap,
        norm=date_norm,
        s=15,
        alpha=0.62,
        linewidths=0,
    )
    finite_values = np.r_[
        data["gnss_horizontal_los_mm"].to_numpy(float),
        data["insar_horizontal_los_mm"].to_numpy(float),
    ]
    limit = float(np.nanpercentile(np.abs(finite_values), 99.0))
    limit = max(limit, 10.0)
    ax.plot([-limit, limit], [-limit, limit], color="0.20", lw=1.2, ls="--")
    ax.set(
        xlim=(-limit, limit),
        ylim=(-limit, limit),
        xlabel="GNSS horizontal LOS (mm)",
        ylabel="InSAR horizontal LOS (mm)",
        title=(
            f"{chr(97 + column)}) {TRACK_LABEL[track]}\n"
            f"r={aggregate.pearson_r:.2f}; "
            f"RMSE={aggregate.rmse_mm:.1f} mm; "
            f"{int(aggregate.station_count)} stations"
        ),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.90", lw=0.7)

    ax = axes[1, column]
    date_summary = (
        data.groupby("date")["residual_insar_minus_gnss_mm"]
        .agg(
            median="median",
            q25=lambda values: np.nanquantile(values, 0.25),
            q75=lambda values: np.nanquantile(values, 0.75),
            station_count="count",
        )
        .reset_index()
    )
    ax.fill_between(
        date_summary["date"],
        date_summary["q25"],
        date_summary["q75"],
        color="#8EC5E8",
        alpha=0.45,
        label="Station IQR",
    )
    ax.plot(
        date_summary["date"],
        date_summary["median"],
        color="#155A9C",
        marker="o",
        ms=3.2,
        lw=1.25,
        label="Station median",
    )
    ax.axhline(0.0, color="0.25", lw=1.0)
    for event_time in EVENTS:
        ax.axvline(event_time, color="#B22222", lw=0.9, ls=":")
    ax.set(
        xlabel="Acquisition date",
        ylabel="InSAR − GNSS horizontal LOS (mm)",
        title=f"{chr(99 + column)}) Residual evolution",
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, color="0.90", lw=0.7)
    ax.legend(loc="best")

colorbar = fig.colorbar(
    mpl.cm.ScalarMappable(norm=date_norm, cmap=cmap),
    ax=axes[0, :],
    orientation="horizontal",
    fraction=0.06,
    pad=0.04,
)
colorbar.set_label("Acquisition date")
tick_dates = pd.date_range(dates.min(), dates.max(), periods=5)
colorbar.set_ticks(mdates.date2num(tick_dates))
colorbar.set_ticklabels([date.strftime("%Y-%m") for date in tick_dates])
fig.suptitle(
    "Leave-one-station-out validation of vertical-corrected InSAR horizontal LOS",
    fontsize=15,
    fontweight="semibold",
)
fig.savefig(
    OUTPUT_DIR / "07_post_correction_hlos_gnss_validation.png",
    bbox_inches="tight",
)
fig.savefig(
    OUTPUT_DIR / "07_post_correction_hlos_gnss_validation.pdf",
    bbox_inches="tight",
)
plt.close(fig)

# %% [markdown]
# ## 7. Reproducibility manifest and hard audits

# %%
aggregate_metrics = (
    metrics.loc[
        (metrics["station"] == "ALL")
        & (metrics["period"] == "all_non_reference_epochs")
    ]
    .set_index("track")
    .to_dict(orient="index")
)
manifest = {
    "purpose": (
        "post-correction station-level validation of cumulative InSAR "
        "horizontal LOS against GNSS horizontal LOS"
    ),
    "formula": (
        "HLOS_LOO = signed_DLOS - referenced(lU * Uhat_LOO); "
        "GNSS_HLOS = referenced(lE * E_GNSS + lN * N_GNSS)"
    ),
    "reference_epoch": str(reference_epoch.date()),
    "reference_station": REFERENCE_STATION,
    "reference_radius_km": REFERENCE_RADIUS_KM,
    "station_sample_radius_km": STATION_SAMPLE_RADIUS_KM,
    "vertical_leave_one_out": True,
    "offset_scale_or_ramp_fitted_during_validation": False,
    "track_signs": {
        "ascending_T64": 1,
        "descending_T71": 1,
    },
    "acquisition_times": {
        track: str(TRACK_TIME[track]) for track in TRACKS
    },
    "validated_station_count": {
        track: int(
            validation.loc[validation["track"] == track, "station"].nunique()
        )
        for track in TRACKS
    },
    "aggregate_metrics_all_non_reference_epochs": aggregate_metrics,
    "outputs": {
        "rows": "post_correction_station_validation_rows.csv",
        "metrics": "post_correction_station_validation_metrics.csv",
        "exclusions": "post_correction_station_validation_exclusions.csv",
        "figure_png": "07_post_correction_hlos_gnss_validation.png",
        "figure_pdf": "07_post_correction_hlos_gnss_validation.pdf",
    },
    "limitations": [
        (
            "Cumulative station-epoch residuals are temporally correlated; "
            "pooled metrics are descriptive and no independence-based p-values "
            "are reported."
        ),
        (
            "The InSAR uncertainty uses the same track-wide noise scale as "
            "Notebook 15 and does not model full spatial or temporal covariance."
        ),
        (
            "GNSS horizontal reference is represented by P463 projected with "
            "the median native reference-disk look vector."
        ),
        (
            "Ascending and descending nominal dates differ by approximately "
            "12.026 hours and are validated at their separate acquisition times."
        ),
        (
            "A station is retained only where the held-out vertical predictor "
            "and at least two persistent InSAR cells are available."
        ),
    ],
}
(OUTPUT_DIR / "post_correction_station_validation_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

if not bool(validation["held_out_from_vertical_interpolator"].all()):
    raise RuntimeError("A validation row was not produced with station LOO.")
if set(validation["track"]) != set(TRACKS):
    raise RuntimeError("Both tracks were not represented in validation.")
if validation.groupby(["track", "station"]).size().min() < MIN_VALIDATION_EPOCHS:
    raise RuntimeError("An under-supported station remained in validation.")
if not np.allclose(
    validation.loc[
        validation["date"] == reference_epoch,
        [
            "insar_horizontal_los_mm",
            "gnss_horizontal_los_mm",
        ],
    ].to_numpy(float),
    0.0,
    atol=1.0e-5,
):
    raise RuntimeError("Reference-epoch HLOS validation identity failed.")

print("Wrote post-correction station validation products to", OUTPUT_DIR)
