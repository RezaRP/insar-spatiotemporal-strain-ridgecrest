# %% [markdown]
# # Cumulative GNSS-vertical correction and dense cumulative 2-D strain
#
# This notebook implements the cumulative-first operation requested for the
# Ridgecrest two-track analysis.  The processing order is
#
# \[
# U_s(t)-U_s(t_0)\rightarrow \widehat U(x,y,t)\rightarrow
# l_U(x,y)\widehat U(x,y,t)\rightarrow D_{\rm HLOS}(x,y,t)
# \rightarrow [E,N](x,y,t)\rightarrow \epsilon(x,y,t).
# \]
#
# Every quantity is cumulative relative to the first common acquisition,
# 27 May 2017.  Statistical change detection is performed later on 12- and
# 24-day innovations of these cumulative strain fields; it is not fitted to
# earthquake or post-earthquake epochs.

# %%
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import math
import os
import sys
import warnings

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay


def find_repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Run from inside the ridgecrest-insar repository.")


ROOT = find_repository_root(Path.cwd().resolve())
sys.path.insert(0, str(ROOT / "src"))

from ridgecrest_cumulative_strain import (  # noqa: E402
    build_fixed_joint_mls,
    evaluate_fixed_joint_mls,
    fixed_joint_mls_component_sigma,
    target_values_to_grid,
)
from ridgecrest_gnss_strain import load_rupture_segments_utm  # noqa: E402
from ridgecrest_jump import load_text_stack  # noqa: E402
from ridgecrest_local_vertical import (  # noqa: E402
    LocalVerticalConfig,
    LocalVerticalModel,
    build_local_support_topology,
    estimate_interval_sill_mm2,
    predict_local_vertical_from_topology,
)
from ridgecrest_strain_change import (  # noqa: E402
    duration_normalize,
    empirical_upper_tail_pvalue,
    fit_robust_baseline,
    leave_one_out_baseline_innovations,
    maximum_signed_cluster_mass,
    positive_page_cusum,
    signed_spatial_clusters,
    sliding_block_cusum_maxima,
    sliding_block_maximum,
    standardized_innovation,
    strain_energy,
)
from ridgecrest_two_track import (  # noqa: E402
    common_utm11_grid,
    correct_vertical_los_on_grid,
    masked_bilinear_resample,
    normalize_look_vectors,
    rupture_point_distance_lower_bound_km,
    solve_two_track_horizontal,
    to_utm11_km,
)
from ridgecrest_vertical_los import (  # noqa: E402
    gnss_interval_table,
    haversine_km,
    load_gnss_network,
)

mpl.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
)

# %% [markdown]
# ## 1. Fixed inputs and validation boundaries
#
# The ascending cumulative series is the Track-64 date-named text stack.  The
# descending cumulative series is the Track-71 no-GACOS full-scene HDF5.
# Tracks are paired by nominal date but differ by about 12 hours.  Pixel-specific
# look vectors are used throughout.
#
# The full-scene event-time GNSS vertical interpolation gate previously failed.
# Therefore cumulative strain is an observational result only farther than
# 18 km from the mapped rupture: a 10-km rupture-continuity buffer plus the
# 8-km derivative support.

# %%
TEXT_DIR = ROOT / "data"
DESC_H5 = ROOT / "data" / "cum_full_scene_no_GACOS.h5"
TRACK64_LOOK = (
    ROOT
    / "outputs"
    / "track64_text_timeseries"
    / "track64_text_pixel_look_vectors.npz"
)
COMMON_DATES_FILE = (
    ROOT
    / "outputs"
    / "track64_text_timeseries"
    / "track64_track71_common_dates.csv"
)
LOCAL_MANIFEST = (
    ROOT
    / "outputs"
    / "gnss_vertical_interpolation_gate"
    / "vertical_interpolation_manifest.json"
)
SIGN_MANIFEST = (
    ROOT
    / "outputs"
    / "station_los_projection_validation"
    / "station_projection_manifest.json"
)
GNSS_ROOT = Path(r"D:\Uni\Thises\GNSS_ridgecrest\data\tenv_data")
FAULT_FILE = ROOT / "data" / "cgs_2019_ridgecrest_fault_ruptures.geojson"
OLD_PRODUCT = (
    ROOT
    / "outputs"
    / "two_track_vertical_corrected_timeseries"
    / "two_track_vertical_corrected_en_timeseries.npz"
)

OUTPUT_DIR = ROOT / "outputs" / "cumulative_two_track_strain"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIRECT_NPZ = OUTPUT_DIR / "direct_cumulative_vertical_corrected_en.npz"
STRAIN_NPZ = OUTPUT_DIR / "dense_cumulative_strain_1km.npz"

FORCE_REBUILD_DIRECT = os.environ.get("RIDGECREST_FORCE_CUMULATIVE", "0") == "1"
FORCE_REBUILD_STRAIN = os.environ.get("RIDGECREST_FORCE_STRAIN", "0") == "1"
SAVE_MAP_PDF = False

TRACK_TIMES = {
    "ascending_T64": pd.Timedelta(
        hours=1, minutes=50, seconds=8, microseconds=490464
    ),
    "descending_T71": pd.Timedelta(
        hours=13, minutes=51, seconds=41, microseconds=812911
    ),
}
NOMINAL_INCIDENCE_DEG = {
    "ascending_T64": 39.6181,
    "descending_T71": 33.7677,
}
EVENTS = (
    pd.Timestamp("2019-07-04T17:33:49"),
    pd.Timestamp("2019-07-06T03:19:53"),
)
REFERENCE_STATION = "P463"
REFERENCE_RADIUS_KM = 1.5
COMMON_GRID_SPACING_KM = 1.0
RUPTURE_BUFFER_KM = 10.0
STRAIN_SUPPORT_RADIUS_KM = 8.0
STRAIN_BANDWIDTH_KM = 4.0
STRAIN_SAFE_DISTANCE_KM = (
    RUPTURE_BUFFER_KM + STRAIN_SUPPORT_RADIUS_KM
)
MIN_STRAIN_SAMPLES = 16
LOS_SIGMA_ASC_MM = 24.335595497061632
LOS_SIGMA_DESC_MM = 16.277079925630176
CALIBRATION_END = pd.Timestamp("2019-05-29")
PRE_EVENT_END = pd.Timestamp("2019-07-04")
EVENT_END = pd.Timestamp("2019-07-16")

for item in (
    DESC_H5,
    TRACK64_LOOK,
    COMMON_DATES_FILE,
    LOCAL_MANIFEST,
    SIGN_MANIFEST,
    GNSS_ROOT,
    FAULT_FILE,
):
    if not item.exists():
        raise FileNotFoundError(item)


def restore_model(raw: dict[str, object]) -> LocalVerticalModel:
    config_raw = dict(raw["config"])
    config = LocalVerticalConfig(
        radii_km=tuple(float(value) for value in config_raw["radii_km"]),
        min_stations=int(config_raw["min_stations"]),
        sector_count=int(config_raw["sector_count"]),
        min_occupied_sectors=int(config_raw["min_occupied_sectors"]),
        require_local_hull=bool(config_raw["require_local_hull"]),
    )
    values = dict(raw)
    values["config"] = config
    return LocalVerticalModel(**values)  # type: ignore[arg-type]


local_manifest = json.loads(LOCAL_MANIFEST.read_text(encoding="utf-8"))
sign_manifest = json.loads(SIGN_MANIFEST.read_text(encoding="utf-8"))
base_models = {
    track: restore_model(raw)
    for track, raw in local_manifest["selected_models"].items()
}
if not all(
    bool(value)
    for value in json.loads(
        (
            ROOT
            / "outputs"
            / "two_track_vertical_corrected_timeseries"
            / "two_track_vertical_corrected_timeseries_manifest.json"
        ).read_text(encoding="utf-8")
    )["event_interval_off_fault_gate"].values()
):
    raise RuntimeError("The required off-fault vertical validation gate failed")

# %% [markdown]
# ## 2. Direct cumulative GNSS-U interpolation and two-track E–N solution
#
# For each track and every common epoch \(t\), the notebook samples each
# station at the actual acquisition time and calculates
# \(U_s(t)-U_s(t_0)\).  One persistent 24-station set, one local support
# topology, one pre-event sill, and one pre-event uncertainty vector are frozen
# before prediction.  Thus the mean interpolation weights cannot change at the
# earthquake.

# %%
def build_direct_cumulative_product() -> None:
    ascending = load_text_stack(TEXT_DIR, align_coordinates=True)
    look = np.load(TRACK64_LOOK)
    if not (
        np.array_equal(look["latitude"], ascending.latitude)
        and np.array_equal(look["longitude"], ascending.longitude)
    ):
        raise RuntimeError("Track-64 look vectors do not match text coordinates")

    asc_latitude = np.sort(np.unique(ascending.latitude))
    asc_longitude = np.sort(np.unique(ascending.longitude))
    asc_row = np.searchsorted(asc_latitude, ascending.latitude)
    asc_col = np.searchsorted(asc_longitude, ascending.longitude)

    def text_vector_to_grid(values: np.ndarray) -> np.ndarray:
        grid = np.full(
            (len(asc_latitude), len(asc_longitude)),
            np.nan,
            dtype=float,
        )
        grid[asc_row, asc_col] = np.asarray(values, dtype=float)
        return grid

    asc_look_grid = {
        name: text_vector_to_grid(look[f"los_{name.lower()}"])
        for name in ("E", "N", "U")
    }
    asc_geometry_valid = (
        np.isfinite(asc_look_grid["E"])
        & np.isfinite(asc_look_grid["N"])
        & np.isfinite(asc_look_grid["U"])
    )

    with h5py.File(DESC_H5, "r") as handle:
        desc_dates = pd.to_datetime(
            np.asarray(handle["imdates"][:], dtype=np.int64).astype(str),
            format="%Y%m%d",
        )
        _, desc_ny, desc_nx = handle["cum"].shape
        desc_latitude = (
            float(handle["corner_lat"][()])
            + np.arange(desc_ny) * float(handle["post_lat"][()])
        )
        desc_longitude = (
            float(handle["corner_lon"][()])
            + np.arange(desc_nx) * float(handle["post_lon"][()])
        )
        desc_look_e = np.asarray(handle["E.geo"][:], dtype=float)
        desc_look_n = np.asarray(handle["N.geo"][:], dtype=float)
        desc_look_u = np.asarray(handle["U.geo"][:], dtype=float)
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
    desc_look_e, desc_look_n, desc_look_u = normalize_look_vectors(
        desc_look_e,
        desc_look_n,
        desc_look_u,
    )
    desc_quality &= (
        np.isfinite(desc_look_e)
        & np.isfinite(desc_look_n)
        & np.isfinite(desc_look_u)
    )

    common_dates = pd.read_csv(
        COMMON_DATES_FILE,
        parse_dates=["date"],
    )["date"].to_list()
    asc_date_index = {
        pd.Timestamp(date): index
        for index, date in enumerate(ascending.dates)
    }
    desc_date_index = {
        pd.Timestamp(date): index
        for index, date in enumerate(desc_dates)
    }
    if any(
        date not in asc_date_index or date not in desc_date_index
        for date in common_dates
    ):
        raise RuntimeError("Common-date manifest does not match source stacks")

    east_grid, north_grid, latitude_grid, longitude_grid, overlap = (
        common_utm11_grid(
            [asc_latitude, desc_latitude],
            [asc_longitude, desc_longitude],
            spacing_km=COMMON_GRID_SPACING_KM,
        )
    )
    targets_xy = np.column_stack(
        [east_grid.ravel(), north_grid.ravel()]
    )
    histories, network = load_gnss_network(GNSS_ROOT)
    network_xy = to_utm11_km(
        network["longitude"].to_numpy(),
        network["latitude"].to_numpy(),
    )
    network_hull = (
        Delaunay(network_xy).find_simplex(targets_xy) >= 0
    ).reshape(east_grid.shape)

    def resample_static(
        latitude: np.ndarray,
        longitude: np.ndarray,
        fields: dict[str, np.ndarray],
        valid: np.ndarray,
    ) -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        support: list[np.ndarray] = []
        for name, values in fields.items():
            sampled, fraction = masked_bilinear_resample(
                latitude,
                longitude,
                values,
                valid,
                latitude_grid,
                longitude_grid,
            )
            output[name] = sampled
            support.append(fraction)
        output["valid"] = (
            overlap
            & network_hull
            & (np.minimum.reduce(support) >= 0.999)
        )
        output["E"], output["N"], output["U"] = normalize_look_vectors(
            output["E"],
            output["N"],
            output["U"],
        )
        for name in ("E", "N", "U"):
            output[name][~output["valid"]] = np.nan
        return output

    asc_static = resample_static(
        asc_latitude,
        asc_longitude,
        asc_look_grid,
        asc_geometry_valid,
    )
    desc_static = resample_static(
        desc_latitude,
        desc_longitude,
        {"E": desc_look_e, "N": desc_look_n, "U": desc_look_u},
        desc_quality,
    )

    reference_station = (
        network.set_index("station").loc[REFERENCE_STATION]
    )
    asc_reference_mask = (
        haversine_km(
            ascending.latitude,
            ascending.longitude,
            float(reference_station["latitude"]),
            float(reference_station["longitude"]),
        )
        <= REFERENCE_RADIUS_KM
    ) & np.isfinite(look["los_u"])
    asc_reference_xy = to_utm11_km(
        ascending.longitude[asc_reference_mask],
        ascending.latitude[asc_reference_mask],
    )
    desc_lat_grid, desc_lon_grid = np.meshgrid(
        desc_latitude,
        desc_longitude,
        indexing="ij",
    )
    desc_reference_mask = desc_quality & (
        haversine_km(
            desc_lat_grid,
            desc_lon_grid,
            float(reference_station["latitude"]),
            float(reference_station["longitude"]),
        )
        <= REFERENCE_RADIUS_KM
    )
    desc_reference_xy = to_utm11_km(
        desc_lon_grid[desc_reference_mask],
        desc_lat_grid[desc_reference_mask],
    )
    if int(asc_reference_mask.sum()) < 10 or int(
        desc_reference_mask.sum()
    ) < 25:
        raise RuntimeError("Insufficient native reference pixels")

    # Build direct cumulative GNSS endpoint tables and freeze one station set.
    table_by_track: dict[str, list[pd.DataFrame | None]] = {
        track: [None] for track in TRACK_TIMES
    }
    station_sets: list[set[str]] = []
    t0 = pd.Timestamp(common_dates[0])
    for track, time_offset in TRACK_TIMES.items():
        for date in common_dates[1:]:
            table = gnss_interval_table(
                histories,
                network,
                start=t0 + time_offset,
                end=pd.Timestamp(date) + time_offset,
                event_times=EVENTS,
                strict=False,
            )
            table_by_track[track].append(table)
            station_sets.append(set(table["station"].astype(str)))
    persistent_stations = sorted(set.intersection(*station_sets))
    if len(persistent_stations) < 20:
        raise RuntimeError(
            "Too few persistent GNSS stations for cumulative interpolation"
        )
    station_metadata = (
        network.set_index("station")
        .loc[persistent_stations]
        .reset_index()
    )
    station_xy = to_utm11_km(
        station_metadata["longitude"].to_numpy(),
        station_metadata["latitude"].to_numpy(),
    )

    frozen_models: dict[str, LocalVerticalModel] = {}
    frozen_sigma: dict[str, np.ndarray] = {}
    target_topology: dict[str, object] = {}
    reference_topology: dict[str, object] = {}
    for track in TRACK_TIMES:
        pre_tables = [
            table.set_index("station").loc[persistent_stations].reset_index()
            for date, table in zip(
                common_dates[1:],
                table_by_track[track][1:],
            )
            if pd.Timestamp(date) <= CALIBRATION_END
            and table is not None
        ]
        frozen_sigma[track] = np.nanmedian(
            np.stack(
                [
                    table["sigma_up_mm"].to_numpy(float)
                    for table in pre_tables
                ]
            ),
            axis=0,
        )
        sill_values = [
            estimate_interval_sill_mm2(
                table["up_mm"].to_numpy(float),
                table["sigma_up_mm"].to_numpy(float),
            )
            for table in pre_tables
        ]
        frozen_models[track] = replace(
            base_models[track],
            sill_mm2=float(np.nanmedian(sill_values)),
        )
        target_topology[track] = build_local_support_topology(
            station_xy,
            targets_xy,
            frozen_models[track].config,
        )
        reference_xy = (
            asc_reference_xy
            if track == "ascending_T64"
            else desc_reference_xy
        )
        reference_topology[track] = build_local_support_topology(
            station_xy,
            reference_xy,
            frozen_models[track].config,
        )

    # Add the zero-displacement baseline using the same persistent network.
    for track in TRACK_TIMES:
        zero = station_metadata.copy()
        zero["up_mm"] = 0.0
        zero["sigma_up_mm"] = frozen_sigma[track]
        table_by_track[track][0] = zero

    long_station_rows: list[dict[str, object]] = []
    direct_fields: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "ascending_vertical_cumulative_mm",
            "descending_vertical_cumulative_mm",
            "ascending_vertical_sigma_cumulative_mm",
            "descending_vertical_sigma_cumulative_mm",
            "ascending_observed_los_cumulative_mm",
            "descending_observed_los_cumulative_mm",
            "ascending_vertical_to_los_cumulative_mm",
            "descending_vertical_to_los_cumulative_mm",
            "ascending_vertical_to_los_sigma_cumulative_mm",
            "descending_vertical_to_los_sigma_cumulative_mm",
            "ascending_pure_horizontal_los_cumulative_mm",
            "descending_pure_horizontal_los_cumulative_mm",
            "cumulative_east_mm",
            "cumulative_north_mm",
            "cumulative_sigma_east_mm",
            "cumulative_sigma_north_mm",
            "cumulative_covariance_east_north_mm2",
            "valid_epoch",
        )
    }

    def ordered_table(track: str, epoch_index: int) -> pd.DataFrame:
        table = table_by_track[track][epoch_index]
        if table is None:
            raise RuntimeError("Missing cumulative GNSS table")
        return (
            table.set_index("station")
            .loc[persistent_stations]
            .reset_index()
        )

    def cumulative_observed_ascending(
        epoch_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        native = (
            ascending.displacement[asc_date_index[common_dates[epoch_index]]]
            - ascending.displacement[asc_date_index[common_dates[0]]]
        )
        reference_valid = asc_reference_mask & np.isfinite(native)
        referenced = native - float(np.nanmedian(native[reference_valid]))
        grid = text_vector_to_grid(referenced)
        return masked_bilinear_resample(
            asc_latitude,
            asc_longitude,
            grid,
            np.isfinite(grid) & asc_geometry_valid,
            latitude_grid,
            longitude_grid,
        )

    def cumulative_observed_descending(
        handle: h5py.File,
        epoch_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        native = (
            np.asarray(
                handle["cum"][desc_date_index[common_dates[epoch_index]]],
                dtype=float,
            )
            - np.asarray(
                handle["cum"][desc_date_index[common_dates[0]]],
                dtype=float,
            )
        )
        reference_valid = desc_reference_mask & np.isfinite(native)
        referenced = native - float(np.nanmedian(native[reference_valid]))
        return masked_bilinear_resample(
            desc_latitude,
            desc_longitude,
            referenced,
            desc_quality & np.isfinite(referenced),
            latitude_grid,
            longitude_grid,
        )

    def predict_cumulative_vertical(
        track: str,
        epoch_index: int,
    ) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
        table = ordered_table(track, epoch_index)
        values = table["up_mm"].to_numpy(float)
        prediction = predict_local_vertical_from_topology(
            frozen_models[track],
            station_xy,
            values,
            frozen_sigma[track],
            target_topology[track],
        )
        reference_prediction = predict_local_vertical_from_topology(
            frozen_models[track],
            station_xy,
            values,
            frozen_sigma[track],
            reference_topology[track],
        )
        reference_u = (
            look["los_u"][asc_reference_mask]
            if track == "ascending_T64"
            else desc_look_u[desc_reference_mask]
        )
        valid_reference = (
            reference_prediction.valid & np.isfinite(reference_u)
        )
        reference_value = float(
            np.nanmedian(
                reference_u[valid_reference]
                * reference_prediction.mean_mm[valid_reference]
            )
        )
        reference_sigma = float(
            np.nanmedian(
                np.abs(reference_u[valid_reference])
                * reference_prediction.sigma_mm[valid_reference]
            )
        )
        return (
            prediction.mean_mm.reshape(east_grid.shape),
            prediction.sigma_mm.reshape(east_grid.shape),
            reference_value,
            reference_sigma,
            prediction.valid.reshape(east_grid.shape),
        )

    with h5py.File(DESC_H5, "r") as handle:
        for epoch_index, date in enumerate(common_dates):
            asc_observed, asc_support = cumulative_observed_ascending(
                epoch_index
            )
            desc_observed, desc_support = cumulative_observed_descending(
                handle,
                epoch_index,
            )
            (
                asc_u,
                asc_u_sigma,
                asc_reference,
                asc_reference_sigma,
                asc_u_valid,
            ) = predict_cumulative_vertical("ascending_T64", epoch_index)
            (
                desc_u,
                desc_u_sigma,
                desc_reference,
                desc_reference_sigma,
                desc_u_valid,
            ) = predict_cumulative_vertical("descending_T71", epoch_index)

            asc_valid = (
                asc_static["valid"]
                & (asc_support >= 0.999)
                & asc_u_valid
            )
            desc_valid = (
                desc_static["valid"]
                & (desc_support >= 0.999)
                & desc_u_valid
            )
            asc_sign = int(
                sign_manifest["selected_insar_sign"]["ascending_T64"]
            )
            desc_sign = int(
                sign_manifest["selected_insar_sign"]["descending_T71"]
            )
            asc_signed = asc_sign * asc_observed
            desc_signed = desc_sign * desc_observed
            (
                asc_hlos,
                asc_vlos,
                asc_vlos_sigma,
                _,
                _,
            ) = correct_vertical_los_on_grid(
                asc_signed,
                asc_static["U"],
                asc_u,
                asc_u_sigma,
                reference_value_mm=asc_reference,
                reference_sigma_mm=asc_reference_sigma,
            )
            (
                desc_hlos,
                desc_vlos,
                desc_vlos_sigma,
                _,
                _,
            ) = correct_vertical_los_on_grid(
                desc_signed,
                desc_static["U"],
                desc_u,
                desc_u_sigma,
                reference_value_mm=desc_reference,
                reference_sigma_mm=desc_reference_sigma,
            )
            for array in (
                asc_signed,
                asc_hlos,
                asc_vlos,
                asc_vlos_sigma,
                asc_u,
                asc_u_sigma,
            ):
                array[~asc_valid] = np.nan
            for array in (
                desc_signed,
                desc_hlos,
                desc_vlos,
                desc_vlos_sigma,
                desc_u,
                desc_u_sigma,
            ):
                array[~desc_valid] = np.nan

            solution = solve_two_track_horizontal(
                asc_hlos,
                desc_hlos,
                asc_static["E"],
                asc_static["N"],
                desc_static["E"],
                desc_static["N"],
                np.full(east_grid.shape, LOS_SIGMA_ASC_MM),
                np.full(east_grid.shape, LOS_SIGMA_DESC_MM),
                vertical_los_sigma_ascending_mm=asc_vlos_sigma,
                vertical_los_sigma_descending_mm=desc_vlos_sigma,
                vertical_correlation=1.0,
                max_condition_number=8.0,
            )

            values_by_name = {
                "ascending_vertical_cumulative_mm": asc_u,
                "descending_vertical_cumulative_mm": desc_u,
                "ascending_vertical_sigma_cumulative_mm": asc_u_sigma,
                "descending_vertical_sigma_cumulative_mm": desc_u_sigma,
                "ascending_observed_los_cumulative_mm": asc_signed,
                "descending_observed_los_cumulative_mm": desc_signed,
                "ascending_vertical_to_los_cumulative_mm": asc_vlos,
                "descending_vertical_to_los_cumulative_mm": desc_vlos,
                "ascending_vertical_to_los_sigma_cumulative_mm": asc_vlos_sigma,
                "descending_vertical_to_los_sigma_cumulative_mm": desc_vlos_sigma,
                "ascending_pure_horizontal_los_cumulative_mm": asc_hlos,
                "descending_pure_horizontal_los_cumulative_mm": desc_hlos,
                "cumulative_east_mm": solution.east_mm,
                "cumulative_north_mm": solution.north_mm,
                "cumulative_sigma_east_mm": solution.sigma_east_mm,
                "cumulative_sigma_north_mm": solution.sigma_north_mm,
                "cumulative_covariance_east_north_mm2": (
                    solution.covariance_east_north_mm2
                ),
                "valid_epoch": solution.valid,
            }
            for name, values in values_by_name.items():
                dtype = bool if name == "valid_epoch" else np.float32
                direct_fields[name].append(
                    np.asarray(values, dtype=dtype)
                )

            for track in TRACK_TIMES:
                table = ordered_table(track, epoch_index)
                nominal_lu = math.cos(
                    math.radians(NOMINAL_INCIDENCE_DEG[track])
                )
                for row in table.itertuples(index=False):
                    long_station_rows.append(
                        {
                            "track": track,
                            "date": pd.Timestamp(date),
                            "station": str(row.station),
                            "cumulative_up_mm": float(row.up_mm),
                            "sigma_up_mm": float(row.sigma_up_mm),
                            "nominal_vertical_to_los_mm": (
                                nominal_lu * float(row.up_mm)
                            ),
                        }
                    )

            if epoch_index % 10 == 0 or epoch_index == len(common_dates) - 1:
                print(
                    "Direct cumulative epoch",
                    f"{epoch_index + 1}/{len(common_dates)}:",
                    pd.Timestamp(date).date(),
                    "valid E-N cells",
                    int(solution.valid.sum()),
                )

    rupture_segments = load_rupture_segments_utm(
        FAULT_FILE,
        certain_only=True,
    )
    distance = rupture_point_distance_lower_bound_km(
        targets_xy,
        rupture_segments,
    ).reshape(east_grid.shape)
    arrays = {
        name: np.stack(values)
        for name, values in direct_fields.items()
    }

    # Exact algebraic gates for the cumulative operation order.
    for prefix in ("ascending", "descending"):
        observed = arrays[f"{prefix}_observed_los_cumulative_mm"]
        vertical = arrays[
            f"{prefix}_vertical_to_los_cumulative_mm"
        ]
        horizontal = arrays[
            f"{prefix}_pure_horizontal_los_cumulative_mm"
        ]
        finite = (
            np.isfinite(observed)
            & np.isfinite(vertical)
            & np.isfinite(horizontal)
        )
        error = float(
            np.nanmax(
                np.abs(
                    horizontal[finite]
                    - (observed[finite] - vertical[finite])
                )
            )
        )
        if error > 1.0e-4:
            raise RuntimeError(
                f"{prefix} cumulative subtraction identity failed: {error:g}"
            )
        print(prefix, "maximum cumulative subtraction error (mm):", error)

    np.savez_compressed(
        DIRECT_NPZ,
        dates=np.asarray(common_dates, dtype="datetime64[ns]"),
        east_km=east_grid,
        north_km=north_grid,
        latitude=latitude_grid,
        longitude=longitude_grid,
        distance_to_mapped_rupture_km=distance.astype(np.float32),
        off_fault_validation_mask=(
            distance > STRAIN_SAFE_DISTANCE_KM
        ),
        ascending_look_e=asc_static["E"].astype(np.float32),
        ascending_look_n=asc_static["N"].astype(np.float32),
        ascending_look_u=asc_static["U"].astype(np.float32),
        descending_look_e=desc_static["E"].astype(np.float32),
        descending_look_n=desc_static["N"].astype(np.float32),
        descending_look_u=desc_static["U"].astype(np.float32),
        persistent_gnss_stations=np.asarray(
            persistent_stations,
            dtype="U8",
        ),
        **arrays,
    )
    pd.DataFrame(long_station_rows).to_csv(
        OUTPUT_DIR / "cumulative_gnss_vertical_by_station.csv",
        index=False,
    )

    interpolation_rows = []
    for track in TRACK_TIMES:
        interpolation_rows.append(
            {
                "track": track,
                "family": frozen_models[track].family,
                "length_scale_km": (
                    frozen_models[track].length_scale_km
                ),
                "frozen_pre_event_sill_mm2": (
                    frozen_models[track].sill_mm2
                ),
                "persistent_station_count": len(persistent_stations),
                "weights_frozen": True,
                "calibration_end": CALIBRATION_END,
            }
        )
    pd.DataFrame(interpolation_rows).to_csv(
        OUTPUT_DIR / "frozen_cumulative_vertical_interpolator.csv",
        index=False,
    )


if FORCE_REBUILD_DIRECT or not DIRECT_NPZ.exists():
    build_direct_cumulative_product()
else:
    print("Loading existing direct cumulative product:", DIRECT_NPZ)

direct = np.load(DIRECT_NPZ)
dates = pd.DatetimeIndex(pd.to_datetime(direct["dates"]))
east_grid = direct["east_km"]
north_grid = direct["north_km"]
latitude_grid = direct["latitude"]
longitude_grid = direct["longitude"]
distance_grid = direct["distance_to_mapped_rupture_km"]
valid_epoch = np.asarray(direct["valid_epoch"], dtype=bool)
print(
    "Direct cumulative cube:",
    len(dates),
    "epochs;",
    direct["cumulative_east_mm"].shape,
)

# %% [markdown]
# ## 3. One fixed dense joint-GLS derivative operator
#
# Cumulative \(E\) and \(N\) are sampled on a persistent 2-km lattice.  A joint
# local affine generalized least-squares model is evaluated at every supported
# 1-km cell.  Neighbours, Gaussian distance weights, E–N covariance weights,
# and the influence matrix are precomputed once and reused for all 80 epochs.
# This prevents changing derivative geometry from appearing as temporal strain.
#
# The result is a dense masked raster over the validated off-fault domain.  No
# strain dots are interpolated afterward, and no values are invented inside the
# rupture-adjacent exclusion.

# %%
def build_dense_cumulative_strain() -> None:
    cumulative_east = np.asarray(
        direct["cumulative_east_mm"],
        dtype=float,
    )
    cumulative_north = np.asarray(
        direct["cumulative_north_mm"],
        dtype=float,
    )
    sigma_east = np.asarray(
        direct["cumulative_sigma_east_mm"],
        dtype=float,
    )
    sigma_north = np.asarray(
        direct["cumulative_sigma_north_mm"],
        dtype=float,
    )
    covariance_en = np.asarray(
        direct["cumulative_covariance_east_north_mm2"],
        dtype=float,
    )
    persistent = (
        np.all(np.isfinite(cumulative_east), axis=0)
        & np.all(np.isfinite(cumulative_north), axis=0)
        & np.all(valid_epoch, axis=0)
    )
    sample_lattice = np.zeros(east_grid.shape, dtype=bool)
    sample_lattice[::2, ::2] = True
    sample_mask = (
        persistent
        & sample_lattice
        & (distance_grid > RUPTURE_BUFFER_KM)
    )
    target_mask = (
        persistent
        & (distance_grid > STRAIN_SAFE_DISTANCE_KM)
    )
    sample_xy = np.column_stack(
        [east_grid[sample_mask], north_grid[sample_mask]]
    )
    target_row, target_column = np.nonzero(target_mask)
    target_xy = np.column_stack(
        [east_grid[target_mask], north_grid[target_mask]]
    )

    calibration_epoch = dates <= CALIBRATION_END
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered")
        variance_east = np.nanmedian(
            np.square(sigma_east[calibration_epoch]),
            axis=0,
        )
        variance_north = np.nanmedian(
            np.square(sigma_north[calibration_epoch]),
            axis=0,
        )
        covariance_fixed = np.nanmedian(
            covariance_en[calibration_epoch],
            axis=0,
        )
    sample_covariance = np.zeros(
        (int(sample_mask.sum()), 2, 2),
        dtype=float,
    )
    sample_covariance[:, 0, 0] = variance_east[sample_mask]
    sample_covariance[:, 1, 1] = variance_north[sample_mask]
    sample_covariance[:, 0, 1] = covariance_fixed[sample_mask]
    sample_covariance[:, 1, 0] = covariance_fixed[sample_mask]

    model = build_fixed_joint_mls(
        sample_xy,
        target_xy,
        covariance_en_mm2=sample_covariance,
        support_radius_km=STRAIN_SUPPORT_RADIUS_KM,
        bandwidth_km=STRAIN_BANDWIDTH_KM,
        min_samples=MIN_STRAIN_SAMPLES,
        max_condition_number=1.0e8,
        covariance_absolute_floor_mm2=1.0,
    )
    values = evaluate_fixed_joint_mls(
        model,
        cumulative_east[:, sample_mask],
        cumulative_north[:, sample_mask],
    )
    sigma = fixed_joint_mls_component_sigma(model)

    grid_values = {
        name: target_values_to_grid(
            component,
            target_row,
            target_column,
            east_grid.shape,
        ).astype(np.float32)
        for name, component in values.items()
    }
    grid_sigma = {
        f"sigma_{name}": target_values_to_grid(
            component,
            target_row,
            target_column,
            east_grid.shape,
        ).astype(np.float32)
        for name, component in sigma.items()
    }
    supported_target_mask = np.zeros(east_grid.shape, dtype=bool)
    supported_target_mask[target_row, target_column] = model.valid
    sample_count_grid = target_values_to_grid(
        model.sample_count,
        target_row,
        target_column,
        east_grid.shape,
    )
    effective_count_grid = target_values_to_grid(
        model.effective_sample_count,
        target_row,
        target_column,
        east_grid.shape,
    )
    condition_grid = target_values_to_grid(
        model.condition_number,
        target_row,
        target_column,
        east_grid.shape,
    )

    np.savez_compressed(
        STRAIN_NPZ,
        dates=np.asarray(dates, dtype="datetime64[ns]"),
        east_km=east_grid,
        north_km=north_grid,
        latitude=latitude_grid,
        longitude=longitude_grid,
        cumulative_east_mm=cumulative_east.astype(np.float32),
        cumulative_north_mm=cumulative_north.astype(np.float32),
        persistent_displacement_mask=persistent,
        displacement_sample_mask=sample_mask,
        cumulative_strain_target_mask=supported_target_mask,
        distance_to_mapped_rupture_km=distance_grid.astype(np.float32),
        local_sample_count=sample_count_grid.astype(np.float32),
        local_effective_sample_count=effective_count_grid.astype(
            np.float32
        ),
        local_normal_condition_number=condition_grid.astype(np.float32),
        **grid_values,
        **grid_sigma,
    )
    diagnostics = pd.DataFrame(
        {
            "east_km": target_xy[:, 0],
            "north_km": target_xy[:, 1],
            "valid": model.valid,
            "sample_count": model.sample_count,
            "effective_sample_count": model.effective_sample_count,
            "normal_condition_number": model.condition_number,
        }
    )
    diagnostics.to_csv(
        OUTPUT_DIR / "dense_cumulative_strain_operator_diagnostics.csv",
        index=False,
    )
    print(
        "Persistent displacement cells:",
        int(persistent.sum()),
        "2-km samples:",
        int(sample_mask.sum()),
        "dense supported 1-km strain cells:",
        int(model.valid.sum()),
        "/",
        len(model.valid),
    )


if FORCE_REBUILD_STRAIN or not STRAIN_NPZ.exists():
    build_dense_cumulative_strain()
else:
    print("Loading existing dense cumulative strain:", STRAIN_NPZ)

strain = np.load(STRAIN_NPZ)
strain_mask = np.asarray(
    strain["cumulative_strain_target_mask"],
    dtype=bool,
)
print(
    "Dense cumulative strain support:",
    int(strain_mask.sum()),
    "1-km cells",
)

# %% [markdown]
# ## 4. Cumulative station and algebraic audits
#
# The first plot makes the cumulative quantity explicit at P595 and CCCC.
# Vertical GNSS displacement is shown relative to 27 May 2017 and projected
# into both nominal track LOS directions.  The pixel correction itself uses
# the saved local pixel-specific \(l_U\), not these nominal angles.

# %%
station_vertical = pd.read_csv(
    OUTPUT_DIR / "cumulative_gnss_vertical_by_station.csv",
    parse_dates=["date"],
)
audit_stations = [
    station
    for station in ("P595", "CCCC")
    if station in set(station_vertical["station"])
]
fig, axes = plt.subplots(
    len(audit_stations),
    1,
    figsize=(13.5, 4.2 * len(audit_stations)),
    sharex=True,
    constrained_layout=True,
)
if len(audit_stations) == 1:
    axes = [axes]
for axis, station_name in zip(axes, audit_stations):
    station_rows = station_vertical.loc[
        station_vertical["station"] == station_name
    ]
    for track, color, linestyle in (
        ("ascending_T64", "#1f77b4", "-"),
        ("descending_T71", "#d62728", "--"),
    ):
        rows = station_rows.loc[
            station_rows["track"] == track
        ].sort_values("date")
        axis.plot(
            rows["date"],
            rows["cumulative_up_mm"],
            color="0.25",
            lw=1.6,
            alpha=0.75,
            label=(
                "GNSS cumulative U"
                if track == "ascending_T64"
                else None
            ),
        )
        axis.plot(
            rows["date"],
            rows["nominal_vertical_to_los_mm"],
            color=color,
            ls=linestyle,
            lw=2.0,
            label=track.replace("_", " "),
        )
    axis.axvline(
        pd.Timestamp("2019-07-04"),
        color="0.15",
        ls=":",
        lw=1.3,
    )
    axis.set_title(f"({chr(97 + audit_stations.index(station_name))}) {station_name}")
    axis.set_ylabel("Cumulative displacement (mm)")
    axis.grid(True, color="0.85", ls="--", lw=0.6)
    axis.legend(ncol=3, loc="upper left")
axes[-1].set_xlabel("Date")
fig.suptitle(
    "GNSS cumulative vertical displacement and its vertical-to-LOS projection",
    fontsize=17,
    fontweight="semibold",
)
fig.savefig(
    OUTPUT_DIR / "01_cumulative_gnss_vertical_projection_timeseries.png",
    bbox_inches="tight",
)
fig.savefig(
    OUTPUT_DIR / "01_cumulative_gnss_vertical_projection_timeseries.pdf",
    bbox_inches="tight",
)
plt.close(fig)

# Compare the corrected cumulative-first result with the previous
# interval-first accumulation to quantify why the operation order matters.
comparison_rows: list[dict[str, object]] = []
if OLD_PRODUCT.exists():
    old = np.load(OLD_PRODUCT)
    if np.array_equal(old["dates"], direct["dates"]):
        for date in (
            pd.Timestamp("2019-07-04"),
            pd.Timestamp("2019-07-16"),
        ):
            index = int(np.flatnonzero(dates == date)[0])
            for name in (
                "ascending_vertical_to_los_cumulative_mm",
                "descending_vertical_to_los_cumulative_mm",
                "cumulative_east_mm",
                "cumulative_north_mm",
            ):
                current = np.asarray(direct[name][index], dtype=float)
                previous = np.asarray(old[name][index], dtype=float)
                finite = np.isfinite(current) & np.isfinite(previous)
                difference = current[finite] - previous[finite]
                comparison_rows.append(
                    {
                        "date": date,
                        "field": name,
                        "common_cells": int(finite.sum()),
                        "median_difference_mm": float(
                            np.nanmedian(difference)
                        ),
                        "median_absolute_difference_mm": float(
                            np.nanmedian(np.abs(difference))
                        ),
                        "p95_absolute_difference_mm": float(
                            np.nanpercentile(np.abs(difference), 95)
                        ),
                    }
                )
pd.DataFrame(comparison_rows).to_csv(
    OUTPUT_DIR / "cumulative_first_vs_interval_first_audit.csv",
    index=False,
)

# %% [markdown]
# ## 5. Continuous cumulative LOS and strain maps
#
# Every panel below is a cumulative map relative to 27 May 2017.  Maps are
# rendered directly from dense grid cells with `pcolormesh`; the white central
# region is the explicit unsupported rupture-adjacent domain, not an
# interpolation gap.

# %%
fault_geojson = json.loads(FAULT_FILE.read_text(encoding="utf-8"))
fault_lines: list[np.ndarray] = []
for feature in fault_geojson["features"]:
    if (
        feature["geometry"]["type"] != "LineString"
        or feature["properties"].get("ExistenceConfidence") != "certain"
        or feature["properties"].get("IdentityConfidence") != "certain"
    ):
        continue
    coordinates = np.asarray(
        feature["geometry"]["coordinates"],
        dtype=float,
    )
    if (
        coordinates.ndim == 2
        and coordinates.shape[0] >= 2
        and coordinates.shape[1] >= 2
    ):
        fault_lines.append(coordinates[:, :2])
fault_points = np.concatenate(fault_lines, axis=0)[::5]


def add_faults(axis: plt.Axes, *, linewidth: float = 0.75) -> None:
    axis.scatter(
        fault_points[:, 0],
        fault_points[:, 1],
        s=max(0.25, 0.55 * linewidth),
        marker=".",
        c="0.08",
        linewidths=0.0,
        alpha=0.90,
        zorder=4,
        rasterized=True,
    )


def date_index(date: str | pd.Timestamp) -> int:
    target = pd.Timestamp(date)
    match = np.flatnonzero(dates == target)
    if len(match) != 1:
        raise KeyError(f"{target.date()} is not a common epoch")
    return int(match[0])


def symmetric_limit(values: np.ndarray, percentile: float = 98.5) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    return max(float(np.nanpercentile(np.abs(finite), percentile)), 1.0e-6)


def plot_cumulative_los_decomposition(
    requested_date: str | pd.Timestamp,
) -> None:
    index = date_index(requested_date)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(17.5, 10.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for row, (prefix, label) in enumerate(
        (
            ("ascending", "Track 64 ascending"),
            ("descending", "Track 71 descending"),
        )
    ):
        observed = direct[
            f"{prefix}_observed_los_cumulative_mm"
        ][index]
        vertical = direct[
            f"{prefix}_vertical_to_los_cumulative_mm"
        ][index]
        horizontal = direct[
            f"{prefix}_pure_horizontal_los_cumulative_mm"
        ][index]
        los_limit = symmetric_limit(
            np.r_[observed.ravel(), horizontal.ravel()]
        )
        vertical_limit = symmetric_limit(vertical)
        los_norm = TwoSlopeNorm(
            vmin=-los_limit,
            vcenter=0.0,
            vmax=los_limit,
        )
        vertical_norm = TwoSlopeNorm(
            vmin=-vertical_limit,
            vcenter=0.0,
            vmax=vertical_limit,
        )
        observed_image = axes[row, 0].pcolormesh(
            longitude_grid,
            latitude_grid,
            observed,
            cmap="RdBu_r",
            norm=los_norm,
            shading="auto",
            rasterized=True,
        )
        vertical_image = axes[row, 1].pcolormesh(
            longitude_grid,
            latitude_grid,
            vertical,
            cmap="RdBu_r",
            norm=vertical_norm,
            shading="auto",
            rasterized=True,
        )
        axes[row, 2].pcolormesh(
            longitude_grid,
            latitude_grid,
            horizontal,
            cmap="RdBu_r",
            norm=los_norm,
            shading="auto",
            rasterized=True,
        )
        axes[row, 0].set_ylabel(f"{label}\nLatitude")
        for axis in axes[row]:
            add_faults(axis)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(True, color="0.82", ls=":", lw=0.55)
        fig.colorbar(
            observed_image,
            ax=[axes[row, 0], axes[row, 2]],
            label="Cumulative LOS displacement (mm)",
            shrink=0.84,
            pad=0.02,
        )
        fig.colorbar(
            vertical_image,
            ax=axes[row, 1],
            label="Cumulative vertical-to-LOS (mm)",
            shrink=0.84,
            pad=0.02,
        )
    for axis, title in zip(
        axes[0],
        (
            r"Observed cumulative $D_{\rm LOS}$",
            r"Cumulative $l_U\widehat U$",
            r"Cumulative horizontal LOS",
        ),
    ):
        axis.set_title(title)
    for axis in axes[-1]:
        axis.set_xlabel("Longitude")
    stamp = pd.Timestamp(requested_date).strftime("%Y%m%d")
    fig.suptitle(
        f"Direct cumulative vertical removal to "
        f"{pd.Timestamp(requested_date):%d %B %Y}",
        fontsize=18,
        fontweight="semibold",
    )
    fig.savefig(
        OUTPUT_DIR / f"02_cumulative_los_decomposition_{stamp}.png",
        bbox_inches="tight",
    )
    if SAVE_MAP_PDF:
        fig.savefig(
            OUTPUT_DIR / f"02_cumulative_los_decomposition_{stamp}.pdf",
            bbox_inches="tight",
        )
    plt.close(fig)


for requested_date in ("2019-07-04", "2019-07-16"):
    plot_cumulative_los_decomposition(requested_date)

# %%
component_specs = (
    (
        "epsilon_EE_microstrain",
        r"$\epsilon_{EE}$",
        "Cumulative normal E (µstrain)",
    ),
    (
        "epsilon_NN_microstrain",
        r"$\epsilon_{NN}$",
        "Cumulative normal N (µstrain)",
    ),
    (
        "gamma_EN_microstrain",
        r"$\gamma_{EN}$",
        "Cumulative engineering shear (µstrain)",
    ),
    (
        "dilatation_microstrain",
        r"$\delta$",
        "Cumulative dilatation (µstrain)",
    ),
    (
        "rotation_microradian",
        r"$\omega$",
        "Cumulative rotation (µrad)",
    ),
)

regional_rows: list[dict[str, object]] = []
for index, date in enumerate(dates):
    for name, symbol, label in component_specs:
        values = np.asarray(strain[name][index], dtype=float)
        finite = strain_mask & np.isfinite(values)
        regional_rows.append(
            {
                "date": date,
                "component": name,
                "symbol": symbol,
                "label": label,
                "supported_cells": int(finite.sum()),
                "spatial_median": float(np.nanmedian(values[finite])),
                "spatial_q25": float(np.nanquantile(values[finite], 0.25)),
                "spatial_q75": float(np.nanquantile(values[finite], 0.75)),
            }
        )
regional = pd.DataFrame(regional_rows)
regional.to_csv(
    OUTPUT_DIR / "cumulative_strain_regional_timeseries.csv",
    index=False,
)

fig, axes = plt.subplots(
    5,
    1,
    figsize=(14.5, 16.8),
    sharex=True,
    constrained_layout=True,
)
for panel, (axis, (name, _, label)) in enumerate(
    zip(axes, component_specs)
):
    rows = regional.loc[regional["component"] == name]
    axis.fill_between(
        rows["date"],
        rows["spatial_q25"],
        rows["spatial_q75"],
        color="#9ecae1",
        alpha=0.45,
        label="Spatial IQR" if panel == 0 else None,
    )
    axis.plot(
        rows["date"],
        rows["spatial_median"],
        color="#08519c",
        lw=2.0,
        label="Spatial median" if panel == 0 else None,
    )
    axis.axhline(0.0, color="0.35", lw=0.8)
    axis.axvspan(
        pd.Timestamp("2019-05-29"),
        pd.Timestamp("2019-07-04"),
        color="#fee391",
        alpha=0.30,
    )
    axis.axvspan(
        pd.Timestamp("2019-07-04"),
        pd.Timestamp("2019-07-16"),
        color="#fcae91",
        alpha=0.28,
    )
    axis.axvline(
        pd.Timestamp("2019-07-04"),
        color="0.20",
        ls="--",
        lw=1.0,
    )
    axis.set_title(f"({chr(97 + panel)}) {label}")
    axis.set_ylabel("µrad" if "rotation" in name else "µstrain")
    axis.grid(True, color="0.85", ls="--", lw=0.55)
axes[0].legend(ncol=2, loc="upper left")
axes[-1].set_xlabel("Date")
fig.suptitle(
    "Off-fault cumulative 2-D horizontal strain relative to 27 May 2017",
    fontsize=18,
    fontweight="semibold",
)
fig.savefig(
    OUTPUT_DIR / "03_cumulative_strain_component_timeseries.png",
    bbox_inches="tight",
)
fig.savefig(
    OUTPUT_DIR / "03_cumulative_strain_component_timeseries.pdf",
    bbox_inches="tight",
)
plt.close(fig)

# %%
key_dates = pd.DatetimeIndex(
    [
        "2019-05-29",
        "2019-06-10",
        "2019-06-22",
        "2019-07-04",
        "2019-07-16",
    ]
)


def plot_cumulative_component_key_dates(
    component: str,
    symbol: str,
    label: str,
) -> None:
    indices = [date_index(date) for date in key_dates]
    selected = np.asarray(strain[component][indices], dtype=float)
    limit = symmetric_limit(
        np.where(strain_mask[None, :, :], selected, np.nan),
        percentile=98.0,
    )
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, axes = plt.subplots(
        1,
        len(indices),
        figsize=(22.5, 5.25),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for column, (axis, index, date) in enumerate(
        zip(axes, indices, key_dates)
    ):
        image = axis.pcolormesh(
            longitude_grid,
            latitude_grid,
            strain[component][index],
            cmap="RdBu_r",
            norm=norm,
            shading="auto",
            rasterized=True,
        )
        axis.contour(
            longitude_grid,
            latitude_grid,
            distance_grid,
            levels=[STRAIN_SAFE_DISTANCE_KM],
            colors="0.15",
            linewidths=1.0,
            linestyles="--",
        )
        add_faults(axis, linewidth=0.65)
        axis.set_title(
            f"({chr(97 + column)}) {date:%d %b %Y}",
            fontsize=14,
        )
        axis.set_xlabel("Longitude")
        if column == 0:
            axis.set_ylabel("Latitude")
        else:
            axis.tick_params(labelleft=False)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, color="0.84", ls=":", lw=0.5)
    if image is None:
        raise RuntimeError("No map image was created")
    fig.colorbar(
        image,
        ax=axes,
        label=label,
        location="right",
        shrink=0.88,
        pad=0.015,
    )
    fig.suptitle(
        f"{symbol}: dense cumulative field relative to 27 May 2017",
        fontsize=18,
        fontweight="semibold",
    )
    filename = (
        "04_cumulative_"
        + component.replace("_microstrain", "").replace(
            "_microradian",
            "",
        )
        + "_key_dates"
    )
    fig.savefig(OUTPUT_DIR / f"{filename}.png", bbox_inches="tight")
    if SAVE_MAP_PDF:
        fig.savefig(OUTPUT_DIR / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


for component, symbol, label in component_specs:
    plot_cumulative_component_key_dates(component, symbol, label)


def plot_cumulative_strain_epoch(
    requested_date: str | pd.Timestamp,
) -> plt.Figure:
    """Interactive-notebook helper for any one of the 80 cumulative epochs."""

    index = date_index(requested_date)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15.8, 9.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, (component, symbol, label) in zip(
        axes.ravel(),
        component_specs,
    ):
        values = np.asarray(strain[component][index], dtype=float)
        limit = symmetric_limit(values)
        image = axis.pcolormesh(
            longitude_grid,
            latitude_grid,
            values,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(
                vmin=-limit,
                vcenter=0.0,
                vmax=limit,
            ),
            shading="auto",
            rasterized=True,
        )
        add_faults(axis)
        axis.set_title(symbol)
        axis.set_aspect("equal", adjustable="box")
        fig.colorbar(image, ax=axis, label=label, shrink=0.82)
    axes.ravel()[-1].axis("off")
    for axis in axes[:, 0]:
        axis.set_ylabel("Latitude")
    for axis in axes[-1, :2]:
        axis.set_xlabel("Longitude")
    fig.suptitle(
        f"Cumulative strain to {pd.Timestamp(requested_date):%d %B %Y}",
        fontsize=17,
    )
    return fig

# %% [markdown]
# ## 6. Change detection from cumulative-strain innovations
#
# The physical fields above remain cumulative.  Formal detection is applied to
# exact 12- and 24-day differences of the cumulative fields,
#
# \[
# \Delta_\tau\epsilon(t)=\epsilon_{\rm cum}(t)-
# \epsilon_{\rm cum}(t-\tau),
# \]
#
# because testing cumulative levels directly would integrate noise and
# reference drift.  A 4-km inference lattice avoids treating overlapping 1-km
# moving-window estimates as independent.  Calibration ends on 29 May 2019;
# 29 May–4 July is held out for surveillance, and 4–16 July is an event
# control.

# %%
component_names = [spec[0] for spec in component_specs]
inference_lattice = np.zeros(east_grid.shape, dtype=bool)
inference_lattice[::4, ::4] = True
inference_mask = strain_mask & inference_lattice
inference_east = east_grid[inference_mask]
inference_north = north_grid[inference_mask]
cumulative_components = np.stack(
    [
        np.asarray(strain[name], dtype=float)[:, inference_mask]
        for name in component_names
    ],
    axis=1,
)
component_sigma = np.stack(
    [
        np.asarray(strain[f"sigma_{name}"], dtype=float)[
            inference_mask
        ]
        for name in component_names
    ],
    axis=0,
)

date_lookup = {
    pd.Timestamp(date): index for index, date in enumerate(dates)
}
window_rows: list[dict[str, object]] = []
window_value: list[np.ndarray] = []
window_sigma: list[np.ndarray] = []
for duration_days in (12, 24):
    delta = pd.Timedelta(days=duration_days)
    for end_index, end in enumerate(dates):
        start = pd.Timestamp(end) - delta
        if start not in date_lookup:
            continue
        start_index = date_lookup[start]
        window_rows.append(
            {
                "start_date": start,
                "end_date": pd.Timestamp(end),
                "duration_days": duration_days,
            }
        )
        window_value.append(
            cumulative_components[end_index]
            - cumulative_components[start_index]
        )
        # Conditional endpoint errors. Shared long-wavelength reference terms
        # largely cancel in a difference; sqrt(2) is retained as a conservative
        # independent-endpoint approximation for formal standardization.
        window_sigma.append(np.sqrt(2.0) * component_sigma)

window_table = (
    pd.DataFrame(window_rows)
    .assign(original_index=np.arange(len(window_rows)))
    .sort_values(["end_date", "duration_days"])
    .reset_index(drop=True)
)
order = window_table["original_index"].to_numpy(int)
window_table = window_table.drop(columns="original_index")
window_value_array = np.stack(window_value)[order]
window_sigma_array = np.stack(window_sigma)[order]
window_rate, window_sigma_rate = duration_normalize(
    window_value_array,
    window_sigma_array,
    window_table["duration_days"].to_numpy(float),
)
baseline_window = (
    window_table["end_date"] <= CALIBRATION_END
).to_numpy()
surveillance_window = (
    (window_table["start_date"] >= CALIBRATION_END)
    & (window_table["end_date"] <= PRE_EVENT_END)
).to_numpy()
event_window = (
    (window_table["start_date"] == PRE_EVENT_END)
    & (window_table["end_date"] == EVENT_END)
    & (window_table["duration_days"] == 12)
).to_numpy()
if int(event_window.sum()) != 1:
    raise RuntimeError("Expected one exact 4-16 July event-control window")

baseline_model = fit_robust_baseline(
    window_rate,
    window_sigma_rate,
    baseline_window,
    min_observations=30,
)
window_z = standardized_innovation(
    window_rate,
    window_sigma_rate,
    baseline_model,
)
baseline_loo_z = leave_one_out_baseline_innovations(
    window_rate,
    window_sigma_rate,
    baseline_window,
    min_observations=29,
)

cluster_mass = np.asarray(
    [
        maximum_signed_cluster_mass(
            window_z[index],
            inference_east,
            inference_north,
            component_names,
            threshold=1.96,
            min_cells=4,
        )
        for index in range(len(window_table))
    ],
    dtype=float,
)
baseline_cluster_mass = np.asarray(
    [
        maximum_signed_cluster_mass(
            baseline_loo_z[index],
            inference_east,
            inference_north,
            component_names,
            threshold=1.96,
            min_cells=4,
        )
        for index in np.flatnonzero(baseline_window)
    ],
    dtype=float,
)
surveillance_count = int(surveillance_window.sum())
cluster_null = sliding_block_maximum(
    baseline_cluster_mass,
    surveillance_count,
)
pre_event_cluster_observed = float(
    np.max(cluster_mass[surveillance_window])
)
pre_event_cluster_p = empirical_upper_tail_pvalue(
    cluster_null,
    pre_event_cluster_observed,
)
event_cluster_observed = float(cluster_mass[event_window][0])
event_cluster_p = empirical_upper_tail_pvalue(
    baseline_cluster_mass,
    event_cluster_observed,
)

cluster_summary = window_table.copy()
cluster_summary["maximum_signed_cluster_mass"] = cluster_mass
cluster_summary["baseline"] = baseline_window
cluster_summary["pre_event_surveillance"] = surveillance_window
cluster_summary["event_control"] = event_window
cluster_summary.to_csv(
    OUTPUT_DIR / "cumulative_strain_innovation_cluster_summary.csv",
    index=False,
)

cluster_records: list[dict[str, object]] = []
for index in np.flatnonzero(surveillance_window | event_window):
    for rank, cluster in enumerate(
        signed_spatial_clusters(
            window_z[index],
            inference_east,
            inference_north,
            component_names,
            threshold=1.96,
            min_cells=4,
        ),
        start=1,
    ):
        cluster_records.append(
            {
                "start_date": window_table.loc[index, "start_date"],
                "end_date": window_table.loc[index, "end_date"],
                "duration_days": int(
                    window_table.loc[index, "duration_days"]
                ),
                "rank": rank,
                **cluster.as_dict(),
            }
        )
pd.DataFrame(cluster_records).to_csv(
    OUTPUT_DIR / "cumulative_strain_innovation_clusters.csv",
    index=False,
)

# Method B uses non-overlapping-duration 12-day windows only.
twelve_day = (
    window_table["duration_days"].to_numpy(int) == 12
)
energy = strain_energy(window_z, quantile=0.90)
baseline_energy_12 = energy[baseline_window & twelve_day]
energy_center = float(np.nanmedian(baseline_energy_12))
energy_scale = float(
    1.4826
    * np.nanmedian(
        np.abs(baseline_energy_12 - energy_center)
    )
)
if not math.isfinite(energy_scale) or energy_scale <= 0.0:
    raise RuntimeError("Invalid baseline strain-energy scale")
standardized_energy = (energy - energy_center) / energy_scale
baseline_energy_standardized = standardized_energy[
    baseline_window & twelve_day
]
pre_12 = surveillance_window & twelve_day
event_12 = event_window & twelve_day
pre_cusum = positive_page_cusum(
    standardized_energy[pre_12],
    reference=0.5,
)
event_cusum = positive_page_cusum(
    standardized_energy[event_12],
    reference=0.5,
)
cusum_null = sliding_block_cusum_maxima(
    baseline_energy_standardized,
    max(1, int(pre_12.sum())),
    reference=0.5,
)
pre_cusum_observed = float(
    np.max(pre_cusum) if len(pre_cusum) else 0.0
)
event_cusum_observed = float(
    np.max(event_cusum) if len(event_cusum) else 0.0
)
pre_cusum_p = empirical_upper_tail_pvalue(
    cusum_null,
    pre_cusum_observed,
)
event_cusum_p = empirical_upper_tail_pvalue(
    sliding_block_cusum_maxima(
        baseline_energy_standardized,
        1,
        reference=0.5,
    ),
    event_cusum_observed,
)

temporal_summary = pd.DataFrame(
    [
        {
            "method": "maximum signed spatial-cluster FWER",
            "period": "pre-event surveillance",
            "observed": pre_event_cluster_observed,
            "empirical_p": pre_event_cluster_p,
            "supported_at_0.05": pre_event_cluster_p <= 0.05,
        },
        {
            "method": "maximum signed spatial-cluster FWER",
            "period": "4-16 July event control",
            "observed": event_cluster_observed,
            "empirical_p": event_cluster_p,
            "supported_at_0.05": event_cluster_p <= 0.05,
        },
        {
            "method": "Page CUSUM of 90th-percentile |z|",
            "period": "pre-event surveillance",
            "observed": pre_cusum_observed,
            "empirical_p": pre_cusum_p,
            "supported_at_0.05": pre_cusum_p <= 0.05,
        },
        {
            "method": "Page CUSUM of 90th-percentile |z|",
            "period": "4-16 July event control",
            "observed": event_cusum_observed,
            "empirical_p": event_cusum_p,
            "supported_at_0.05": event_cusum_p <= 0.05,
        },
    ]
)
temporal_summary.to_csv(
    OUTPUT_DIR / "cumulative_strain_change_detection_summary.csv",
    index=False,
)
print(temporal_summary.to_string(index=False))

# %%
fig, axes = plt.subplots(
    2,
    1,
    figsize=(14.0, 8.4),
    sharex=True,
    constrained_layout=True,
)
for duration, marker, color in (
    (12, "o", "#08519c"),
    (24, "s", "#cb181d"),
):
    use = window_table["duration_days"].to_numpy(int) == duration
    axes[0].plot(
        window_table.loc[use, "end_date"],
        cluster_mass[use],
        marker=marker,
        ms=4,
        lw=1.25,
        color=color,
        label=f"{duration}-day innovation",
    )
axes[0].set_ylabel("Maximum cluster mass")
axes[0].set_title(
    "(a) Maximum signed spatial-cluster statistic"
)
axes[0].legend(ncol=2, loc="upper left")
use_12 = twelve_day
axes[1].plot(
    window_table.loc[use_12, "end_date"],
    standardized_energy[use_12],
    color="#6a51a3",
    lw=1.7,
    marker="o",
    ms=4,
)
axes[1].axhline(0.0, color="0.35", lw=0.8)
axes[1].set_ylabel("Standardized 90th-percentile |z|")
axes[1].set_title("(b) Map-level strain-innovation energy")
axes[1].set_xlabel("Window end date")
for axis in axes:
    axis.axvspan(
        CALIBRATION_END,
        PRE_EVENT_END,
        color="#fee391",
        alpha=0.30,
    )
    axis.axvspan(
        PRE_EVENT_END,
        EVENT_END,
        color="#fcae91",
        alpha=0.28,
    )
    axis.axvline(
        PRE_EVENT_END,
        color="0.20",
        ls="--",
        lw=1.0,
    )
    axis.grid(True, color="0.85", ls="--", lw=0.55)
fig.suptitle(
    "Change detection derived from cumulative 2-D strain",
    fontsize=17,
    fontweight="semibold",
)
fig.savefig(
    OUTPUT_DIR / "05_cumulative_strain_change_detection_timeseries.png",
    bbox_inches="tight",
)
fig.savefig(
    OUTPUT_DIR / "05_cumulative_strain_change_detection_timeseries.pdf",
    bbox_inches="tight",
)
plt.close(fig)

# %% [markdown]
# ## 7. Dense innovation maps for visual interpretation
#
# Formal family-wise inference uses the 4-km lattice above.  The maps below
# display the same fixed-operator 12-day innovations on the full supported 1-km
# raster.  They are visualizations of the tested quantity, not extra
# independent tests.

# %%
dense_cumulative_components = np.stack(
    [
        np.asarray(strain[name], dtype=float)
        for name in component_names
    ],
    axis=1,
)
dense_sigma = np.stack(
    [
        np.asarray(strain[f"sigma_{name}"], dtype=float)
        for name in component_names
    ],
    axis=0,
)
dense_window_value: list[np.ndarray] = []
dense_window_sigma: list[np.ndarray] = []
dense_window_rows: list[dict[str, object]] = []
for end in dates:
    start = pd.Timestamp(end) - pd.Timedelta(days=12)
    if start not in date_lookup:
        continue
    dense_window_rows.append(
        {"start_date": start, "end_date": pd.Timestamp(end)}
    )
    dense_window_value.append(
        dense_cumulative_components[date_lookup[pd.Timestamp(end)]]
        - dense_cumulative_components[date_lookup[start]]
    )
    dense_window_sigma.append(np.sqrt(2.0) * dense_sigma)
dense_window_table = pd.DataFrame(dense_window_rows)
dense_value = np.stack(dense_window_value)
dense_sigma_array = np.stack(dense_window_sigma)
dense_rate, dense_sigma_rate = duration_normalize(
    dense_value,
    dense_sigma_array,
    np.full(len(dense_window_table), 12.0),
)
dense_baseline_mask = (
    dense_window_table["end_date"] <= CALIBRATION_END
).to_numpy()
dense_baseline = fit_robust_baseline(
    dense_rate,
    dense_sigma_rate,
    dense_baseline_mask,
    min_observations=30,
)
dense_z = standardized_innovation(
    dense_rate,
    dense_sigma_rate,
    dense_baseline,
)


def plot_dense_innovation(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> None:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    match = np.flatnonzero(
        (dense_window_table["start_date"] == start)
        & (dense_window_table["end_date"] == end)
    )
    if len(match) != 1:
        raise KeyError(f"No exact 12-day window {start.date()}-{end.date()}")
    index = int(match[0])
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15.8, 9.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for axis, (_, symbol, _) , component_index in zip(
        axes.ravel(),
        component_specs,
        range(len(component_specs)),
    ):
        image = axis.pcolormesh(
            longitude_grid,
            latitude_grid,
            dense_z[index, component_index],
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-5.0, vcenter=0.0, vmax=5.0),
            shading="auto",
            rasterized=True,
        )
        axis.contour(
            longitude_grid,
            latitude_grid,
            distance_grid,
            levels=[STRAIN_SAFE_DISTANCE_KM],
            colors="0.15",
            linestyles="--",
            linewidths=1.0,
        )
        add_faults(axis)
        axis.set_title(symbol)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, color="0.84", ls=":", lw=0.5)
    axes.ravel()[-1].axis("off")
    for axis in axes[:, 0]:
        axis.set_ylabel("Latitude")
    for axis in axes[-1, :2]:
        axis.set_xlabel("Longitude")
    if image is None:
        raise RuntimeError("No innovation map was created")
    fig.colorbar(
        image,
        ax=axes.ravel()[:-1],
        label="Standardized 12-day strain innovation z",
        shrink=0.86,
        pad=0.02,
    )
    fig.suptitle(
        f"Dense strain innovation: {start:%d %b}–{end:%d %b %Y}",
        fontsize=18,
        fontweight="semibold",
    )
    stamp = f"{start:%Y%m%d}_{end:%Y%m%d}"
    fig.savefig(
        OUTPUT_DIR / f"06_dense_strain_innovation_{stamp}.png",
        bbox_inches="tight",
    )
    if SAVE_MAP_PDF:
        fig.savefig(
            OUTPUT_DIR / f"06_dense_strain_innovation_{stamp}.pdf",
            bbox_inches="tight",
        )
    plt.close(fig)


plot_dense_innovation("2019-06-22", "2019-07-04")
plot_dense_innovation("2019-07-04", "2019-07-16")

# %% [markdown]
# ## 8. Save provenance and interpretation boundary
#
# A statistically detectable change in a cumulative strain-derived innovation
# is not, by itself, evidence of fault slip or earthquake preparation.  The
# direct interferograms, cross-track replication, GNSS validation, and source
# inversion remain separate evidentiary requirements.

# %%
np.savez_compressed(
    OUTPUT_DIR / "cumulative_strain_change_detection_arrays.npz",
    start_dates=np.asarray(
        window_table["start_date"],
        dtype="datetime64[ns]",
    ),
    end_dates=np.asarray(
        window_table["end_date"],
        dtype="datetime64[ns]",
    ),
    duration_days=window_table["duration_days"].to_numpy(int),
    inference_east_km=inference_east,
    inference_north_km=inference_north,
    component_names=np.asarray(component_names, dtype="U40"),
    strain_increment=window_value_array.astype(np.float32),
    strain_rate_per_day=window_rate.astype(np.float32),
    standardized_innovation=window_z.astype(np.float32),
    maximum_signed_cluster_mass=cluster_mass,
    map_energy=energy,
)

map_manifest_rows = []
for path in sorted(OUTPUT_DIR.glob("*.png")):
    map_manifest_rows.append(
        {
            "file": path.name,
            "format": "PNG",
            "cumulative": (
                "innovation" not in path.name
                and "change_detection" not in path.name
            ),
            "continuous_dense_grid": (
                "strain_component_timeseries" not in path.name
                and "gnss_vertical_projection_timeseries" not in path.name
                and "change_detection_timeseries" not in path.name
            ),
        }
    )
pd.DataFrame(map_manifest_rows).to_csv(
    OUTPUT_DIR / "output_figure_manifest.csv",
    index=False,
)

comparison = pd.DataFrame(comparison_rows)
manifest = {
    "status": (
        "direct cumulative GNSS-U interpolation, cumulative vertical-to-LOS "
        "removal, cumulative E-N solution, and dense cumulative strain complete"
    ),
    "reference_epoch": str(dates[0].date()),
    "last_epoch": str(dates[-1].date()),
    "epoch_count": int(len(dates)),
    "common_grid_shape": list(east_grid.shape),
    "common_grid_spacing_km": COMMON_GRID_SPACING_KM,
    "source_tracks": {
        "ascending": "Track-64 date-named cumulative text stack",
        "descending": "Track-71 cum_full_scene_no_GACOS.h5",
    },
    "track_times_utc": {
        key: str(value) for key, value in TRACK_TIMES.items()
    },
    "vertical_processing": {
        "station_quantity": "U_s(t)-U_s(t0) at exact track time",
        "persistent_station_count": int(
            len(direct["persistent_gnss_stations"])
        ),
        "persistent_stations": [
            str(value)
            for value in direct["persistent_gnss_stations"]
        ],
        "interpolator": "pre-event-frozen local gp_rbf",
        "length_scale_km": {
            track: float(base_models[track].length_scale_km)
            for track in TRACK_TIMES
        },
        "mean_weights_frozen": True,
        "projection": "pixel-specific lU",
        "subtraction": "D_HLOS = D_LOS - referenced(lU * Uhat)",
    },
    "horizontal_solution": {
        "method": "pixelwise two-track 2x2 E-N inversion",
        "maximum_geometry_condition_number": 8.0,
        "full_EN_covariance_saved": True,
    },
    "cumulative_strain": {
        "method": "fixed joint E-N local affine GLS",
        "sample_lattice_km": 2.0,
        "dense_output_grid_km": 1.0,
        "support_radius_km": STRAIN_SUPPORT_RADIUS_KM,
        "bandwidth_km": STRAIN_BANDWIDTH_KM,
        "minimum_samples": MIN_STRAIN_SAMPLES,
        "rupture_buffer_km": RUPTURE_BUFFER_KM,
        "minimum_target_distance_from_rupture_km": (
            STRAIN_SAFE_DISTANCE_KM
        ),
        "supported_target_cells": int(strain_mask.sum()),
        "maps_interpolated_from_sparse_strain_dots": False,
    },
    "change_detection": {
        "source": "exact 12- and 24-day innovations of cumulative strain",
        "inference_lattice_km": 4.0,
        "calibration_end": str(CALIBRATION_END.date()),
        "pre_event_surveillance_end": str(PRE_EVENT_END.date()),
        "event_control_end": str(EVENT_END.date()),
        "pre_event_max_cluster_p": pre_event_cluster_p,
        "event_max_cluster_p": event_cluster_p,
        "pre_event_page_cusum_p": pre_cusum_p,
        "event_page_cusum_p": event_cusum_p,
        "earthquake_or_post_event_used_for_calibration": False,
    },
    "operation_order_audit": {
        "old_interval_first_product_compared": bool(len(comparison)),
        "comparison_file": (
            "cumulative_first_vs_interval_first_audit.csv"
        ),
    },
    "validation_boundary": (
        "Primary observational strain is restricted to >18 km from mapped "
        "rupture because the full-scene event vertical interpolation gate "
        "failed; the off-fault gate passed."
    ),
    "limitations": [
        (
            "Ascending and descending nominal-date acquisitions differ by "
            "about 12.026 hours."
        ),
        (
            "North displacement and north-related strain remain weakly "
            "constrained by two near-polar LOS geometries."
        ),
        (
            "Conditional GLS uncertainty does not fully represent long-range "
            "spatial correlation in InSAR."
        ),
        (
            "No generic interpolation or gap filling is applied across the "
            "rupture-adjacent exclusion."
        ),
        (
            "Cumulative strain levels are descriptive; formal testing uses "
            "their duration-normalized innovations."
        ),
    ],
}
(OUTPUT_DIR / "cumulative_two_track_strain_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

readme = f"""# Direct cumulative two-track strain

This directory replaces the earlier interval-first strain visualization.

The operation order is:

1. cumulative GNSS U at each station relative to {dates[0].date()};
2. fixed local interpolation of cumulative U at every common epoch;
3. pixel-specific vertical-to-LOS projection;
4. subtraction from each track's cumulative LOS;
5. cumulative two-track E-N inversion;
6. one fixed dense joint-GLS derivative operator;
7. cumulative strain maps on the supported 1-km grid.

The dense maps are direct MLS target estimates rendered with `pcolormesh`.
They are not interpolated from the old 4-km strain dots. Unsupported cells
within 18 km of mapped rupture remain missing.

Formal change detection uses exact 12- and 24-day innovations of the cumulative
strain cube. Calibration ends on {CALIBRATION_END.date()}; earthquake and
post-earthquake data are not used for baseline fitting or threshold calibration.

Primary files:

- `direct_cumulative_vertical_corrected_en.npz`
- `dense_cumulative_strain_1km.npz`
- `03_cumulative_strain_component_timeseries.png`
- `04_cumulative_*_key_dates.png`
- `05_cumulative_strain_change_detection_timeseries.png`
- `06_dense_strain_innovation_*.png`
- `cumulative_two_track_strain_manifest.json`
"""
(OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")
print("Saved cumulative workflow manifest and outputs to", OUTPUT_DIR)
