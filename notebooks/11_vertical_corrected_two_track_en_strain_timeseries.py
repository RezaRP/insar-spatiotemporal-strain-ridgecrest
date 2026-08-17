# %% [markdown]
# # Vertical-corrected ascending–descending E–N and incremental-strain time series
#
# This notebook applies the validated all-station local vertical model to every
# shared Track-64/Track-71 acquisition interval. The processing order is fixed:
#
# \[
# \widehat U(x,y,t)\;\to\;l_U(x,y)\widehat U(x,y,t)\;\to\;
# h_{asc},h_{desc}\;\to\;E,N\;\to\;\epsilon.
# \]
#
# It uses no global vertical plane and no ten-nearest-station interpolation.
# Each target uses all stations in its smallest geometry-adequate local radius.

# %%
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import sys

import h5py
from IPython.display import display
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

from ridgecrest_gnss_strain import load_rupture_segments_utm  # noqa: E402
from ridgecrest_jump import load_text_stack  # noqa: E402
from ridgecrest_local_vertical import (  # noqa: E402
    LocalVerticalConfig,
    LocalVerticalModel,
    build_local_support_topology,
    estimate_interval_sill_mm2,
    evaluate_local_vertical_model,
    predict_local_vertical_from_topology,
)
from ridgecrest_two_track import (  # noqa: E402
    common_utm11_grid,
    correct_vertical_los_on_grid,
    masked_bilinear_resample,
    normalize_look_vectors,
    rmls_incremental_strain,
    rupture_point_distance_lower_bound_km,
    solve_two_track_horizontal,
    to_utm11_km,
)
from ridgecrest_vertical_los import (  # noqa: E402
    gnss_interval_table,
    haversine_km,
    load_gnss_network,
)


mpl.rcParams.update({"figure.dpi": 130, "savefig.dpi": 300, "font.size": 11, "axes.titlesize": 13})

# %% [markdown]
# ## 1. Inputs and model restoration
#
# `Notebook 13` selected the local covariance family and parameters using only
# pre-event GNSS controls and then passed an independent 2019 temporal holdout.
# Here they are held fixed. The interval-specific
# vertical sill is allowed to increase with the observed all-station GNSS
# vertical variance, so event-time uncertainty is not artificially constrained
# to quiet pre-event levels.

# %%
TEXT_DIR = ROOT / "data"
DESC_H5 = ROOT / "data" / "cum_full_scene_no_GACOS.h5"
TRACK64_LOOK = ROOT / "outputs" / "track64_text_timeseries" / "track64_text_pixel_look_vectors.npz"
COMMON_DATES_FILE = ROOT / "outputs" / "track64_text_timeseries" / "track64_track71_common_dates.csv"
LOCAL_MANIFEST = ROOT / "outputs" / "gnss_vertical_interpolation_gate" / "vertical_interpolation_manifest.json"
SIGN_MANIFEST = ROOT / "outputs" / "station_los_projection_validation" / "station_projection_manifest.json"
GNSS_ROOT = Path(r"D:\Uni\Thises\GNSS_ridgecrest\data\tenv_data")
FAULT_FILE = ROOT / "data" / "cgs_2019_ridgecrest_fault_ruptures.geojson"
OUTPUT_DIR = ROOT / "outputs" / "two_track_vertical_corrected_timeseries"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for item in (DESC_H5, TRACK64_LOOK, COMMON_DATES_FILE, LOCAL_MANIFEST, SIGN_MANIFEST, GNSS_ROOT, FAULT_FILE):
    if not item.exists():
        raise FileNotFoundError(item)

TRACK_TIMES = {
    "ascending_T64": pd.Timedelta(hours=1, minutes=50, seconds=8, microseconds=490464),
    "descending_T71": pd.Timedelta(hours=13, minutes=51, seconds=41, microseconds=812911),
}
EVENTS = (pd.Timestamp("2019-07-04T17:33:49"), pd.Timestamp("2019-07-06T03:19:53"))
# P463 is the geometrically most distant reference-station candidate with
# adequate native pixels in both cumulative products and the independent
# 4-16 July IFG maps. The previous P597 disk lies outside this T71 H5.
REFERENCE_STATION = "P463"
REFERENCE_RADIUS_KM = 1.5
COMMON_GRID_SPACING_KM = 1.0
RUPTURE_BUFFER_KM = 10.0
STRAIN_SUPPORT_RADIUS_KM = 8.0
STRAIN_SAFE_DISTANCE_KM = RUPTURE_BUFFER_KM + STRAIN_SUPPORT_RADIUS_KM
INCREMENT_SIGMA_ASC_MM = 24.335595497061632
INCREMENT_SIGMA_DESC_MM = 16.277079925630176


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
base_models = {track: restore_model(raw) for track, raw in local_manifest["selected_models"].items()}
print("Restored local models:", {track: (model.family, model.length_scale_km, model.nugget_mm) for track, model in base_models.items()})

# %% [markdown]
# ## 2. Load text and HDF5 cumulative fields onto native grids
#
# The text stack is aligned by coordinates by `load_text_stack`; neither its
# changing row order nor its small missing-pixel changes are treated as time
# series displacement. The descending HDF5 is read one cumulative epoch at a
# time to avoid loading the full cube into memory.

# %%
ascending = load_text_stack(TEXT_DIR, align_coordinates=True)
look = np.load(TRACK64_LOOK)
if not (np.array_equal(look["latitude"], ascending.latitude) and np.array_equal(look["longitude"], ascending.longitude)):
    raise RuntimeError("Track-64 look vectors do not match canonical text coordinates")

asc_latitude = np.sort(np.unique(ascending.latitude))
asc_longitude = np.sort(np.unique(ascending.longitude))
asc_row = np.searchsorted(asc_latitude, ascending.latitude)
asc_col = np.searchsorted(asc_longitude, ascending.longitude)
if not (np.allclose(asc_latitude[asc_row], ascending.latitude) and np.allclose(asc_longitude[asc_col], ascending.longitude)):
    raise RuntimeError("Cannot build a rectilinear Track-64 grid from text coordinates")


def text_vector_to_grid(values: np.ndarray) -> np.ndarray:
    grid = np.full((len(asc_latitude), len(asc_longitude)), np.nan, dtype=float)
    grid[asc_row, asc_col] = np.asarray(values, dtype=float)
    return grid


asc_look_grid = {name: text_vector_to_grid(look[f"los_{name.lower()}"]) for name in ("E", "N", "U")}
asc_geometry_valid = np.isfinite(asc_look_grid["E"]) & np.isfinite(asc_look_grid["N"]) & np.isfinite(asc_look_grid["U"])

with h5py.File(DESC_H5, "r") as handle:
    desc_dates = pd.to_datetime(np.asarray(handle["imdates"][:], dtype=np.int64).astype(str), format="%Y%m%d")
    _, desc_ny, desc_nx = handle["cum"].shape
    desc_latitude = float(handle["corner_lat"][()]) + np.arange(desc_ny) * float(handle["post_lat"][()])
    desc_longitude = float(handle["corner_lon"][()]) + np.arange(desc_nx) * float(handle["post_lon"][()])
    desc_look_e = np.asarray(handle["E.geo"][:], dtype=float)
    desc_look_n = np.asarray(handle["N.geo"][:], dtype=float)
    desc_look_u = np.asarray(handle["U.geo"][:], dtype=float)
    desc_quality = (
        np.isfinite(handle["coh_avg"][:]) & (handle["coh_avg"][:] >= 0.30)
        & np.isfinite(handle["resid_rms"][:]) & (handle["resid_rms"][:] <= 5.0)
        & np.isfinite(handle["n_gap"][:]) & (handle["n_gap"][:] <= 2)
        & np.isfinite(handle["n_loop_err"][:]) & (handle["n_loop_err"][:] <= 10)
    )
desc_look_e, desc_look_n, desc_look_u = normalize_look_vectors(desc_look_e, desc_look_n, desc_look_u)
desc_quality &= np.isfinite(desc_look_e) & np.isfinite(desc_look_n) & np.isfinite(desc_look_u)
desc_date_index = {pd.Timestamp(date): index for index, date in enumerate(desc_dates)}
asc_date_index = {pd.Timestamp(date): index for index, date in enumerate(ascending.dates)}

common_dates = pd.read_csv(COMMON_DATES_FILE, parse_dates=["date"])["date"].to_list()
if any(date not in asc_date_index or date not in desc_date_index for date in common_dates):
    raise RuntimeError("Common-date manifest does not match source stacks")
print("Paired cumulative epochs:", len(common_dates), "intervals:", len(common_dates) - 1)

# %% [markdown]
# ## 3. Establish a shared analysis grid and cached all-station local support
#
# The two native grids are not index-aligned. They are explicitly sampled to a
# 1 km UTM 11N grid. Local support topology is computed once from the complete
# 25-station network and reused at every epoch; values and reported GNSS
# uncertainties remain interval-specific.

# %%
histories, network = load_gnss_network(GNSS_ROOT)
network_xy = to_utm11_km(network["longitude"].to_numpy(), network["latitude"].to_numpy())
east_grid, north_grid, latitude_grid, longitude_grid, geographic_overlap = common_utm11_grid(
    [asc_latitude, desc_latitude], [asc_longitude, desc_longitude],
    spacing_km=COMMON_GRID_SPACING_KM,
)
targets_xy = np.column_stack([east_grid.ravel(), north_grid.ravel()])
network_hull = Delaunay(network_xy).find_simplex(targets_xy) >= 0


def resample_static(latitude: np.ndarray, longitude: np.ndarray, fields: dict[str, np.ndarray], valid: np.ndarray) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    supports: list[np.ndarray] = []
    for name, values in fields.items():
        sampled, support = masked_bilinear_resample(latitude, longitude, values, valid, latitude_grid, longitude_grid)
        output[name] = sampled
        supports.append(support)
    output["valid"] = geographic_overlap & network_hull.reshape(east_grid.shape) & (np.minimum.reduce(supports) >= 0.999)
    output["E"], output["N"], output["U"] = normalize_look_vectors(output["E"], output["N"], output["U"])
    for name in ("E", "N", "U"):
        output[name][~output["valid"]] = np.nan
    return output


asc_static = resample_static(asc_latitude, asc_longitude, asc_look_grid, asc_geometry_valid)
desc_static = resample_static(
    desc_latitude, desc_longitude,
    {"E": desc_look_e, "N": desc_look_n, "U": desc_look_u}, desc_quality,
)
print("Common target cells with static geometry and network-hull support:", int((asc_static["valid"] & desc_static["valid"]).sum()))

reference_station = network.set_index("station").loc[REFERENCE_STATION]
asc_reference_vector = haversine_km(
    ascending.latitude, ascending.longitude,
    float(reference_station["latitude"]), float(reference_station["longitude"]),
) <= REFERENCE_RADIUS_KM
asc_reference_mask = asc_reference_vector & np.isfinite(look["los_u"])
asc_reference_xy = to_utm11_km(ascending.longitude[asc_reference_mask], ascending.latitude[asc_reference_mask])
desc_lat_grid, desc_lon_grid = np.meshgrid(desc_latitude, desc_longitude, indexing="ij")
desc_reference_mask = desc_quality & (
    haversine_km(desc_lat_grid, desc_lon_grid, float(reference_station["latitude"]), float(reference_station["longitude"])) <= REFERENCE_RADIUS_KM
)
desc_reference_xy = to_utm11_km(desc_lon_grid[desc_reference_mask], desc_lat_grid[desc_reference_mask])
if int(asc_reference_mask.sum()) < 10 or int(desc_reference_mask.sum()) < 25:
    raise RuntimeError(f"Insufficient native {REFERENCE_STATION} reference pixels")

# Later epochs can legitimately have a missing daily GNSS station.  Cache
# topology by the actual endpoint-valid station set, retaining every available
# station rather than forcing a fixed 25-station network or inventing data.
topology_cache: dict[tuple[str, tuple[str, ...]], tuple[object, object]] = {}


def topologies_for(track: str, table: pd.DataFrame):
    station_key = tuple(table["station"].astype(str))
    key = (track, station_key)
    if key not in topology_cache:
        xy = table[["east_km", "north_km"]].to_numpy(float)
        config = base_models[track].config
        reference_xy = asc_reference_xy if track == "ascending_T64" else desc_reference_xy
        topology_cache[key] = (
            build_local_support_topology(xy, targets_xy, config),
            build_local_support_topology(xy, reference_xy, config),
        )
    return topology_cache[key]

# %% [markdown]
# ## 4. Interval-specific GNSS vertical prediction and independent LOS correction
#
# The GNSS vertical field is predicted for each exact track interval, then
# projected with the matching `lU` field. Observed LOS increments are also
# independently re-referenced over the P463 disk. After the fixed sign audit,
# the signed projected vertical term is subtracted; a negative vertical term
# is therefore added automatically by the same equation.

# %%
def gnss_table(track: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    table = gnss_interval_table(
        histories, network,
        start=start + TRACK_TIMES[track], end=end + TRACK_TIMES[track],
        event_times=EVENTS, strict=False,
    )
    if len(table) < base_models[track].config.min_stations:
        raise RuntimeError(f"{track} {start.date()}-{end.date()}: fewer than the minimum local GNSS stations are endpoint-valid")
    xy = to_utm11_km(table["longitude"].to_numpy(), table["latitude"].to_numpy())
    table["east_km"] = xy[:, 0]
    table["north_km"] = xy[:, 1]
    return table


def dynamic_model(track: str, table: pd.DataFrame) -> LocalVerticalModel:
    event_sill = estimate_interval_sill_mm2(
        table["up_mm"].to_numpy(float),
        table["sigma_up_mm"].to_numpy(float),
    )
    return replace(base_models[track], sill_mm2=event_sill)


def asc_increment(start: pd.Timestamp, end: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, float]:
    values = ascending.displacement[asc_date_index[end]] - ascending.displacement[asc_date_index[start]]
    reference_valid = asc_reference_mask & np.isfinite(values)
    if int(reference_valid.sum()) < 10:
        raise RuntimeError(f"T64 insufficient observed reference pixels for {start.date()}-{end.date()}")
    referenced = values - float(np.nanmedian(values[reference_valid]))
    grid = text_vector_to_grid(referenced)
    sampled, support = masked_bilinear_resample(
        asc_latitude, asc_longitude, grid, np.isfinite(grid) & asc_geometry_valid,
        latitude_grid, longitude_grid,
    )
    return sampled, support, float(np.nanmedian(values[reference_valid]))


def desc_increment(handle: h5py.File, start: pd.Timestamp, end: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(handle["cum"][desc_date_index[end]], dtype=float) - np.asarray(handle["cum"][desc_date_index[start]], dtype=float)
    reference_valid = desc_reference_mask & np.isfinite(values)
    if int(reference_valid.sum()) < 25:
        raise RuntimeError(f"T71 insufficient observed reference pixels for {start.date()}-{end.date()}")
    referenced = values - float(np.nanmedian(values[reference_valid]))
    sampled, support = masked_bilinear_resample(
        desc_latitude, desc_longitude, referenced, desc_quality & np.isfinite(referenced),
        latitude_grid, longitude_grid,
    )
    return sampled, support, float(np.nanmedian(values[reference_valid]))


def vertical_reference(track: str, model: LocalVerticalModel, table: pd.DataFrame, reference_topology) -> tuple[float, float, int]:
    if track == "ascending_T64":
        reference_u = look["los_u"][asc_reference_mask]
    else:
        reference_u = desc_look_u[desc_reference_mask]
    prediction = predict_local_vertical_from_topology(
        model, table[["east_km", "north_km"]].to_numpy(float), table["up_mm"].to_numpy(float), table["sigma_up_mm"].to_numpy(float), reference_topology
    )
    valid = prediction.valid & np.isfinite(reference_u)
    if int(valid.sum()) < 10:
        raise RuntimeError(f"{track}: no valid local vertical reference prediction")
    raw = reference_u[valid] * prediction.mean_mm[valid]
    raw_sigma = np.abs(reference_u[valid]) * prediction.sigma_mm[valid]
    return float(np.nanmedian(raw)), float(np.nanmedian(raw_sigma)), int(valid.sum())


def corrected_los_interval(
    track: str, observed: np.ndarray, observed_support: np.ndarray, table: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = dynamic_model(track, table)
    target_topology, reference_topology = topologies_for(track, table)
    prediction = predict_local_vertical_from_topology(
        model, table[["east_km", "north_km"]].to_numpy(float), table["up_mm"].to_numpy(float), table["sigma_up_mm"].to_numpy(float), target_topology
    )
    u_mean = prediction.mean_mm.reshape(east_grid.shape)
    u_sigma = prediction.sigma_mm.reshape(east_grid.shape)
    static = asc_static if track == "ascending_T64" else desc_static
    valid = static["valid"] & (observed_support >= 0.999) & prediction.valid.reshape(east_grid.shape)
    reference_value, reference_sigma, _ = vertical_reference(track, model, table, reference_topology)
    sign = int(sign_manifest["selected_insar_sign"][track])
    corrected, vertical_los, vertical_los_sigma, _, _ = correct_vertical_los_on_grid(
        sign * observed, static["U"], u_mean, u_sigma,
        reference_value_mm=reference_value, reference_sigma_mm=reference_sigma,
    )
    for item in (corrected, vertical_los, vertical_los_sigma, u_mean, u_sigma):
        item[~valid] = np.nan
    return corrected, vertical_los, vertical_los_sigma, u_mean, u_sigma

# %% [markdown]
# ## 5. Fixed-model event-interval spatial validation
#
# The covariance family, range, nugget, and uncertainty scale remain frozen
# from the pre-event experiment. They are evaluated here by leaving out each
# interior GNSS station during 4–16 July. Event observations cannot tune the
# model. Because only one event interval exists, this gate does not use an
# interval bootstrap; it requires complete station coverage, practical RMSE
# equivalence to the same-neighbourhood local constant, improved mean NLPD,
# and calibrated 90% coverage.

# %%
rupture_segments = load_rupture_segments_utm(FAULT_FILE, certain_only=True)
network_distance_to_rupture = pd.Series(
    rupture_point_distance_lower_bound_km(network_xy, rupture_segments),
    index=network["station"].astype(str),
)
event_validation_summaries: dict[str, dict[str, float | int | bool]] = {}
event_gate_results: dict[str, bool] = {}
event_far_field_summaries: dict[str, dict[str, float | int | bool]] = {}
event_far_field_gate_results: dict[str, bool] = {}
for track in TRACK_TIMES:
    event_table = gnss_table(
        track, pd.Timestamp("2019-07-04"), pd.Timestamp("2019-07-16")
    )
    event_summary, event_predictions = evaluate_local_vertical_model(
        [event_table],
        base_models[track],
        rmse_relative_tolerance=0.10,
        coverage_bounds=(0.65, 1.00),
        uncertainty_scale=float(
            local_manifest["calibration_uncertainty_scales"][track]
        ),
    )
    event_gate = bool(
        event_summary["complete_fixed_holdout_coverage"]
        and float(event_summary["rmse_mm"])
        <= 1.10 * float(event_summary["baseline_rmse_mm"])
        and float(event_summary["mean_nlpd"])
        < float(event_summary["baseline_mean_nlpd"])
        and 0.65 <= float(event_summary["coverage90"]) <= 1.00
    )
    event_predictions["distance_to_rupture_km"] = event_predictions[
        "holdout_station"
    ].map(network_distance_to_rupture)
    far_field = event_predictions.loc[
        event_predictions["distance_to_rupture_km"]
        > STRAIN_SAFE_DISTANCE_KM
    ].copy()
    far_field_rmse = float(
        np.sqrt(np.mean(np.square(far_field["residual_mm"].to_numpy(float))))
    )
    far_field_baseline_rmse = float(
        np.sqrt(
            np.mean(
                np.square(
                    far_field["baseline_residual_mm"].to_numpy(float)
                )
            )
        )
    )
    far_field_mean_nlpd = float(far_field["nlpd"].mean())
    far_field_baseline_nlpd = float(far_field["baseline_nlpd"].mean())
    far_field_coverage90 = float(far_field["covered_90"].mean())
    far_field_summary: dict[str, float | int | bool] = {
        "rupture_safe_distance_km": STRAIN_SAFE_DISTANCE_KM,
        "n_predictions": int(len(far_field)),
        "rmse_mm": far_field_rmse,
        "baseline_rmse_mm": far_field_baseline_rmse,
        "mean_nlpd": far_field_mean_nlpd,
        "baseline_mean_nlpd": far_field_baseline_nlpd,
        "coverage90": far_field_coverage90,
    }
    far_field_gate = bool(
        len(far_field) >= 8
        and far_field_rmse <= 1.10 * far_field_baseline_rmse
        and far_field_mean_nlpd < far_field_baseline_nlpd
        and 0.65 <= far_field_coverage90 <= 1.00
    )
    far_field_summary["passed"] = far_field_gate
    event_validation_summaries[track] = {
        key: (
            None
            if isinstance(value, float) and not np.isfinite(value)
            else value
        )
        for key, value in event_summary.items()
    }
    event_gate_results[track] = event_gate
    event_far_field_summaries[track] = far_field_summary
    event_far_field_gate_results[track] = far_field_gate
    event_predictions.to_csv(
        OUTPUT_DIR / f"{track}_20190704_20190716_vertical_loo.csv",
        index=False,
    )
    print(
        track,
        "event spatial gate:",
        "PASS" if event_gate else "FAIL",
        {
            "rmse_mm": event_summary["rmse_mm"],
            "baseline_rmse_mm": event_summary["baseline_rmse_mm"],
            "mean_nlpd": event_summary["mean_nlpd"],
            "baseline_mean_nlpd": event_summary["baseline_mean_nlpd"],
            "coverage90": event_summary["coverage90"],
        },
    )
    print(
        track,
        f"event off-fault gate (>{STRAIN_SAFE_DISTANCE_KM:g} km):",
        "PASS" if far_field_gate else "FAIL",
        far_field_summary,
    )

# %% [markdown]
# ## 6. Build the 79 interval horizontal-LOS, E–N, and RMLS strain increments
#
# Each row below is a matched nominal-date interval. The two actual track
# endpoints remain about 12 hours apart; intervals spanning 4–16 July are
# labelled earthquake-sequence increments, not exactly simultaneous or purely
# coseismic deformation.

# %%
distance = rupture_point_distance_lower_bound_km(targets_xy, rupture_segments).reshape(east_grid.shape)
sample_lattice = np.zeros(east_grid.shape, dtype=bool)
sample_lattice[::2, ::2] = True
strain_targets = np.column_stack([east_grid[::4, ::4].ravel(), north_grid[::4, ::4].ravel()])
strain_targets = strain_targets[rupture_point_distance_lower_bound_km(strain_targets, rupture_segments) > STRAIN_SAFE_DISTANCE_KM]

east_increment: list[np.ndarray] = []
north_increment: list[np.ndarray] = []
sigma_east: list[np.ndarray] = []
sigma_north: list[np.ndarray] = []
valid_increment: list[np.ndarray] = []
asc_vertical: list[np.ndarray] = []
desc_vertical: list[np.ndarray] = []
asc_observed_los: list[np.ndarray] = []
desc_observed_los: list[np.ndarray] = []
asc_vertical_los: list[np.ndarray] = []
desc_vertical_los: list[np.ndarray] = []
asc_horizontal_los: list[np.ndarray] = []
desc_horizontal_los: list[np.ndarray] = []
asc_vertical_los_sigma: list[np.ndarray] = []
desc_vertical_los_sigma: list[np.ndarray] = []
summary_rows: list[dict[str, object]] = []
strain_frames: list[pd.DataFrame] = []

with h5py.File(DESC_H5, "r") as handle:
    for interval_index, (start, end) in enumerate(zip(common_dates[:-1], common_dates[1:])):
        asc_observed, asc_support, asc_reference = asc_increment(start, end)
        desc_observed, desc_support, desc_reference = desc_increment(handle, start, end)
        asc_table = gnss_table("ascending_T64", start, end)
        desc_table = gnss_table("descending_T71", start, end)
        asc_los, asc_vlos, asc_vlos_sigma, asc_u, _ = corrected_los_interval(
            "ascending_T64", asc_observed, asc_support, asc_table
        )
        desc_los, desc_vlos, desc_vlos_sigma, desc_u, _ = corrected_los_interval(
            "descending_T71", desc_observed, desc_support, desc_table
        )
        asc_signed_observed = (
            int(sign_manifest["selected_insar_sign"]["ascending_T64"])
            * asc_observed
        )
        desc_signed_observed = (
            int(sign_manifest["selected_insar_sign"]["descending_T71"])
            * desc_observed
        )
        asc_signed_observed[~np.isfinite(asc_los)] = np.nan
        desc_signed_observed[~np.isfinite(desc_los)] = np.nan
        solution = solve_two_track_horizontal(
            asc_los, desc_los,
            asc_static["E"], asc_static["N"], desc_static["E"], desc_static["N"],
            np.full(east_grid.shape, INCREMENT_SIGMA_ASC_MM),
            np.full(east_grid.shape, INCREMENT_SIGMA_DESC_MM),
            vertical_los_sigma_ascending_mm=asc_vlos_sigma,
            vertical_los_sigma_descending_mm=desc_vlos_sigma,
            vertical_correlation=1.0,
            max_condition_number=8.0,
        )
        safe = solution.valid & (distance > STRAIN_SAFE_DISTANCE_KM)
        sample_mask = safe & sample_lattice
        strain = rmls_incremental_strain(
            np.column_stack([east_grid[sample_mask], north_grid[sample_mask]]),
            solution.east_mm[sample_mask], solution.north_mm[sample_mask],
            solution.sigma_east_mm[sample_mask], solution.sigma_north_mm[sample_mask],
            strain_targets, support_radius_km=STRAIN_SUPPORT_RADIUS_KM,
            bandwidth_km=4.0, min_samples=16,
        )
        strain["interval_index"] = interval_index
        strain["start_date"] = start
        strain["end_date"] = end
        strain["duration_days"] = int((end - start).days)
        strain["dilatation_z"] = strain["dilatation_nstrain"] / np.maximum(strain["sigma_dilatation_nstrain"], 1.0e-12)
        strain["dilatation_resolved_95pct"] = np.abs(strain["dilatation_z"]) >= 1.96
        strain_frames.append(strain)
        east_increment.append(solution.east_mm.astype(np.float32))
        north_increment.append(solution.north_mm.astype(np.float32))
        sigma_east.append(solution.sigma_east_mm.astype(np.float32))
        sigma_north.append(solution.sigma_north_mm.astype(np.float32))
        valid_increment.append(solution.valid)
        asc_vertical.append(asc_u.astype(np.float32))
        desc_vertical.append(desc_u.astype(np.float32))
        asc_observed_los.append(asc_signed_observed.astype(np.float32))
        desc_observed_los.append(desc_signed_observed.astype(np.float32))
        asc_vertical_los.append(asc_vlos.astype(np.float32))
        desc_vertical_los.append(desc_vlos.astype(np.float32))
        asc_horizontal_los.append(asc_los.astype(np.float32))
        desc_horizontal_los.append(desc_los.astype(np.float32))
        asc_vertical_los_sigma.append(asc_vlos_sigma.astype(np.float32))
        desc_vertical_los_sigma.append(desc_vlos_sigma.astype(np.float32))
        summary_rows.append({
            "interval_index": interval_index,
            "start_date": start, "end_date": end,
            "duration_days": int((end - start).days),
            "earthquake_sequence_interval": bool(start <= pd.Timestamp("2019-07-04") and end >= pd.Timestamp("2019-07-16")),
            "valid_en_cells": int(solution.valid.sum()),
            "median_sigma_east_mm": float(np.nanmedian(solution.sigma_east_mm)),
            "median_sigma_north_mm": float(np.nanmedian(solution.sigma_north_mm)),
            "median_condition_number": float(np.nanmedian(solution.condition_number)),
            "median_asc_vertical_mm": float(np.nanmedian(asc_u)),
            "median_desc_vertical_mm": float(np.nanmedian(desc_u)),
            "median_abs_asc_vertical_los_mm": float(
                np.nanmedian(np.abs(asc_vlos))
            ),
            "median_abs_desc_vertical_los_mm": float(
                np.nanmedian(np.abs(desc_vlos))
            ),
            "valid_asc_horizontal_los_cells": int(np.isfinite(asc_los).sum()),
            "valid_desc_horizontal_los_cells": int(np.isfinite(desc_los).sum()),
            "asc_observed_reference_mm": asc_reference,
            "desc_observed_reference_mm": desc_reference,
            "rmls_valid_targets": int(strain["valid"].sum()),
            "rmls_dilatation_resolved_95pct_targets": int((strain["valid"] & strain["dilatation_resolved_95pct"]).sum()),
        })
        if interval_index % 10 == 0 or interval_index == len(common_dates) - 2:
            print(f"Completed interval {interval_index + 1}/{len(common_dates) - 1}: {start.date()} to {end.date()}")

summary = pd.DataFrame(summary_rows)
strain_timeseries = pd.concat(strain_frames, ignore_index=True)
summary.to_csv(OUTPUT_DIR / "two_track_en_interval_summary.csv", index=False)
strain_timeseries.to_csv(OUTPUT_DIR / "two_track_rmls_incremental_strain_timeseries.csv", index=False)

east_increment_array = np.stack(east_increment)
north_increment_array = np.stack(north_increment)
sigma_east_array = np.stack(sigma_east)
sigma_north_array = np.stack(sigma_north)
valid_increment_array = np.stack(valid_increment)
asc_vertical_array = np.stack(asc_vertical)
desc_vertical_array = np.stack(desc_vertical)
asc_observed_los_array = np.stack(asc_observed_los)
desc_observed_los_array = np.stack(desc_observed_los)
asc_vertical_los_array = np.stack(asc_vertical_los)
desc_vertical_los_array = np.stack(desc_vertical_los)
asc_horizontal_los_array = np.stack(asc_horizontal_los)
desc_horizontal_los_array = np.stack(desc_horizontal_los)
asc_vertical_los_sigma_array = np.stack(asc_vertical_los_sigma)
desc_vertical_los_sigma_array = np.stack(desc_vertical_los_sigma)

# A cumulative E-N series is retained only where every prior interval was valid;
# missing intervals are never replaced by zero displacement.
cumulative_east = np.full((len(common_dates), *east_grid.shape), np.nan, dtype=np.float32)
cumulative_north = np.full((len(common_dates), *east_grid.shape), np.nan, dtype=np.float32)
running_valid = np.ones(east_grid.shape, dtype=bool)
cumulative_east[0, running_valid] = 0.0
cumulative_north[0, running_valid] = 0.0
for index in range(len(east_increment_array)):
    running_valid &= valid_increment_array[index]
    prior_east = cumulative_east[index]
    prior_north = cumulative_north[index]
    cumulative_east[index + 1, running_valid] = prior_east[running_valid] + east_increment_array[index][running_valid]
    cumulative_north[index + 1, running_valid] = prior_north[running_valid] + north_increment_array[index][running_valid]


def strict_cumulative(increments: np.ndarray) -> np.ndarray:
    """Accumulate only where every preceding interval is observed."""

    cumulative = np.full(
        (len(increments) + 1, *increments.shape[1:]),
        np.nan,
        dtype=np.float32,
    )
    running_valid = np.ones(increments.shape[1:], dtype=bool)
    cumulative[0, running_valid] = 0.0
    for index, increment in enumerate(increments):
        running_valid &= np.isfinite(increment)
        cumulative[index + 1, running_valid] = (
            cumulative[index, running_valid] + increment[running_valid]
        )
    return cumulative


asc_observed_los_cumulative = strict_cumulative(asc_observed_los_array)
desc_observed_los_cumulative = strict_cumulative(desc_observed_los_array)
asc_vertical_los_cumulative = strict_cumulative(asc_vertical_los_array)
desc_vertical_los_cumulative = strict_cumulative(desc_vertical_los_array)
asc_horizontal_los_cumulative = strict_cumulative(asc_horizontal_los_array)
desc_horizontal_los_cumulative = strict_cumulative(desc_horizontal_los_array)

# Algebraic audit of the requested signed subtraction.
for observed, vertical_los, horizontal_los, label in (
    (
        asc_observed_los_array,
        asc_vertical_los_array,
        asc_horizontal_los_array,
        "ascending",
    ),
    (
        desc_observed_los_array,
        desc_vertical_los_array,
        desc_horizontal_los_array,
        "descending",
    ),
):
    finite = (
        np.isfinite(observed)
        & np.isfinite(vertical_los)
        & np.isfinite(horizontal_los)
    )
    maximum_error = float(
        np.nanmax(
            np.abs(
                horizontal_los[finite]
                - (observed[finite] - vertical_los[finite])
            )
        )
    )
    if maximum_error > 1.0e-4:
        raise RuntimeError(
            f"{label} horizontal-LOS subtraction audit failed: "
            f"{maximum_error:g} mm"
        )
    print(label, "maximum subtraction identity error (mm):", maximum_error)

np.savez_compressed(
    OUTPUT_DIR / "two_track_vertical_corrected_en_timeseries.npz",
    dates=np.asarray(common_dates, dtype="datetime64[ns]"),
    east_km=east_grid, north_km=north_grid, latitude=latitude_grid, longitude=longitude_grid,
    distance_to_mapped_rupture_km=distance.astype(np.float32),
    off_fault_vertical_validation_mask=(
        distance > STRAIN_SAFE_DISTANCE_KM
    ),
    east_increment_mm=east_increment_array, north_increment_mm=north_increment_array,
    sigma_east_mm=sigma_east_array, sigma_north_mm=sigma_north_array,
    valid_increment=valid_increment_array,
    ascending_vertical_increment_mm=asc_vertical_array,
    descending_vertical_increment_mm=desc_vertical_array,
    ascending_observed_los_increment_mm=asc_observed_los_array,
    descending_observed_los_increment_mm=desc_observed_los_array,
    ascending_vertical_to_los_increment_mm=asc_vertical_los_array,
    descending_vertical_to_los_increment_mm=desc_vertical_los_array,
    ascending_vertical_to_los_sigma_mm=asc_vertical_los_sigma_array,
    descending_vertical_to_los_sigma_mm=desc_vertical_los_sigma_array,
    ascending_pure_horizontal_los_increment_mm=asc_horizontal_los_array,
    descending_pure_horizontal_los_increment_mm=desc_horizontal_los_array,
    ascending_observed_los_cumulative_mm=asc_observed_los_cumulative,
    descending_observed_los_cumulative_mm=desc_observed_los_cumulative,
    ascending_vertical_to_los_cumulative_mm=asc_vertical_los_cumulative,
    descending_vertical_to_los_cumulative_mm=desc_vertical_los_cumulative,
    ascending_pure_horizontal_los_cumulative_mm=asc_horizontal_los_cumulative,
    descending_pure_horizontal_los_cumulative_mm=desc_horizontal_los_cumulative,
    cumulative_east_mm=cumulative_east, cumulative_north_mm=cumulative_north,
)
print("Saved interval fields:", len(summary), "and RMLS records:", len(strain_timeseries))

# %% [markdown]
# ## 7. Verify and visualize the event-interval LOS subtraction
#
# The observed and pure-horizontal maps share one colour range within each
# track. The projected vertical contribution uses its own range so that a
# smaller but spatially structured correction remains visible.

# %%
event_indices = np.flatnonzero(
    summary["earthquake_sequence_interval"].to_numpy(bool)
)
if len(event_indices) != 1:
    raise RuntimeError("Expected exactly one 4–16 July sequence interval")
event_index = int(event_indices[0])
event_products = (
    (
        "Track 64 ascending",
        asc_observed_los_array[event_index],
        asc_vertical_los_array[event_index],
        asc_horizontal_los_array[event_index],
    ),
    (
        "Track 71 descending",
        desc_observed_los_array[event_index],
        desc_vertical_los_array[event_index],
        desc_horizontal_los_array[event_index],
    ),
)
fig, axes = plt.subplots(
    2,
    3,
    figsize=(16.5, 9.2),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)
for row, (track_label, observed, vertical_los, horizontal_los) in enumerate(
    event_products
):
    horizontal_scale = float(
        np.nanpercentile(
            np.abs(np.r_[observed.ravel(), horizontal_los.ravel()]), 98.0
        )
    )
    vertical_scale = float(np.nanpercentile(np.abs(vertical_los), 98.0))
    horizontal_norm = TwoSlopeNorm(
        vmin=-horizontal_scale, vcenter=0.0, vmax=horizontal_scale
    )
    vertical_norm = TwoSlopeNorm(
        vmin=-vertical_scale, vcenter=0.0, vmax=vertical_scale
    )
    observed_image = axes[row, 0].pcolormesh(
        east_grid,
        north_grid,
        observed,
        cmap="RdBu_r",
        norm=horizontal_norm,
        shading="auto",
    )
    vertical_image = axes[row, 1].pcolormesh(
        east_grid,
        north_grid,
        vertical_los,
        cmap="RdBu_r",
        norm=vertical_norm,
        shading="auto",
    )
    axes[row, 2].pcolormesh(
        east_grid,
        north_grid,
        horizontal_los,
        cmap="RdBu_r",
        norm=horizontal_norm,
        shading="auto",
    )
    axes[row, 0].set_ylabel(f"{track_label}\nNorthing (km)")
    for column in range(3):
        axes[row, column].set_aspect("equal")
        axes[row, column].grid(True, color="0.83", ls="--", lw=0.45)
    fig.colorbar(
        observed_image,
        ax=[axes[row, 0], axes[row, 2]],
        label="LOS displacement (mm)",
        shrink=0.88,
    )
    fig.colorbar(
        vertical_image,
        ax=axes[row, 1],
        label="Vertical-to-LOS (mm)",
        shrink=0.88,
    )
for axis, title in zip(
    axes[0],
    (
        r"Observed $D_{\mathrm{LOS}}$",
        r"Projected $l_U\widehat{U}$",
        r"Vertical-removed HLOS sensitivity $D_{\mathrm{LOS}}-l_U\widehat{U}$",
    ),
):
    axis.set_title(title)
for axis in axes[-1]:
    axis.set_xlabel("Easting (km)")
fig.suptitle(
    "4–16 July 2019: reference-consistent vertical removal "
    "(full-scene validation failed)",
    fontsize=16,
)
fig.savefig(
    OUTPUT_DIR / "event_los_vertical_horizontal_decomposition.png",
    bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ## 8. Plot E–N cumulative time series at diagnostic GNSS-neighbour pixels
#
# These are spatially resolved two-track displacement estimates with propagated
# vertical-model uncertainty. They are not a replacement for the direct GNSS
# components and are not plotted as a precursor diagnostic.

# %%
# P595 lies in a persistently masked near-rupture zone in this particular
# two-track overlap.  Use three fully sampled diagnostic pixels instead of
# drawing an empty time series and falsely suggesting a zero displacement.
diagnostic_stations = [station for station in ("P580", "CCCC", "P593") if station in set(network["station"])]
fig, axes = plt.subplots(len(diagnostic_stations), 1, figsize=(12, 3.4 * len(diagnostic_stations)), sharex=True, constrained_layout=True)
if len(diagnostic_stations) == 1:
    axes = [axes]
for axis, station in zip(axes, diagnostic_stations):
    row = network.set_index("station").loc[station]
    xy = to_utm11_km(np.array([row["longitude"]]), np.array([row["latitude"]]))[0]
    distance_to_pixel = np.hypot(east_grid - xy[0], north_grid - xy[1])
    grid_row, grid_col = np.unravel_index(np.nanargmin(distance_to_pixel), distance_to_pixel.shape)
    axis.plot(common_dates, cumulative_east[:, grid_row, grid_col], marker="o", ms=3, color="#1f77b4", label="E cumulative")
    axis.plot(common_dates, cumulative_north[:, grid_row, grid_col], marker="o", ms=3, color="#d62728", label="N cumulative")
    axis.axvline(pd.Timestamp("2019-07-04"), color="0.25", ls="--", lw=1)
    axis.axvline(pd.Timestamp("2019-07-06"), color="0.25", ls="--", lw=1)
    axis.set(title=f"Two-track cumulative E-N at nearest 1-km pixel to {station}", ylabel="Cumulative increment (mm)")
    axis.grid(True, color="0.88", ls="--", lw=0.5)
    axis.legend(loc="best")
axes[-1].set_xlabel("Nominal acquisition date")
fig.savefig(OUTPUT_DIR / "two_track_en_cumulative_timeseries_diagnostic_pixels.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Strain time-series uncertainty summary
#
# Incremental strain is shown only as an uncertainty-screened off-fault
# diagnostic. A zero resolved-cell count is an informative result: it means the
# current two-track geometry and GNSS vertical uncertainty do not support a
# direct observed 2-D strain claim at this resolution.

# %%
strain_summary = (
    strain_timeseries.loc[strain_timeseries["valid"]]
    .groupby(["interval_index", "start_date", "end_date"], as_index=False)
    .agg(
        valid_targets=("valid", "size"),
        median_abs_dilatation_nstrain=("dilatation_nstrain", lambda x: float(np.nanmedian(np.abs(x)))),
        median_sigma_dilatation_nstrain=("sigma_dilatation_nstrain", "median"),
        resolved_dilatation_targets=("dilatation_resolved_95pct", "sum"),
    )
)
strain_summary.to_csv(OUTPUT_DIR / "two_track_rmls_strain_timeseries_summary.csv", index=False)

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
axes[0].plot(summary["end_date"], summary["median_sigma_north_mm"], color="#6a3d9a", marker="o", ms=3)
axes[0].set(title="Two-track north uncertainty by interval", ylabel="Median 1 sigma north (mm)")
axes[0].grid(True, color="0.88", ls="--", lw=0.5)
axes[1].step(strain_summary["end_date"], strain_summary["resolved_dilatation_targets"], where="mid", color="#b2182b", lw=2)
axes[1].set(title="Off-fault RMLS targets with |dilatation z| >= 1.96", ylabel="Resolved targets", xlabel="Interval end date")
axes[1].grid(True, color="0.88", ls="--", lw=0.5)
fig.savefig(OUTPUT_DIR / "two_track_strain_uncertainty_timeseries.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 10. Save analysis manifest and interpretation boundary

# %%
all_vertical_gates_pass = bool(
    local_manifest["vertical_interpolation_passed_both_tracks"]
    and all(event_gate_results.values())
)
off_fault_vertical_gates_pass = bool(
    local_manifest["vertical_interpolation_passed_both_tracks"]
    and all(event_far_field_gate_results.values())
)
manifest = {
    "status": (
        "validated vertical-removed horizontal-LOS, two-track E-N, and strain time series"
        if all_vertical_gates_pass
        else (
            "full-scene vertical-removal sensitivity; off-fault domain passed "
            "the event vertical-interpolation gate"
            if off_fault_vertical_gates_pass
            else "vertical-removal sensitivity product; at least one validation gate failed"
        )
    ),
    "epoch_count": len(common_dates),
    "interval_count": len(summary),
    "source_tracks": {
        "ascending": "Track-64 date-named cumulative text stack",
        "descending": "Track-71 cum_full_scene_no_GACOS.h5",
    },
    "vertical_interpolator": {
        "shared_model": local_manifest["shared_spatial_model"],
        "candidate_families": local_manifest["candidate_families"],
        "all_locally_eligible_stations_used": True,
        "map_wide_plane_used": False,
    },
    "pre_event_temporal_holdout_passed": bool(
        local_manifest["vertical_interpolation_passed_both_tracks"]
    ),
    "event_interval_spatial_gate": event_gate_results,
    "event_interval_spatial_validation": event_validation_summaries,
    "event_interval_off_fault_gate": event_far_field_gate_results,
    "event_interval_off_fault_validation": event_far_field_summaries,
    "all_vertical_gates_passed": all_vertical_gates_pass,
    "off_fault_vertical_gates_passed": off_fault_vertical_gates_pass,
    "off_fault_validation_mask": (
        f"distance to mapped rupture > {STRAIN_SAFE_DISTANCE_KM:g} km"
    ),
    "correction": {
        "ascending": (
            "D_LOS_PURE_HORIZONTAL_Asc = signed_referenced_D_LOS_Ascending "
            "- referenced(lU_Ascending * Uhat_GNSS_Ascending)"
        ),
        "descending": (
            "D_LOS_PURE_HORIZONTAL_Desc = signed_referenced_D_LOS_Descending "
            "- referenced(lU_Descending * Uhat_GNSS_Descending)"
        ),
        "sign_rule": (
            "always subtract the signed projected vertical term; subtraction "
            "automatically adds it where lU*Uhat is negative"
        ),
    },
    "reference": f"{REFERENCE_STATION}, {REFERENCE_RADIUS_KM:.1f}-km native disk common to both cumulative products",
    "look_geometry": "pixel-specific LiCSAR look vectors; no nominal incidence-angle approximation",
    "saved_horizontal_los_fields": [
        "ascending_observed_los_increment_mm",
        "ascending_vertical_to_los_increment_mm",
        "ascending_pure_horizontal_los_increment_mm",
        "descending_observed_los_increment_mm",
        "descending_vertical_to_los_increment_mm",
        "descending_pure_horizontal_los_increment_mm",
        "and cumulative equivalents",
        "off_fault_vertical_validation_mask",
    ],
    "grid_spacing_km": COMMON_GRID_SPACING_KM,
    "strain_support_radius_km": STRAIN_SUPPORT_RADIUS_KM,
    "strain_safe_distance_from_mapped_rupture_km": STRAIN_SAFE_DISTANCE_KM,
    "mean_valid_en_cells_per_interval": float(summary["valid_en_cells"].mean()),
    "median_sigma_east_mm": float(summary["median_sigma_east_mm"].median()),
    "median_sigma_north_mm": float(summary["median_sigma_north_mm"].median()),
    "resolved_dilatation_target_total": int(strain_summary["resolved_dilatation_targets"].sum()),
    "limitations": [
        "Ascending and descending nominal-date acquisitions differ by about 12 hours.",
        "Intervals spanning 4-16 July are earthquake-sequence intervals, not purely coseismic or exactly simultaneous products.",
        "North is weakly constrained by the two LOS geometries; use the saved uncertainty fields with every E-N or strain interpretation.",
        "Off-fault RMLS strain is an incremental diagnostic and is not a direct strain claim where uncertainty gates fail.",
        (
            "The pure-horizontal wording is not authorized for the full scene "
            "because the near-fault event interpolation gate failed. The "
            "off-fault mask identifies the independently supported domain."
        ),
    ],
}
(OUTPUT_DIR / "two_track_vertical_corrected_timeseries_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
display(pd.DataFrame([manifest]).T)
