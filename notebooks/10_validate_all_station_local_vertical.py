# %% [markdown]
# # All-station adaptive local GNSS vertical interpolation and LOS correction
#
# This notebook replaces the earlier fixed P595/ten-station sensitivity branch.
# It uses every endpoint-valid GNSS station as a candidate at every InSAR target,
# retains all stations in the smallest geometry-adequate local radius, and fits
# no spatial plane. Covariance parameters are selected from pre-event GNSS
# controls before the 4–16 July interval is evaluated.

# %%
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import sys

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
from ridgecrest_local_vertical import (  # noqa: E402
    LocalVerticalConfig,
    LocalVerticalModel,
    predict_local_vertical,
    select_local_vertical_model,
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


mpl.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 11,
})

# %% [markdown]
# ## 1. Inputs and predeclared no-plane local-model rules
#
# A target must lie inside its adaptive local GNSS hull and have at least five
# contributing stations in three of eight azimuth sectors. The maximum 120 km
# radius is selected by geometry, not by an arbitrary nearest-station count.
# Every station in the first adequate radius contributes to the prediction.

# %%
GNSS_ROOT = Path(r"D:\Uni\Thises\GNSS_ridgecrest\data\tenv_data")
PHASE1_DIR = ROOT / "outputs" / "gnss_vertical_los_phase1"
COMMON_DATES_FILE = ROOT / "outputs" / "track64_text_timeseries" / "track64_track71_common_dates.csv"
FAULT_FILE = ROOT / "data" / "cgs_2019_ridgecrest_fault_ruptures.geojson"
OUTPUT_DIR = ROOT / "outputs" / "all_station_local_vertical"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for item in (GNSS_ROOT, PHASE1_DIR, COMMON_DATES_FILE, FAULT_FILE):
    if not item.exists():
        raise FileNotFoundError(item)

TRACKS = {
    "ascending_T64": {
        "utc_time": pd.Timedelta(hours=1, minutes=50, seconds=8, microseconds=490464),
        "phase1_file": PHASE1_DIR / "ascending_T64_20190704_20190716_vertical_corrected_hlos.npz",
        "noise_floor_mm": 5.0,
    },
    "descending_T71": {
        "utc_time": pd.Timedelta(hours=13, minutes=51, seconds=41, microseconds=812911),
        "phase1_file": PHASE1_DIR / "descending_T71_20190704_20190716_vertical_corrected_hlos.npz",
        "noise_floor_mm": 5.0,
    },
}
EVENTS = (pd.Timestamp("2019-07-04T17:33:49"), pd.Timestamp("2019-07-06T03:19:53"))
EVENT_START = pd.Timestamp("2019-07-04")
EVENT_END = pd.Timestamp("2019-07-16")
PRE_EVENT_START = pd.Timestamp("2018-01-01")
PRE_EVENT_END = pd.Timestamp("2019-05-31")
CONFIG = LocalVerticalConfig(
    radii_km=(35.0, 45.0, 55.0, 65.0, 75.0, 90.0, 105.0, 120.0),
    min_stations=5,
    sector_count=8,
    min_occupied_sectors=3,
    require_local_hull=True,
)
MODEL_FAMILIES = ("ok_exponential", "gp_matern32")
MODEL_LENGTH_SCALES_KM = (25.0, 40.0, 60.0)
MODEL_NUGGETS_MM = (0.0, 5.0, 10.0)
RMSE_RELATIVE_TOLERANCE = 0.02
COMMON_GRID_SPACING_KM = 1.0
RUPTURE_BUFFER_KM = 10.0
STRAIN_SUPPORT_RADIUS_KM = 8.0
STRAIN_SAFE_DISTANCE_KM = RUPTURE_BUFFER_KM + STRAIN_SUPPORT_RADIUS_KM
# P463 is the farthest geometry-selected GNSS reference candidate with an
# adequate native disk in both cumulative products *and* the independent
# 4-16 July IFG maps. P597, used in the older Phase-I product, is outside the
# cropped Track-71 cumulative HDF5.
REFERENCE_STATION = "P463"
REFERENCE_RADIUS_KM = 1.5

phase1_manifest = json.loads((PHASE1_DIR / "phase1_manifest.json").read_text(encoding="utf-8"))

# %% [markdown]
# ## 2. Fixed pre-event controls and endpoint GNSS tables
#
# Non-overlapping 12-day controls are fixed from the shared ascending–descending
# calendar before the earthquake sequence. Their GNSS endpoint values are used
# only to choose the local covariance model, never to fit an InSAR ramp.

# %%
common_dates = pd.read_csv(COMMON_DATES_FILE, parse_dates=["date"])["date"]
pre_dates = common_dates[(common_dates >= PRE_EVENT_START) & (common_dates <= PRE_EVENT_END)]
pre_date_set = set(pre_dates)
control_pairs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
next_allowed: pd.Timestamp | None = None
for start in pre_dates:
    end = start + pd.Timedelta(days=12)
    if end in pre_date_set and (next_allowed is None or start >= next_allowed):
        control_pairs.append((start, end))
        next_allowed = end + pd.Timedelta(days=12)
print("Fixed non-overlapping pre-event 12-day controls:", len(control_pairs))
print(control_pairs)

histories, network = load_gnss_network(GNSS_ROOT)


def interval_table(track: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    time = TRACKS[track]["utc_time"]
    table = gnss_interval_table(
        histories, network,
        start=start_date + time,
        end=end_date + time,
        event_times=EVENTS,
        strict=False,
    )
    if len(table) != len(network):
        raise RuntimeError(f"{track}: a control did not retain the complete GNSS network")
    xy = to_utm11_km(table["longitude"].to_numpy(), table["latitude"].to_numpy())
    table["east_km"] = xy[:, 0]
    table["north_km"] = xy[:, 1]
    return table


controls = {
    track: [interval_table(track, start, end) for start, end in control_pairs]
    for track in TRACKS
}
print("Stations in every control:", len(controls["ascending_T64"][0]))

# %% [markdown]
# ## 3. Pre-event leave-one-station-out model selection
#
# The local ordinary-Kriging and local Matérn-3/2 GP candidates are compared
# against an uncertainty-weighted local constant baseline over exactly the same
# non-extrapolative held-out GNSS stations. A spatial candidate is accepted only
# if it improves predictive density with a positive 95% interval-bootstrap
# lower bound, has calibrated 90% intervals, and does not worsen RMSE by more
# than the predeclared 2% practical-equivalence margin.

# %%
selected_models: dict[str, LocalVerticalModel] = {}
for track in TRACKS:
    model, score, predictions = select_local_vertical_model(
        controls[track],
        configs=(CONFIG,),
        families=MODEL_FAMILIES,
        length_scales_km=MODEL_LENGTH_SCALES_KM,
        nuggets_mm=MODEL_NUGGETS_MM,
        rmse_relative_tolerance=RMSE_RELATIVE_TOLERANCE,
    )
    selected_models[track] = model
    score.to_csv(OUTPUT_DIR / f"{track}_pre_event_local_vertical_scores.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / f"{track}_pre_event_local_vertical_loo.csv", index=False)
    print(track, model)

model_table = pd.DataFrame({track: asdict(model) for track, model in selected_models.items()}).T
display(model_table)

# %% [markdown]
# ## 4. Apply selected all-station models to the exact 4–16 July endpoints
#
# The vertical field is estimated separately at each track's actual UTC
# endpoints. After sign alignment, correction is deterministic:
#
# \[
# h_i=s_i d_i-\left[l_{Ui}\widehat U_i-
# \operatorname{median}_{R}(l_{Ui}\widehat U_i)\right].
# \]
#
# It is therefore neither an arbitrary addition nor an incidence-angle
# approximation.

# %%
event_tables = {track: interval_table(track, EVENT_START, EVENT_END) for track in TRACKS}
for track, table in event_tables.items():
    table.to_csv(OUTPUT_DIR / f"{track}_all_station_20190704_20190716_enu.csv", index=False)
    print(track, "event endpoint-valid stations:", len(table))

products = {track: np.load(config["phase1_file"]) for track, config in TRACKS.items()}
east_grid, north_grid, latitude_grid, longitude_grid, geographic_overlap = common_utm11_grid(
    [products[track]["latitude"] for track in TRACKS],
    [products[track]["longitude"] for track in TRACKS],
    spacing_km=COMMON_GRID_SPACING_KM,
)
targets_xy = np.column_stack([east_grid.ravel(), north_grid.ravel()])
network_xy = event_tables["ascending_T64"][["east_km", "north_km"]].to_numpy(float)
reference_station = network.set_index("station").loc[REFERENCE_STATION]
inside_network_hull = Delaunay(network_xy).find_simplex(targets_xy) >= 0
inside_network_hull = inside_network_hull.reshape(east_grid.shape)
print("Common 1-km grid:", east_grid.shape, "inside all-station hull:", int(inside_network_hull.sum()))


def resample_phase1(track: str) -> dict[str, np.ndarray]:
    product = products[track]
    native_valid = product["valid"].astype(bool)
    native_latitude, native_longitude = np.meshgrid(product["latitude"], product["longitude"], indexing="ij")
    observed_reference_mask = native_valid & (
        haversine_km(
            native_latitude, native_longitude,
            float(reference_station["latitude"]), float(reference_station["longitude"]),
        ) <= REFERENCE_RADIUS_KM
    )
    if int(observed_reference_mask.sum()) < 25:
        raise RuntimeError(f"{track}: insufficient native {REFERENCE_STATION} reference pixels")
    observed_reference = float(np.nanmedian(product["referenced_los_mm"][observed_reference_mask]))
    fields: dict[str, np.ndarray] = {}
    supports: list[np.ndarray] = []
    for name in ("referenced_los_mm", "los_e", "los_n", "los_u"):
        source = product[name] - observed_reference if name == "referenced_los_mm" else product[name]
        sampled, support = masked_bilinear_resample(
            product["latitude"], product["longitude"], source, native_valid,
            latitude_grid, longitude_grid,
        )
        fields[name] = sampled
        supports.append(support)
    fields["valid"] = geographic_overlap & inside_network_hull & (np.minimum.reduce(supports) >= 0.999)
    fields["los_e"], fields["los_n"], fields["los_u"] = normalize_look_vectors(
        fields["los_e"], fields["los_n"], fields["los_u"]
    )
    for name in ("referenced_los_mm", "los_e", "los_n", "los_u"):
        fields[name][~fields["valid"]] = np.nan
    fields["observed_reference_mm"] = observed_reference
    return fields


common_fields = {track: resample_phase1(track) for track in TRACKS}


def native_reference_summary(track: str, model: LocalVerticalModel, table: pd.DataFrame) -> tuple[float, float, int]:
    product = products[track]
    native_latitude, native_longitude = np.meshgrid(product["latitude"], product["longitude"], indexing="ij")
    reference_mask = (
        product["valid"].astype(bool)
        & (haversine_km(native_latitude, native_longitude, float(reference_station["latitude"]), float(reference_station["longitude"])) <= REFERENCE_RADIUS_KM)
    )
    reference_xy = to_utm11_km(native_longitude[reference_mask], native_latitude[reference_mask])
    prediction = predict_local_vertical(
        model,
        table[["east_km", "north_km"]].to_numpy(float),
        table["up_mm"].to_numpy(float), table["sigma_up_mm"].to_numpy(float),
        reference_xy,
    )
    usable = prediction.valid
    if int(usable.sum()) < 25:
        raise RuntimeError(f"{track}: insufficient all-station local reference predictions")
    look_up = product["los_u"][reference_mask][usable]
    raw = look_up * prediction.mean_mm[usable]
    raw_sigma = np.abs(look_up) * prediction.sigma_mm[usable]
    return float(np.nanmedian(raw)), float(np.nanmedian(raw_sigma)), int(usable.sum())


vertical_outputs: dict[str, dict[str, np.ndarray | float | int]] = {}
for track in TRACKS:
    table = event_tables[track]
    prediction = predict_local_vertical(
        selected_models[track],
        table[["east_km", "north_km"]].to_numpy(float),
        table["up_mm"].to_numpy(float), table["sigma_up_mm"].to_numpy(float),
        targets_xy,
    )
    field = common_fields[track]
    u_mean = prediction.mean_mm.reshape(east_grid.shape)
    u_sigma = prediction.sigma_mm.reshape(east_grid.shape)
    local_valid = prediction.valid.reshape(east_grid.shape) & field["valid"]
    reference_value, reference_sigma, reference_count = native_reference_summary(track, selected_models[track], table)
    sign = int(phase1_manifest["tracks"][track]["sign_audit"]["selected_insar_sign"])
    corrected, vlos, vlos_sigma, _, _ = correct_vertical_los_on_grid(
        sign * field["referenced_los_mm"], field["los_u"], u_mean, u_sigma,
        reference_value_mm=reference_value, reference_sigma_mm=reference_sigma,
    )
    for item in (u_mean, u_sigma, corrected, vlos, vlos_sigma):
        item[~local_valid] = np.nan
    vertical_outputs[track] = {
        "vertical_mm": u_mean, "vertical_sigma_mm": u_sigma,
        "vertical_los_mm": vlos, "vertical_los_sigma_mm": vlos_sigma,
        "corrected_los_mm": corrected,
        "support_count": prediction.support_count.reshape(east_grid.shape),
        "support_radius_km": prediction.support_radius_km.reshape(east_grid.shape),
        "sector_count": prediction.occupied_sector_count.reshape(east_grid.shape),
        "local_valid": local_valid,
        "reference_vertical_los_mm": reference_value,
        "reference_vertical_los_sigma_mm": reference_sigma,
        "reference_prediction_count": reference_count,
        "sign": sign,
    }
    print(track, "valid all-station local predictions:", int(local_valid.sum()), "median support:", float(np.nanmedian(prediction.support_count[prediction.valid])))

# %% [markdown]
# ## 5. Correct each LOS separately, then solve E–N and off-fault strain
#
# The following map is a 4–16 July earthquake-sequence increment. It is not a
# purely coseismic or exactly simultaneous two-track observation. Strain is
# calculated only after the E–N solution and only outside the rupture-safe
# exclusion distance.

# %%
def base_los_sigma(track: str) -> float:
    ramp = float(phase1_manifest["tracks"][track]["insar_ramp_scale_mm"])
    return float(max(TRACKS[track]["noise_floor_mm"], ramp))


asc, desc = vertical_outputs["ascending_T64"], vertical_outputs["descending_T71"]
solution = solve_two_track_horizontal(
    asc["corrected_los_mm"], desc["corrected_los_mm"],
    common_fields["ascending_T64"]["los_e"], common_fields["ascending_T64"]["los_n"],
    common_fields["descending_T71"]["los_e"], common_fields["descending_T71"]["los_n"],
    np.full(east_grid.shape, base_los_sigma("ascending_T64")),
    np.full(east_grid.shape, base_los_sigma("descending_T71")),
    vertical_los_sigma_ascending_mm=asc["vertical_los_sigma_mm"],
    vertical_los_sigma_descending_mm=desc["vertical_los_sigma_mm"],
    vertical_correlation=1.0,
    max_condition_number=8.0,
)

rupture_segments = load_rupture_segments_utm(FAULT_FILE, certain_only=True)
distance = rupture_point_distance_lower_bound_km(targets_xy, rupture_segments).reshape(east_grid.shape)
safe = solution.valid & (distance > STRAIN_SAFE_DISTANCE_KM)
sample_lattice = np.zeros(east_grid.shape, dtype=bool)
sample_lattice[::2, ::2] = True
sample_mask = safe & sample_lattice
target_lattice = np.column_stack([east_grid[::4, ::4].ravel(), north_grid[::4, ::4].ravel()])
target_distance = rupture_point_distance_lower_bound_km(target_lattice, rupture_segments)
target_lattice = target_lattice[target_distance > STRAIN_SAFE_DISTANCE_KM]
strain = rmls_incremental_strain(
    np.column_stack([east_grid[sample_mask], north_grid[sample_mask]]),
    solution.east_mm[sample_mask], solution.north_mm[sample_mask],
    solution.sigma_east_mm[sample_mask], solution.sigma_north_mm[sample_mask],
    target_lattice,
    support_radius_km=STRAIN_SUPPORT_RADIUS_KM,
    bandwidth_km=4.0,
    min_samples=16,
)
strain["rupture_distance_lower_bound_km"] = rupture_point_distance_lower_bound_km(strain[["east_km", "north_km"]].to_numpy(float), rupture_segments)
strain["dilatation_z"] = strain["dilatation_nstrain"] / np.maximum(strain["sigma_dilatation_nstrain"], 1.0e-12)
strain["dilatation_resolved_95pct"] = np.abs(strain["dilatation_z"]) >= 1.96
strain.to_csv(OUTPUT_DIR / "all_station_local_vertical_20190704_20190716_rmls_strain.csv", index=False)
print(
    "E-N valid pixels:", int(solution.valid.sum()),
    "RMLS valid targets:", int(strain["valid"].sum()),
    "95% resolved dilatation targets:", int((strain["valid"] & strain["dilatation_resolved_95pct"]).sum()),
)

# %% [markdown]
# ## 6. Plot the all-station local vertical field, corrected E–N, and uncertainty

# %%
def plot_map(ax: plt.Axes, values: np.ndarray, title: str, label: str, *, positive: bool = False) -> None:
    finite = values[np.isfinite(values)]
    vmax = max(float(np.nanpercentile(np.abs(finite), 98)), 1.0) if len(finite) else 1.0
    cmap = "magma_r" if positive else "RdBu_r"
    norm = None if positive else TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    image = ax.pcolormesh(east_grid, north_grid, values, shading="nearest", cmap=cmap, norm=norm)
    ax.set(title=title, xlabel="UTM 11N easting (km)", ylabel="UTM 11N northing (km)", aspect="equal")
    ax.grid(True, color="0.86", ls="--", lw=0.5)
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(label)


fig, axes = plt.subplots(2, 3, figsize=(17.5, 11.2), constrained_layout=True)
plot_map(axes[0, 0], asc["vertical_mm"], "All-station local U — ascending endpoint", "Vertical increment (mm)")
plot_map(axes[0, 1], desc["vertical_mm"], "All-station local U — descending endpoint", "Vertical increment (mm)")
plot_map(axes[0, 2], asc["support_count"].astype(float), "Ascending local GNSS support", "Contributing stations", positive=True)
plot_map(axes[1, 0], solution.east_mm, "Two-track east increment", "East increment (mm)")
plot_map(axes[1, 1], solution.north_mm, "Two-track north increment", "North increment (mm)")
plot_map(axes[1, 2], solution.sigma_north_mm, "North uncertainty", "1 sigma north (mm)", positive=True)
fig.suptitle("All-station adaptive local vertical correction: U -> lU U -> corrected LOS -> E-N", fontsize=16)
fig.savefig(OUTPUT_DIR / "all_station_local_vertical_to_en_20190704_20190716.png", bbox_inches="tight")
plt.show()


def plot_strain(ax: plt.Axes, field: str, title: str, label: str, *, symmetric: bool = True) -> None:
    accepted = strain.loc[strain["valid"] & np.isfinite(strain[field])]
    values = accepted[field].to_numpy(float)
    limit = max(float(np.nanpercentile(np.abs(values), 98)), 1.0) if len(values) else 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit) if symmetric else None
    image = ax.scatter(
        accepted["east_km"], accepted["north_km"], c=values, s=58, marker="s",
        cmap="RdBu_r" if symmetric else "magma_r", norm=norm, edgecolors="none",
    )
    ax.set(title=title, xlabel="UTM 11N easting (km)", ylabel="UTM 11N northing (km)", aspect="equal")
    ax.grid(True, color="0.86", ls="--", lw=0.5)
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(label)


fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), constrained_layout=True)
plot_strain(axes[0], "dilatation_nstrain", "Off-fault RMLS dilatation", "Dilatation (nstrain)")
plot_strain(axes[1], "sigma_dilatation_nstrain", "Formal dilatation uncertainty", "1 sigma (nstrain)", symmetric=False)
plot_strain(axes[2], "dilatation_z", "Dilatation signal-to-uncertainty", "Dilatation z score")
fig.suptitle("4-16 July all-station vertical correction: strain uncertainty diagnostic, not a resolved strain claim", fontsize=15)
fig.savefig(OUTPUT_DIR / "all_station_local_vertical_rmls_strain_diagnostic.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Save fields, validation evidence, and interpretation guardrails

# %%
np.savez_compressed(
    OUTPUT_DIR / "all_station_local_vertical_20190704_20190716.npz",
    east_km=east_grid, north_km=north_grid, latitude=latitude_grid, longitude=longitude_grid,
    valid=solution.valid, condition_number=solution.condition_number,
    east_mm=solution.east_mm, north_mm=solution.north_mm,
    sigma_east_mm=solution.sigma_east_mm, sigma_north_mm=solution.sigma_north_mm,
    ascending_vertical_mm=asc["vertical_mm"], ascending_vertical_sigma_mm=asc["vertical_sigma_mm"],
    ascending_vertical_los_mm=asc["vertical_los_mm"], ascending_corrected_los_mm=asc["corrected_los_mm"],
    descending_vertical_mm=desc["vertical_mm"], descending_vertical_sigma_mm=desc["vertical_sigma_mm"],
    descending_vertical_los_mm=desc["vertical_los_mm"], descending_corrected_los_mm=desc["corrected_los_mm"],
)
manifest = {
    "status": "all-station, pre-event-validated local vertical correction",
    "vertical_interpolator": "adaptive local ordinary Kriging or local Matérn-3/2 GP; no global plane",
    "selected_models": {track: asdict(model) for track, model in selected_models.items()},
    "all_station_count": int(len(network)),
    "local_support_rule": asdict(CONFIG),
    "pre_event_control_pairs": [[str(start.date()), str(end.date())] for start, end in control_pairs],
    "correction": "h_i = s_i d_i - [lU_i Uhat_i - median_reference(lU_i Uhat_i)]",
    "reference": f"{REFERENCE_STATION}, {REFERENCE_RADIUS_KM:.1f}-km native disk",
    "look_geometry": "native LiCSAR E/N/U look vectors; lU used instead of nominal incidence angle",
    "event_interval": "2019-07-04 to 2019-07-16, separate track UTC endpoints",
    "valid_en_pixels": int(solution.valid.sum()),
    "median_sigma_east_mm": float(np.nanmedian(solution.sigma_east_mm)),
    "median_sigma_north_mm": float(np.nanmedian(solution.sigma_north_mm)),
    "median_condition_number": float(np.nanmedian(solution.condition_number)),
    "rmls_valid_targets": int(strain["valid"].sum()),
    "rmls_dilatation_resolved_95pct_targets": int((strain["valid"] & strain["dilatation_resolved_95pct"]).sum()),
    "limitations": [
        "Ascending and descending dates have a roughly 12-hour acquisition-time separation.",
        "The 4-16 July interval includes both earthquakes and early postseismic deformation.",
        "Two-track LOS geometry weakly resolves north; north and strain uncertainty must accompany every result.",
        "RMLS is evaluated outside an 18 km rupture-safe distance and is an incremental earthquake-sequence strain diagnostic.",
    ],
}
(OUTPUT_DIR / "all_station_local_vertical_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
display(pd.DataFrame([manifest]).T)
