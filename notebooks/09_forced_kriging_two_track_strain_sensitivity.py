# %% [markdown]
# # Phase III — Forced-Kriging vertical sensitivity, two-track EN, and strain
#
# This notebook completes the requested algebraic sequence for the 4–16 July
# 2019 *earthquake-sequence* interval:
#
# \[
# \widehat U(x,y) \;\longrightarrow\; l_U(x,y)\widehat U(x,y)
# \;\longrightarrow\; h_{\rm asc},h_{\rm desc}
# \;\longrightarrow\; E,N \;\longrightarrow\; \boldsymbol{\epsilon}.
# \]
#
# It is deliberately a **sensitivity experiment**, not the paper's primary
# result.  Phase I tested ordinary kriging, Matérn GP, and a constant vertical
# field with leave-one-GNSS-station-out prediction.  The spatial models did not
# beat the constant field.  This notebook nevertheless forces the best ordinary
# Kriging candidate to answer a narrow diagnostic question:
#
# > How much would a plausible, but unvalidated, spatial vertical field change
# > the two-track E–N solution and its off-fault strain sensitivity?
#
# It cannot establish a resolved vertical field, pure horizontal LOS, a
# pixel-scale strain measurement, a purely coseismic product, or a pre-event
# signal.

# %% [markdown]
# ## Fixed equations and protection against common mistakes
#
# For each common-grid analysis pixel and track \(i\), vertical correction is
# applied with the *pixel-specific* LiCSAR look-up component:
#
# \[
# h_i=s_i d_{i,\rm LOS}-
# \left[l_{Ui}\widehat U_i-
# {\rm median}_{R}(l_{Ui}\widehat U_i)\right].
# \]
#
# The reference disk \(R\) is the same P597 disk used for the InSAR maps.  The
# corrected ascending and descending observations are then decomposed as
#
# \[
# \begin{bmatrix}h_a\\h_d\end{bmatrix}=
# \begin{bmatrix}l_{Ea}&l_{Na}\\l_{Ed}&l_{Nd}\end{bmatrix}
# \begin{bmatrix}E\\N\end{bmatrix}.
# \]
#
# The two source grids are offset by fractional pixels, so they are explicitly
# sampled onto a common **1 km UTM 11N analysis grid**.  They are never aligned
# by array index.  Strain is later fit at 4 km output spacing using an 8 km
# fault-safe local support; there is no finite-differencing of native 100 m
# pixels.

# %%
from __future__ import annotations

from pathlib import Path
import json
import sys

from IPython.display import display
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd


def find_repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Run from inside the ridgecrest-insar repository.")


ROOT = find_repository_root(Path.cwd().resolve())
sys.path.insert(0, str(ROOT / "src"))

from ridgecrest_gnss_strain import (  # noqa: E402
    load_rupture_segments_utm,
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
    SpatialModel,
    gnss_interval_table,
    haversine_km,
    load_gnss_network,
    predict_vertical_field,
    select_station_circle,
)


mpl.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
    }
)

# %% [markdown]
# ## 1. Inputs and predeclared sensitivity settings
#
# The forced ordinary-Kriging model is the best *ordinary-Kriging* candidate in
# the Phase-I score table, not a post-hoc choice from the E–N maps.  Its 10 km
# range and 5 mm nugget are held fixed for both tracks.  The Phase-I score table
# is copied into the output folder so the failed spatial-validation gate stays
# alongside every sensitivity figure.

# %%
GNSS_ROOT = Path(r"D:\Uni\Thises\GNSS_ridgecrest\data\tenv_data")
PHASE1_DIR = ROOT / "outputs" / "gnss_vertical_los_phase1"
FAULT_FILE = ROOT / "data" / "cgs_2019_ridgecrest_fault_ruptures.geojson"
OUTPUT_DIR = ROOT / "outputs" / "forced_kriging_two_track_sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRODUCTS = {
    "ascending_T64": PHASE1_DIR / "ascending_T64_20190704_20190716_vertical_corrected_hlos.npz",
    "descending_T71": PHASE1_DIR / "descending_T71_20190704_20190716_vertical_corrected_hlos.npz",
}
for path in (*PRODUCTS.values(), GNSS_ROOT, FAULT_FILE):
    if not path.exists():
        raise FileNotFoundError(path)

M64 = {"latitude": 35.705, "longitude": -117.504, "time": pd.Timestamp("2019-07-04T17:33:49")}
M71 = {"latitude": 35.770, "longitude": -117.599, "time": pd.Timestamp("2019-07-06T03:19:53")}
EVENT_TIMES = (M64["time"], M71["time"])

TRACKS = {
    "ascending_T64": {
        "label": "Track 64 ascending",
        "start": pd.Timestamp("2019-07-04T01:50:08.490464"),
        "end": pd.Timestamp("2019-07-16T01:50:08.490464"),
        "insar_noise_floor_mm": 5.0,
    },
    "descending_T71": {
        "label": "Track 71 descending",
        "start": pd.Timestamp("2019-07-04T13:51:41.812911"),
        "end": pd.Timestamp("2019-07-16T13:51:41.812911"),
        "insar_noise_floor_mm": 5.0,
    },
}

# Fixed sensitivity configuration.
FORCED_OK_RANGE_KM = 10.0
FORCED_OK_NUGGET_MM = 5.0
# One kilometre sampling is deliberately finer than the 1.5-km P597 reference
# disk, so that the disk contains several analysis cells before re-referencing.
# It is still much coarser than the native approximately 100-m LiCSAR pixels.
COMMON_GRID_SPACING_KM = 1.0
MAX_EN_CONDITION_NUMBER = 8.0
VERTICAL_CORRELATION = 1.0  # conservative shared-GNSS vertical-error bound
RUPTURE_BUFFER_KM = 10.0
STRAIN_SUPPORT_RADIUS_KM = 8.0
STRAIN_BANDWIDTH_KM = 4.0
STRAIN_SAFE_DISTANCE_KM = RUPTURE_BUFFER_KM + STRAIN_SUPPORT_RADIUS_KM

phase1_manifest = json.loads((PHASE1_DIR / "phase1_manifest.json").read_text(encoding="utf-8"))
products = {track: np.load(path) for track, path in PRODUCTS.items()}
for track, product in products.items():
    if bool(product["vertical_field_resolved"]):
        raise RuntimeError("This sensitivity notebook assumes the Phase-I vertical field was unresolved")
    print(track, "Phase-I vertical field resolved?", bool(product["vertical_field_resolved"]))

# %% [markdown]
# ## 2. Rebuild the fixed P595-centred GNSS circle and exact track endpoints
#
# The vertical field is estimated separately for each track interval because
# their acquisition times are not identical.  The 4 July endpoints are before
# Mw 6.4 and are estimated from the pre-event daily-GNSS trend; no daily GNSS
# coordinate is interpolated through the event step.

# %%
histories, network = load_gnss_network(GNSS_ROOT)
centre_station, circle_radius_km, circle = select_station_circle(
    network,
    event_latitude=M64["latitude"],
    event_longitude=M64["longitude"],
    station_count=10,
    margin_km=0.20,
)
reference_station = network.set_index("station").loc[phase1_manifest["reference_station"]]
print("Circle centre:", centre_station["station"], f"radius={circle_radius_km:.3f} km")
print("Reference station:", reference_station.name)

gnss_intervals: dict[str, pd.DataFrame] = {}
for track, config in TRACKS.items():
    table = gnss_interval_table(
        histories,
        circle,
        start=config["start"],
        end=config["end"],
        event_times=EVENT_TIMES,
        strict=True,
    )
    gnss_intervals[track] = table
    table.to_csv(OUTPUT_DIR / f"{track}_ten_station_enu_interval.csv", index=False)
    display(table[["station", "east_mm", "north_mm", "up_mm", "sigma_up_mm"]])

# %% [markdown]
# ## 3. Build a common 1 km UTM analysis grid and resample the native LOS fields
#
# The source grids are approximately 100 m and offset from one another.  We
# retain a target only when both source interpolations have essentially complete
# bilinear support, are coherent/valid, and lie inside the Phase-I GNSS convex
# hull.  Thus no value outside the vertical-network support enters the E–N
# sensitivity solution.

# %%
east_grid, north_grid, latitude_grid, longitude_grid, in_geographic_overlap = common_utm11_grid(
    [products[track]["latitude"] for track in TRACKS],
    [products[track]["longitude"] for track in TRACKS],
    spacing_km=COMMON_GRID_SPACING_KM,
)
target_xy = np.column_stack([east_grid.ravel(), north_grid.ravel()])
print("Common analysis grid:", east_grid.shape, "cells", east_grid.size)


def resample_phase1(track: str) -> dict[str, np.ndarray]:
    product = products[track]
    source_valid = product["valid"].astype(bool) & product["inside_gnss_hull"].astype(bool)
    fields: dict[str, np.ndarray] = {}
    support_fields: list[np.ndarray] = []
    for name in ("referenced_los_mm", "los_e", "los_n", "los_u"):
        sampled, support = masked_bilinear_resample(
            product["latitude"], product["longitude"], product[name], source_valid,
            latitude_grid, longitude_grid,
        )
        fields[name] = sampled
        support_fields.append(support)
    fields["support"] = np.minimum.reduce(support_fields)
    fields["valid"] = in_geographic_overlap & (fields["support"] >= 0.999)
    fields["los_e"], fields["los_n"], fields["los_u"] = normalize_look_vectors(
        fields["los_e"], fields["los_n"], fields["los_u"]
    )
    fields["valid"] &= (
        np.isfinite(fields["referenced_los_mm"])
        & np.isfinite(fields["los_e"])
        & np.isfinite(fields["los_n"])
        & np.isfinite(fields["los_u"])
    )
    for name in ("referenced_los_mm", "los_e", "los_n", "los_u"):
        fields[name][~fields["valid"]] = np.nan
    return fields


common_fields = {track: resample_phase1(track) for track in TRACKS}
for track, fields in common_fields.items():
    print(track, "common valid/hull support:", int(fields["valid"].sum()), "/", fields["valid"].size)

# %% [markdown]
# ## 4. Force ordinary Kriging at every common-grid pixel, then project vertically
#
# Kriging is carried out **before** projection.  The predicted vertical field
# differs slightly by track because the GNSS interval endpoints differ.  Its
# prediction uncertainty is retained and treated as perfectly correlated
# between tracks during the E–N covariance calculation; that is conservative
# for a shared GNSS-derived vertical field.

# %%
def forced_kriging_model(table: pd.DataFrame) -> SpatialModel:
    sill = float(max(np.var(table["up_mm"].to_numpy(float), ddof=1), 1.0))
    return SpatialModel(
        method="ordinary_kriging",
        length_scale_km=FORCED_OK_RANGE_KM,
        nugget_mm=FORCED_OK_NUGGET_MM,
        sill_mm2=sill,
        loo_rmse_mm=np.nan,
        loo_mae_mm=np.nan,
        loo_nlpd=np.nan,
        loo_standardized_rms=np.nan,
    )


vertical_products: dict[str, dict[str, np.ndarray | float]] = {}
for track, config in TRACKS.items():
    station_table = gnss_intervals[track]
    station_xy = to_utm11_km(
        station_table["longitude"].to_numpy(), station_table["latitude"].to_numpy()
    )
    u_mean, u_sigma = predict_vertical_field(
        forced_kriging_model(station_table),
        station_xy,
        station_table["up_mm"].to_numpy(float),
        station_table["sigma_up_mm"].to_numpy(float),
        target_xy,
    )
    u_mean = u_mean.reshape(east_grid.shape)
    u_sigma = u_sigma.reshape(east_grid.shape)
    fields = common_fields[track]
    # Preserve the actual native-resolution P597 reference disk.  A one-km
    # common grid may have only a few quality-controlled cells in this small
    # disk, but the native LiCSAR product has many; reference the *projected*
    # vertical term there before combining tracks on the common grid.
    native = products[track]
    native_latitude, native_longitude = np.meshgrid(
        native["latitude"], native["longitude"], indexing="ij"
    )
    native_reference_mask = (
        haversine_km(
            native_latitude, native_longitude,
            float(reference_station["latitude"]), float(reference_station["longitude"]),
        ) <= float(phase1_manifest["reference_disk_radius_km"])
    ) & native["valid"].astype(bool) & native["inside_gnss_hull"].astype(bool)
    if int(native_reference_mask.sum()) < 25:
        raise RuntimeError(f"{track}: insufficient native reference samples")
    native_xy = to_utm11_km(
        native_longitude[native_reference_mask], native_latitude[native_reference_mask]
    )
    native_u_mean, native_u_sigma = predict_vertical_field(
        forced_kriging_model(station_table),
        station_xy,
        station_table["up_mm"].to_numpy(float),
        station_table["sigma_up_mm"].to_numpy(float),
        native_xy,
    )
    native_raw_vertical_los = native["los_u"][native_reference_mask] * native_u_mean
    native_raw_vertical_los_sigma = np.abs(native["los_u"][native_reference_mask]) * native_u_sigma
    native_reference_value = float(np.nanmedian(native_raw_vertical_los))
    native_reference_sigma = float(np.nanmedian(native_raw_vertical_los_sigma))
    corrected, vertical_los, vertical_los_sigma, reference_value, reference_sigma = correct_vertical_los_on_grid(
        fields["referenced_los_mm"], fields["los_u"], u_mean, u_sigma,
        reference_value_mm=native_reference_value,
        reference_sigma_mm=native_reference_sigma,
    )
    corrected[~fields["valid"]] = np.nan
    vertical_los[~fields["valid"]] = np.nan
    vertical_los_sigma[~fields["valid"]] = np.nan
    vertical_products[track] = {
        "vertical_mm": u_mean,
        "vertical_sigma_mm": u_sigma,
        "vertical_los_mm": vertical_los,
        "vertical_los_sigma_mm": vertical_los_sigma,
        "corrected_los_mm": corrected,
        "reference_vertical_los_mm": reference_value,
        "reference_vertical_los_sigma_mm": reference_sigma,
        "native_reference_sample_count": int(native_reference_mask.sum()),
    }
    print(
        f"{track}: forced-OK U p2–p98 = {np.nanpercentile(u_mean, [2, 98])}; "
        f"median sigmaU={np.nanmedian(u_sigma):.2f} mm"
    )

# %% [markdown]
# ## 5. Solve two-track E–N and compare it with the constant-U scenario
#
# Both models use the same regridded raw LOS and look vectors.  This isolates
# only the effect of forcing a spatial vertical predictor.  The two track times
# still differ by approximately 12 hours, so the output is labelled a
# near-synchronous 4–16 July sensitivity field—not an exactly common-time
# observation.

# %%
# The actual 12-hour interval difference inferred from daily-GNSS endpoint
# estimates is retained as a small additional per-LOS uncertainty.
all_network_intervals = {
    track: gnss_interval_table(
        histories, network,
        start=config["start"], end=config["end"], event_times=EVENT_TIMES,
        strict=False,
    ).set_index("station")
    for track, config in TRACKS.items()
}
common_network = all_network_intervals["ascending_T64"].index.intersection(
    all_network_intervals["descending_T71"].index
)
timing_difference = all_network_intervals["ascending_T64"].loc[common_network, ["east_mm", "north_mm", "up_mm"]] - all_network_intervals["descending_T71"].loc[common_network, ["east_mm", "north_mm", "up_mm"]]
TIMING_MISMATCH_MM = float(np.sqrt(np.mean(np.sum(np.square(timing_difference.to_numpy(float)), axis=1))))
print(f"GNSS endpoint timing-difference RMS: {TIMING_MISMATCH_MM:.3f} mm")


def base_insar_sigma(track: str) -> float:
    item = phase1_manifest["tracks"][track]
    return float(max(TRACKS[track]["insar_noise_floor_mm"], item["insar_ramp_scale_mm"]))


ascending = common_fields["ascending_T64"]
descending = common_fields["descending_T71"]
forced_a = vertical_products["ascending_T64"]
forced_d = vertical_products["descending_T71"]

forced_solution = solve_two_track_horizontal(
    forced_a["corrected_los_mm"], forced_d["corrected_los_mm"],
    ascending["los_e"], ascending["los_n"], descending["los_e"], descending["los_n"],
    np.full(east_grid.shape, np.hypot(base_insar_sigma("ascending_T64"), TIMING_MISMATCH_MM)),
    np.full(east_grid.shape, np.hypot(base_insar_sigma("descending_T71"), TIMING_MISMATCH_MM)),
    vertical_los_sigma_ascending_mm=forced_a["vertical_los_sigma_mm"],
    vertical_los_sigma_descending_mm=forced_d["vertical_los_sigma_mm"],
    vertical_correlation=VERTICAL_CORRELATION,
    max_condition_number=MAX_EN_CONDITION_NUMBER,
)

constant_los_a, _ = masked_bilinear_resample(
    products["ascending_T64"]["latitude"], products["ascending_T64"]["longitude"],
    products["ascending_T64"]["hlos_mm"],
    products["ascending_T64"]["valid"].astype(bool) & products["ascending_T64"]["inside_gnss_hull"].astype(bool),
    latitude_grid, longitude_grid,
)
constant_los_d, _ = masked_bilinear_resample(
    products["descending_T71"]["latitude"], products["descending_T71"]["longitude"],
    products["descending_T71"]["hlos_mm"],
    products["descending_T71"]["valid"].astype(bool) & products["descending_T71"]["inside_gnss_hull"].astype(bool),
    latitude_grid, longitude_grid,
)
constant_solution = solve_two_track_horizontal(
    constant_los_a, constant_los_d,
    ascending["los_e"], ascending["los_n"], descending["los_e"], descending["los_n"],
    np.full(east_grid.shape, base_insar_sigma("ascending_T64")),
    np.full(east_grid.shape, base_insar_sigma("descending_T71")),
    max_condition_number=MAX_EN_CONDITION_NUMBER,
)

common_valid = forced_solution.valid & constant_solution.valid
delta_east = forced_solution.east_mm - constant_solution.east_mm
delta_north = forced_solution.north_mm - constant_solution.north_mm
for array in (delta_east, delta_north):
    array[~common_valid] = np.nan
print("Forced-OK minus constant-U E p2–p98 (mm):", np.nanpercentile(delta_east, [2, 98]))
print("Forced-OK minus constant-U N p2–p98 (mm):", np.nanpercentile(delta_north, [2, 98]))
print("Median propagated sigmaE/sigmaN (mm):", np.nanmedian(forced_solution.sigma_east_mm), np.nanmedian(forced_solution.sigma_north_mm))
print("Median two-track condition number:", np.nanmedian(forced_solution.condition_number))

# %% [markdown]
# ## 6. Visualize the forced vertical fields and the E–N sensitivity
#
# The first row proves that Kriging has now been explicitly applied at every
# common-grid analysis pixel.  The final two panels show why the output remains
# a sensitivity study: the north solution changes materially when the
# unvalidated vertical predictor is exchanged for the constant model.

# %%
def map_extent(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("UTM 11N easting (km)")
    ax.set_ylabel("UTM 11N northing (km)")
    ax.grid(True, color="0.86", linewidth=0.55, linestyle="--")


def draw_map(ax: plt.Axes, data: np.ndarray, title: str, label: str, *, symmetric: bool = True) -> None:
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        raise RuntimeError(f"No finite values for {title}")
    limit = max(float(np.nanpercentile(np.abs(finite), 98)), 1.0)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit) if symmetric else None
    image = ax.pcolormesh(east_grid, north_grid, data, shading="nearest", cmap="RdBu_r", norm=norm)
    map_extent(ax)
    ax.set_title(title)
    colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label(label)


fig, axes = plt.subplots(2, 3, figsize=(17.5, 11.2), constrained_layout=True)
draw_map(axes[0, 0], vertical_products["ascending_T64"]["vertical_mm"], "Forced OK ΔU — ascending interval", "Vertical increment (mm)")
draw_map(axes[0, 1], vertical_products["descending_T71"]["vertical_mm"], "Forced OK ΔU — descending interval", "Vertical increment (mm)")
draw_map(axes[0, 2], forced_a["vertical_los_mm"], "Forced OK vertical-to-LOS term — ascending", "LOS contribution (mm)")
draw_map(axes[1, 0], forced_solution.east_mm, "Two-track east increment (forced OK)", "East increment (mm)")
draw_map(axes[1, 1], forced_solution.north_mm, "Two-track north increment (forced OK)", "North increment (mm)")
draw_map(axes[1, 2], delta_north, "Forced OK − constant-U north", "North difference (mm)")
fig.suptitle(
    "Forced ordinary-Kriging sensitivity: vertical → LOS → two-track E–N\n"
    "Not a validated pure-horizontal or common-time E–N product",
    fontsize=16,
)
fig.savefig(OUTPUT_DIR / "01_forced_kriging_vertical_to_en_sensitivity.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. 2-D off-fault strain sensitivity using RMLS—not native-pixel differences
#
# The RMLS output is generated only outside a conservative **18 km** mapped
# rupture exclusion zone: 10 km physical no-inference buffer plus the full 8 km
# local fitting support.  This ensures that a local fit cannot use samples from
# across a mapped displacement discontinuity.  It is a 4 km output-grid,
# kilometre-scale sensitivity field—not a 100 m strain map.

# %%
rupture_segments = load_rupture_segments_utm(FAULT_FILE, certain_only=True)
distance_lower = rupture_point_distance_lower_bound_km(target_xy, rupture_segments).reshape(east_grid.shape)
safe_for_strain = forced_solution.valid & (distance_lower > STRAIN_SAFE_DISTANCE_KM)

# To prevent false precision, use every second 1-km analysis cell as the RMLS
# input and a disjoint 4-km output lattice.  Vertical projection itself was
# nevertheless calculated on *every* valid 1-km common-grid pixel above.
sample_lattice = np.zeros(east_grid.shape, dtype=bool)
sample_lattice[::2, ::2] = True
sample_mask = safe_for_strain & sample_lattice
sample_xy = np.column_stack([east_grid[sample_mask], north_grid[sample_mask]])
strain_targets = np.column_stack([east_grid[::4, ::4].ravel(), north_grid[::4, ::4].ravel()])
target_distance = rupture_point_distance_lower_bound_km(strain_targets, rupture_segments)
strain_targets = strain_targets[target_distance > STRAIN_SAFE_DISTANCE_KM]

strain = rmls_incremental_strain(
    sample_xy,
    forced_solution.east_mm[sample_mask],
    forced_solution.north_mm[sample_mask],
    forced_solution.sigma_east_mm[sample_mask],
    forced_solution.sigma_north_mm[sample_mask],
    strain_targets,
    support_radius_km=STRAIN_SUPPORT_RADIUS_KM,
    bandwidth_km=STRAIN_BANDWIDTH_KM,
    min_samples=16,
)
strain["rupture_distance_lower_bound_km"] = rupture_point_distance_lower_bound_km(
    strain[["east_km", "north_km"]].to_numpy(), rupture_segments
)
strain["formal_z_dilatation"] = strain["dilatation_nstrain"] / np.maximum(
    strain["sigma_dilatation_nstrain"], 1.0e-12
)
strain.to_csv(OUTPUT_DIR / "forced_kriging_rmls_strain_sensitivity.csv", index=False)
print("RMLS strain targets accepted by numerical geometry:", int(strain["valid"].sum()), "/", len(strain))


def plot_strain_scatter(ax: plt.Axes, field: str, label: str, *, symmetric: bool = True) -> None:
    accepted = strain.loc[strain["valid"] & np.isfinite(strain[field])]
    value = accepted[field].to_numpy(float)
    limit = max(float(np.nanpercentile(np.abs(value), 98)), 1.0) if len(value) else 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit) if symmetric else Normalize(vmin=0.0, vmax=limit)
    scatter = ax.scatter(
        accepted["east_km"], accepted["north_km"], c=value, s=60,
        cmap="RdBu_r" if symmetric else "magma_r", norm=norm,
        marker="s", edgecolors="none", zorder=2,
    )
    # Display thinning only; all rupture segments enter the safety distance.
    ax.add_collection(LineCollection(rupture_segments[::8], colors="black", linewidths=0.20, alpha=0.45, zorder=3))
    map_extent(ax)
    ax.set_title(label)
    colorbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label(label.split("(")[0].strip())


fig, axes = plt.subplots(1, 3, figsize=(17, 5.7), constrained_layout=True)
plot_strain_scatter(axes[0], "dilatation_nstrain", "RMLS dilatation sensitivity (nstrain)")
plot_strain_scatter(axes[1], "rotation_nrad", "RMLS rotation sensitivity (nrad)")
plot_strain_scatter(axes[2], "sigma_dilatation_nstrain", "Formal σ dilatation (nstrain)", symmetric=False)
fig.suptitle(
    "Forced-Kriging two-track RMLS strain sensitivity — off fault only\n"
    "Conditional uncertainty; not a validated observed strain field",
    fontsize=15,
)
fig.savefig(OUTPUT_DIR / "02_forced_kriging_rmls_strain_sensitivity.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Save uncertainty and scope checks
#
# The notebook exits with a summary rather than silently promoting the result.
# The principal decision point is whether the forced-Kriging versus constant-U
# difference is small relative to the inferred E–N field and its uncertainty.
# A large north difference means the requested strain map is a model-scenario
# sensitivity, not a stable measurement.

# %%
summary = {
    "status": "forced ordinary-Kriging sensitivity only",
    "phase1_spatial_vertical_validation": "failed; constant field selected",
    "ordinary_kriging_sensitivity_parameters": {
        "range_km": FORCED_OK_RANGE_KM,
        "nugget_mm": FORCED_OK_NUGGET_MM,
    },
    "analysis_grid_spacing_km": COMMON_GRID_SPACING_KM,
    "common_grid_shape": list(east_grid.shape),
    "paired_valid_common_grid_cells": int(forced_solution.valid.sum()),
    "median_two_track_condition_number": float(np.nanmedian(forced_solution.condition_number)),
    "median_sigma_east_mm": float(np.nanmedian(forced_solution.sigma_east_mm)),
    "median_sigma_north_mm": float(np.nanmedian(forced_solution.sigma_north_mm)),
    "forced_minus_constant_east_p2_p98_mm": [float(value) for value in np.nanpercentile(delta_east, [2, 98])],
    "forced_minus_constant_north_p2_p98_mm": [float(value) for value in np.nanpercentile(delta_north, [2, 98])],
    "gnss_endpoint_timing_difference_rms_mm": TIMING_MISMATCH_MM,
    "vertical_correlation_assumption": VERTICAL_CORRELATION,
    "strain_input_spacing_km": 2.0 * COMMON_GRID_SPACING_KM,
    "strain_output_spacing_km": 4.0 * COMMON_GRID_SPACING_KM,
    "strain_support_radius_km": STRAIN_SUPPORT_RADIUS_KM,
    "strain_safe_distance_from_mapped_rupture_km": STRAIN_SAFE_DISTANCE_KM,
    "rmls_targets_valid": int(strain["valid"].sum()),
    "critical_limitations": [
        "The ten-station vertical field did not pass leave-one-out spatial validation.",
        "Ascending and descending endpoint times differ by about 12 hours.",
        "Two-track InSAR weakly constrains north motion; vertical-model uncertainty strongly projects into north.",
        "RMLS conditional uncertainty does not remove spatial correlation or source-model uncertainty.",
        "This 4–16 July interval includes earthquakes and early postseismic deformation; it is not a purely coseismic interval.",
    ],
    "allowed_interpretation": "Scenario sensitivity to a forced vertical interpolation, useful for method comparison only.",
    "not_allowed_interpretation": "A validated pure-horizontal InSAR displacement or observed two-dimensional strain product.",
}
(OUTPUT_DIR / "sensitivity_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "sensitivity_summary.csv", index=False)

for score in PHASE1_DIR.glob("*_vertical_model_scores.csv"):
    pd.read_csv(score).to_csv(OUTPUT_DIR / score.name, index=False)

display(pd.DataFrame([summary]).T)
print(f"Saved Phase III sensitivity outputs to: {OUTPUT_DIR}")
