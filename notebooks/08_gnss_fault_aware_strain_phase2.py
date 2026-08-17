# %% [markdown]
# # Phase II — Fault-aware GNSS horizontal incremental strain
#
# This notebook is the defensible continuation of Phase I.  Phase I verified
# the InSAR LOS polarity and geometry but did **not** resolve a spatial GNSS
# vertical field.  It therefore does not produce a pixelwise "pure horizontal"
# InSAR raster, and this notebook does not differentiate InSAR or a corrected
# HLOS raster.
#
# Instead, it uses the independently observed horizontal GNSS increments to
# estimate **network-scale, off-fault incremental strain**.  It deliberately
# replaces ordinary kriging / a conventional RMLS map with a fault-aware
# triangular finite-element estimate:
#
# 1. GNSS endpoints are sampled at each track's actual SAR acquisition times;
# 2. a Delaunay triangle carries one local affine EN displacement field;
# 3. every triangle touching a conservative mapped-rupture buffer is masked;
# 4. weak triangle geometry is masked; and
# 5. a duration-matched, pre-event GNSS control is reported alongside the
#    earthquake-sequence increment.
#
# Thus the result is an **incremental** strain/rotation diagnostic for the
# 12-day earthquake sequence, not a strain-rate map, a pixel-scale InSAR
# product, a near-fault tensor, or a claim about preparation before an event.

# %% [markdown]
# ## Method and fixed quality gates
#
# At the centroid of each triangle, we fit the local horizontal displacement
# field exactly through its three GNSS vertices:
#
# \[
# u_E=a_E+\frac{\partial u_E}{\partial x}(x-x_c)+
# \frac{\partial u_E}{\partial y}(y-y_c),\qquad
# u_N=a_N+\frac{\partial u_N}{\partial x}(x-x_c)+
# \frac{\partial u_N}{\partial y}(y-y_c).
# \]
#
# The incremental strain and rotation are then
#
# \[
# \epsilon_{xx}=\partial u_E/\partial x,\quad
# \epsilon_{yy}=\partial u_N/\partial y,\quad
# \epsilon_{xy}=\tfrac12(\partial u_E/\partial y+\partial u_N/\partial x),
# \]
# \[
# \Delta=\epsilon_{xx}+\epsilon_{yy},\qquad
# \omega=\tfrac12(\partial u_N/\partial x-\partial u_E/\partial y).
# \]
#
# The following gates are declared before plotting values:
#
# - mapped rupture lower-distance bound must exceed **10 km**;
# - longest triangle edge must be <= **60 km**;
# - minimum interior angle must be >= **15 degrees**;
# - triangle Jacobian condition number must be <= **10**.
#
# The 10 km rupture buffer is used as a conservative no-inference zone, not as
# a physical width of the fault.  A 0.25 km trace-sampling resolution enlarges
# the mask slightly at ambiguous edges; it cannot promote a questionable
# near-fault triangle to a valid result.

# %%
from __future__ import annotations

from pathlib import Path
import json
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd


def find_repository_root(start: Path) -> Path:
    """Find the repository root in VS Code or Jupyter execution."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Run from inside the ridgecrest-insar repository.")


ROOT = find_repository_root(Path.cwd().resolve())
sys.path.insert(0, str(ROOT / "src"))

from ridgecrest_gnss_strain import (  # noqa: E402
    finite_element_triangle_strain,
    load_rupture_segments_utm,
    to_utm11_km,
)
from ridgecrest_vertical_los import (  # noqa: E402
    gnss_interval_table,
    load_gnss_network,
)


mpl.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    }
)
print(f"Repository: {ROOT}")

# %% [markdown]
# ## Inputs and immutable configuration
#
# The event endpoints are the same actual UTC SAR epochs used in Phase I.  The
# two tracks remain separate because their endpoint times differ by about
# twelve hours.  Daily GNSS solutions cannot resolve the intra-day rupture;
# `gnss_interval_table` uses a pre-event trend for the 4 July endpoint and a
# post-Mw 7.1 local trend for the 16 July endpoint, never interpolation through
# an earthquake step.

# %%
GNSS_ROOT = Path(r"D:\Uni\Thises\GNSS_ridgecrest\data\tenv_data")
FAULT_FILE = ROOT / "data" / "cgs_2019_ridgecrest_fault_ruptures.geojson"
OUTPUT_DIR = ROOT / "outputs" / "gnss_fault_aware_strain_phase2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

M64_TIME = pd.Timestamp("2019-07-04T17:33:49")
M71_TIME = pd.Timestamp("2019-07-06T03:19:53")
EVENT_TIMES = (M64_TIME, M71_TIME)

# Fixed before inspecting strain results.
RUPTURE_BUFFER_KM = 10.0
RUPTURE_MASK_SAMPLE_SPACING_KM = 0.25
MAX_TRIANGLE_EDGE_KM = 60.0
MIN_TRIANGLE_ANGLE_DEG = 15.0
MAX_TRIANGLE_CONDITION = 10.0
VECTOR_DISPLAY_KM_PER_MM = 0.04  # graphical scale only; 100 mm = 4 km arrow.

TRACKS = {
    "ascending_T64": {
        "label": "Track 64 ascending",
        "event_start": pd.Timestamp("2019-07-04T01:50:08.490464"),
        "event_end": pd.Timestamp("2019-07-16T01:50:08.490464"),
        "control_start": pd.Timestamp("2019-06-10T01:50:08.490464"),
        "control_end": pd.Timestamp("2019-06-22T01:50:08.490464"),
    },
    "descending_T71": {
        "label": "Track 71 descending",
        "event_start": pd.Timestamp("2019-07-04T13:51:41.812911"),
        "event_end": pd.Timestamp("2019-07-16T13:51:41.812911"),
        "control_start": pd.Timestamp("2019-06-10T13:51:41.812911"),
        "control_end": pd.Timestamp("2019-06-22T13:51:41.812911"),
    },
}

for path in (GNSS_ROOT, FAULT_FILE):
    if not path.exists():
        raise FileNotFoundError(path)

print(json.dumps({"quality_gates": {
    "rupture_buffer_km": RUPTURE_BUFFER_KM,
    "maximum_edge_km": MAX_TRIANGLE_EDGE_KM,
    "minimum_angle_deg": MIN_TRIANGLE_ANGLE_DEG,
    "maximum_condition_number": MAX_TRIANGLE_CONDITION,
}}, indent=2))

# %% [markdown]
# ## 1. Build track-specific GNSS increments and a pre-event control
#
# The control has the same nominal 12-day duration as the earthquake-sequence
# increment, but it is entirely before the July earthquakes.  It is a GNSS
# negative control—not a matched InSAR interferogram—and is used only to show
# the scale of the network's ordinary short-window variability.

# %%
histories, station_metadata = load_gnss_network(GNSS_ROOT)


def common_stations(*tables: pd.DataFrame) -> list[str]:
    shared = set(tables[0]["station"])
    for table in tables[1:]:
        shared &= set(table["station"])
    if len(shared) < 10:
        raise RuntimeError(f"Only {len(shared)} stations overlap all required intervals")
    return sorted(shared)


def select_stations(table: pd.DataFrame, stations: list[str]) -> pd.DataFrame:
    return (
        table.set_index("station")
        .loc[stations]
        .reset_index()
        .copy()
    )


intervals: dict[str, dict[str, pd.DataFrame]] = {}
raw_tables: list[pd.DataFrame] = []
for track, config in TRACKS.items():
    event_table = gnss_interval_table(
        histories,
        station_metadata,
        start=config["event_start"],
        end=config["event_end"],
        event_times=EVENT_TIMES,
        strict=False,
    )
    control_table = gnss_interval_table(
        histories,
        station_metadata,
        start=config["control_start"],
        end=config["control_end"],
        event_times=EVENT_TIMES,
        strict=False,
    )
    intervals[track] = {"event": event_table, "control": control_table}
    raw_tables.extend([event_table, control_table])

SHARED_STATIONS = common_stations(*raw_tables)
for track in TRACKS:
    for interval_name in ("event", "control"):
        intervals[track][interval_name] = select_stations(
            intervals[track][interval_name], SHARED_STATIONS
        )

print(f"Common GNSS stations: {len(SHARED_STATIONS)}")
print(", ".join(SHARED_STATIONS))
for track, config in TRACKS.items():
    print(
        f"{config['label']}: event {config['event_start']} -> {config['event_end']} | "
        f"control {config['control_start']} -> {config['control_end']}"
    )

# %% [markdown]
# ## 2. Build the fixed finite-element geometry and conservative rupture mask
#
# A finite element estimates displacement derivatives only within its triangle.
# It never predicts across a mapped rupture.  The map will retain rejected
# triangles in gray, so the spatial-resolution limit is visible rather than
# hidden.

# %%
geometry_table = intervals["descending_T71"]["event"]
station_xy_km = to_utm11_km(
    geometry_table["longitude"].to_numpy(),
    geometry_table["latitude"].to_numpy(),
)

# Loading the complete certain CGS trace ensures a triangle cannot appear valid
# just because a local plotting subset was chosen.  The finite-element routine
# uses a conservative 0.25 km trace-sampling safety mask for speed and safety.
rupture_segments_xy_km = load_rupture_segments_utm(FAULT_FILE, certain_only=True)
print(f"Mapped certain rupture segments: {len(rupture_segments_xy_km):,}")


def triangle_table(displacements: pd.DataFrame) -> pd.DataFrame:
    """Run the fixed, fault-aware finite-element estimator for one interval."""
    return finite_element_triangle_strain(
        station_xy_km,
        displacements["east_mm"].to_numpy(),
        displacements["north_mm"].to_numpy(),
        displacements["sigma_east_mm"].to_numpy(),
        displacements["sigma_north_mm"].to_numpy(),
        station_ids=displacements["station"].to_numpy(),
        rupture_segments_xy_km=rupture_segments_xy_km,
        rupture_buffer_km=RUPTURE_BUFFER_KM,
        rupture_mask_sample_spacing_km=RUPTURE_MASK_SAMPLE_SPACING_KM,
        max_edge_km=MAX_TRIANGLE_EDGE_KM,
        min_angle_deg=MIN_TRIANGLE_ANGLE_DEG,
        max_condition_number=MAX_TRIANGLE_CONDITION,
    )


triangles: dict[str, dict[str, pd.DataFrame]] = {}
for track in TRACKS:
    triangles[track] = {
        interval_name: triangle_table(intervals[track][interval_name])
        for interval_name in ("event", "control")
    }
    for interval_name, result in triangles[track].items():
        result.to_csv(
            OUTPUT_DIR / f"{track}_{interval_name}_triangle_strain.csv",
            index=False,
        )
        print(
            f"{track} {interval_name}: {int(result['valid'].sum())}/{len(result)} "
            "off-fault, well-conditioned triangles"
        )

# %% [markdown]
# ## 3. Plot the accepted and withheld finite elements
#
# Green triangles pass all gates. Orange triangles are too near a mapped surface
# rupture; gray triangles fail their geometric-resolution gate.  They are not
# data gaps to be filled by interpolation.

# %%
def triangle_vertices(row: pd.Series) -> np.ndarray:
    return station_xy_km[[
        int(row["station_index_1"]),
        int(row["station_index_2"]),
        int(row["station_index_3"]),
    ]]


def add_ruptures(ax: plt.Axes, *, stride: int = 8) -> None:
    # The CGS trace is exceptionally dense.  The subset is only a display
    # thinning; all segments are used for the scientific safety mask above.
    ax.add_collection(
        LineCollection(
            rupture_segments_xy_km[::stride], colors="black", linewidths=0.22,
            alpha=0.55, zorder=3,
        )
    )


def set_map_extent(ax: plt.Axes) -> None:
    padding = 6.0
    ax.set_xlim(station_xy_km[:, 0].min() - padding, station_xy_km[:, 0].max() + padding)
    ax.set_ylim(station_xy_km[:, 1].min() - padding, station_xy_km[:, 1].max() + padding)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.86", linewidth=0.6, linestyle="--")
    ax.set_xlabel("UTM 11N easting (km)")
    ax.set_ylabel("UTM 11N northing (km)")


def draw_quality_map(ax: plt.Axes, table: pd.DataFrame, title: str) -> None:
    for state, colour, label in (
        (table["valid"], "#51a66a", "Accepted off-fault triangle"),
        (table["within_rupture_buffer"], "#e39d48", "Masked: rupture buffer"),
        (~table["passes_geometry"], "#a9adb3", "Masked: geometry"),
    ):
        selected = table.loc[state]
        if selected.empty:
            continue
        collection = PolyCollection(
            [triangle_vertices(row) for _, row in selected.iterrows()],
            facecolors=colour, edgecolors="white", linewidths=0.65,
            alpha=0.70, label=label, zorder=1,
        )
        ax.add_collection(collection)
    add_ruptures(ax)
    ax.scatter(station_xy_km[:, 0], station_xy_km[:, 1], c="black", s=28, zorder=5)
    for station, point in zip(SHARED_STATIONS, station_xy_km):
        ax.annotate(station, point, xytext=(3, 3), textcoords="offset points", fontsize=7)
    set_map_extent(ax)
    ax.set_title(title)


fig, axes = plt.subplots(1, 2, figsize=(15, 7.8), constrained_layout=True)
for axis, (track, config) in zip(axes, TRACKS.items()):
    draw_quality_map(axis, triangles[track]["event"], config["label"])
handles, labels = axes[0].get_legend_handles_labels()
unique = dict(zip(labels, handles))
fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=3, frameon=True)
fig.suptitle(
    "Fault-aware GNSS finite-element domain: 4–16 July earthquake-sequence increment",
    fontsize=16,
)
fig.savefig(OUTPUT_DIR / "01_fault_aware_strain_domain.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Plot off-fault incremental strain, rotation, and GNSS vectors
#
# Color is assigned only to triangles that passed every gate.  Values are in
# nanostrain (or nanoradians for rotation).  The formal 95% uncertainty applies
# only to propagated GNSS endpoint uncertainty; it does not validate an
# unresolved sub-triangle deformation pattern.

# %%
def robust_symmetric_limit(values: list[np.ndarray]) -> float:
    joined = np.concatenate([value[np.isfinite(value)] for value in values])
    if len(joined) == 0:
        return 1.0
    return float(max(np.quantile(np.abs(joined), 0.96), 1.0))


event_tables = [triangles[track]["event"] for track in TRACKS]
dilatation_limit = robust_symmetric_limit([
    table.loc[table["valid"], "dilatation_nstrain"].to_numpy()
    for table in event_tables
])
rotation_limit = robust_symmetric_limit([
    table.loc[table["valid"], "rotation_nrad"].to_numpy()
    for table in event_tables
])
principal_limit = robust_symmetric_limit([
    table.loc[table["valid"], "principal_max_nstrain"].to_numpy()
    for table in event_tables
])


def draw_value_map(
    ax: plt.Axes,
    table: pd.DataFrame,
    field: str,
    label: str,
    norm: Normalize,
    cmap: str = "RdBu_r",
) -> mpl.cm.ScalarMappable:
    colour_map = mpl.colormaps[cmap]
    invalid = table.loc[~table["valid"]]
    if not invalid.empty:
        ax.add_collection(
            PolyCollection(
                [triangle_vertices(row) for _, row in invalid.iterrows()],
                facecolors="0.88", edgecolors="white", linewidths=0.55,
                alpha=0.72, zorder=1,
            )
        )
    valid = table.loc[table["valid"] & np.isfinite(table[field])]
    if not valid.empty:
        ax.add_collection(
            PolyCollection(
                [triangle_vertices(row) for _, row in valid.iterrows()],
                facecolors=[colour_map(norm(value)) for value in valid[field]],
                edgecolors="white", linewidths=0.7, zorder=2,
            )
        )
    add_ruptures(ax)
    set_map_extent(ax)
    ax.set_title(label)
    return mpl.cm.ScalarMappable(norm=norm, cmap=colour_map)


fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
component_specs = (
    ("dilatation_nstrain", "Dilatation (nstrain)", TwoSlopeNorm(vcenter=0.0, vmin=-dilatation_limit, vmax=dilatation_limit)),
    ("rotation_nrad", "Incremental rotation (nrad)", TwoSlopeNorm(vcenter=0.0, vmin=-rotation_limit, vmax=rotation_limit)),
    ("principal_max_nstrain", "Maximum principal increment (nstrain)", Normalize(vmin=-principal_limit, vmax=principal_limit)),
)
for row, (track, config) in enumerate(TRACKS.items()):
    displacement = intervals[track]["event"]
    for column, (field, title, norm) in enumerate(component_specs):
        axis = axes[row, column]
        mappable = draw_value_map(
            axis, triangles[track]["event"], field,
            f"{config['label']} — {title}", norm,
        )
        quiver = axis.quiver(
            station_xy_km[:, 0], station_xy_km[:, 1],
            displacement["east_mm"].to_numpy() * VECTOR_DISPLAY_KM_PER_MM,
            displacement["north_mm"].to_numpy() * VECTOR_DISPLAY_KM_PER_MM,
            angles="xy", scale_units="xy", scale=1.0, width=0.0032,
            headwidth=3.7, color="0.16", zorder=6,
        )
        if row == 0:
            colorbar = fig.colorbar(mappable, ax=axis, fraction=0.046, pad=0.03)
            colorbar.set_label(title, fontsize=11)
        if column == 0:
            axis.quiverkey(
                quiver, X=0.06, Y=0.05, U=100.0 * VECTOR_DISPLAY_KM_PER_MM,
                label="100 mm GNSS increment", labelpos="E", coordinates="axes",
                fontproperties={"size": 10}, color="0.16",
            )
fig.suptitle(
    "GNSS-only off-fault incremental strain: 4–16 July earthquake sequence\n"
    "Gray triangles fail the fault/geometry gates and are not interpreted",
    fontsize=16,
)
fig.savefig(OUTPUT_DIR / "02_off_fault_incremental_strain.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Cross-track and pre-event-control diagnostics
#
# The ascending and descending endpoint windows differ slightly, so this is a
# consistency diagnostic rather than an independent replication.  The control
# comparison makes no claim of a null field; it simply checks whether the
# earthquake-sequence increments are much larger than ordinary 12-day GNSS
# variability at the same network scale.

# %%
def merge_triangle_values(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    field: str,
    sigma_field: str,
) -> pd.DataFrame:
    columns = ["triangle_index", "valid", field, sigma_field]
    a = left[columns].rename(columns={
        "valid": "valid_left", field: "left", sigma_field: "sigma_left",
    })
    b = right[columns].rename(columns={
        "valid": "valid_right", field: "right", sigma_field: "sigma_right",
    })
    merged = a.merge(b, on="triangle_index", how="inner")
    merged = merged.loc[merged["valid_left"] & merged["valid_right"]].copy()
    merged["difference"] = merged["left"] - merged["right"]
    merged["sigma_difference"] = np.hypot(merged["sigma_left"], merged["sigma_right"])
    merged["formal_z_difference"] = merged["difference"] / np.maximum(
        merged["sigma_difference"], 1.0e-12
    )
    return merged


event_consistency = merge_triangle_values(
    triangles["ascending_T64"]["event"],
    triangles["descending_T71"]["event"],
    field="dilatation_nstrain", sigma_field="sigma_dilatation_nstrain",
)
control_consistency = merge_triangle_values(
    triangles["ascending_T64"]["control"],
    triangles["descending_T71"]["control"],
    field="dilatation_nstrain", sigma_field="sigma_dilatation_nstrain",
)
event_consistency.to_csv(OUTPUT_DIR / "event_track_dilatation_consistency.csv", index=False)
control_consistency.to_csv(OUTPUT_DIR / "control_track_dilatation_consistency.csv", index=False)


def event_control_values(track: str) -> pd.DataFrame:
    event = triangles[track]["event"]
    control = triangles[track]["control"]
    merged = event[["triangle_index", "valid", "dilatation_nstrain"]].merge(
        control[["triangle_index", "valid", "dilatation_nstrain"]],
        on="triangle_index", suffixes=("_event", "_control"),
    )
    return merged.loc[merged["valid_event"] & merged["valid_control"]].copy()


fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
for axis, table, title in (
    (axes[0], event_consistency, "Earthquake-sequence: Ascending vs descending"),
    (axes[1], control_consistency, "Pre-event control: Ascending vs descending"),
):
    if len(table):
        axis.scatter(table["right"], table["left"], s=44, c="#2673b8", edgecolor="white", linewidth=0.6)
        limits = np.nanmax(np.abs(table[["left", "right"]].to_numpy())) * 1.1
        axis.plot([-limits, limits], [-limits, limits], "--", color="0.45", linewidth=1.0)
        axis.set_xlim(-limits, limits)
        axis.set_ylim(-limits, limits)
        correlation = float(np.corrcoef(table["right"], table["left"])[0, 1]) if len(table) > 1 else np.nan
        axis.text(0.03, 0.97, f"n = {len(table)}\nr = {correlation:.2f}", transform=axis.transAxes,
                  va="top", bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.90})
    axis.axhline(0.0, color="0.75", linewidth=0.7)
    axis.axvline(0.0, color="0.75", linewidth=0.7)
    axis.grid(True, color="0.88", linestyle="--")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Descending dilatation (nstrain)")
    axis.set_ylabel("Ascending dilatation (nstrain)")
    axis.set_title(title)

for track, color in (("ascending_T64", "#2070b4"), ("descending_T71", "#c94b51")):
    comparison = event_control_values(track)
    axes[2].scatter(
        comparison["dilatation_nstrain_control"],
        comparison["dilatation_nstrain_event"],
        s=46, alpha=0.86, color=color, edgecolor="white", linewidth=0.5,
        label=TRACKS[track]["label"],
    )
axes[2].axhline(0.0, color="0.75", linewidth=0.7)
axes[2].axvline(0.0, color="0.75", linewidth=0.7)
axes[2].grid(True, color="0.88", linestyle="--")
axes[2].set_xlabel("Pre-event control dilatation (nstrain)")
axes[2].set_ylabel("Earthquake-sequence dilatation (nstrain)")
axes[2].set_title("Duration-matched GNSS control")
axes[2].legend(frameon=True, loc="best")
fig.savefig(OUTPUT_DIR / "03_track_consistency_and_control.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Export a compact, reproducible validation summary
#
# Read this summary before interpreting maps.  The most important outcome is
# not the largest colored triangle: it is the number of triangles surviving the
# predeclared fault and geometry gates.  A high formal Z-score means only that
# GNSS endpoint noise is small compared with the triangle difference; it does
# not resolve unmodelled fault-zone deformation or upgrade the product to an
# InSAR pixel-scale result.

# %%
summary_rows: list[dict[str, object]] = []
for track, config in TRACKS.items():
    for interval_name in ("event", "control"):
        table = triangles[track][interval_name]
        accepted = table.loc[table["valid"]].copy()
        formal_z = accepted["dilatation_nstrain"] / np.maximum(
            accepted["sigma_dilatation_nstrain"], 1.0e-12
        )
        summary_rows.append(
            {
                "track": track,
                "track_label": config["label"],
                "interval": interval_name,
                "start_utc": str(config[f"{interval_name}_start"]),
                "end_utc": str(config[f"{interval_name}_end"]),
                "station_count": len(SHARED_STATIONS),
                "triangles_total": len(table),
                "triangles_accepted": int(table["valid"].sum()),
                "triangles_masked_rupture": int(table["within_rupture_buffer"].sum()),
                "triangles_failed_geometry": int((~table["passes_geometry"]).sum()),
                "median_abs_dilatation_nstrain": float(np.median(np.abs(accepted["dilatation_nstrain"]))) if len(accepted) else np.nan,
                "median_sigma_dilatation_nstrain": float(np.median(accepted["sigma_dilatation_nstrain"])) if len(accepted) else np.nan,
                "median_abs_formal_z_dilatation": float(np.median(np.abs(formal_z))) if len(accepted) else np.nan,
                "uncertainty_scope": "GNSS endpoint uncertainty only; no sub-triangle model error",
                "interpretation_scope": "off-fault GNSS network-scale incremental strain only",
            }
        )

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_DIR / "phase2_validation_summary.csv", index=False)

manifest = {
    "phase": "GNSS fault-aware incremental strain",
    "status": "diagnostic off-fault GNSS product",
    "not_a_pixelwise_insar_strain_product": True,
    "not_a_pure_horizontal_los_product": True,
    "not_a_strain_rate_product": True,
    "not_a_precursory_analysis": True,
    "vertical_phase1_conclusion": (
        "The ten-station vertical field did not pass out-of-sample spatial-resolution "
        "validation; its constant-field sensitivity correction must not be interpreted "
        "as pure horizontal LOS."
    ),
    "strain_method": "fault-aware Delaunay linear finite elements",
    "quality_gates": {
        "rupture_buffer_km": RUPTURE_BUFFER_KM,
        "rupture_mask_sample_spacing_km": RUPTURE_MASK_SAMPLE_SPACING_KM,
        "max_triangle_edge_km": MAX_TRIANGLE_EDGE_KM,
        "min_triangle_angle_deg": MIN_TRIANGLE_ANGLE_DEG,
        "max_triangle_condition_number": MAX_TRIANGLE_CONDITION,
    },
    "event_time_scope": "12-day earthquake-sequence increment; not purely coseismic",
    "track_endpoint_timing": {
        key: {
            "event_start": str(value["event_start"]),
            "event_end": str(value["event_end"]),
        }
        for key, value in TRACKS.items()
    },
    "common_station_count": len(SHARED_STATIONS),
    "recommended_next_stage": (
        "Jointly invert native ascending/descending LOS and GNSS ENU observations "
        "using the full LOS look vector, then derive posterior model-predicted "
        "off-fault strain; do not subtract an unresolved interpolated vertical field."
    ),
}
(OUTPUT_DIR / "phase2_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

display(summary)
print(f"Saved results to: {OUTPUT_DIR}")
