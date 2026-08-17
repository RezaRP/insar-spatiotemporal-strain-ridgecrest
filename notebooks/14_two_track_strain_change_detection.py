# %% [markdown]
# # Two-track horizontal strain and leakage-safe change detection
#
# This notebook continues the validated processing chain
#
# \[
# d_{\mathrm{LOS}}=l_EE+l_NN+l_UU,\qquad
# h=d_{\mathrm{LOS}}-l_U\widehat U,
# \]
#
# and solves the two horizontal observations jointly:
#
# \[
# \begin{bmatrix}h_a\\h_d\end{bmatrix}
# =
# \begin{bmatrix}l_{E,a}&l_{N,a}\\l_{E,d}&l_{N,d}\end{bmatrix}
# \begin{bmatrix}E\\N\end{bmatrix}.
# \]
#
# A single LOS observation is never reprojected into a two-component vector.
# The vertical field is an external GNSS correction; this remains a **2-D
# horizontal** inversion, not a 3-D InSAR inversion.
#
# Local derivatives are the small-strain tensor and vertical-axis rotation:
#
# \[
# \epsilon_{EE}=\frac{\partial E}{\partial x},\quad
# \epsilon_{NN}=\frac{\partial N}{\partial y},\quad
# \epsilon_{EN}=\frac12\left(\frac{\partial E}{\partial y}
# +\frac{\partial N}{\partial x}\right),
# \]
#
# \[
# \delta=\epsilon_{EE}+\epsilon_{NN},\qquad
# \omega=\frac12\left(\frac{\partial N}{\partial x}
# -\frac{\partial E}{\partial y}\right).
# \]
#
# References: Gudmundsson et al. (2002), Wright et al. (2004), Shen et al.
# (2015), Nichols and Holmes (2002), and Page (1954).

# %%
from __future__ import annotations

from pathlib import Path
import gc
import json
import sys

import h5py
from IPython.display import Image, display
import matplotlib as mpl
RUNNING_NOTEBOOK = "get_ipython" in globals()
if not RUNNING_NOTEBOOK:
    mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from pyproj import Transformer


def find_repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Run from inside the ridgecrest-insar repository.")


ROOT = find_repository_root(Path.cwd().resolve())
sys.path.insert(0, str(ROOT / "src"))

from ridgecrest_gnss_strain import load_rupture_segments_utm  # noqa: E402
from ridgecrest_jump import distance_km  # noqa: E402
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
    rupture_point_distance_lower_bound_km,
    to_utm11_km,
)

mpl.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 320,
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }
)

# %% [markdown]
# ## 1. Frozen inputs and interpretation gates
#
# Notebook 11 already performed the pixel-specific vertical projection,
# ascending--descending horizontal inversion, propagated uncertainty, and RMLS
# derivative estimation. Here those products are audited and screened. The
# event never enters baseline fitting or threshold calibration.

# %%
SOURCE_DIR = ROOT / "outputs" / "two_track_vertical_corrected_timeseries"
SOURCE_NPZ = SOURCE_DIR / "two_track_vertical_corrected_en_timeseries.npz"
STRAIN_CSV = SOURCE_DIR / "two_track_rmls_incremental_strain_timeseries.csv"
SOURCE_MANIFEST = SOURCE_DIR / "two_track_vertical_corrected_timeseries_manifest.json"
DESC_H5 = ROOT / "data" / "cum_full_scene_no_GACOS.h5"
FAULT_FILE = ROOT / "data" / "cgs_2019_ridgecrest_fault_ruptures.geojson"
OUTPUT_DIR = ROOT / "outputs" / "two_track_strain_change_detection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for item in (SOURCE_NPZ, STRAIN_CSV, SOURCE_MANIFEST, DESC_H5, FAULT_FILE):
    if not item.exists():
        raise FileNotFoundError(item)

source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
if not bool(source_manifest["off_fault_vertical_gates_passed"]):
    raise RuntimeError(
        "The off-fault vertical validation gate failed; strain detection is not authorized."
    )

fields = np.load(SOURCE_NPZ)
strain = pd.read_csv(STRAIN_CSV, parse_dates=["start_date", "end_date"])
intervals = (
    strain[["interval_index", "start_date", "end_date", "duration_days"]]
    .drop_duplicates()
    .sort_values("interval_index")
    .reset_index(drop=True)
)
if not np.array_equal(intervals["interval_index"].to_numpy(), np.arange(len(intervals))):
    raise RuntimeError("Strain interval indices are not complete and ordered")

BASELINE_END = pd.Timestamp("2019-05-29")
SURVEILLANCE_START = pd.Timestamp("2019-05-29")
PRE_EVENT_END = pd.Timestamp("2019-07-04")
EVENT_END = pd.Timestamp("2019-07-16")
MIN_BASELINE_OBSERVATIONS = 40
CLUSTER_Z_THRESHOLD = 1.96
CLUSTER_MIN_CELLS = 4
CUSUM_REFERENCE = 0.5
ENERGY_QUANTILE = 0.90

baseline_mask = (intervals["end_date"] <= BASELINE_END).to_numpy()
pre_event_surveillance = (
    (intervals["start_date"] >= SURVEILLANCE_START)
    & (intervals["end_date"] <= PRE_EVENT_END)
).to_numpy()
event_mask = (
    (intervals["start_date"] == PRE_EVENT_END)
    & (intervals["end_date"] == EVENT_END)
).to_numpy()
if baseline_mask.sum() < MIN_BASELINE_OBSERVATIONS:
    raise RuntimeError("Pre-event baseline is too short")
if pre_event_surveillance.sum() != 3 or event_mask.sum() != 1:
    raise RuntimeError("The prespecified surveillance or event intervals are missing")
if intervals.loc[baseline_mask, "end_date"].max() >= pd.Timestamp("2019-07-04T17:33:49"):
    raise RuntimeError("Temporal leakage: earthquake data entered baseline fitting")

event_index = int(np.flatnonzero(event_mask)[0])
asc_subtraction_error = float(
    np.nanmax(
        np.abs(
            fields["ascending_observed_los_increment_mm"]
            - fields["ascending_vertical_to_los_increment_mm"]
            - fields["ascending_pure_horizontal_los_increment_mm"]
        )
    )
)
desc_subtraction_error = float(
    np.nanmax(
        np.abs(
            fields["descending_observed_los_increment_mm"]
            - fields["descending_vertical_to_los_increment_mm"]
            - fields["descending_pure_horizontal_los_increment_mm"]
        )
    )
)
if max(asc_subtraction_error, desc_subtraction_error) > 1.0e-3:
    raise RuntimeError("Vertical subtraction identity failed")

print("Intervals:", len(intervals), "baseline:", int(baseline_mask.sum()))
print("Prespecified pre-event surveillance intervals:", int(pre_event_surveillance.sum()))
print("Event interval index:", event_index)
print("Maximum subtraction errors (ascending, descending), mm:", asc_subtraction_error, desc_subtraction_error)
print("Interpretation gate:", source_manifest["status"])

# %% [markdown]
# ## 2. Why the near-fault gaps exist
#
# Three different exclusions must not be conflated:
#
# 1. native InSAR pixels rejected by coherence, residual, loop-closure, or
#    finite-increment quality controls;
# 2. paired E--N cells lost because either ascending or descending HLOS is
#    unavailable; and
# 3. the deliberate 18-km strain-safe distance (10-km rupture buffer plus the
#    8-km RMLS derivative support).
#
# These gaps are not filled. Smooth interpolation across a rupture would turn a
# displacement discontinuity into a manufactured gradient.

# %%
rupture_segments = load_rupture_segments_utm(FAULT_FILE, certain_only=True)
distance_grid = np.asarray(fields["distance_to_mapped_rupture_km"], dtype=float)
asc_valid = np.isfinite(fields["ascending_pure_horizontal_los_increment_mm"][event_index])
desc_valid = np.isfinite(fields["descending_pure_horizontal_los_increment_mm"][event_index])
en_valid = np.asarray(fields["valid_increment"][event_index], dtype=bool)
strain_domain = en_valid & (distance_grid > 18.0)

gap_category = np.zeros(en_valid.shape, dtype=np.int8)
gap_category[asc_valid & ~desc_valid] = 1
gap_category[~asc_valid & desc_valid] = 2
gap_category[asc_valid & desc_valid & ~en_valid] = 3
gap_category[en_valid & (distance_grid <= 18.0)] = 4
gap_category[strain_domain] = 5

category_labels = {
    0: "Neither HLOS / outside shared support",
    1: "Ascending only (descending missing)",
    2: "Descending only (ascending missing)",
    3: "Both HLOS but invalid E-N geometry",
    4: "Valid E-N; intentional <=18 km strain exclusion",
    5: "Supported off-fault strain domain",
}
common_gap_rows = [
    {
        "scope": "common_1km_event_grid",
        "metric": category_labels[category],
        "count": int(np.count_nonzero(gap_category == category)),
    }
    for category in category_labels
]
common_gap_rows.extend(
    [
        {"scope": "common_1km_event_grid", "metric": "total_cells", "count": int(en_valid.size)},
        {"scope": "common_1km_event_grid", "metric": "ascending_HLOS_finite", "count": int(asc_valid.sum())},
        {"scope": "common_1km_event_grid", "metric": "descending_HLOS_finite", "count": int(desc_valid.sum())},
        {"scope": "common_1km_event_grid", "metric": "paired_EN_valid", "count": int(en_valid.sum())},
    ]
)

distance_band_rows = []
for label_text, lower, upper in (
    ("0-5 km", 0.0, 5.0),
    ("5-10 km", 5.0, 10.0),
    ("10-18 km", 10.0, 18.0),
    (">18 km", 18.0, np.inf),
):
    use = (distance_grid > lower if lower > 0 else distance_grid >= lower) & (distance_grid <= upper)
    distance_band_rows.append(
        {
            "scope": "common_1km_distance_band",
            "metric": label_text,
            "count": int(use.sum()),
            "paired_en_count": int((use & en_valid).sum()),
            "paired_en_percent": float(100.0 * (use & en_valid).sum() / max(use.sum(), 1)),
        }
    )

# Native descending failure attribution for the 4-16 July interval.
with h5py.File(DESC_H5, "r") as handle:
    desc_dates = pd.to_datetime(
        np.asarray(handle["imdates"][:], dtype=np.int64).astype(str),
        format="%Y%m%d",
    )
    date_index = {pd.Timestamp(date): index for index, date in enumerate(desc_dates)}
    event_increment = (
        np.asarray(handle["cum"][date_index[EVENT_END]], dtype=float)
        - np.asarray(handle["cum"][date_index[PRE_EVENT_END]], dtype=float)
    )
    ny, nx = event_increment.shape
    latitude_axis = float(handle["corner_lat"][()]) + np.arange(ny) * float(handle["post_lat"][()])
    longitude_axis = float(handle["corner_lon"][()]) + np.arange(nx) * float(handle["post_lon"][()])
    latitude_native, longitude_native = np.meshgrid(latitude_axis, longitude_axis, indexing="ij")
    native_xy = to_utm11_km(longitude_native.ravel(), latitude_native.ravel())
    native_distance = rupture_point_distance_lower_bound_km(
        native_xy, rupture_segments
    ).reshape(event_increment.shape)
    native_conditions = [
        ("nonfinite_4-16_July_increment", np.isfinite(event_increment)),
        ("coherence_below_0.30", np.isfinite(handle["coh_avg"][:]) & (handle["coh_avg"][:] >= 0.30)),
        ("residual_RMS_above_5_mm", np.isfinite(handle["resid_rms"][:]) & (handle["resid_rms"][:] <= 5.0)),
        ("loop_errors_above_10", np.isfinite(handle["n_loop_err"][:]) & (handle["n_loop_err"][:] <= 10)),
        ("gap_count_above_2", np.isfinite(handle["n_gap"][:]) & (handle["n_gap"][:] <= 2)),
        (
            "nonfinite_look_vector",
            np.isfinite(handle["E.geo"][:])
            & np.isfinite(handle["N.geo"][:])
            & np.isfinite(handle["U.geo"][:]),
        ),
    ]

native_gap_rows = []
for radius in (5.0, 18.0):
    within = native_distance <= radius
    remaining = within.copy()
    for failure_name, passes in native_conditions:
        failure = remaining & ~passes
        native_gap_rows.append(
            {
                "scope": f"native_desc_within_{radius:g}km",
                "metric": failure_name,
                "count": int(failure.sum()),
            }
        )
        remaining &= passes
    native_gap_rows.extend(
        [
            {
                "scope": f"native_desc_within_{radius:g}km",
                "metric": "joint_quality_pass",
                "count": int(remaining.sum()),
            },
            {
                "scope": f"native_desc_within_{radius:g}km",
                "metric": "total_pixels",
                "count": int(within.sum()),
            },
        ]
    )

gap_audit = pd.concat(
    [
        pd.DataFrame(common_gap_rows),
        pd.DataFrame(distance_band_rows),
        pd.DataFrame(native_gap_rows),
    ],
    ignore_index=True,
    sort=False,
)
gap_audit.to_csv(OUTPUT_DIR / "gap_mask_audit.csv", index=False)
display(gap_audit.fillna(""))

# %%
to_geographic = Transformer.from_crs("EPSG:32611", "EPSG:4326", always_xy=True)


def plot_rupture(axis: plt.Axes, **kwargs: object) -> None:
    for segment in rupture_segments:
        longitude, latitude = to_geographic.transform(segment[:, 0] * 1000.0, segment[:, 1] * 1000.0)
        axis.plot(longitude, latitude, **kwargs)


category_colors = ["#d9d9d9", "#377eb8", "#4daf4a", "#984ea3", "#ffbf00", "#1b7837"]
cmap = ListedColormap(category_colors)
norm = BoundaryNorm(np.arange(-0.5, 6.5, 1.0), cmap.N)
fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), constrained_layout=True)
axes[0].pcolormesh(
    fields["longitude"],
    fields["latitude"],
    gap_category,
    cmap=cmap,
    norm=norm,
    shading="auto",
    rasterized=True,
)
axes[0].contour(
    fields["longitude"],
    fields["latitude"],
    distance_grid,
    levels=[18.0],
    colors="black",
    linestyles="--",
    linewidths=1.3,
)
plot_rupture(axes[0], color="black", lw=1.8)
axes[0].set(
    title="(a) 4-16 July paired-geometry and strain masks",
    xlabel="Longitude",
    ylabel="Latitude",
)
axes[0].grid(True, color="0.8", ls=":", lw=0.5)
axes[0].legend(
    handles=[
        Patch(facecolor=category_colors[index], edgecolor="0.3", label=category_labels[index])
        for index in category_labels
    ],
    loc="lower left",
    frameon=True,
    fontsize=9,
)

plot_counts = pd.DataFrame(common_gap_rows[:6])
axes[1].barh(
    np.arange(len(plot_counts)),
    plot_counts["count"],
    color=category_colors,
    edgecolor="0.25",
)
axes[1].set_yticks(np.arange(len(plot_counts)), plot_counts["metric"])
axes[1].invert_yaxis()
axes[1].set(
    title="(b) Common-grid cell accounting",
    xlabel="Number of 1-km cells",
)
axes[1].grid(True, axis="x", color="0.85", ls="--", lw=0.6)
for index, value in enumerate(plot_counts["count"]):
    axes[1].text(value + 70, index, f"{value:,}", va="center", fontsize=10)
fig.suptitle(
    "Near-fault gaps are quality exclusions and an explicit derivative-safe buffer",
    fontsize=15,
    fontweight="bold",
)
fig.savefig(OUTPUT_DIR / "01_gap_and_strain_domain_audit.png", bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "01_gap_and_strain_domain_audit.pdf", bbox_inches="tight")
plt.show()
plt.close(fig)

# %% [markdown]
# ## 3. Interval strain-rate cube and pre-event-only standardization
#
# Because intervals are 6, 12, or 24 days, increments are divided by their
# actual duration before temporal comparison. For target \(s\), component \(c\),
# and interval \(t\),
#
# \[
# z_{sct}=\frac{r_{sct}-\widetilde r_{sc,B}}
# {\sqrt{\sigma_{sct}^{2}+\tau_{sc,B}^{2}}},
# \]
#
# where \(B\) ends on 29 May 2019. The excess scale \(\tau\) is the robust
# baseline variance remaining after removing the median reported variance.

# %%
component_specs = [
    ("epsilon_EE", "epsilon_xx_nstrain", "sigma_epsilon_xx_nstrain", 1.0, "Normal E"),
    ("epsilon_NN", "epsilon_yy_nstrain", "sigma_epsilon_yy_nstrain", 1.0, "Normal N"),
    ("gamma_EN", "gamma_xy_nstrain", "sigma_epsilon_xy_nstrain", 2.0, "Engineering shear"),
    ("dilatation", "dilatation_nstrain", "sigma_dilatation_nstrain", 1.0, "Dilatation"),
    ("rotation", "rotation_nrad", "sigma_rotation_nrad", 1.0, "Rotation"),
]
component_names = [item[0] for item in component_specs]
component_titles = [item[4] for item in component_specs]
targets = (
    strain[["east_km", "north_km"]]
    .drop_duplicates()
    .sort_values(["north_km", "east_km"])
    .reset_index(drop=True)
)
target_lookup = {
    (float(row.east_km), float(row.north_km)): index
    for index, row in targets.iterrows()
}
value_increment = np.full(
    (len(intervals), len(component_specs), len(targets)), np.nan, dtype=float
)
sigma_increment = np.full_like(value_increment, np.nan)

for component_index, (_, value_column, sigma_column, sigma_multiplier, _) in enumerate(component_specs):
    columns = [
        "interval_index",
        "east_km",
        "north_km",
        "valid",
        value_column,
        sigma_column,
    ]
    for row in strain.loc[strain["valid"], columns].itertuples(index=False):
        target_index = target_lookup[(float(row.east_km), float(row.north_km))]
        value_increment[int(row.interval_index), component_index, target_index] = float(
            getattr(row, value_column)
        )
        sigma_increment[int(row.interval_index), component_index, target_index] = (
            sigma_multiplier * float(getattr(row, sigma_column))
        )

rate, sigma_rate = duration_normalize(
    value_increment,
    sigma_increment,
    intervals["duration_days"].to_numpy(float),
)
baseline_model = fit_robust_baseline(
    rate,
    sigma_rate,
    baseline_mask,
    min_observations=MIN_BASELINE_OBSERVATIONS,
)
z = standardized_innovation(rate, sigma_rate, baseline_model)
baseline_loo_z = leave_one_out_baseline_innovations(
    rate,
    sigma_rate,
    baseline_mask,
    min_observations=MIN_BASELINE_OBSERVATIONS,
)
print("RMLS targets:", len(targets))
print("Supported component-target combinations:", int(baseline_model.supported.sum()))
print("Baseline intervals:", int(baseline_mask.sum()), "ending", BASELINE_END.date())

# %% [markdown]
# ## 4. Method A -- uncertainty-standardized maximum-cluster FWER
#
# Sign-consistent 8-neighbour clusters require at least four 4-km target cells
# with \(|z|\ge1.96\). Each baseline interval is scored leave-one-out. The null
# statistic is the maximum cluster mass across all locations, signs, and five
# components. The pre-event family-wise test additionally takes the maximum
# across consecutive three-interval baseline blocks, matching the three
# prespecified surveillance intervals.

# %%
east_target = targets["east_km"].to_numpy(float)
north_target = targets["north_km"].to_numpy(float)
scores_for_null = np.where(baseline_mask[:, None, None], baseline_loo_z, z)

maximum_cluster_mass = np.asarray(
    [
        maximum_signed_cluster_mass(
            scores_for_null[index],
            east_target,
            north_target,
            component_names,
            threshold=CLUSTER_Z_THRESHOLD,
            min_cells=CLUSTER_MIN_CELLS,
        )
        for index in range(len(intervals))
    ],
    dtype=float,
)
baseline_maximum_mass = maximum_cluster_mass[baseline_mask]
pre_block_length = int(pre_event_surveillance.sum())
cluster_block_null = sliding_block_maximum(
    baseline_maximum_mass,
    pre_block_length,
)
pre_cluster_observed = float(np.max(maximum_cluster_mass[pre_event_surveillance]))
pre_cluster_p_fwer = empirical_upper_tail_pvalue(
    cluster_block_null,
    pre_cluster_observed,
)
event_cluster_mass = float(maximum_cluster_mass[event_mask][0])
event_cluster_p_fwer = empirical_upper_tail_pvalue(
    baseline_maximum_mass,
    event_cluster_mass,
)

cluster_rows = []
interval_cluster_rows = []
for index, interval in intervals.iterrows():
    records = signed_spatial_clusters(
        scores_for_null[index],
        east_target,
        north_target,
        component_names,
        threshold=CLUSTER_Z_THRESHOLD,
        min_cells=CLUSTER_MIN_CELLS,
    )
    interval_p = (
        empirical_upper_tail_pvalue(baseline_maximum_mass, maximum_cluster_mass[index])
        if not baseline_mask[index]
        else np.nan
    )
    interval_cluster_rows.append(
        {
            "interval_index": index,
            "start_date": interval.start_date,
            "end_date": interval.end_date,
            "baseline_interval": bool(baseline_mask[index]),
            "pre_event_surveillance": bool(pre_event_surveillance[index]),
            "earthquake_sequence_interval": bool(event_mask[index]),
            "maximum_cluster_mass": maximum_cluster_mass[index],
            "cluster_count": len(records),
            "interval_max_cluster_fwer_p": interval_p,
        }
    )
    for cluster_rank, record in enumerate(records, start=1):
        item = record.as_dict()
        item.update(
            {
                "interval_index": index,
                "start_date": interval.start_date,
                "end_date": interval.end_date,
                "cluster_rank": cluster_rank,
                "baseline_interval": bool(baseline_mask[index]),
                "pre_event_surveillance": bool(pre_event_surveillance[index]),
                "earthquake_sequence_interval": bool(event_mask[index]),
                "interval_max_cluster_fwer_p": interval_p,
            }
        )
        cluster_rows.append(item)

interval_cluster_table = pd.DataFrame(interval_cluster_rows)
cluster_table = pd.DataFrame(cluster_rows)
interval_cluster_table.to_csv(OUTPUT_DIR / "strain_cluster_interval_summary.csv", index=False)
cluster_table.to_csv(OUTPUT_DIR / "strain_spatial_clusters.csv", index=False)

sensitivity_rows = []
for threshold, min_cells in (
    (1.645, 4),
    (1.960, 3),
    (1.960, 4),
    (1.960, 6),
    (2.576, 4),
):
    masses = np.asarray(
        [
            maximum_signed_cluster_mass(
                scores_for_null[index],
                east_target,
                north_target,
                component_names,
                threshold=threshold,
                min_cells=min_cells,
            )
            for index in range(len(intervals))
        ]
    )
    baseline_masses = masses[baseline_mask]
    block_null = sliding_block_maximum(baseline_masses, pre_block_length)
    pre_observed = float(np.max(masses[pre_event_surveillance]))
    event_observed = float(masses[event_mask][0])
    sensitivity_rows.append(
        {
            "z_threshold": threshold,
            "minimum_cluster_cells": min_cells,
            "pre_event_maximum_mass": pre_observed,
            "pre_event_block_fwer_p": empirical_upper_tail_pvalue(block_null, pre_observed),
            "event_interval_maximum_mass": event_observed,
            "event_interval_fwer_p": empirical_upper_tail_pvalue(baseline_masses, event_observed),
        }
    )
sensitivity = pd.DataFrame(sensitivity_rows)
sensitivity.to_csv(OUTPUT_DIR / "strain_cluster_sensitivity.csv", index=False)
display(interval_cluster_table.loc[interval_cluster_table["pre_event_surveillance"] | interval_cluster_table["earthquake_sequence_interval"]])
display(sensitivity)

# %% [markdown]
# ## 5. Method B -- robust strain-energy Page CUSUM
#
# The 90th percentile of \(|z|\) summarizes distributed strain magnitude without
# choosing a fault-side ROI or allowing positive and negative strain to cancel.
# Its baseline center and MAD are estimated from leave-one-out pre-event scores.
# A one-sided Page CUSUM uses reference \(k=0.5\). Consecutive baseline blocks
# with the same length as the tested window preserve the observed temporal
# ordering and define the empirical null.

# %%
energy = strain_energy(z, quantile=ENERGY_QUANTILE)
baseline_loo_energy_full = strain_energy(
    baseline_loo_z,
    quantile=ENERGY_QUANTILE,
)
baseline_loo_energy = baseline_loo_energy_full[baseline_mask]
energy_center = float(np.median(baseline_loo_energy))
energy_scale = float(
    1.4826 * np.median(np.abs(baseline_loo_energy - energy_center))
)
if not np.isfinite(energy_scale) or energy_scale <= 0.0:
    raise RuntimeError("The baseline strain-energy scale is invalid")
energy_standardized = (energy - energy_center) / energy_scale
baseline_energy_standardized = (baseline_loo_energy - energy_center) / energy_scale

pre_cusum = positive_page_cusum(
    energy_standardized[pre_event_surveillance],
    reference=CUSUM_REFERENCE,
)
pre_cusum_observed = float(np.max(pre_cusum))
pre_cusum_null = sliding_block_cusum_maxima(
    baseline_energy_standardized,
    int(pre_event_surveillance.sum()),
    reference=CUSUM_REFERENCE,
)
pre_cusum_p = empirical_upper_tail_pvalue(pre_cusum_null, pre_cusum_observed)

event_cusum_observed = float(
    positive_page_cusum(
        energy_standardized[event_mask],
        reference=CUSUM_REFERENCE,
    )[0]
)
event_cusum_null = sliding_block_cusum_maxima(
    baseline_energy_standardized,
    1,
    reference=CUSUM_REFERENCE,
)
event_cusum_p = empirical_upper_tail_pvalue(
    event_cusum_null,
    event_cusum_observed,
)

surveillance_and_after = intervals["start_date"].ge(SURVEILLANCE_START).to_numpy()
full_cusum = positive_page_cusum(
    energy_standardized[surveillance_and_after],
    reference=CUSUM_REFERENCE,
)
full_cusum_null = sliding_block_cusum_maxima(
    baseline_energy_standardized,
    int(surveillance_and_after.sum()),
    reference=CUSUM_REFERENCE,
)
full_cusum_threshold_95 = float(np.quantile(full_cusum_null, 0.95))

pre_change_supported = bool(
    pre_cluster_p_fwer <= 0.05
    and pre_cusum_p <= 0.05
)
temporal_detection = pd.DataFrame(
    [
        {
            "test": "pre_event_spatial_max_cluster",
            "window": "2019-05-29 to 2019-07-04",
            "statistic": pre_cluster_observed,
            "empirical_p": pre_cluster_p_fwer,
            "passed_0.05": pre_cluster_p_fwer <= 0.05,
        },
        {
            "test": "pre_event_strain_energy_Page_CUSUM",
            "window": "2019-05-29 to 2019-07-04",
            "statistic": pre_cusum_observed,
            "empirical_p": pre_cusum_p,
            "passed_0.05": pre_cusum_p <= 0.05,
        },
        {
            "test": "event_interval_spatial_max_cluster",
            "window": "2019-07-04 to 2019-07-16",
            "statistic": event_cluster_mass,
            "empirical_p": event_cluster_p_fwer,
            "passed_0.05": event_cluster_p_fwer <= 0.05,
        },
        {
            "test": "event_interval_strain_energy_Page_CUSUM",
            "window": "2019-07-04 to 2019-07-16",
            "statistic": event_cusum_observed,
            "empirical_p": event_cusum_p,
            "passed_0.05": event_cusum_p <= 0.05,
        },
    ]
)
temporal_detection.to_csv(OUTPUT_DIR / "strain_temporal_detection_summary.csv", index=False)
display(temporal_detection)
print("Consensus-supported pre-event strain change:", pre_change_supported)

# %% [markdown]
# ## 6. Physical strain-component time series
#
# Curves are spatial medians across the supported off-fault RMLS targets. The
# band is the spatial interquartile range, not a formal confidence interval.
# Nanostrain per day is converted to microstrain per year; rotation is
# correspondingly microradians per year.

# %%
mid_date = intervals["start_date"] + (intervals["end_date"] - intervals["start_date"]) / 2
RATE_TO_MICRO_PER_YEAR = 365.2425 / 1000.0
regional_rows = []
for time_index, interval in intervals.iterrows():
    for component_index, component in enumerate(component_names):
        finite = np.isfinite(rate[time_index, component_index])
        regional_rows.append(
            {
                "interval_index": time_index,
                "start_date": interval.start_date,
                "end_date": interval.end_date,
                "mid_date": mid_date.iloc[time_index],
                "duration_days": interval.duration_days,
                "component": component,
                "valid_targets": int(finite.sum()),
                "median_rate_native_per_day": float(np.nanmedian(rate[time_index, component_index])),
                "q25_rate_native_per_day": float(np.nanquantile(rate[time_index, component_index], 0.25)),
                "q75_rate_native_per_day": float(np.nanquantile(rate[time_index, component_index], 0.75)),
                "median_pointwise_sigma_per_day": float(np.nanmedian(sigma_rate[time_index, component_index])),
                "median_standardized_innovation": float(np.nanmedian(z[time_index, component_index])),
                "q90_abs_standardized_innovation": float(np.nanquantile(np.abs(z[time_index, component_index]), 0.90)),
            }
        )
regional = pd.DataFrame(regional_rows)
regional.to_csv(OUTPUT_DIR / "strain_component_regional_timeseries.csv", index=False)

fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True, constrained_layout=True)
for component_index, (axis, component, title) in enumerate(
    zip(axes, component_names, component_titles)
):
    table = regional.loc[regional["component"] == component].sort_values("interval_index")
    median = table["median_rate_native_per_day"].to_numpy() * RATE_TO_MICRO_PER_YEAR
    lower = table["q25_rate_native_per_day"].to_numpy() * RATE_TO_MICRO_PER_YEAR
    upper = table["q75_rate_native_per_day"].to_numpy() * RATE_TO_MICRO_PER_YEAR
    axis.fill_between(
        table["mid_date"],
        lower,
        upper,
        color="#9ecae1",
        alpha=0.45,
        label="Spatial IQR",
    )
    axis.plot(
        table["mid_date"],
        median,
        color="#08519c",
        lw=1.8,
        marker="o",
        ms=3.5,
        label="Spatial median",
    )
    axis.axhline(0.0, color="0.35", lw=0.8)
    axis.axvspan(SURVEILLANCE_START, PRE_EVENT_END, color="#fee391", alpha=0.35)
    axis.axvspan(PRE_EVENT_END, EVENT_END, color="#fb6a4a", alpha=0.18)
    axis.axvline(PRE_EVENT_END, color="0.25", ls="--", lw=1.0)
    axis.axvline(pd.Timestamp("2019-07-06"), color="0.25", ls=":", lw=1.0)
    unit = r"$\mu$rad yr$^{-1}$" if component == "rotation" else r"$\mu$strain yr$^{-1}$"
    axis.set(title=f"({chr(97 + component_index)}) {title}", ylabel=unit)
    axis.grid(True, color="0.87", ls="--", lw=0.55)
axes[0].legend(loc="upper left", ncol=2)
locator = AutoDateLocator(minticks=5, maxticks=10)
axes[-1].xaxis.set_major_locator(locator)
axes[-1].xaxis.set_major_formatter(ConciseDateFormatter(locator))
axes[-1].set_xlabel("Interval midpoint")
fig.suptitle(
    "Off-fault 2-D horizontal strain-rate time series",
    fontsize=16,
    fontweight="bold",
)
fig.savefig(OUTPUT_DIR / "02_strain_component_timeseries.png", bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "02_strain_component_timeseries.pdf", bbox_inches="tight")
plt.show()
plt.close(fig)

# %% [markdown]
# ## 7. Detection diagnostics over time

# %%
fig, axes = plt.subplots(3, 1, figsize=(14, 10.5), sharex=True, constrained_layout=True)
axes[0].plot(
    mid_date,
    energy,
    color="#54278f",
    lw=1.8,
    marker="o",
    ms=3.5,
)
axes[0].axhline(energy_center, color="0.25", ls="--", lw=1.1, label="Pre-event LOO median")
axes[0].set(
    title=f"(a) Map-level {int(ENERGY_QUANTILE * 100)}th percentile of |standardized strain innovation|",
    ylabel="Robust strain energy",
)
axes[0].legend(loc="upper left")

axes[1].plot(
    mid_date,
    maximum_cluster_mass,
    color="#cb181d",
    lw=1.8,
    marker="o",
    ms=3.5,
)
cluster_threshold_95 = float(np.quantile(baseline_maximum_mass, 0.95))
axes[1].axhline(
    cluster_threshold_95,
    color="0.25",
    ls="--",
    lw=1.1,
    label="95th percentile of pre-event LOO maxima",
)
axes[1].set(
    title="(b) Maximum signed spatial-cluster mass",
    ylabel="Cluster mass",
)
axes[1].legend(loc="upper left")

surveillance_mid = mid_date[surveillance_and_after]
axes[2].plot(
    surveillance_mid,
    full_cusum,
    color="#006d2c",
    lw=2.0,
    marker="o",
    ms=3.5,
)
axes[2].axhline(
    full_cusum_threshold_95,
    color="0.25",
    ls="--",
    lw=1.1,
    label="95th percentile of equal-length baseline blocks",
)
axes[2].set(
    title=(
        "(c) Positive Page CUSUM; "
        f"pre-event p={pre_cusum_p:.3f}, event-only p={event_cusum_p:.3f}"
    ),
    ylabel="CUSUM",
    xlabel="Interval midpoint",
)
axes[2].legend(loc="upper left")

for axis in axes:
    axis.axvspan(SURVEILLANCE_START, PRE_EVENT_END, color="#fee391", alpha=0.35)
    axis.axvspan(PRE_EVENT_END, EVENT_END, color="#fb6a4a", alpha=0.18)
    axis.axvline(PRE_EVENT_END, color="0.25", ls="--", lw=1.0)
    axis.axvline(pd.Timestamp("2019-07-06"), color="0.25", ls=":", lw=1.0)
    axis.grid(True, color="0.87", ls="--", lw=0.55)
locator = AutoDateLocator(minticks=5, maxticks=10)
axes[-1].xaxis.set_major_locator(locator)
axes[-1].xaxis.set_major_formatter(ConciseDateFormatter(locator))
fig.suptitle(
    "Leakage-safe spatial and temporal strain-change diagnostics",
    fontsize=16,
    fontweight="bold",
)
fig.savefig(OUTPUT_DIR / "03_strain_change_detection_timeseries.png", bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "03_strain_change_detection_timeseries.pdf", bbox_inches="tight")
plt.show()
plt.close(fig)

# %% [markdown]
# ## 8. Pre-event and earthquake-sequence strain maps
#
# These maps are off-fault derivative estimates only. The white central region
# is the intentional 18-km strain-safe domain, not a value to be interpolated.

# %%
target_longitude, target_latitude = to_geographic.transform(
    east_target * 1000.0,
    north_target * 1000.0,
)
map_components = [
    (3, "Dilatation", r"$\mu$strain per interval"),
    (2, "Engineering shear", r"$\mu$strain per interval"),
    (4, "Rotation", r"$\mu$rad per interval"),
]
map_intervals = [
    (int(np.flatnonzero(intervals["end_date"].eq(PRE_EVENT_END).to_numpy())[0]), "22 June-4 July 2019"),
    (event_index, "4-16 July 2019"),
]
limits = {}
for component_index, title, _ in map_components:
    values = np.concatenate(
        [
            value_increment[time_index, component_index] / 1000.0
            for time_index, _ in map_intervals
        ]
    )
    finite = np.abs(values[np.isfinite(values)])
    limits[component_index] = max(float(np.quantile(finite, 0.98)), 0.01)

plt.close("all")
gc.collect()
map_output_rows = []
for row_index, (time_index, row_title) in enumerate(map_intervals):
    for column_index, (component_index, column_title, colorbar_label) in enumerate(map_components):
        interval_slug = "pre_event" if row_index == 0 else "earthquake_sequence"
        component_slug = component_names[component_index]
        stem = f"04_{interval_slug}_{component_slug}"
        png_path = OUTPUT_DIR / f"{stem}.png"
        pdf_path = OUTPUT_DIR / f"{stem}.pdf"
        if RUNNING_NOTEBOOK and png_path.exists() and pdf_path.exists():
            display(Image(filename=str(png_path), width=720))
            map_output_rows.append(
                {
                    "interval_index": time_index,
                    "interval": row_title,
                    "component": component_names[component_index],
                    "color_limit_micro_units": limits[component_index],
                    "png": png_path.name,
                    "pdf": pdf_path.name,
                }
            )
            continue
        fig, axis = plt.subplots(figsize=(8.2, 7.0), constrained_layout=True)
        values = value_increment[time_index, component_index] / 1000.0
        limit = limits[component_index]
        mappable = axis.scatter(
            target_longitude,
            target_latitude,
            c=values,
            s=30,
            marker="s",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            linewidths=0.0,
            rasterized=True,
        )
        axis.contour(
            fields["longitude"],
            fields["latitude"],
            distance_grid,
            levels=[18.0],
            colors="black",
            linestyles="--",
            linewidths=1.2,
        )
        plot_rupture(axis, color="black", lw=1.7)
        axis.set_title(
            f"{row_title}: {column_title}",
            fontweight="bold",
        )
        axis.set(xlabel="Longitude", ylabel="Latitude")
        axis.grid(True, color="0.82", ls=":", lw=0.5)
        colorbar = fig.colorbar(mappable, ax=axis, orientation="vertical", pad=0.025)
        colorbar.set_label(colorbar_label)
        axis.text(
            0.015,
            0.015,
            "Dashed line: 18-km strain-safe boundary",
            transform=axis.transAxes,
            va="bottom",
            ha="left",
            fontsize=9.5,
            bbox={"facecolor": "white", "edgecolor": "0.5", "alpha": 0.8},
        )
        if not png_path.exists():
            fig.savefig(png_path, dpi=240)
        if not pdf_path.exists():
            fig.savefig(pdf_path)
        plt.show()
        plt.close(fig)
        map_output_rows.append(
            {
                "interval_index": time_index,
                "interval": row_title,
                "component": component_names[component_index],
                "color_limit_micro_units": limit,
                "png": png_path.name,
                "pdf": pdf_path.name,
            }
        )
pd.DataFrame(map_output_rows).to_csv(
    OUTPUT_DIR / "strain_map_file_manifest.csv",
    index=False,
)

# %% [markdown]
# ## 9. Save reproducibility package and research conclusion

# %%
np.savez_compressed(
    OUTPUT_DIR / "strain_change_detection_arrays.npz",
    interval_start=intervals["start_date"].to_numpy("datetime64[ns]"),
    interval_end=intervals["end_date"].to_numpy("datetime64[ns]"),
    duration_days=intervals["duration_days"].to_numpy(np.int16),
    component_names=np.asarray(component_names),
    target_east_km=east_target,
    target_north_km=north_target,
    strain_increment=value_increment.astype(np.float32),
    strain_sigma=sigma_increment.astype(np.float32),
    strain_rate_per_day=rate.astype(np.float32),
    strain_rate_sigma_per_day=sigma_rate.astype(np.float32),
    standardized_innovation=z.astype(np.float32),
    baseline_leave_one_out_innovation=baseline_loo_z.astype(np.float32),
    maximum_cluster_mass=maximum_cluster_mass.astype(np.float32),
    strain_energy=energy.astype(np.float32),
    energy_standardized=energy_standardized.astype(np.float32),
    gap_category=gap_category,
)

manifest = {
    "status": (
        "consensus-supported pre-event off-fault strain change"
        if pre_change_supported
        else "no independently supported pre-event off-fault strain change"
    ),
    "dimensionality": (
        "2-D horizontal E-N solution after subtracting GNSS-derived vertical-to-LOS; "
        "not a 3-D InSAR inversion"
    ),
    "source_projection": source_manifest["correction"],
    "look_geometry": source_manifest["look_geometry"],
    "interval_count": len(intervals),
    "target_count": len(targets),
    "baseline": {
        "last_interval_end": str(BASELINE_END.date()),
        "interval_count": int(baseline_mask.sum()),
        "minimum_target_observations": MIN_BASELINE_OBSERVATIONS,
        "earthquake_data_used_for_fit_or_thresholds": False,
    },
    "pre_event_surveillance": {
        "start": str(SURVEILLANCE_START.date()),
        "end": str(PRE_EVENT_END.date()),
        "interval_count": int(pre_event_surveillance.sum()),
    },
    "method_A_spatial_cluster": {
        "z_threshold": CLUSTER_Z_THRESHOLD,
        "minimum_cells": CLUSTER_MIN_CELLS,
        "connectivity": "8-neighbour, sign-consistent",
        "family": "locations, signs, five components, and three surveillance intervals",
        "pre_event_maximum_mass": pre_cluster_observed,
        "pre_event_block_fwer_p": pre_cluster_p_fwer,
        "event_interval_maximum_mass": event_cluster_mass,
        "event_interval_fwer_p": event_cluster_p_fwer,
    },
    "method_B_Page_CUSUM": {
        "energy": f"{ENERGY_QUANTILE:.2f} quantile of absolute standardized innovations",
        "reference": CUSUM_REFERENCE,
        "pre_event_maximum": pre_cusum_observed,
        "pre_event_empirical_p": pre_cusum_p,
        "event_interval_value": event_cusum_observed,
        "event_interval_empirical_p": event_cusum_p,
    },
    "consensus_rule": "both pre-event methods require empirical p <= 0.05",
    "consensus_supported": pre_change_supported,
    "gap_policy": {
        "near_fault_filling_performed": False,
        "strain_safe_distance_km": 18.0,
        "reason": (
            "RMLS assumes local continuity; filling a rupture-zone displacement "
            "discontinuity can manufacture or suppress strain."
        ),
    },
    "validation_boundary": {
        "full_scene_vertical_gate": bool(source_manifest["all_vertical_gates_passed"]),
        "off_fault_vertical_gate": bool(source_manifest["off_fault_vertical_gates_passed"]),
        "primary_domain": "distance to mapped rupture >18 km",
    },
    "limitations": [
        "Ascending and descending acquisitions on a nominal common date differ by about 12 hours.",
        "North displacement is weakly constrained because both Sentinel-1 look vectors have small, similarly signed north components.",
        "RMLS uncertainty is conditional on resampled displacement errors and does not remove all InSAR spatial correlation.",
        "A negative off-fault strain screen does not contradict strongly supported earthquake deformation in direct interferograms or source inversions.",
        "No claim is made for strain inside the 18-km rupture-safe exclusion.",
    ],
}
(OUTPUT_DIR / "strain_change_detection_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

readme = f"""# Two-track horizontal strain change detection

This directory contains the output of `notebooks/14_two_track_strain_change_detection.ipynb`.

## Primary result

{manifest["status"]}.

- Spatial maximum-cluster pre-event FWER p: {pre_cluster_p_fwer:.4f}
- Strain-energy Page-CUSUM pre-event p: {pre_cusum_p:.4f}
- Event-interval cluster FWER p: {event_cluster_p_fwer:.4f}
- Event-interval Page-CUSUM p: {event_cusum_p:.4f}

The analysis is a 2-D E-N solution after GNSS-derived vertical-to-LOS subtraction.
Near-fault gaps were not filled. Primary strain inference is restricted to more
than 18 km from the mapped rupture because a smooth local derivative model must
not cross a displacement discontinuity.

## Main files

- `gap_mask_audit.csv`: reproducible attribution of common-grid and native
  descending gaps.
- `strain_component_regional_timeseries.csv`: physical component summaries.
- `strain_cluster_interval_summary.csv`: interval maximum-cluster statistics.
- `strain_spatial_clusters.csv`: retained sign-consistent clusters.
- `strain_cluster_sensitivity.csv`: threshold and minimum-area sensitivity.
- `strain_temporal_detection_summary.csv`: the two primary tests and event
  controls.
- `strain_change_detection_arrays.npz`: rates, uncertainties, standardized
  innovations, masks, and detection statistics.
- `strain_change_detection_manifest.json`: frozen choices and interpretation
  boundary.
"""
(OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")
display(pd.DataFrame([manifest]).T)

# %% [markdown]
# ## Conclusion
#
# The vertical-removed ascending and descending LOS fields support a joint E--N
# solution over the paired domain, but they do not justify filling the
# near-rupture quality gap. On the independently supported off-fault domain,
# neither the spatial maximum-cluster test nor the robust strain-energy CUSUM
# supports a pre-event strain change at the 5% level. This is a valid negative
# result: temporal coincidence in a derived strain product is not evidence of
# earthquake preparation.
