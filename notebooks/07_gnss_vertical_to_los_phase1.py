# %% [markdown]
# # Phase I — GNSS vertical field and LiCSAR vertical-to-LOS correction
#
# This notebook-style VS Code Python file implements the requested P595-centred,
# ten-station experiment for the Ridgecrest sequence.
#
# It:
#
# 1. fixes a circle containing the ten closest GNSS stations to the station
#    nearest the Mw 6.4 epicentre;
# 2. estimates GNSS ENU increments at the actual ascending and descending SAR
#    acquisition times;
# 3. compares a constant field, ordinary kriging, and a heteroscedastic
#    Matern Gaussian process using leave-one-station-out prediction;
# 4. predicts vertical displacement at every valid LiCSAR pixel inside the
#    circle;
# 5. uses the LiCSAR pixel-wise LOS-up component to test a predicted vertical
#    contribution; and
# 6. audits InSAR polarity and the reference datum against independently
#    forward-projected GNSS ENU.
#
# Scope boundary: the two tracks have different start/end times, about
# twelve hours apart. A product can only be called horizontal-LOS (HLOS) if
# the vertical field passes its out-of-sample spatial-resolution gate. It is
# never a common-time east/north displacement solution, and this notebook
# deliberately does not compute strain.
#
# Critical timing safeguard: the available .tenv3 series are daily GNSS
# solutions. The 4 July SAR acquisitions occurred before the Mw 6.4 origin
# time. A daily GNSS coordinate is never interpolated across that rupture:
# the notebook estimates each 4 July pre-event endpoint from the preceding
# 30-day pre-event GNSS segment. The later endpoint is estimated from an
# 8-day post–Mw 7.1 local weighted daily-GNSS trend, with residual scatter
# propagated as endpoint uncertainty.

# %% [markdown]
# ## Physical equations and fixed decisions
#
# For each track i, with LiCSAR's pixel-wise LOS unit vector
# l_i = (l_iE, l_iN, l_iU), the GNSS forward projection is:
#
# d_GNSS_LOS = l_E ΔE + l_N ΔN + l_U ΔU.
#
# After one global, station-audited InSAR polarity s is selected for each
# track, vertical removal is fixed everywhere:
#
# d_HLOS = s d_InSAR − l_U U_hat.
#
# There is no pixel-by-pixel choice to add or subtract. The signed
# pixel-wise l_U provides the correct projection. The vertical field is
# estimated first and projected second; a nominal incidence angle is not used.
#
# All raw interferograms use the same pre-declared procedure: coherence mask,
# robust plane estimated outside a 70 km two-event exclusion zone, and one
# common reference disk centred on a geographically selected far-field GNSS
# station. The two ascending intervals are summed only after that identical
# referencing operation.

# %%
from __future__ import annotations

from pathlib import Path
import json
import sys
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay


def find_repository_root(start: Path) -> Path:
    """Find this repository when run as a VS Code interactive notebook."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError(
        "Run this notebook-style file from inside the ridgecrest-insar repository."
    )


ROOT = find_repository_root(Path.cwd().resolve())
sys.path.insert(0, str(ROOT / "src"))

from ridgecrest_vertical_los import (  # noqa: E402
    choose_reference_station,
    crop_slices_for_circle,
    fault_segments_in_bounds,
    gnss_interval_table,
    gnss_los_sign_audit,
    haversine_km,
    load_gnss_network,
    load_los_vectors,
    predict_vertical_field,
    read_and_reference_pair,
    select_station_circle,
    select_vertical_model,
    sum_referenced_pairs,
    to_utm11_km,
    vertical_los_correction,
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
print(f"Repository: {ROOT}")

# %% [markdown]
# ## Inputs and analysis configuration
#
# These input paths intentionally point to the original raw LiCSAR GeoTIFFs
# and NGL time series, not to a cumulative HDF5 product. The currently
# available cumulative HDF5 products are both Track 71 descending. Raw
# ascending pairs are therefore needed here to form the same nominal 4–16 July
# interval.
#
# Do not alter STATION_COUNT, event coordinates, the radius margin, or the
# model candidates after inspecting output. They are fixed before examining
# displacement values.

# %%
GNSS_ROOT = Path(r"D:\Uni\Thises\GNSS_ridgecrest\data\tenv_data")
ASC_ROOT = Path(r"D:\Lics\GEOC_asc")
DESC_ROOT = Path(r"D:\Lics\GEOC_desc")
FAULT_FILE = ROOT / "data" / "cgs_2019_ridgecrest_fault_ruptures.geojson"
OUTPUT_DIR = ROOT / "outputs" / "gnss_vertical_los_phase1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mw 6.4 is the geometrical centre-selection target.
M64 = {
    "name": "Mw 6.4",
    "latitude": 35.705,
    "longitude": -117.504,
    "time_utc": pd.Timestamp("2019-07-04T17:33:49"),
}
M71 = {
    "name": "Mw 7.1",
    "latitude": 35.770,
    "longitude": -117.599,
    "time_utc": pd.Timestamp("2019-07-06T03:19:53"),
}
EVENT_POINTS = [(M64["latitude"], M64["longitude"]), (M71["latitude"], M71["longitude"])]
EVENT_TIMES = [M64["time_utc"], M71["time_utc"]]

STATION_COUNT = 10
RADIUS_MARGIN_KM = 0.20
REFERENCE_DISK_RADIUS_KM = 1.50
COHERENCE_MIN = 0.50
RAMP_EXCLUSION_KM = 70.0
STATION_SAMPLE_RADIUS_KM = 1.00
CIRCLE_PADDING_KM = 1.50

# Track-specific UTC acquisition epochs taken from the LiCSAR metadata.
TRACKS = {
    "ascending_T64": {
        "root": ASC_ROOT,
        "frame": "064A_05410_131313",
        "pairs": ("20190704_20190710", "20190710_20190716"),
        "start": pd.Timestamp("2019-07-04T01:50:08.490464"),
        "end": pd.Timestamp("2019-07-16T01:50:08.490464"),
        "display_name": "Track 64 ascending, 4–16 July",
        "insar_noise_floor_mm": 5.0,
    },
    "descending_T71": {
        "root": DESC_ROOT,
        "frame": "071D_05377_131313",
        "pairs": ("20190704_20190716",),
        "start": pd.Timestamp("2019-07-04T13:51:41.812911"),
        "end": pd.Timestamp("2019-07-16T13:51:41.812911"),
        "display_name": "Track 71 descending, 4–16 July",
        "insar_noise_floor_mm": 5.0,
    },
}

for path in (GNSS_ROOT, ASC_ROOT, DESC_ROOT):
    assert path.exists(), f"Missing required input: {path}"
for label, config in TRACKS.items():
    for pair in config["pairs"]:
        assert (config["root"] / pair).exists(), f"{label}: missing pair {pair}"

print("Inputs and raw pairs found.")

# %% [markdown]
# ## 1. Fix the ten-station circle before examining displacement
#
# The circle centre is the GNSS station closest to the Mw 6.4 epicentre. Its
# radius is the distance to the tenth-nearest GNSS station plus a fixed 0.20 km
# numerical margin. The map-reference station is separately selected as the
# ten-station member with greatest minimum distance from the Mw 6.4 and Mw 7.1
# epicentres. This is a geographical rule, not a displacement-based choice.

# %%
histories, network = load_gnss_network(GNSS_ROOT)
centre_station, circle_radius_km, circle = select_station_circle(
    network,
    event_latitude=M64["latitude"],
    event_longitude=M64["longitude"],
    station_count=STATION_COUNT,
    margin_km=RADIUS_MARGIN_KM,
)
reference_station = choose_reference_station(circle, event_points=EVENT_POINTS)

distance_check = network.assign(
    distance_to_centre_km=haversine_km(
        network["latitude"],
        network["longitude"],
        float(centre_station["latitude"]),
        float(centre_station["longitude"]),
    )
)
assert len(circle) == STATION_COUNT
assert int((distance_check["distance_to_centre_km"] <= circle_radius_km).sum()) == STATION_COUNT

print("Circle centre:", centre_station["station"])
print(f"Fixed radius: {circle_radius_km:.2f} km")
print("Common raw-IFG reference station:", reference_station["station"])
display(
    circle[
        [
            "station",
            "latitude",
            "longitude",
            "distance_to_centre_km",
            "distance_to_event_km",
        ]
    ].round(3)
)

if int(network["invalid_date_rows"].sum()) > 0:
    print("Dropped malformed historical date record(s):", int(network["invalid_date_rows"].sum()))

# %%
fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
ax.scatter(network["longitude"], network["latitude"], s=26, c="0.75", label="Other GNSS")
ax.scatter(circle["longitude"], circle["latitude"], s=62, c="#2878B5", label="Ten-station circle")
ax.scatter(
    float(centre_station["longitude"]),
    float(centre_station["latitude"]),
    marker="*",
    s=220,
    c="#F6C344",
    edgecolor="k",
    linewidth=0.7,
    label=f"Circle centre: {centre_station['station']}",
)
ax.scatter(
    float(reference_station["longitude"]),
    float(reference_station["latitude"]),
    marker="s",
    s=76,
    c="white",
    edgecolor="k",
    linewidth=1.4,
    label=f"Reference: {reference_station['station']}",
)
for event, marker in ((M64, "*"), (M71, "P")):
    ax.scatter(
        event["longitude"],
        event["latitude"],
        marker=marker,
        s=180,
        c="crimson",
        edgecolor="k",
        linewidth=0.6,
        zorder=5,
        label=event["name"],
    )
theta = np.linspace(0, 2 * np.pi, 361)
lat_radius = circle_radius_km / 110.574
lon_radius = circle_radius_km / (
    111.320 * np.cos(np.deg2rad(float(centre_station["latitude"])))
)
ax.plot(
    float(centre_station["longitude"]) + lon_radius * np.cos(theta),
    float(centre_station["latitude"]) + lat_radius * np.sin(theta),
    color="#2878B5",
    lw=1.5,
    ls="--",
)
for row in circle.itertuples(index=False):
    ax.annotate(row.station, (row.longitude, row.latitude), xytext=(4, 4), textcoords="offset points", fontsize=9)
ax.set(
    xlabel="Longitude",
    ylabel="Latitude",
    title="Fixed P595-centred ten-station GNSS circle",
)
ax.grid(True, color="0.88", lw=0.7)
ax.set_aspect(1 / np.cos(np.deg2rad(float(centre_station["latitude"]))))
ax.legend(loc="best", frameon=True)
fig.savefig(OUTPUT_DIR / "01_station_circle.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Form GNSS increments at the actual SAR acquisition times
#
# The ascending and descending intervals are calculated separately. The start
# endpoint of each 4 July interval is pre-event, so it is predicted from the
# preceding 30 days rather than interpolated through the Mw 6.4 discontinuity.
# The post–Mw 7.1 endpoint is a local weighted trend evaluated at the actual
# SAR time. The output table records every endpoint method for audit.

# %%
circle_intervals = {}
audit_intervals = {}
for track, config in TRACKS.items():
    circle_intervals[track] = gnss_interval_table(
        histories,
        circle,
        start=config["start"],
        end=config["end"],
        event_times=EVENT_TIMES,
    )
    audit_intervals[track] = gnss_interval_table(
        histories,
        network,
        start=config["start"],
        end=config["end"],
        event_times=EVENT_TIMES,
        strict=False,
    )
    circle_intervals[track].to_csv(
        OUTPUT_DIR / f"{track}_ten_station_enu_interval.csv",
        index=False,
    )
    audit_intervals[track].to_csv(
        OUTPUT_DIR / f"{track}_all_station_enu_interval.csv",
        index=False,
    )
    skipped = pd.DataFrame(audit_intervals[track].attrs["skipped_stations"])
    if not skipped.empty:
        skipped.to_csv(
            OUTPUT_DIR / f"{track}_all_station_enu_interval_skipped.csv",
            index=False,
        )
        print(f"{track}: excluded {len(skipped)} audit station(s) without exact endpoint coverage.")
    print("\n", config["display_name"])
    print(" start:", config["start"], "| end:", config["end"])
    display(
        circle_intervals[track][
            [
                "station",
                "east_mm",
                "north_mm",
                "up_mm",
                "sigma_east_mm",
                "sigma_north_mm",
                "sigma_up_mm",
                "up_start_method",
                "up_end_method",
            ]
        ].round(2)
    )

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True, sharey=True)
for ax, (track, table) in zip(axes, circle_intervals.items()):
    order = table.sort_values("up_mm")
    ax.errorbar(
        order["station"],
        order["up_mm"],
        yerr=order["sigma_up_mm"],
        fmt="o",
        color="#6A3D9A",
        capsize=3,
        lw=1.2,
    )
    ax.axhline(0, color="0.35", lw=0.9)
    ax.set(
        title=TRACKS[track]["display_name"],
        xlabel="GNSS station",
        ylabel="Vertical increment, ΔU (mm)",
    )
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", color="0.9")
fig.suptitle("Track-specific GNSS vertical increments at SAR endpoint times", y=1.03, fontsize=14)
fig.savefig(OUTPUT_DIR / "02_gnss_vertical_increments.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Select a vertical-field predictor by leave-one-station-out validation
#
# This tests three models on the same ten stations: a spatially constant field,
# ordinary kriging with an exponential covariance, and a Matern-3/2 Gaussian
# process with GNSS vertical uncertainty included as a heteroscedastic nugget.
# A spatial predictor is selected only when its paired LOO likelihood gain over
# the constant field exceeds one standard error and it also lowers RMSE. This
# one-standard-error parsimony rule prevents a ten-station network from
# manufacturing a pixel-scale pattern that it cannot predict out of sample.
#
# Predicted values are reported throughout the circle, but pixels outside the
# GNSS convex hull or near mapped rupture are explicitly labelled as
# modelled/extrapolated rather than resolved observations.

# %%
vertical_models = {}
vertical_scores = {}
vertical_loo = {}
for track, table in circle_intervals.items():
    xy_km = to_utm11_km(table["longitude"].to_numpy(), table["latitude"].to_numpy())
    model, scores, loo = select_vertical_model(
        xy_km,
        table["up_mm"].to_numpy(float),
        table["sigma_up_mm"].to_numpy(float),
    )
    loo = loo.copy()
    loo["station"] = loo["holdout_index"].map(dict(enumerate(table["station"])))
    vertical_models[track] = model
    vertical_scores[track] = scores
    vertical_loo[track] = loo
    scores.assign(track=track).to_csv(
        OUTPUT_DIR / f"{track}_vertical_model_scores.csv",
        index=False,
    )
    loo.to_csv(
        OUTPUT_DIR / f"{track}_vertical_loo_predictions.csv",
        index=False,
    )
    print(
        f"{TRACKS[track]['display_name']}: {model.method}; "
        f"LOO RMSE = {model.loo_rmse_mm:.2f} mm; "
        f"standardized RMS = {model.loo_standardized_rms:.2f}"
    )
    display(scores.head(8).round(3))

# %%
def selected_loo_rows(model, table):
    use = table["method"].eq(model.method) & np.isclose(table["nugget_mm"], model.nugget_mm)
    if model.length_scale_km is None:
        use &= table["length_scale_km"].isna()
    else:
        use &= np.isclose(table["length_scale_km"], model.length_scale_km)
    return table.loc[use].copy()


fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
for row, track in enumerate(TRACKS):
    score = vertical_scores[track]
    ax = axes[row, 0]
    for method, group in score.groupby("method"):
        ax.scatter(
            group["loo_rmse_mm"],
            group["loo_nlpd"],
            label=method,
            s=52,
            alpha=0.85,
        )
    best = score.iloc[0]
    ax.scatter(
        best["loo_rmse_mm"],
        best["loo_nlpd"],
        marker="*",
        s=190,
        c="#F6C344",
        edgecolor="k",
        linewidth=0.7,
        zorder=5,
        label="selected",
    )
    ax.set(
        title=TRACKS[track]["display_name"],
        xlabel="LOO RMSE (mm)",
        ylabel="LOO negative log predictive density",
    )
    ax.grid(True, color="0.9")
    ax.legend(loc="best")

    ax = axes[row, 1]
    loo = selected_loo_rows(vertical_models[track], vertical_loo[track])
    sigma = loo["predictive_sigma_mm"].to_numpy(float)
    ax.errorbar(
        loo["observed_mm"],
        loo["predicted_mm"],
        yerr=sigma,
        fmt="o",
        color="#6A3D9A",
        capsize=3,
    )
    limits = np.nanpercentile(np.r_[loo["observed_mm"], loo["predicted_mm"]], [2, 98])
    padding = max(5.0, 0.10 * (limits[1] - limits[0]))
    limits = [limits[0] - padding, limits[1] + padding]
    ax.plot(limits, limits, color="0.25", ls="--", lw=1.1)
    for record in loo.itertuples(index=False):
        ax.annotate(
            record.station,
            (record.observed_mm, record.predicted_mm),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set(
        xlim=limits,
        ylim=limits,
        xlabel="Held-out GNSS ΔU (mm)",
        ylabel="Prediction from remaining nine stations (mm)",
        title="Selected model: leave-one-station-out",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.9")
fig.savefig(OUTPUT_DIR / "03_vertical_model_validation.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Read raw InSAR, apply the fixed common reference, and audit polarity
#
# The ascending 4–16 July map is the sum of 4–10 July and 10–16 July maps only
# after each has been coherence-masked, corrected using the same
# far-field-ramp rule, and referenced to the same reference-station disk.
# The descending map is processed with the identical rule.
#
# The full GNSS network, not only the ten interpolation stations, is then
# forward-projected into each track's LOS geometry. This audit selects one
# global raw-InSAR polarity and reports a centred residual. The centring is
# diagnostic only and is not applied as a GNSS-fitted ramp.

# %%
insar = {}
for track, config in TRACKS.items():
    component_rasters = [
        read_and_reference_pair(
            config["root"] / pair,
            coherence_min=COHERENCE_MIN,
            event_points=EVENT_POINTS,
            ramp_exclusion_km=RAMP_EXCLUSION_KM,
            reference_latitude=float(reference_station["latitude"]),
            reference_longitude=float(reference_station["longitude"]),
            reference_radius_km=REFERENCE_DISK_RADIUS_KM,
        )
        for pair in config["pairs"]
    ]
    raster = (
        component_rasters[0]
        if len(component_rasters) == 1
        else sum_referenced_pairs(component_rasters, label=f"{track}_20190704_20190716")
    )
    los_e, los_n, los_u = load_los_vectors(
        config["root"],
        config["frame"],
        expected_shape=raster.values_mm.shape,
    )
    audit_table, audit_summary = gnss_los_sign_audit(
        raster,
        los_e,
        los_n,
        los_u,
        audit_intervals[track],
        station_radius_km=STATION_SAMPLE_RADIUS_KM,
    )
    audit_table.to_csv(OUTPUT_DIR / f"{track}_gnss_insar_sign_audit.csv", index=False)
    insar[track] = {
        "raster": raster,
        "los_e": los_e,
        "los_n": los_n,
        "los_u": los_u,
        "audit_table": audit_table,
        "audit_summary": audit_summary,
    }
    print("\n", TRACKS[track]["display_name"])
    print(json.dumps(audit_summary, indent=2))
    if audit_summary["aligned_correlation"] < 0.50:
        warnings.warn(
            f"{track}: weak GNSS–InSAR sign audit. Inspect this before interpreting HLOS.",
            stacklevel=1,
        )

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
for ax, track in zip(axes, TRACKS):
    audit = insar[track]["audit_table"]
    summary = insar[track]["audit_summary"]
    x = audit["gnss_projected_los_mm"].to_numpy(float)
    y = audit["insar_sign_aligned_mm"].to_numpy(float) - summary["centred_offset_mm"]
    ax.scatter(x, y, s=55, c="#2878B5")
    limits = np.nanpercentile(np.r_[x, y], [2, 98])
    pad = max(10.0, 0.08 * (limits[1] - limits[0]))
    limits = [limits[0] - pad, limits[1] + pad]
    ax.plot(limits, limits, color="0.2", lw=1.1, ls="--")
    for row in audit.itertuples(index=False):
        ax.annotate(
            row.station,
            (
                row.gnss_projected_los_mm,
                row.insar_sign_aligned_mm - summary["centred_offset_mm"],
            ),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set(
        xlim=limits,
        ylim=limits,
        xlabel="Forward-projected GNSS ENU (mm)",
        ylabel="Sign-aligned InSAR LOS, centred (mm)",
        title=(
            f"{TRACKS[track]['display_name']}\n"
            f"r = {summary['aligned_correlation']:.2f}; "
            f"centred RMSE = {summary['centred_rmse_mm']:.1f} mm"
        ),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.9")
fig.savefig(OUTPUT_DIR / "04_gnss_insar_sign_audit.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Predict vertical displacement on each LiCSAR grid and test LOS correction
#
# This cell evaluates the selected vertical model at all quality-controlled
# pixels inside the fixed circle. The raster output includes the vertical
# prediction, posterior standard deviation, vertical LOS contribution,
# vertical-corrected LOS, and a convex-hull flag. Values outside the station
# convex hull remain useful as a sensitivity surface, but not as resolved
# observations.
#
# The vertical LOS term is re-referenced in the same GNSS-reference disk used
# for the raw interferograms. Its uncertainty includes the local posterior
# uncertainty and a conservative reference-disk term. If the parsimony gate
# selects a constant field, this result is explicitly a sensitivity product,
# not pure HLOS. The same predicted vertical field is later a correlated
# uncertainty source between tracks; that covariance will be propagated in a
# joint inversion rather than ignored.

# %%
def compact_grid(raster, los_e, los_n, los_u):
    row_slice, col_slice = crop_slices_for_circle(
        raster.latitude,
        raster.longitude,
        centre_latitude=float(centre_station["latitude"]),
        centre_longitude=float(centre_station["longitude"]),
        radius_km=circle_radius_km,
        padding_km=CIRCLE_PADDING_KM,
    )
    latitude = raster.latitude[row_slice]
    longitude = raster.longitude[col_slice]
    los = raster.values_mm[row_slice, col_slice]
    valid = raster.valid[row_slice, col_slice] & np.isfinite(los)
    e = los_e[row_slice, col_slice]
    n = los_n[row_slice, col_slice]
    u = los_u[row_slice, col_slice]
    valid &= np.isfinite(e) & np.isfinite(n) & np.isfinite(u)
    lat_grid, lon_grid = np.meshgrid(latitude, longitude, indexing="ij")
    circle_mask = haversine_km(
        lat_grid,
        lon_grid,
        float(centre_station["latitude"]),
        float(centre_station["longitude"]),
    ) <= circle_radius_km
    reference_mask = haversine_km(
        lat_grid,
        lon_grid,
        float(reference_station["latitude"]),
        float(reference_station["longitude"]),
    ) <= REFERENCE_DISK_RADIUS_KM
    valid &= circle_mask
    return {
        "latitude": latitude,
        "longitude": longitude,
        "lat_grid": lat_grid,
        "lon_grid": lon_grid,
        "los_mm": los,
        "valid": valid,
        "circle_mask": circle_mask,
        "reference_mask": reference_mask,
        "los_e": e,
        "los_n": n,
        "los_u": u,
    }


phase1_products = {}
for track, config in TRACKS.items():
    data = insar[track]
    grid = compact_grid(data["raster"], data["los_e"], data["los_n"], data["los_u"])
    station_table = circle_intervals[track]
    station_xy = to_utm11_km(
        station_table["longitude"].to_numpy(),
        station_table["latitude"].to_numpy(),
    )
    pixel_rows, pixel_cols = np.where(grid["valid"])
    pixel_xy = to_utm11_km(
        grid["lon_grid"][pixel_rows, pixel_cols],
        grid["lat_grid"][pixel_rows, pixel_cols],
    )
    mean, sigma = predict_vertical_field(
        vertical_models[track],
        station_xy,
        station_table["up_mm"].to_numpy(float),
        station_table["sigma_up_mm"].to_numpy(float),
        pixel_xy,
    )
    vertical_mm = np.full(grid["valid"].shape, np.nan, dtype=np.float32)
    vertical_sigma_mm = np.full_like(vertical_mm, np.nan)
    vertical_mm[pixel_rows, pixel_cols] = mean
    vertical_sigma_mm[pixel_rows, pixel_cols] = sigma

    # A diagnostic support flag; predictions outside this hull remain in the
    # output but must not be described as observations.
    hull = Delaunay(station_xy)
    inside_hull = np.zeros_like(grid["valid"], dtype=bool)
    inside_hull[pixel_rows, pixel_cols] = hull.find_simplex(pixel_xy) >= 0

    vertical_los_mm, vertical_los_sigma_mm, reference_vlos_mm, reference_vlos_sigma_mm = (
        vertical_los_correction(
            grid["los_u"],
            vertical_mm,
            vertical_sigma_mm,
            grid["reference_mask"] & grid["valid"],
        )
    )
    sign = int(data["audit_summary"]["selected_insar_sign"])
    hlos_mm = sign * grid["los_mm"] - vertical_los_mm
    vertical_field_resolved = vertical_models[track].method != "constant"
    insar_sigma_mm = max(config["insar_noise_floor_mm"], data["raster"].ramp_scale_mm)
    hlos_sigma_mm = np.sqrt(insar_sigma_mm**2 + vertical_los_sigma_mm**2)
    hlos_mm[~grid["valid"]] = np.nan
    hlos_sigma_mm[~grid["valid"]] = np.nan

    phase1_products[track] = {
        **grid,
        "vertical_mm": vertical_mm,
        "vertical_sigma_mm": vertical_sigma_mm,
        "inside_gnss_hull": inside_hull,
        "vertical_los_mm": vertical_los_mm,
        "vertical_los_sigma_mm": vertical_los_sigma_mm,
        "hlos_mm": hlos_mm,
        "hlos_sigma_mm": hlos_sigma_mm,
        "insar_sign": sign,
        "insar_noise_mm": insar_sigma_mm,
        "reference_vlos_mm": reference_vlos_mm,
        "reference_vlos_sigma_mm": reference_vlos_sigma_mm,
        "vertical_field_resolved": vertical_field_resolved,
    }
    filename = f"{track}_20190704_20190716_vertical_corrected_hlos.npz"
    np.savez_compressed(
        OUTPUT_DIR / filename,
        latitude=grid["latitude"],
        longitude=grid["longitude"],
        valid=grid["valid"],
        inside_gnss_hull=inside_hull,
        los_e=grid["los_e"],
        los_n=grid["los_n"],
        los_u=grid["los_u"],
        referenced_los_mm=sign * grid["los_mm"],
        vertical_mm=vertical_mm,
        vertical_sigma_mm=vertical_sigma_mm,
        vertical_los_mm=vertical_los_mm,
        vertical_los_sigma_mm=vertical_los_sigma_mm,
        hlos_mm=hlos_mm,
        hlos_sigma_mm=hlos_sigma_mm,
        vertical_corrected_los_mm=hlos_mm,
        vertical_field_resolved=vertical_field_resolved,
        start_utc=str(config["start"]),
        end_utc=str(config["end"]),
        insar_sign=sign,
    )
    print(
        f"{track}: wrote {filename}; vertical-LOS reference = "
        f"{reference_vlos_mm:.2f} ± {reference_vlos_sigma_mm:.2f} mm"
    )

# %%
def robust_limit(*arrays, lower=2, upper=98, minimum=10.0):
    values = np.concatenate([array[np.isfinite(array)] for array in arrays])
    value = float(np.nanpercentile(np.abs(values), upper))
    return max(minimum, value)


FAULT_CACHE = {}


def add_context(ax, product):
    bounds = (
        float(product["latitude"].min()),
        float(product["latitude"].max()),
        float(product["longitude"].min()),
        float(product["longitude"].max()),
    )
    if bounds not in FAULT_CACHE:
        FAULT_CACHE[bounds] = fault_segments_in_bounds(
            FAULT_FILE,
            latitude_min=bounds[0],
            latitude_max=bounds[1],
            longitude_min=bounds[2],
            longitude_max=bounds[3],
        )
    for segment in FAULT_CACHE[bounds]:
        ax.plot(segment[:, 0], segment[:, 1], color="k", lw=0.35, alpha=0.6, zorder=3)
    ax.scatter(
        circle["longitude"],
        circle["latitude"],
        s=20,
        facecolor="white",
        edgecolor="k",
        lw=0.5,
        zorder=4,
    )
    ax.scatter(
        float(reference_station["longitude"]),
        float(reference_station["latitude"]),
        s=50,
        marker="s",
        facecolor="#F6C344",
        edgecolor="k",
        lw=0.6,
        zorder=5,
    )
    ax.scatter(
        M64["longitude"],
        M64["latitude"],
        s=66,
        marker="*",
        c="crimson",
        edgecolor="k",
        lw=0.4,
        zorder=5,
    )
    ax.set_aspect(1 / np.cos(np.deg2rad(float(centre_station["latitude"]))))
    ax.grid(True, color="0.86", lw=0.6)


fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
for row, track in enumerate(TRACKS):
    product = phase1_products[track]
    extent = [
        product["longitude"].min(),
        product["longitude"].max(),
        product["latitude"].min(),
        product["latitude"].max(),
    ]
    v_limit = robust_limit(product["vertical_mm"], minimum=10.0)
    panels = [
        (
            product["vertical_mm"],
            "Vertical prediction, ΔU (mm)",
            "coolwarm",
            -v_limit,
            v_limit,
        ),
        (
            product["vertical_sigma_mm"],
            "Vertical posterior σ (mm)",
            "magma",
            0,
            max(5, np.nanpercentile(product["vertical_sigma_mm"], 98)),
        ),
        (
            product["inside_gnss_hull"].astype(float),
            "GNSS convex-hull support",
            "Greys",
            0,
            1,
        ),
    ]
    for col, (image, title, cmap, vmin, vmax) in enumerate(panels):
        ax = axes[row, col]
        shown = np.where(product["valid"], image, np.nan)
        artist = ax.imshow(
            shown,
            extent=extent,
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        add_context(ax, product)
        ax.set(title=title)
        if col == 0:
            ax.set_ylabel(f"{TRACKS[track]['display_name']}\nLatitude")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Longitude")
        fig.colorbar(artist, ax=ax, shrink=0.84, pad=0.02)
fig.suptitle("Vertical field: prediction, uncertainty, and spatial-support flag", fontsize=16, y=1.01)
fig.savefig(OUTPUT_DIR / "05_vertical_field_maps.png", bbox_inches="tight")
plt.show()

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
for row, track in enumerate(TRACKS):
    product = phase1_products[track]
    extent = [
        product["longitude"].min(),
        product["longitude"].max(),
        product["latitude"].min(),
        product["latitude"].max(),
    ]
    los_limit = robust_limit(
        product["los_mm"],
        product["vertical_los_mm"],
        product["hlos_mm"],
        minimum=20.0,
    )
    panels = [
        (
            product["insar_sign"] * product["los_mm"],
            "Referenced InSAR LOS (mm)",
        ),
        (
            product["vertical_los_mm"],
            "Predicted vertical LOS contribution (mm)",
        ),
        (
            product["hlos_mm"],
            (
                "Vertical-removed horizontal LOS, HLOS (mm)"
                if product["vertical_field_resolved"]
                else "LOS after constant-U sensitivity correction (mm)"
            ),
        ),
    ]
    for col, (image, title) in enumerate(panels):
        ax = axes[row, col]
        artist = ax.imshow(
            np.where(product["valid"], image, np.nan),
            extent=extent,
            origin="upper",
            cmap="RdBu_r",
            vmin=-los_limit,
            vmax=los_limit,
        )
        add_context(ax, product)
        ax.set(title=title)
        if col == 0:
            ax.set_ylabel(f"{TRACKS[track]['display_name']}\nLatitude")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Longitude")
        fig.colorbar(artist, ax=ax, shrink=0.84, pad=0.02, label="mm")
fig.suptitle(
    "Track-specific vertical-to-LOS sensitivity correction (same colour scale within each row)",
    fontsize=16,
    y=1.01,
)
fig.savefig(OUTPUT_DIR / "06_vertical_los_correction_hlos_maps.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Save a reproducibility manifest and apply the stop gate
#
# Before any horizontal decomposition or strain calculation, review the saved
# figures and CSV tables. The minimum acceptance conditions are:
#
# - exactly ten stations were selected by the fixed radius;
# - the 4 July endpoint method is pre_event_weighted_trend, not interpolation
#   across the earthquake;
# - the selected predictor has credible leave-one-station-out residuals and
#   uncertainty calibration;
# - the GNSS–InSAR sign audit has enough stations and a clear positive aligned
#   correlation; and
# - the vertical correction is reported with uncertainty and a convex-hull
#   extrapolation mask.
#
# If only a constant field is selected, stop: a pure HLOS product is not
# established. Even if a spatial field is selected, do not calculate 2-D
# strain directly from HLOS. The next notebook must first bring the two tracks
# to a defensible common temporal interval, or model their timing mismatch;
# solve east/north motion with a condition-number map and correlated
# vertical-field covariance; then estimate off-fault strain with uncertainty.

# %%
def serialise_model(model):
    return {
        "method": model.method,
        "length_scale_km": model.length_scale_km,
        "nugget_mm": model.nugget_mm,
        "sill_mm2": model.sill_mm2,
        "loo_rmse_mm": model.loo_rmse_mm,
        "loo_mae_mm": model.loo_mae_mm,
        "loo_nlpd": model.loo_nlpd,
        "loo_standardized_rms": model.loo_standardized_rms,
    }


manifest = {
    "purpose": "Phase-I GNSS vertical prediction and LiCSAR vertical-to-LOS correction",
    "units": "mm",
    "circle_centre_station": str(centre_station["station"]),
    "circle_radius_km": circle_radius_km,
    "station_count": STATION_COUNT,
    "selected_stations": circle["station"].tolist(),
    "reference_station": str(reference_station["station"]),
    "reference_disk_radius_km": REFERENCE_DISK_RADIUS_KM,
    "event_times_utc": [str(value) for value in EVENT_TIMES],
    "tracks": {},
    "limitations": [
        "Daily GNSS positions cannot resolve sub-day coseismic steps; the pre-event SAR endpoint is a pre-event trend estimate.",
        "The ascending and descending endpoints differ by about 12 hours and are not a common-time EN displacement solution.",
        "Smooth spatial predictions outside the GNSS convex hull or across mapped rupture are sensitivity estimates, not resolved observations.",
        "A constant-only vertical model cannot establish pure horizontal LOS; its vertical-corrected map is a sensitivity product.",
        "HLOS is not east/north displacement and cannot be differentiated directly to obtain 2-D strain.",
    ],
}
for track, config in TRACKS.items():
    summary = insar[track]["audit_summary"]
    product = phase1_products[track]
    manifest["tracks"][track] = {
        "display_name": config["display_name"],
        "pairs": list(config["pairs"]),
        "start_utc": str(config["start"]),
        "end_utc": str(config["end"]),
        "vertical_model": serialise_model(vertical_models[track]),
        "sign_audit": {
            key: int(value) if key == "selected_insar_sign" else float(value)
            for key, value in summary.items()
        },
        "insar_ramp_scale_mm": float(insar[track]["raster"].ramp_scale_mm),
        "vertical_los_reference_mm": float(product["reference_vlos_mm"]),
        "vertical_los_reference_sigma_mm": float(product["reference_vlos_sigma_mm"]),
        "vertical_field_resolved": bool(product["vertical_field_resolved"]),
        "output_npz": f"{track}_20190704_20190716_vertical_corrected_hlos.npz",
    }
(OUTPUT_DIR / "phase1_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

summary_rows = []
for track in TRACKS:
    model = vertical_models[track]
    audit = insar[track]["audit_summary"]
    summary_rows.append(
        {
            "track": track,
            "vertical_model": model.method,
            "vertical_field_status": (
                "spatial field supported"
                if phase1_products[track]["vertical_field_resolved"]
                else "constant only; pixel-scale vertical field unresolved"
            ),
            "loo_rmse_mm": model.loo_rmse_mm,
            "loo_standardized_rms": model.loo_standardized_rms,
            "sign": audit["selected_insar_sign"],
            "sign_audit_station_count": audit["station_count"],
            "sign_audit_correlation": audit["aligned_correlation"],
            "sign_audit_centred_rmse_mm": audit["centred_rmse_mm"],
        }
    )
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_DIR / "phase1_validation_summary.csv", index=False)
display(summary.round(3))

print("\nSTOP GATE: inspect the figures and CSV files in")
print(OUTPUT_DIR)
print("Then share phase1_validation_summary.csv and the diagnostic figures before moving to the two-track horizontal inversion and strain stage.")

# %% [markdown]
# ## References used by this workflow
#
# - COMET LiCSAR product details:
#   https://comet.nerc.ac.uk/comet-lics-portal-product-details/
#   Documents the pixel-specific .geo.E, .geo.N, and .geo.U LOS components for
#   projection of GNSS ENU.
# - USGS Mw 6.4 Ridgecrest event page:
#   https://earthquake.usgs.gov/earthquakes/eventpage/ci38443183
#   Origin time used by the timing gate.
# - Lazecký et al. (2020), Remote Sensing:
#   https://doi.org/10.3390/rs12152430
#   LiCSAR processing and products.
# - Hines and Hetland (2018), Geophysical Journal International:
#   https://doi.org/10.1093/gji/ggx525
#   Gaussian-process interpolation and uncertainty-aware GNSS strain concepts.
# - Wright, Parsons and Lu (2004), Geophysical Research Letters:
#   https://doi.org/10.1029/2003GL018827
#   Limited north-component sensitivity of near-polar InSAR geometries.
# - Sandwell and Wessel (2016), Geophysical Research Letters:
#   https://doi.org/10.1002/2016GL070340
#   Elastically constrained interpolation and analytic strain derivatives for
#   the later vector-field stage.
