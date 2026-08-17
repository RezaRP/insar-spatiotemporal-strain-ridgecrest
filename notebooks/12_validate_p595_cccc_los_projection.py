# %% [markdown]
# # P595 and CCCC: GNSS ENU projection into Track 64 and Track 71 LOS
#
# This notebook is the geometry gate that must precede any vertical-field
# correction or two-track strain analysis. It answers three separate questions:
#
# 1. Does the heading/incidence formula reproduce the LiCSAR LOS-vector signs?
# 2. How different is the frame-average portal geometry from the native local
#    LiCSAR E/N/U vector at P595 and CCCC?
# 3. Does a full ENU projection agree with independent 4–16 July
#    interferograms after one track-wide reference offset and polarity audit?
#
# The cumulative-series overlays are diagnostic only. They receive only a
# temporal median removal; no trend, scale, phase ramp, or GNSS-fitted spatial
# correction is applied. The independent interferogram comparison is therefore
# the decisive projection check.

# %%
from __future__ import annotations

from pathlib import Path
import json
import math
import sys
import warnings

import h5py
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import tifffile


def find_repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Run this notebook from inside the repository.")


ROOT = find_repository_root(Path.cwd().resolve())
sys.path.insert(0, str(ROOT / "src"))

from ridgecrest_jump import load_text_stack  # noqa: E402
from ridgecrest_los_projection import (  # noqa: E402
    PortalGeometry,
    heading_incidence_from_look_vector,
    project_enu_mm,
    project_enu_covariance_mm,
    project_tenv3_history,
    remove_baseline_median,
)
from ridgecrest_two_track import normalize_look_vectors  # noqa: E402
from ridgecrest_vertical_los import (  # noqa: E402
    geotiff_axes,
    gnss_interval_table,
    haversine_km,
    load_gnss_network,
)


mpl.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
    }
)

# %% [markdown]
# ## 1. Inputs and fixed conventions
#
# Heading is satellite flight azimuth clockwise from north. Incidence is
# measured from local vertical. Under the LiCSAR ground-to-satellite convention,
#
# \[
# \boldsymbol l =
# [-\sin\theta\cos H,\ \sin\theta\sin H,\ \cos\theta],
# \qquad
# d_{\mathrm{LOS}} = l_E E + l_N N + l_U U.
# \]
#
# Positive LOS therefore means motion towards the satellite. The portal
# metadata reconstruct one frame-average vector. Native `geo.E/N/U.tif` values
# retain the local imaging geometry and topographic contribution.

# %%
GNSS_ROOT = Path(r"D:\Uni\Thises\GNSS_ridgecrest\data\tenv_data")
ASC_GEOMETRY_ROOT = Path(r"D:\Lics\GEOC_asc")
DESC_GEOMETRY_ROOT = Path(r"D:\Lics\GEOC_desc")
DESC_H5 = ROOT / "data" / "cum_full_scene_no_GACOS.h5"
PHASE1_DIR = ROOT / "outputs" / "gnss_vertical_los_phase1"
OUTPUT_DIR = ROOT / "outputs" / "station_los_projection_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATIONS = ("P595", "CCCC")
BASELINE_START = pd.Timestamp("2018-01-01")
BASELINE_END = pd.Timestamp("2018-03-31")
STATION_SAMPLE_RADIUS_KM = 1.0
EVENT_TIMES = (
    pd.Timestamp("2019-07-04T17:33:49"),
    pd.Timestamp("2019-07-06T03:19:53"),
)

TRACKS = {
    "ascending_T64": {
        "short": "Ascending T64",
        "geometry_root": ASC_GEOMETRY_ROOT,
        "geometry_stem": "064A_05410_131313.geo",
        "portal": PortalGeometry(
            "ascending_T64", -10.146887, 39.6181, "01:50:08.490464"
        ),
        "start_utc": pd.Timestamp("2019-07-04T01:50:08.490464"),
        "end_utc": pd.Timestamp("2019-07-16T01:50:08.490464"),
        "audit_csv": PHASE1_DIR / "ascending_T64_gnss_insar_sign_audit.csv",
    },
    "descending_T71": {
        "short": "Descending T71",
        "geometry_root": DESC_GEOMETRY_ROOT,
        "geometry_stem": "071D_05377_131313.geo",
        "portal": PortalGeometry(
            "descending_T71", -169.84503, 33.7677, "13:51:41.812911"
        ),
        "start_utc": pd.Timestamp("2019-07-04T13:51:41.812911"),
        "end_utc": pd.Timestamp("2019-07-16T13:51:41.812911"),
        "audit_csv": PHASE1_DIR / "descending_T71_gnss_insar_sign_audit.csv",
    },
}

for required in (
    GNSS_ROOT,
    ASC_GEOMETRY_ROOT,
    DESC_GEOMETRY_ROOT,
    DESC_H5,
    PHASE1_DIR / "phase1_manifest.json",
):
    if not required.exists():
        raise FileNotFoundError(required)

histories, network = load_gnss_network(GNSS_ROOT)
network_two = network.loc[network["station"].isin(STATIONS)].copy()
if set(network_two["station"]) != set(STATIONS):
    raise RuntimeError("P595 and CCCC were not both found in the GNSS network")
network_two = network_two.set_index("station").loc[list(STATIONS)].reset_index()
network_two

# %% [markdown]
# ## 2. Reconstruct portal vectors and sample local native vectors
#
# A 1-km median is used to prevent a single raster cell from deciding the local
# geometry. The sampled E/N/U components are normalized before use.

# %%
def sample_local_look_vector(
    geometry_root: Path,
    geometry_stem: str,
    *,
    latitude: float,
    longitude: float,
    radius_km: float,
) -> tuple[np.ndarray, int]:
    paths = [geometry_root / f"{geometry_stem}.{component}.tif" for component in "ENU"]
    if any(not path.exists() for path in paths):
        raise FileNotFoundError(paths)
    with Image.open(paths[0]) as image:
        latitude_axis, longitude_axis = geotiff_axes(image)
    arrays = normalize_look_vectors(
        *(tifffile.imread(path).astype(float) for path in paths)
    )
    dy = (latitude_axis[:, None] - latitude) * 111.195
    dx = (
        (longitude_axis[None, :] - longitude)
        * 111.195
        * math.cos(math.radians(latitude))
    )
    mask = np.square(dx) + np.square(dy) <= radius_km**2
    vector = np.array([np.nanmedian(array[mask]) for array in arrays], dtype=float)
    vector /= np.linalg.norm(vector)
    return vector, int(mask.sum())


geometry_rows: list[dict[str, object]] = []
local_vectors: dict[tuple[str, str], np.ndarray] = {}
for track, config in TRACKS.items():
    portal = config["portal"]
    portal_vector = portal.look_vector
    for station_row in network_two.itertuples(index=False):
        station = str(station_row.station)
        local_vector, pixel_count = sample_local_look_vector(
            config["geometry_root"],
            config["geometry_stem"],
            latitude=float(station_row.latitude),
            longitude=float(station_row.longitude),
            radius_km=STATION_SAMPLE_RADIUS_KM,
        )
        local_vectors[(track, station)] = local_vector
        local_heading, local_incidence = heading_incidence_from_look_vector(local_vector)
        angular_difference = math.degrees(
            math.acos(float(np.clip(portal_vector @ local_vector, -1.0, 1.0)))
        )
        geometry_rows.append(
            {
                "track": track,
                "station": station,
                "portal_heading_deg": portal.heading_deg,
                "portal_incidence_deg": portal.incidence_deg,
                "portal_los_e": portal_vector[0],
                "portal_los_n": portal_vector[1],
                "portal_los_u": portal_vector[2],
                "local_heading_deg": local_heading,
                "local_incidence_deg": local_incidence,
                "local_los_e": local_vector[0],
                "local_los_n": local_vector[1],
                "local_los_u": local_vector[2],
                "portal_local_angular_difference_deg": angular_difference,
                "native_geometry_pixel_count": pixel_count,
            }
        )

geometry_table = pd.DataFrame(geometry_rows)
geometry_table.to_csv(OUTPUT_DIR / "station_geometry_comparison.csv", index=False)
geometry_table[
    [
        "track",
        "station",
        "portal_incidence_deg",
        "local_incidence_deg",
        "portal_local_angular_difference_deg",
        "local_los_e",
        "local_los_n",
        "local_los_u",
    ]
].round(5)

# %% [markdown]
# ## 3. Project the complete P595 and CCCC GNSS histories
#
# Both portal-average and native-local projections are retained. All curves
# below have only the 1 January–31 March 2018 temporal median removed. This
# alignment changes neither sign, trend, scale, nor coseismic amplitude.

# %%
projection_rows: list[pd.DataFrame] = []
for station in STATIONS:
    history = histories[station]
    for track, config in TRACKS.items():
        portal = project_tenv3_history(history, config["portal"].look_vector)
        native = project_tenv3_history(history, local_vectors[(track, station)])
        use = (portal["date"] >= "2017-05-21") & (portal["date"] <= "2019-11-25")
        portal = portal.loc[use].copy()
        native = native.loc[use].copy()
        portal_centred, portal_baseline, _ = remove_baseline_median(
            portal["los_mm"].to_numpy(),
            pd.DatetimeIndex(portal["date"]),
            start=BASELINE_START,
            end=BASELINE_END,
        )
        native_centred, native_baseline, _ = remove_baseline_median(
            native["los_mm"].to_numpy(),
            pd.DatetimeIndex(native["date"]),
            start=BASELINE_START,
            end=BASELINE_END,
        )
        projection_rows.append(
            pd.DataFrame(
                {
                    "date": portal["date"].to_numpy(),
                    "station": station,
                    "track": track,
                    "east_mm": native["east_mm"].to_numpy(),
                    "north_mm": native["north_mm"].to_numpy(),
                    "up_mm": native["up_mm"].to_numpy(),
                    "portal_los_mm": portal["los_mm"].to_numpy(),
                    "native_los_mm": native["los_mm"].to_numpy(),
                    "portal_los_centred_mm": portal_centred,
                    "native_los_centred_mm": native_centred,
                    "native_los_sigma_mm": native["los_sigma_mm"].to_numpy(),
                    "portal_baseline_mm": portal_baseline,
                    "native_baseline_mm": native_baseline,
                }
            )
        )

projected_daily = pd.concat(projection_rows, ignore_index=True)
projected_daily.to_csv(
    OUTPUT_DIR / "p595_cccc_projected_gnss_los_daily.csv", index=False
)

fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.2), sharex=True, constrained_layout=True)
colors = {"ascending_T64": "#1768AC", "descending_T71": "#C73E1D"}
for ax, station in zip(axes, STATIONS):
    for track, config in TRACKS.items():
        data = projected_daily.loc[
            (projected_daily["station"] == station)
            & (projected_daily["track"] == track)
        ]
        ax.plot(
            data["date"],
            data["native_los_centred_mm"],
            color=colors[track],
            lw=1.4,
            label=f"{config['short']}: native local vector",
        )
        ax.plot(
            data["date"],
            data["portal_los_centred_mm"],
            color=colors[track],
            lw=1.0,
            ls="--",
            alpha=0.75,
            label=f"{config['short']}: portal-average angles",
        )
    for event_time in EVENT_TIMES:
        ax.axvline(event_time, color="0.2", lw=0.9, ls=":")
    ax.set_title(f"{station}: GNSS ENU projected into both LOS geometries")
    ax.set_ylabel("Projected LOS (mm)")
    ax.grid(True, color="0.90", lw=0.7)
    ax.legend(ncol=2, loc="upper left")
axes[-1].set_xlabel("Date")
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.savefig(OUTPUT_DIR / "01_gnss_dual_geometry_los_timeseries.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Diagnostic comparison with both cumulative InSAR series
#
# Track 64 is read from the date-named ascending text maps; no ascending HDF5 is
# required. Track 71 is read from the descending HDF5. A 1-km station median is
# used in both cases.
#
# P595 has finite descending values but fails the primary Track-71 time-series
# quality mask because its local `n_loop_err` is 12 (>10). It is plotted only as
# a flagged sensitivity curve. CCCC passes the mask.

# %%
ascending = load_text_stack(ROOT / "data", align_coordinates=True)
ascending_station_series: dict[str, np.ndarray] = {}
ascending_counts: dict[str, int] = {}
for station_row in network_two.itertuples(index=False):
    station = str(station_row.station)
    distance = haversine_km(
        ascending.latitude,
        ascending.longitude,
        float(station_row.latitude),
        float(station_row.longitude),
    )
    mask = distance <= STATION_SAMPLE_RADIUS_KM
    ascending_counts[station] = int(mask.sum())
    ascending_station_series[station] = np.nanmedian(
        ascending.displacement[:, mask], axis=1
    )

descending_station_series: dict[str, np.ndarray] = {}
descending_counts: dict[str, tuple[int, int]] = {}
with h5py.File(DESC_H5, "r") as handle:
    descending_dates = pd.to_datetime(
        np.asarray(handle["imdates"][:], dtype=np.int64).astype(str),
        format="%Y%m%d",
    )
    ny, nx = handle["cum"].shape[1:]
    descending_latitude = float(handle["corner_lat"][()]) + np.arange(ny) * float(
        handle["post_lat"][()]
    )
    descending_longitude = float(handle["corner_lon"][()]) + np.arange(nx) * float(
        handle["post_lon"][()]
    )
    quality = (
        np.isfinite(handle["coh_avg"][:])
        & (handle["coh_avg"][:] >= 0.30)
        & np.isfinite(handle["resid_rms"][:])
        & (handle["resid_rms"][:] <= 5.0)
        & (handle["n_gap"][:] <= 2)
        & (handle["n_loop_err"][:] <= 10)
    )
    for station_row in network_two.itertuples(index=False):
        station = str(station_row.station)
        latitude = float(station_row.latitude)
        longitude = float(station_row.longitude)
        row_index = np.flatnonzero(np.abs(descending_latitude - latitude) < 0.02)
        col_index = np.flatnonzero(np.abs(descending_longitude - longitude) < 0.02)
        row_slice = slice(int(row_index.min()), int(row_index.max() + 1))
        col_slice = slice(int(col_index.min()), int(col_index.max() + 1))
        local_cube = np.asarray(
            handle["cum"][:, row_slice, col_slice], dtype=float
        )
        local_latitude = descending_latitude[row_slice]
        local_longitude = descending_longitude[col_slice]
        dy = (local_latitude[:, None] - latitude) * 111.195
        dx = (
            (local_longitude[None, :] - longitude)
            * 111.195
            * math.cos(math.radians(latitude))
        )
        disk = np.square(dx) + np.square(dy) <= STATION_SAMPLE_RADIUS_KM**2
        quality_disk = disk & quality[row_slice, col_slice]
        descending_counts[station] = (int(disk.sum()), int(quality_disk.sum()))
        use = quality_disk if int(quality_disk.sum()) > 0 else disk
        descending_station_series[station] = np.nanmedian(
            np.where(use[None, :, :], local_cube, np.nan), axis=(1, 2)
        )


def acquisition_comparison(
    station: str,
    track: str,
    dates: pd.DatetimeIndex,
    insar_values: np.ndarray,
    *,
    insar_quality_supported: bool,
    insar_pixel_count: int,
) -> pd.DataFrame:
    daily = projected_daily.loc[
        (projected_daily["station"] == station)
        & (projected_daily["track"] == track)
    ].set_index("date")
    selected = daily.reindex(dates)
    insar_centred, insar_baseline, _ = remove_baseline_median(
        insar_values,
        dates,
        start=BASELINE_START,
        end=BASELINE_END,
    )
    return pd.DataFrame(
        {
            "date": dates,
            "station": station,
            "track": track,
            "insar_los_centred_mm": insar_centred,
            "native_gnss_los_centred_mm": selected[
                "native_los_centred_mm"
            ].to_numpy(),
            "portal_gnss_los_centred_mm": selected[
                "portal_los_centred_mm"
            ].to_numpy(),
            "insar_temporal_baseline_mm": insar_baseline,
            "insar_quality_supported": insar_quality_supported,
            "insar_pixel_count": insar_pixel_count,
            "event_day_daily_gnss_unsafe": dates.isin(
                pd.DatetimeIndex([event.normalize() for event in EVENT_TIMES])
            ),
        }
    )


comparison_parts: list[pd.DataFrame] = []
for station in STATIONS:
    comparison_parts.append(
        acquisition_comparison(
            station,
            "ascending_T64",
            ascending.dates,
            ascending_station_series[station],
            insar_quality_supported=True,
            insar_pixel_count=ascending_counts[station],
        )
    )
    raw_count, quality_count = descending_counts[station]
    comparison_parts.append(
        acquisition_comparison(
            station,
            "descending_T71",
            descending_dates,
            descending_station_series[station],
            insar_quality_supported=quality_count > 0,
            insar_pixel_count=quality_count if quality_count > 0 else raw_count,
        )
    )

acquisition_table = pd.concat(comparison_parts, ignore_index=True)
acquisition_table.to_csv(
    OUTPUT_DIR / "p595_cccc_insar_cumulative_comparison.csv", index=False
)

fig, axes = plt.subplots(
    2, 2, figsize=(14.2, 8.5), sharex=True, constrained_layout=True
)
for row, station in enumerate(STATIONS):
    for column, (track, config) in enumerate(TRACKS.items()):
        ax = axes[row, column]
        data = acquisition_table.loc[
            (acquisition_table["station"] == station)
            & (acquisition_table["track"] == track)
        ]
        daily = projected_daily.loc[
            (projected_daily["station"] == station)
            & (projected_daily["track"] == track)
        ]
        ax.plot(
            daily["date"],
            daily["native_los_centred_mm"],
            color="#1768AC",
            lw=1.25,
            label="GNSS ENU → native local LOS",
        )
        ax.plot(
            daily["date"],
            daily["portal_los_centred_mm"],
            color="#E07A1F",
            lw=1.0,
            ls="--",
            label="GNSS ENU → portal-average LOS",
        )
        supported = bool(data["insar_quality_supported"].iloc[0])
        ax.plot(
            data["date"],
            data["insar_los_centred_mm"],
            marker="o",
            ms=3.1,
            lw=0.8,
            color="0.15" if supported else "0.55",
            alpha=0.9,
            label=(
                "Cumulative InSAR, 1-km median"
                if supported
                else "Cumulative InSAR, QC-failed sensitivity"
            ),
        )
        for event_time in EVENT_TIMES:
            ax.axvline(event_time, color="#9B2226", lw=0.85, ls=":")
        ax.set_title(f"{station} — {config['short']}")
        ax.set_ylabel("Baseline-centred LOS (mm)")
        ax.grid(True, color="0.91", lw=0.7)
        if row == 0 and column == 0:
            ax.legend(loc="upper left")
        if not supported:
            ax.text(
                0.02,
                0.05,
                "Track-71 primary time-series QC failed at P595",
                transform=ax.transAxes,
                color="0.35",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
            )
for ax in axes[-1]:
    ax.set_xlabel("Date")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
for ax in axes.ravel():
    ax.set_xlim(pd.Timestamp("2017-05-01"), pd.Timestamp("2019-12-15"))
fig.savefig(
    OUTPUT_DIR / "02_cumulative_insar_projection_diagnostic.png",
    bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ## 5. Decisive check against independent 4–16 July interferograms
#
# Daily GNSS coordinates cannot identify the sub-day pre-earthquake position on
# 4 July. The interval table therefore uses a pre-event weighted trend for the
# 4 July SAR endpoint and a post-event weighted trend for 16 July. No daily
# interpolation crosses either earthquake.
#
# The independent interferograms retain an arbitrary spatial reference. The
# earlier all-station polarity audit selected one global sign and one robust
# reference offset per track. Here we compare P595 and CCCC after applying that
# already-audited offset. No station-specific offset or scale is fitted.

# %%
phase1_manifest = json.loads(
    (PHASE1_DIR / "phase1_manifest.json").read_text(encoding="utf-8")
)
event_rows: list[dict[str, object]] = []
for track, config in TRACKS.items():
    interval = gnss_interval_table(
        histories,
        network_two,
        start=config["start_utc"],
        end=config["end_utc"],
        event_times=EVENT_TIMES,
        strict=True,
    ).set_index("station")
    audit = pd.read_csv(config["audit_csv"]).set_index("station")
    audit_summary = phase1_manifest["tracks"][track]["sign_audit"]
    sign = int(audit_summary["selected_insar_sign"])
    offset = float(audit_summary["centred_offset_mm"])
    for station in STATIONS:
        row = interval.loc[station]
        enu = np.array([row.east_mm, row.north_mm, row.up_mm], dtype=float)
        sigma_enu = np.array(
            [row.sigma_east_mm, row.sigma_north_mm, row.sigma_up_mm], dtype=float
        )
        native_vector = local_vectors[(track, station)]
        portal_vector = config["portal"].look_vector
        native_projection = float(project_enu_mm(*enu, native_vector))
        portal_projection = float(project_enu_mm(*enu, portal_vector))
        native_sigma = float(
            project_enu_covariance_mm(*sigma_enu, native_vector)
        )
        direct_ifg = float(sign * audit.loc[station, "insar_los_mm"] - offset)
        standardized_native_residual = (
            direct_ifg - native_projection
        ) / max(native_sigma, 1.0)
        acquisition = acquisition_table.loc[
            (acquisition_table["station"] == station)
            & (acquisition_table["track"] == track)
        ].set_index("date")
        cumulative_delta = float(
            acquisition.loc[pd.Timestamp("2019-07-16"), "insar_los_centred_mm"]
            - acquisition.loc[pd.Timestamp("2019-07-04"), "insar_los_centred_mm"]
        )
        quality_supported = bool(
            acquisition["insar_quality_supported"].iloc[0]
        )
        event_rows.append(
            {
                "track": track,
                "station": station,
                "east_mm": row.east_mm,
                "north_mm": row.north_mm,
                "up_mm": row.up_mm,
                "sigma_east_mm": row.sigma_east_mm,
                "sigma_north_mm": row.sigma_north_mm,
                "sigma_up_mm": row.sigma_up_mm,
                "native_projected_los_mm": native_projection,
                "native_projected_sigma_mm": native_sigma,
                "portal_projected_los_mm": portal_projection,
                "portal_minus_native_mm": portal_projection - native_projection,
                "independent_ifg_sign": sign,
                "independent_ifg_reference_offset_mm": offset,
                "independent_ifg_centred_los_mm": direct_ifg,
                "independent_ifg_minus_native_mm": direct_ifg - native_projection,
                "native_standardized_residual": standardized_native_residual,
                "cumulative_4_16_july_delta_mm": cumulative_delta,
                "cumulative_timeseries_quality_supported": quality_supported,
            }
        )

event_table = pd.DataFrame(event_rows)
event_table.to_csv(
    OUTPUT_DIR / "station_event_interval_projection.csv", index=False
)
event_table[
    [
        "track",
        "station",
        "native_projected_los_mm",
        "portal_projected_los_mm",
        "portal_minus_native_mm",
        "independent_ifg_centred_los_mm",
        "independent_ifg_minus_native_mm",
        "native_standardized_residual",
        "cumulative_4_16_july_delta_mm",
        "cumulative_timeseries_quality_supported",
    ]
].round(2)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), constrained_layout=True)
for ax, (track, config) in zip(axes, TRACKS.items()):
    data = event_table.loc[event_table["track"] == track].set_index("station")
    x = np.arange(len(STATIONS), dtype=float)
    ax.errorbar(
        x - 0.16,
        data.loc[list(STATIONS), "native_projected_los_mm"],
        yerr=data.loc[list(STATIONS), "native_projected_sigma_mm"],
        fmt="o",
        color="#1768AC",
        capsize=3,
        label="GNSS, native local vector",
    )
    ax.scatter(
        x,
        data.loc[list(STATIONS), "portal_projected_los_mm"],
        marker="s",
        color="#E07A1F",
        label="GNSS, portal-average angles",
    )
    for index, station in enumerate(STATIONS):
        supported = bool(
            data.loc[station, "cumulative_timeseries_quality_supported"]
        )
        ax.scatter(
            index + 0.16,
            data.loc[station, "independent_ifg_centred_los_mm"],
            marker="D" if supported else "X",
            s=55,
            color="0.15" if supported else "0.55",
            label=(
                "Independent IFG, centred"
                if index == 0
                else None
            ),
        )
    ax.axhline(0.0, color="0.65", lw=0.8)
    ax.set_xticks(x, STATIONS)
    ax.set_ylabel("4–16 July LOS change (mm)")
    ax.set_title(config["short"])
    ax.grid(True, axis="y", color="0.90")
    ax.legend(loc="best")
fig.savefig(
    OUTPUT_DIR / "03_independent_ifg_projection_check.png", bbox_inches="tight"
)
plt.show()

# %% [markdown]
# ## 6. Explicit geometry decision
#
# The portal formula is accepted only as a reconstruction of the frame-average
# geometry. Pixel correction must use native LiCSAR unit vectors. A station
# comparison is considered supported only where the independent interferogram
# is locally usable and the native-vector residual is within two propagated
# GNSS standard errors. P595–Track 71 remains a failed/flagged validation point,
# not evidence against the geometry, because its cumulative pixel fails the
# loop-error quality gate and its independent-IFG residual is anomalous.

# %%
geometry_decision = {
    "formula": "l=[-sin(i)cos(H), sin(i)sin(H), cos(i)]; LOS=lE*E+lN*N+lU*U",
    "los_positive_direction": "ground toward satellite",
    "portal_vectors": {
        track: config["portal"].look_vector.tolist()
        for track, config in TRACKS.items()
    },
    "maximum_portal_local_angular_difference_deg": float(
        geometry_table["portal_local_angular_difference_deg"].max()
    ),
    "native_pixel_geometry_required": True,
    "selected_insar_sign": {
        track: int(phase1_manifest["tracks"][track]["sign_audit"]["selected_insar_sign"])
        for track in TRACKS
    },
    "station_support": {
        f"{row.track}:{row.station}": bool(
            row.cumulative_timeseries_quality_supported
            and abs(row.native_standardized_residual) <= 2.0
        )
        for row in event_table.itertuples(index=False)
    },
    "cumulative_overlay_role": (
        "diagnostic only; temporal median removed, spatial references not fitted"
    ),
    "vertical_interpolation_status": (
        "separate stage; no LOS correction or strain is authorized by this notebook"
    ),
}
(OUTPUT_DIR / "station_projection_manifest.json").write_text(
    json.dumps(geometry_decision, indent=2), encoding="utf-8"
)
pd.DataFrame(
    {
        "item": list(geometry_decision["station_support"]),
        "projection_supported": list(geometry_decision["station_support"].values()),
    }
)

# %% [markdown]
# ### Interpretation rule
#
# - Use `geo.E/N/U.tif` at each pixel for all subsequent projections.
# - Retain a fixed `+1` InSAR polarity for both tracks.
# - Accept the native-vector projection at P595 and CCCC for Track 64.
# - Accept it at CCCC for Track 71.
# - Do not use P595 as a Track-71 time-series validation point; retain it only
#   as a visibly flagged sensitivity comparison.
# - Do not apply a vertical correction or compute strain until the independent
#   all-station vertical interpolation gate has been evaluated.
