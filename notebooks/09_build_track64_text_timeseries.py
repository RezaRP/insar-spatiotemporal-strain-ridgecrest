# %% [markdown]
# # Track-64 ascending cumulative text stack: verified input for two-track time series
#
# The date-named text files are the authoritative Track-64 ascending cumulative
# series. Each file has three whitespace-separated columns:
# `LOS displacement (mm), latitude, longitude`. This notebook verifies the
# stack, samples native Track-64 LiCSAR look vectors, and records dates shared
# with the Track-71 descending cumulative HDF5. No ascending HDF5 is required.

# %%
from __future__ import annotations

from pathlib import Path
import json
import sys

import h5py
from IPython.display import display
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import tifffile


def find_repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Run from inside the ridgecrest-insar repository.")


ROOT = find_repository_root(Path.cwd().resolve())
sys.path.insert(0, str(ROOT / "src"))

from ridgecrest_jump import load_text_stack  # noqa: E402
from ridgecrest_two_track import masked_bilinear_resample, normalize_look_vectors  # noqa: E402
from ridgecrest_vertical_los import geotiff_axes, haversine_km  # noqa: E402

mpl.rcParams.update({"figure.dpi": 125, "savefig.dpi": 300, "font.size": 11})

# %% [markdown]
# ## 1. Fixed inputs and UTC convention
#
# Exact Track-64 and Track-71 UTC times are retained so that GNSS vertical
# endpoints are sampled separately for each track later.

# %%
TEXT_DIR = ROOT / "data"
DESC_H5 = ROOT / "data" / "cum_full_scene_no_GACOS.h5"
ASC_GEOMETRY_ROOT = Path(r"D:\Lics\GEOC_asc")
ASC_GEOMETRY_STEM = "064A_05410_131313.geo"
PHASE1_ASC = ROOT / "outputs" / "gnss_vertical_los_phase1" / "ascending_T64_20190704_20190716_vertical_corrected_hlos.npz"
OUTPUT_DIR = ROOT / "outputs" / "track64_text_timeseries"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for required in (
    DESC_H5,
    PHASE1_ASC,
    *(ASC_GEOMETRY_ROOT / f"{ASC_GEOMETRY_STEM}.{component}.tif" for component in ("E", "N", "U")),
):
    if not required.exists():
        raise FileNotFoundError(required)

TRACK64_UTC_TIME = pd.Timedelta(hours=1, minutes=50, seconds=8, microseconds=490464)
TRACK71_UTC_TIME = pd.Timedelta(hours=13, minutes=51, seconds=41, microseconds=812911)
P597 = {"latitude": 35.684349, "longitude": -117.595507, "radius_km": 1.5}

# %% [markdown]
# ## 2. Reconstruct the ascending stack by coordinate, never by row number

# %%
ascending_stack = load_text_stack(TEXT_DIR, align_coordinates=True)
latitude_axis = np.sort(np.unique(ascending_stack.latitude))
longitude_axis = np.sort(np.unique(ascending_stack.longitude))
print("Track-64 text epochs:", len(ascending_stack.dates))
print("Date range:", ascending_stack.dates.min().date(), "to", ascending_stack.dates.max().date())
print("Canonical points:", len(ascending_stack.latitude))
print("Nominal grid:", len(latitude_axis), "latitude x", len(longitude_axis), "longitude")
availability = pd.DataFrame({
    "date": ascending_stack.dates,
    "valid_pixels": np.isfinite(ascending_stack.displacement).sum(axis=1),
    "valid_fraction": np.isfinite(ascending_stack.displacement).mean(axis=1),
})
display(availability.head())

# %% [markdown]
# ## 3. Sample native T64 pixel-wise look vectors
#
# `lU` from LiCSAR `.geo.U.tif` is the actual vertical LOS coefficient; it is
# safer than a nominal incidence angle because it preserves local geometry.

# %%
def load_geometry(component: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = ASC_GEOMETRY_ROOT / f"{ASC_GEOMETRY_STEM}.{component}.tif"
    with Image.open(path) as image:
        latitude, longitude = geotiff_axes(image)
    values = tifffile.imread(path).astype(float)
    if values.shape != (len(latitude), len(longitude)):
        raise ValueError(f"Unexpected Track-64 {component} geometry shape: {values.shape}")
    return latitude, longitude, values


geometry = {component: load_geometry(component) for component in ("E", "N", "U")}
geom_latitude, geom_longitude, _ = geometry["U"]
all_valid_geometry = np.isfinite(geometry["E"][2]) & np.isfinite(geometry["N"][2]) & np.isfinite(geometry["U"][2])
look_values: dict[str, np.ndarray] = {}
for component, (_, _, values) in geometry.items():
    sampled, support = masked_bilinear_resample(
        geom_latitude, geom_longitude, values, all_valid_geometry,
        ascending_stack.latitude, ascending_stack.longitude,
    )
    if np.count_nonzero(support >= 0.999) != len(ascending_stack.latitude):
        raise RuntimeError(f"Track-64 geometry did not cover every text pixel ({component})")
    look_values[component] = sampled
look_values["E"], look_values["N"], look_values["U"] = normalize_look_vectors(
    look_values["E"], look_values["N"], look_values["U"]
)
print("Median T64 look vector:", [float(np.nanmedian(look_values[c])) for c in ("E", "N", "U")])
np.savez_compressed(
    OUTPUT_DIR / "track64_text_pixel_look_vectors.npz",
    latitude=ascending_stack.latitude,
    longitude=ascending_stack.longitude,
    los_e=look_values["E"], los_n=look_values["N"], los_u=look_values["U"],
)

# %% [markdown]
# ## 4. Find shared nominal cumulative dates
#
# Shared dates are not automatically common-time observations: their UTC times
# differ by roughly 12 hours and must remain separate during the earthquake
# sequence.

# %%
with h5py.File(DESC_H5, "r") as handle:
    descending_dates = pd.to_datetime(
        np.asarray(handle["imdates"][:], dtype=np.int64).astype(str), format="%Y%m%d"
    )
common_dates = ascending_stack.dates.intersection(descending_dates)
common = pd.DataFrame({"date": common_dates})
common["ascending_utc"] = common["date"] + TRACK64_UTC_TIME
common["descending_utc"] = common["date"] + TRACK71_UTC_TIME
common["track_time_separation_hours"] = (common["descending_utc"] - common["ascending_utc"]).dt.total_seconds() / 3600.0
common["earthquake_sequence_date"] = common["date"].between("2019-07-04", "2019-07-16")
common.to_csv(OUTPUT_DIR / "track64_track71_common_dates.csv", index=False)
print("Exact nominal common dates:", len(common))
display(common.loc[common["date"].between("2019-06-01", "2019-08-01")])

# %% [markdown]
# ## 5. Direct 4–16 July consistency check against the independent ascending IFG
#
# This checks text-stack provenance and sign only. Both maps are re-referenced
# over the same P597 disk, so their arbitrary input reference constants vanish.

# %%
index_04 = int(ascending_stack.dates.get_loc(pd.Timestamp("2019-07-04")))
index_16 = int(ascending_stack.dates.get_loc(pd.Timestamp("2019-07-16")))
text_delta = ascending_stack.displacement[index_16] - ascending_stack.displacement[index_04]
phase1 = np.load(PHASE1_ASC)
ifg_delta, ifg_support = masked_bilinear_resample(
    phase1["latitude"], phase1["longitude"], phase1["referenced_los_mm"], phase1["valid"].astype(bool),
    ascending_stack.latitude, ascending_stack.longitude,
)
reference = haversine_km(
    ascending_stack.latitude, ascending_stack.longitude, P597["latitude"], P597["longitude"]
) <= P597["radius_km"]
valid = np.isfinite(text_delta) & np.isfinite(ifg_delta) & (ifg_support >= 0.999)
valid_reference = valid & reference
if int(valid_reference.sum()) < 10:
    raise RuntimeError("Too few common P597 pixels for text/IFG consistency check")
text_delta_ref = text_delta - np.nanmedian(text_delta[valid_reference])
ifg_delta_ref = ifg_delta - np.nanmedian(ifg_delta[valid_reference])
valid &= np.isfinite(text_delta_ref) & np.isfinite(ifg_delta_ref)
correlation = float(np.corrcoef(text_delta_ref[valid], ifg_delta_ref[valid])[0, 1])
scale = float(np.linalg.lstsq(ifg_delta_ref[valid, None], text_delta_ref[valid], rcond=None)[0][0])
print(f"Text delta(16-4 July) vs independent T64 IFG: r={correlation:.3f}; scale={scale:.3f}; n={valid.sum()}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
axes[0].hist(text_delta_ref[valid], bins=80, color="#2878B5", alpha=0.82)
axes[0].set(title="Track-64 text cumulative increment", xlabel="Referenced LOS increment (mm)", ylabel="Pixel count")
sample = np.flatnonzero(valid)[::max(1, int(valid.sum() // 12000))]
axes[1].scatter(ifg_delta_ref[sample], text_delta_ref[sample], s=2, alpha=0.25, color="#D1495B")
limit = float(np.nanpercentile(np.abs(np.r_[ifg_delta_ref[valid], text_delta_ref[valid]]), 99))
axes[1].plot([-limit, limit], [-limit, limit], color="0.2", lw=1)
axes[1].set(title=f"Text increment vs independent IFG (r={correlation:.3f})", xlabel="Independent T64 IFG (mm)", ylabel="Text increment (mm)", xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
fig.savefig(OUTPUT_DIR / "track64_text_ifg_consistency.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Save a reproducible input manifest

# %%
manifest = {
    "status": "verified ascending text cumulative input",
    "text_schema": "displacement_mm latitude_deg longitude_deg",
    "track": "Sentinel-1 Track 64 ascending",
    "text_epoch_count": int(len(ascending_stack.dates)),
    "text_date_range": [str(ascending_stack.dates.min().date()), str(ascending_stack.dates.max().date())],
    "canonical_text_pixel_count": int(len(ascending_stack.latitude)),
    "common_track64_track71_dates": int(len(common)),
    "track64_utc_time": str(TRACK64_UTC_TIME),
    "track71_utc_time": str(TRACK71_UTC_TIME),
    "look_vector_source": str(ASC_GEOMETRY_ROOT / f"{ASC_GEOMETRY_STEM}.{{E,N,U}}.tif"),
    "text_ifg_20190704_20190716_correlation": correlation,
    "text_ifg_20190704_20190716_scale": scale,
    "reference": "P597, 1.5 km disk used only for this consistency comparison",
    "temporal_warning": "Nominally common track dates differ by about 12 h; do not call 4–16 July exactly simultaneous or purely coseismic.",
}
(OUTPUT_DIR / "track64_text_timeseries_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
availability.to_csv(OUTPUT_DIR / "track64_text_epoch_availability.csv", index=False)
display(pd.DataFrame([manifest]).T)
