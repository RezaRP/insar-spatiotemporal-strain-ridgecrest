"""Station-level GNSS ENU to LiCSAR LOS projection utilities.

The module keeps two distinct geometries explicit:

* a frame-average vector reconstructed from LiCSAR portal heading and
  incidence metadata; and
* the local E/N/U unit vector sampled from LiCSAR geometry rasters.

LiCSAR LOS unit vectors point from the ground towards the satellite, so a
positive dot product denotes motion towards the satellite.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortalGeometry:
    """Frame-average LiCSAR viewing metadata."""

    track: str
    heading_deg: float
    incidence_deg: float
    acquisition_time_utc: str

    @property
    def look_vector(self) -> np.ndarray:
        return look_vector_from_heading_incidence(
            self.heading_deg, self.incidence_deg
        )


def look_vector_from_heading_incidence(
    heading_deg: float,
    incidence_deg: float,
) -> np.ndarray:
    """Return the ground-to-satellite LOS unit vector in local ENU.

    ``heading_deg`` is satellite flight azimuth clockwise from north, with
    negative angles allowed. ``incidence_deg`` is measured from local vertical.
    Sentinel-1 is right-looking. Under the LiCSAR ground-to-satellite
    convention the components are

    ``[-sin(i) cos(h), sin(i) sin(h), cos(i)]``.
    """

    heading = math.radians(float(heading_deg))
    incidence = math.radians(float(incidence_deg))
    vector = np.array(
        [
            -math.sin(incidence) * math.cos(heading),
            math.sin(incidence) * math.sin(heading),
            math.cos(incidence),
        ],
        dtype=float,
    )
    return vector / np.linalg.norm(vector)


def heading_incidence_from_look_vector(
    look_vector: np.ndarray,
) -> tuple[float, float]:
    """Recover LiCSAR-style heading and incidence from one ENU LOS vector."""

    vector = np.asarray(look_vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("LOS vector must be finite and non-zero")
    east, north, up = vector / norm
    incidence = math.degrees(math.acos(float(np.clip(up, -1.0, 1.0))))
    heading = math.degrees(math.atan2(float(north), float(-east)))
    return heading, incidence


def project_enu_mm(
    east_mm: np.ndarray | float,
    north_mm: np.ndarray | float,
    up_mm: np.ndarray | float,
    look_vector: np.ndarray,
) -> np.ndarray:
    """Project ENU displacement in millimetres into LiCSAR LOS."""

    vector = np.asarray(look_vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("LOS vector must be finite and non-zero")
    vector = vector / norm
    east, north, up = np.broadcast_arrays(
        np.asarray(east_mm, dtype=float),
        np.asarray(north_mm, dtype=float),
        np.asarray(up_mm, dtype=float),
    )
    return vector[0] * east + vector[1] * north + vector[2] * up


def project_enu_covariance_mm(
    sigma_east_mm: np.ndarray | float,
    sigma_north_mm: np.ndarray | float,
    sigma_up_mm: np.ndarray | float,
    look_vector: np.ndarray,
    *,
    corr_en: np.ndarray | float = 0.0,
    corr_eu: np.ndarray | float = 0.0,
    corr_nu: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Propagate ENU standard errors and correlations into LOS uncertainty."""

    vector = np.asarray(look_vector, dtype=float).reshape(3)
    vector = vector / np.linalg.norm(vector)
    se, sn, su, ren, reu, rnu = np.broadcast_arrays(
        np.asarray(sigma_east_mm, dtype=float),
        np.asarray(sigma_north_mm, dtype=float),
        np.asarray(sigma_up_mm, dtype=float),
        np.asarray(corr_en, dtype=float),
        np.asarray(corr_eu, dtype=float),
        np.asarray(corr_nu, dtype=float),
    )
    covariance_en = np.clip(ren, -1.0, 1.0) * se * sn
    covariance_eu = np.clip(reu, -1.0, 1.0) * se * su
    covariance_nu = np.clip(rnu, -1.0, 1.0) * sn * su
    variance = (
        vector[0] ** 2 * se**2
        + vector[1] ** 2 * sn**2
        + vector[2] ** 2 * su**2
        + 2.0 * vector[0] * vector[1] * covariance_en
        + 2.0 * vector[0] * vector[2] * covariance_eu
        + 2.0 * vector[1] * vector[2] * covariance_nu
    )
    return np.sqrt(np.maximum(variance, 0.0))


def project_tenv3_history(
    history: pd.DataFrame,
    look_vector: np.ndarray,
) -> pd.DataFrame:
    """Project an NGL ``.tenv3`` history while retaining component provenance."""

    required = {
        "date",
        "__east(m)",
        "_north(m)",
        "____up(m)",
        "sig_e(m)",
        "sig_n(m)",
        "sig_u(m)",
        "__corr_en",
        "__corr_eu",
        "__corr_nu",
    }
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"GNSS history is missing columns: {missing}")
    output = history.loc[:, sorted(required)].copy()
    output["east_mm"] = history["__east(m)"].to_numpy(float) * 1000.0
    output["north_mm"] = history["_north(m)"].to_numpy(float) * 1000.0
    output["up_mm"] = history["____up(m)"].to_numpy(float) * 1000.0
    output["los_mm"] = project_enu_mm(
        output["east_mm"], output["north_mm"], output["up_mm"], look_vector
    )
    output["los_sigma_mm"] = project_enu_covariance_mm(
        history["sig_e(m)"].to_numpy(float) * 1000.0,
        history["sig_n(m)"].to_numpy(float) * 1000.0,
        history["sig_u(m)"].to_numpy(float) * 1000.0,
        look_vector,
        corr_en=history["__corr_en"].to_numpy(float),
        corr_eu=history["__corr_eu"].to_numpy(float),
        corr_nu=history["__corr_nu"].to_numpy(float),
    )
    return output.sort_values("date").reset_index(drop=True)


def remove_baseline_median(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[np.ndarray, float, int]:
    """Remove one temporal median without fitting trend, scale, or ramp."""

    array = np.asarray(values, dtype=float)
    date_index = pd.DatetimeIndex(dates)
    use = (
        (date_index >= pd.Timestamp(start))
        & (date_index <= pd.Timestamp(end))
        & np.isfinite(array)
    )
    count = int(use.sum())
    if count < 3:
        raise ValueError("Fewer than three finite values occur in the baseline")
    baseline = float(np.nanmedian(array[use]))
    return array - baseline, baseline, count
