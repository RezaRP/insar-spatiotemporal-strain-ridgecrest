"""Utilities for conservative change-point analysis of Ridgecrest text epochs.

The text files are expected to contain three whitespace-separated columns:
displacement, latitude, longitude.  The module deliberately separates data
loading, spatial referencing, and temporal model comparison so that a change
point is not mistaken for a physical interpretation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

DATE_FILE = re.compile(r"^(\d{8})\.txt$")


@dataclass(frozen=True)
class Stack:
    dates: pd.DatetimeIndex
    displacement: np.ndarray  # (time, point), float32
    latitude: np.ndarray
    longitude: np.ndarray
    files: tuple[Path, ...]


@dataclass(frozen=True)
class ReferencedSeries:
    dates: pd.DatetimeIndex
    values: np.ndarray
    roi_raw: np.ndarray
    reference: np.ndarray
    roi_mad: np.ndarray
    roi_pixel_count: int
    reference_pixel_count: int
    roi_mask: np.ndarray
    reference_mask: np.ndarray


def discover_epoch_files(data_dir: str | Path) -> tuple[tuple[Path, ...], pd.DatetimeIndex]:
    data_dir = Path(data_dir)
    pairs: list[tuple[pd.Timestamp, Path]] = []
    for path in data_dir.glob("*.txt"):
        match = DATE_FILE.match(path.name)
        if match:
            pairs.append((pd.to_datetime(match.group(1), format="%Y%m%d"), path))
    if not pairs:
        raise FileNotFoundError(f"No YYYYMMDD.txt epoch files found in {data_dir}")
    pairs.sort(key=lambda item: item[0])
    dates = pd.DatetimeIndex([item[0] for item in pairs])
    if dates.has_duplicates:
        raise ValueError("Duplicate acquisition dates were found")
    return tuple(item[1] for item in pairs), dates


def _coordinate_keys(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """Stable integer keys for coordinates written at approximately 1e-6 degree."""
    lat_i = np.rint(np.asarray(latitude, dtype=float) * 1_000_000).astype(np.int64)
    lon_i = np.rint(np.asarray(longitude, dtype=float) * 1_000_000).astype(np.int64)
    return lat_i * np.int64(1_000_000_000) + lon_i


def load_text_stack(data_dir: str | Path, *, align_coordinates: bool = True) -> Stack:
    files, dates = discover_epoch_files(data_dir)
    first = np.loadtxt(files[0], dtype=np.float32)
    if first.ndim != 2 or first.shape[1] != 3:
        raise ValueError(f"Expected three columns in {files[0]}, got {first.shape}")

    n_time, n_point = len(files), first.shape[0]
    displacement = np.empty((n_time, n_point), dtype=np.float32)
    displacement[0] = first[:, 0]
    latitude = first[:, 1].copy()
    longitude = first[:, 2].copy()
    canonical_keys = _coordinate_keys(latitude, longitude)
    if np.unique(canonical_keys).size != canonical_keys.size:
        raise ValueError(f"The canonical grid in {files[0]} contains duplicate coordinates")

    for i, path in enumerate(files[1:], start=1):
        epoch = np.loadtxt(path, dtype=np.float32)
        if epoch.ndim != 2 or epoch.shape[1] != 3:
            raise ValueError(f"Expected three columns in {path}, got {epoch.shape}")
        if epoch.shape == first.shape and np.array_equal(epoch[:, 1:], first[:, 1:]):
            displacement[i] = epoch[:, 0]
            continue
        if not align_coordinates:
            raise ValueError(f"Coordinate grid/order changed in {path}")

        # Some exports omit different pixels while retaining nearly the same
        # row count. Align explicitly; never let a row shift masquerade as a
        # temporal displacement jump.
        epoch_keys = _coordinate_keys(epoch[:, 1], epoch[:, 2])
        unique_keys, unique_indices = np.unique(epoch_keys, return_index=True)
        positions = np.searchsorted(unique_keys, canonical_keys)
        matched = positions < unique_keys.size
        matched[matched] &= unique_keys[positions[matched]] == canonical_keys[matched]
        row = np.full(n_point, np.nan, dtype=np.float32)
        row[matched] = epoch[unique_indices[positions[matched]], 0]
        displacement[i] = row

    return Stack(dates, displacement, latitude, longitude, files)


def distance_km(latitude: np.ndarray, longitude: np.ndarray, center: tuple[float, float]) -> np.ndarray:
    """Local equirectangular distance, adequate for this regional grid."""
    lat0, lon0 = center
    dy = (latitude - lat0) * 111.195
    dx = (longitude - lon0) * 111.195 * np.cos(np.deg2rad(lat0))
    return np.hypot(dx, dy)


def spatial_median_series(displacement: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        raise ValueError("The spatial mask selects no points")
    return np.nanmedian(displacement[:, mask], axis=1)


def spatial_mad_series(displacement: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    values = displacement[:, mask]
    med = np.nanmedian(values, axis=1, keepdims=True)
    return 1.4826 * np.nanmedian(np.abs(values - med), axis=1)


def _time_years(dates: pd.DatetimeIndex) -> np.ndarray:
    return ((dates - dates[0]).total_seconds() / (365.2425 * 86400.0)).to_numpy(float)


def _design_matrix(t: np.ndarray, model: str, tau: float | None = None) -> np.ndarray:
    columns = [np.ones_like(t), t, np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)]
    if model == "step":
        if tau is None:
            raise ValueError("tau is required for the step model")
        columns.append((t >= tau).astype(float))
    elif model == "hinge":
        if tau is None:
            raise ValueError("tau is required for the hinge model")
        columns.append(np.maximum(0.0, t - tau))
    elif model != "baseline":
        raise ValueError(f"Unknown model: {model}")
    return np.column_stack(columns)


def huber_irls(X: np.ndarray, y: np.ndarray, *, tuning: float = 1.345, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Robust linear fit using Huber iteratively reweighted least squares."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(max_iter):
        resid = y - X @ beta
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        scale = max(float(scale), np.finfo(float).eps)
        u = np.abs(resid) / (tuning * scale)
        weights = np.ones_like(u)
        large = u > 1.0
        weights[large] = 1.0 / u[large]
        root_w = np.sqrt(weights)
        updated = np.linalg.lstsq(X * root_w[:, None], y * root_w, rcond=None)[0]
        if np.allclose(updated, beta, rtol=1e-9, atol=1e-9):
            beta = updated
            break
        beta = updated
    return beta, y - X @ beta


def _bic(residual: np.ndarray, n_parameters: int) -> float:
    n = residual.size
    rss = max(float(np.sum(residual**2)), np.finfo(float).tiny)
    return n * np.log(rss / n) + n_parameters * np.log(n)


def scan_change_points(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    *,
    min_before: int = 20,
    min_after: int = 3,
) -> pd.DataFrame:
    """Compare baseline, step, and continuous slope-change models.

    The searched change date is counted as an additional free parameter in BIC.
    A positive delta_bic favors the change model over the baseline.
    """
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    dates = pd.DatetimeIndex(dates[valid])
    y = values[valid]
    if y.size < min_before + min_after:
        raise ValueError("Too few valid epochs for the requested segment sizes")
    t = _time_years(dates)
    X0 = _design_matrix(t, "baseline")
    beta0, resid0 = huber_irls(X0, y)
    bic0 = _bic(resid0, X0.shape[1])

    rows: list[dict[str, object]] = []
    for split in range(min_before, y.size - min_after + 1):
        tau = t[split]
        for model in ("step", "hinge"):
            X = _design_matrix(t, model, tau)
            beta, resid = huber_irls(X, y)
            bic = _bic(resid, X.shape[1] + 1)  # +1 for the searched change date
            rows.append(
                {
                    "model": model,
                    "change_date": dates[split],
                    "delta_bic": bic0 - bic,
                    "amplitude": beta[-1],
                    "bic": bic,
                    "baseline_bic": bic0,
                    "n": y.size,
                }
            )
    return pd.DataFrame(rows).sort_values("delta_bic", ascending=False, ignore_index=True)


def fitted_curve(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    *,
    model: str,
    change_date: pd.Timestamp | str | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    t = _time_years(pd.DatetimeIndex(dates))
    if model == "baseline":
        X = _design_matrix(t, model)
    else:
        if change_date is None:
            raise ValueError("change_date is required for a change model")
        tau = float((pd.Timestamp(change_date) - pd.Timestamp(dates[0])).total_seconds() / (365.2425 * 86400.0))
        X = _design_matrix(t, model, tau)
    beta, _ = huber_irls(X[valid], values[valid])
    return X @ beta


def moving_block_bootstrap_pvalue(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    observed_delta_bic: float,
    *,
    n_boot: int = 500,
    block_length: int = 4,
    min_before: int = 20,
    min_after: int = 3,
    seed: int = 20260721,
) -> tuple[float, np.ndarray]:
    """Family-wise p-value for the maximum searched change statistic."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    dates_valid = pd.DatetimeIndex(dates[valid])
    y = values[valid]
    baseline = fitted_curve(dates_valid, y, model="baseline")
    resid = y - baseline
    resid = resid - np.mean(resid)
    n = y.size
    rng = np.random.default_rng(seed)
    maxima = np.empty(n_boot, dtype=float)
    starts = np.arange(max(1, n - block_length + 1))
    for b in range(n_boot):
        sampled: list[float] = []
        while len(sampled) < n:
            start = int(rng.choice(starts))
            sampled.extend(resid[start : start + block_length])
        synthetic = baseline + np.asarray(sampled[:n])
        scan = scan_change_points(
            dates_valid,
            synthetic,
            min_before=min_before,
            min_after=min_after,
        )
        maxima[b] = float(scan.iloc[0]["delta_bic"])
    pvalue = (1.0 + np.sum(maxima >= observed_delta_bic)) / (n_boot + 1.0)
    return float(pvalue), maxima


def load_referenced_h5_series(
    h5_file: str | Path,
    *,
    center: tuple[float, float] = (35.74, -117.55),
    roi_radius_km: float = 10.0,
    reference_box: tuple[int, int, int, int] = (497, 518, 50, 71),
    coherence_min: float = 0.30,
    residual_rms_max_mm: float = 2.0,
    max_gaps: float = 0.0,
    max_loop_errors: float = 5.0,
) -> ReferencedSeries:
    """Extract a quality-masked ROI median and apply a fixed spatial reference.

    ``reference_box`` is ``(x1, x2, y1, y2)`` with Python/LiCSBAS-exclusive
    upper bounds.  Only the small ROI and reference box are read from each
    cumulative epoch; the full cube is never loaded into memory.
    """
    h5_file = Path(h5_file)
    with h5py.File(h5_file, "r") as h5:
        dates = pd.to_datetime(
            np.asarray(h5["imdates"][:], dtype=np.int64).astype(str),
            format="%Y%m%d",
        )
        _, ny, nx = h5["cum"].shape
        y, x = np.indices((ny, nx))
        latitude = float(h5["corner_lat"][()]) + y * float(h5["post_lat"][()])
        longitude = float(h5["corner_lon"][()]) + x * float(h5["post_lon"][()])
        dist = distance_km(latitude.ravel(), longitude.ravel(), center).reshape(
            ny, nx
        )

        quality = (
            np.isfinite(h5["coh_avg"][:])
            & (h5["coh_avg"][:] >= coherence_min)
            & np.isfinite(h5["resid_rms"][:])
            & (h5["resid_rms"][:] <= residual_rms_max_mm)
            & np.isfinite(h5["n_gap"][:])
            & (h5["n_gap"][:] <= max_gaps)
            & np.isfinite(h5["n_loop_err"][:])
            & (h5["n_loop_err"][:] <= max_loop_errors)
        )
        roi_mask = quality & (dist <= roi_radius_km)

        x1, x2, y1, y2 = reference_box
        if not (0 <= x1 < x2 <= nx and 0 <= y1 < y2 <= ny):
            raise ValueError(
                f"Reference box {reference_box} is outside grid {(ny, nx)}"
            )
        reference_mask = np.zeros((ny, nx), dtype=bool)
        reference_mask[y1:y2, x1:x2] = quality[y1:y2, x1:x2]
        if np.count_nonzero(roi_mask) == 0:
            raise ValueError("The quality mask removed every ROI pixel")
        if np.count_nonzero(reference_mask) < 25:
            raise ValueError("Fewer than 25 valid pixels remain in reference box")

        roi_raw = np.empty(len(dates), dtype=float)
        reference = np.empty(len(dates), dtype=float)
        roi_mad = np.empty(len(dates), dtype=float)
        for i in range(len(dates)):
            epoch = np.asarray(h5["cum"][i], dtype=np.float32)
            roi_values = epoch[roi_mask]
            reference_values = epoch[reference_mask]
            roi_raw[i] = float(np.nanmedian(roi_values))
            reference[i] = float(np.nanmedian(reference_values))
            roi_median = roi_raw[i]
            roi_mad[i] = float(
                1.4826 * np.nanmedian(np.abs(roi_values - roi_median))
            )

    return ReferencedSeries(
        dates=pd.DatetimeIndex(dates),
        values=roi_raw - reference,
        roi_raw=roi_raw,
        reference=reference,
        roi_mad=roi_mad,
        roi_pixel_count=int(np.count_nonzero(roi_mask)),
        reference_pixel_count=int(np.count_nonzero(reference_mask)),
        roi_mask=roi_mask,
        reference_mask=reference_mask,
    )


def _standardized_design(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float).copy()
    for j in range(1, X.shape[1]):
        mean = float(np.mean(X[:, j]))
        scale = float(np.std(X[:, j]))
        X[:, j] = X[:, j] - mean
        if scale > np.finfo(float).eps:
            X[:, j] /= scale
    return X


def _log_bayesian_linear_evidence(
    y: np.ndarray,
    X: np.ndarray,
    *,
    prior_scale: float = 10.0,
    a0: float = 1.0,
    b0: float = 1.0,
) -> float:
    """Conjugate normal-inverse-gamma marginal likelihood."""
    y = np.asarray(y, dtype=float)
    X = _standardized_design(X)
    n, p = X.shape
    prior_precision = np.eye(p, dtype=float) / prior_scale**2
    posterior_precision = prior_precision + X.T @ X
    sign0, logdet_precision0 = np.linalg.slogdet(prior_precision)
    signn, logdet_precisionn = np.linalg.slogdet(posterior_precision)
    if sign0 <= 0 or signn <= 0:
        raise np.linalg.LinAlgError("Non-positive-definite Bayesian precision")
    rhs = X.T @ y
    beta_n = np.linalg.solve(posterior_precision, rhs)
    an = a0 + n / 2.0
    bn = b0 + 0.5 * max(
        float(y @ y - beta_n @ posterior_precision @ beta_n),
        np.finfo(float).eps,
    )
    return float(
        -0.5 * n * np.log(2.0 * np.pi)
        + 0.5 * (logdet_precision0 - logdet_precisionn)
        + a0 * np.log(b0)
        - an * np.log(bn)
        + gammaln(an)
        - gammaln(a0)
    )


def bayesian_change_point(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    *,
    min_before: int = 20,
    min_after: int = 3,
    prior_scale: float = 10.0,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Bayesian comparison of baseline, level-step, and slope-change models.

    Model classes have equal prior probability.  Candidate dates within each
    change class have a uniform discrete prior.  The reported date interval is
    conditional on there being either a step or hinge change.
    """
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    dates = pd.DatetimeIndex(dates[valid])
    y = values[valid]
    if y.size < min_before + min_after:
        raise ValueError("Too few valid epochs for Bayesian change-point scan")

    y_scale = float(np.std(y))
    if y_scale <= np.finfo(float).eps:
        raise ValueError("The time series has zero variance")
    y_std = (y - np.mean(y)) / y_scale
    t = _time_years(dates)
    X0 = _design_matrix(t, "baseline")
    log_evidence_baseline = _log_bayesian_linear_evidence(
        y_std, X0, prior_scale=prior_scale
    )

    rows: list[dict[str, object]] = []
    for split in range(min_before, y.size - min_after + 1):
        tau = float(t[split])
        for model in ("step", "hinge"):
            X = _design_matrix(t, model, tau)
            log_evidence = _log_bayesian_linear_evidence(
                y_std, X, prior_scale=prior_scale
            )
            beta, _ = huber_irls(X, y)
            rows.append(
                {
                    "model": model,
                    "change_date": dates[split],
                    "log_evidence": log_evidence,
                    "amplitude": float(beta[-1]),
                }
            )
    table = pd.DataFrame(rows)
    class_log_evidence: dict[str, float] = {
        "baseline": log_evidence_baseline,
    }
    for model in ("step", "hinge"):
        model_logs = table.loc[table["model"] == model, "log_evidence"].to_numpy(
            float
        )
        class_log_evidence[model] = float(
            logsumexp(model_logs) - np.log(model_logs.size)
        )

    class_names = ("baseline", "step", "hinge")
    log_class = np.array([class_log_evidence[name] for name in class_names])
    class_probabilities = np.exp(log_class - logsumexp(log_class))
    probability_by_class = dict(zip(class_names, class_probabilities))

    change_rows = table.copy()
    model_prior_within_change = np.log(0.5)
    change_rows["log_joint_change"] = (
        change_rows["log_evidence"]
        + model_prior_within_change
        - change_rows.groupby("model")["model"].transform("count").map(np.log)
    )
    change_log_norm = float(logsumexp(change_rows["log_joint_change"]))
    change_rows["row_probability_given_change"] = np.exp(
        change_rows["log_joint_change"] - change_log_norm
    )
    date_posterior = (
        change_rows.groupby("change_date", as_index=False)[
            "row_probability_given_change"
        ]
        .sum()
        .rename(
            columns={
                "row_probability_given_change": "date_probability_given_change"
            }
        )
        .sort_values("change_date")
        .reset_index(drop=True)
    )
    cumulative = date_posterior["date_probability_given_change"].cumsum()
    lower_index = int(np.searchsorted(cumulative.to_numpy(), 0.025))
    upper_index = int(np.searchsorted(cumulative.to_numpy(), 0.975))
    lower_index = min(lower_index, len(date_posterior) - 1)
    upper_index = min(upper_index, len(date_posterior) - 1)

    best_row = table.loc[table["log_evidence"].idxmax()]
    log_evidence_change_mixture = float(
        logsumexp(
            [
                class_log_evidence["step"] + np.log(0.5),
                class_log_evidence["hinge"] + np.log(0.5),
            ]
        )
    )
    summary: dict[str, object] = {
        "p_baseline": float(probability_by_class["baseline"]),
        "p_step": float(probability_by_class["step"]),
        "p_hinge": float(probability_by_class["hinge"]),
        "bayes_factor_change_vs_baseline": float(
            np.exp(log_evidence_change_mixture - log_evidence_baseline)
        ),
        "best_model": str(best_row["model"]),
        "best_change_date": pd.Timestamp(best_row["change_date"]),
        "best_amplitude": float(best_row["amplitude"]),
        "change_date_ci95_start": pd.Timestamp(
            date_posterior.iloc[lower_index]["change_date"]
        ),
        "change_date_ci95_end": pd.Timestamp(
            date_posterior.iloc[upper_index]["change_date"]
        ),
    }
    table = table.merge(
        change_rows[
            ["model", "change_date", "row_probability_given_change"]
        ],
        on=["model", "change_date"],
        how="left",
    ).merge(date_posterior, on="change_date", how="left")
    return summary, table.sort_values(
        "log_evidence", ascending=False, ignore_index=True
    )


def _segment_constant_cost_prefix(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    return np.r_[0.0, np.cumsum(values)], np.r_[0.0, np.cumsum(values**2)]


def _segment_constant_cost(
    prefix: np.ndarray, prefix_square: np.ndarray, start: int, end: int
) -> float:
    n = end - start
    if n <= 0:
        return np.inf
    total = prefix[end] - prefix[start]
    total_square = prefix_square[end] - prefix_square[start]
    return max(float(total_square - total**2 / n), 0.0)


def optimal_partition_mean_shifts(
    values: np.ndarray,
    *,
    penalty: float,
    min_size: int = 5,
) -> tuple[list[int], float]:
    """Exact optimal partitioning for the PELT penalized objective.

    At n≈100, the unpruned dynamic program is inexpensive and returns the same
    optimum as PELT for this additive mean-shift cost.
    """
    values = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("Optimal partitioning requires finite values")
    n = values.size
    if n < 2 * min_size:
        raise ValueError("Too few values for requested minimum segment size")
    prefix, prefix_square = _segment_constant_cost_prefix(values)
    objective = np.full(n + 1, np.inf, dtype=float)
    previous = np.full(n + 1, -1, dtype=int)
    objective[0] = -penalty

    for end in range(min_size, n + 1):
        starts = np.arange(0, end - min_size + 1)
        starts = starts[(starts == 0) | (starts >= min_size)]
        candidates = np.array(
            [
                objective[start]
                + _segment_constant_cost(prefix, prefix_square, start, end)
                + penalty
                for start in starts
            ]
        )
        best = int(np.argmin(candidates))
        objective[end] = candidates[best]
        previous[end] = int(starts[best])

    breakpoints: list[int] = []
    end = n
    while previous[end] > 0:
        start = int(previous[end])
        breakpoints.append(start)
        end = start
    breakpoints.reverse()
    return breakpoints, float(objective[n])


def pelt_change_detection(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    *,
    penalty_multipliers: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
    min_size: int = 5,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Run penalty-stability segmentation on robust baseline residuals."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    dates = pd.DatetimeIndex(dates[valid])
    y = values[valid]
    baseline = fitted_curve(dates, y, model="baseline")
    residual = y - baseline
    robust_sigma = float(
        1.4826 * np.median(np.abs(residual - np.median(residual)))
    )
    robust_sigma = max(robust_sigma, np.finfo(float).eps)

    rows = []
    for multiplier in penalty_multipliers:
        penalty = float(multiplier * np.log(y.size) * robust_sigma**2)
        breakpoints, objective = optimal_partition_mean_shifts(
            residual, penalty=penalty, min_size=min_size
        )
        rows.append(
            {
                "penalty_multiplier": multiplier,
                "penalty": penalty,
                "n_breakpoints": len(breakpoints),
                "breakpoint_indices": tuple(breakpoints),
                "breakpoint_dates": tuple(dates[index] for index in breakpoints),
                "objective": objective,
            }
        )
    return pd.DataFrame(rows), residual


def frequentist_breakpoint_bootstrap(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    *,
    min_before: int = 20,
    min_after: int = 3,
    n_boot: int = 999,
    block_length: int = 4,
    seed: int = 20260727,
) -> dict[str, object]:
    """Calibrate the strongest residual mean shift with a moving-block null."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    dates = pd.DatetimeIndex(dates[valid])
    y = values[valid]
    baseline = fitted_curve(dates, y, model="baseline")
    residual = y - baseline
    residual -= np.mean(residual)
    n = residual.size
    prefix, prefix_square = _segment_constant_cost_prefix(residual)
    total_cost = _segment_constant_cost(prefix, prefix_square, 0, n)
    candidates = np.arange(min_before, n - min_after + 1)
    split_cost = np.array(
        [
            _segment_constant_cost(prefix, prefix_square, 0, split)
            + _segment_constant_cost(prefix, prefix_square, split, n)
            for split in candidates
        ]
    )
    improvements = total_cost - split_cost
    best_position = int(np.argmax(improvements))
    best_split = int(candidates[best_position])
    observed = float(improvements[best_position])

    rng = np.random.default_rng(seed)
    starts = np.arange(max(1, n - block_length + 1))
    null_maxima = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled: list[float] = []
        while len(sampled) < n:
            start = int(rng.choice(starts))
            sampled.extend(residual[start : start + block_length])
        synthetic = np.asarray(sampled[:n], dtype=float)
        synthetic -= np.mean(synthetic)
        pfx, pfx2 = _segment_constant_cost_prefix(synthetic)
        base_cost = _segment_constant_cost(pfx, pfx2, 0, n)
        null_maxima[b] = max(
            base_cost
            - (
                _segment_constant_cost(pfx, pfx2, 0, split)
                + _segment_constant_cost(pfx, pfx2, split, n)
            )
            for split in candidates
        )
    pvalue = float((1 + np.sum(null_maxima >= observed)) / (n_boot + 1))
    return {
        "change_date": pd.Timestamp(dates[best_split]),
        "split_index": best_split,
        "improvement": observed,
        "pvalue": pvalue,
        "n_boot": n_boot,
        "block_length": block_length,
        "null_maxima": null_maxima,
        "residual": residual,
        "baseline": baseline,
    }
