"""Conservative transient-acceleration analysis for Ridgecrest InSAR time series.

The functions in this module separate two complementary questions:

1. Does a prespecified regional time series favor a gradual velocity change or
   acceleration over a linear-plus-annual background?
2. Does a compatible transient form a spatially coherent pattern after the
   complete search over cells, onset dates, and transient shapes is corrected?

All times are expressed in decimal years from the first acquisition.  The
spatial analysis uses fixed grid cells rather than selecting individual pixels
after inspecting their time series.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

from ridgecrest_jump import huber_irls

SECONDS_PER_YEAR = 365.2425 * 86400.0


@dataclass(frozen=True)
class PatchSeries:
    dates: pd.DatetimeIndex
    values: np.ndarray  # (time, patch)
    metadata: pd.DataFrame
    reference: np.ndarray
    quality_mask: np.ndarray
    target_mask: np.ndarray


def time_years(dates: pd.DatetimeIndex) -> np.ndarray:
    dates = pd.DatetimeIndex(dates)
    return ((dates - dates[0]).total_seconds() / SECONDS_PER_YEAR).to_numpy(float)


def transient_design(
    t: np.ndarray,
    model: str,
    tau: float | None = None,
) -> np.ndarray:
    """Return linear-plus-annual, hinge, or accelerating design matrices."""
    t = np.asarray(t, dtype=float)
    columns = [
        np.ones_like(t),
        t,
        np.sin(2.0 * np.pi * t),
        np.cos(2.0 * np.pi * t),
    ]
    if model == "baseline":
        return np.column_stack(columns)
    if tau is None:
        raise ValueError("tau is required for a transient model")
    hinge = np.maximum(0.0, t - tau)
    columns.append(hinge)
    if model == "acceleration":
        columns.append(0.5 * hinge**2)
    elif model != "hinge":
        raise ValueError(f"Unknown model: {model}")
    return np.column_stack(columns)


def _standardize_design(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    means = np.zeros(X.shape[1], dtype=float)
    scales = np.ones(X.shape[1], dtype=float)
    standardized = X.copy()
    for j in range(1, X.shape[1]):
        means[j] = float(np.mean(X[:, j]))
        scales[j] = max(float(np.std(X[:, j])), np.finfo(float).eps)
        standardized[:, j] = (X[:, j] - means[j]) / scales[j]
    return standardized, means, scales


def _prewhiten(y: np.ndarray, X: np.ndarray, rho: float) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if y.size != X.shape[0]:
        raise ValueError("y and X must have the same number of rows")
    factor = np.sqrt(max(1.0 - rho**2, np.finfo(float).eps))
    yw = np.empty_like(y)
    Xw = np.empty_like(X)
    yw[0] = factor * y[0]
    Xw[0] = factor * X[0]
    yw[1:] = y[1:] - rho * y[:-1]
    Xw[1:] = X[1:] - rho * X[:-1]
    return yw, Xw


def estimate_ar1(values: np.ndarray, X: np.ndarray) -> float:
    """Estimate a conservative acquisition-index AR(1) coefficient."""
    _, residual = huber_irls(np.asarray(X, float), np.asarray(values, float))
    denominator = float(residual[:-1] @ residual[:-1])
    if denominator <= np.finfo(float).eps:
        return 0.0
    rho = float((residual[:-1] @ residual[1:]) / denominator)
    return float(np.clip(rho, -0.80, 0.80))


def _nig_evidence_and_posterior(
    y: np.ndarray,
    X: np.ndarray,
    *,
    rho: float,
    prior_scale: float,
    a0: float = 1.0,
    b0: float = 1.0,
) -> dict[str, np.ndarray | float]:
    Xs, means, scales = _standardize_design(X)
    yw, Xw = _prewhiten(y, Xs, rho)
    n, p = Xw.shape
    prior_precision = np.eye(p, dtype=float) / prior_scale**2
    posterior_precision = prior_precision + Xw.T @ Xw
    sign0, logdet0 = np.linalg.slogdet(prior_precision)
    signn, logdetn = np.linalg.slogdet(posterior_precision)
    if sign0 <= 0 or signn <= 0:
        raise np.linalg.LinAlgError("Non-positive-definite posterior precision")
    rhs = Xw.T @ yw
    beta_n = np.linalg.solve(posterior_precision, rhs)
    an = a0 + n / 2.0
    bn = b0 + 0.5 * max(
        float(yw @ yw - beta_n @ posterior_precision @ beta_n),
        np.finfo(float).eps,
    )
    log_evidence = float(
        -0.5 * n * np.log(2.0 * np.pi)
        + 0.5 * (logdet0 - logdetn)
        + a0 * np.log(b0)
        - an * np.log(bn)
        + gammaln(an)
        - gammaln(a0)
    )
    return {
        "log_evidence": log_evidence,
        "beta": beta_n,
        "precision": posterior_precision,
        "a": float(an),
        "b": float(bn),
        "means": means,
        "scales": scales,
    }


def bayesian_transient_analysis(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    *,
    min_before: int = 20,
    min_after: int = 4,
    candidate_start_date: pd.Timestamp | str | None = None,
    candidate_end_date: pd.Timestamp | str | None = None,
    prior_scale: float = 3.0,
    n_samples: int = 4000,
    seed: int = 20260727,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, np.ndarray]]:
    """Compare background, velocity-change, and accelerating transient models.

    Model classes receive equal prior probability. Candidate onset dates have a
    uniform prior within each transient class. A single AR(1) coefficient,
    estimated under the background model, is held fixed for every comparison.
    Posterior curves are conditional on the maximum-evidence transient model
    and onset; the onset interval itself integrates over both transient models.
    """
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    dates = pd.DatetimeIndex(dates[valid])
    y_original = values[valid]
    if y_original.size < min_before + min_after:
        raise ValueError("Too few valid epochs for the requested segment sizes")
    y_mean = float(np.mean(y_original))
    y_scale = float(np.std(y_original))
    if y_scale <= np.finfo(float).eps:
        raise ValueError("The time series has zero variance")
    y = (y_original - y_mean) / y_scale
    t = time_years(dates)
    X0 = transient_design(t, "baseline")
    rho = estimate_ar1(y, X0)
    base = _nig_evidence_and_posterior(
        y, X0, rho=rho, prior_scale=prior_scale
    )

    rows: list[dict[str, object]] = []
    candidate_splits = list(range(min_before, y.size - min_after + 1))
    if candidate_start_date is not None:
        start_date = pd.Timestamp(candidate_start_date)
        candidate_splits = [
            split for split in candidate_splits if dates[split] >= start_date
        ]
    if candidate_end_date is not None:
        end_date = pd.Timestamp(candidate_end_date)
        candidate_splits = [
            split for split in candidate_splits if dates[split] <= end_date
        ]
    if not candidate_splits:
        raise ValueError("No onset dates remain inside the requested candidate window")

    for split in candidate_splits:
        tau = float(t[split])
        for model in ("hinge", "acceleration"):
            X = transient_design(t, model, tau)
            posterior = _nig_evidence_and_posterior(
                y, X, rho=rho, prior_scale=prior_scale
            )
            beta_robust, _ = huber_irls(X, y_original)
            rows.append(
                {
                    "model": model,
                    "onset_date": dates[split],
                    "split_index": split,
                    "log_evidence": posterior["log_evidence"],
                    "velocity_change_mm_per_year": float(beta_robust[4]),
                    "acceleration_mm_per_year2": (
                        float(beta_robust[5]) if model == "acceleration" else 0.0
                    ),
                }
            )
    table = pd.DataFrame(rows)
    class_logs = {"baseline": float(base["log_evidence"])}
    for model in ("hinge", "acceleration"):
        logs = table.loc[table["model"] == model, "log_evidence"].to_numpy(float)
        class_logs[model] = float(logsumexp(logs) - np.log(logs.size))
    names = ("baseline", "hinge", "acceleration")
    log_classes = np.array([class_logs[name] for name in names])
    probabilities = np.exp(log_classes - logsumexp(log_classes))
    probability_by_class = dict(zip(names, probabilities))

    transient_rows = table.copy()
    counts = transient_rows.groupby("model")["model"].transform("count")
    transient_rows["log_joint_transient"] = (
        transient_rows["log_evidence"] + np.log(0.5) - np.log(counts)
    )
    transient_norm = float(logsumexp(transient_rows["log_joint_transient"]))
    transient_rows["row_probability_given_transient"] = np.exp(
        transient_rows["log_joint_transient"] - transient_norm
    )
    onset_posterior = (
        transient_rows.groupby("onset_date", as_index=False)[
            "row_probability_given_transient"
        ]
        .sum()
        .rename(
            columns={
                "row_probability_given_transient": "onset_probability_given_transient"
            }
        )
        .sort_values("onset_date")
        .reset_index(drop=True)
    )
    cumulative = onset_posterior["onset_probability_given_transient"].cumsum()
    lower = min(
        int(np.searchsorted(cumulative.to_numpy(), 0.025)),
        len(onset_posterior) - 1,
    )
    upper = min(
        int(np.searchsorted(cumulative.to_numpy(), 0.975)),
        len(onset_posterior) - 1,
    )

    best = table.loc[table["log_evidence"].idxmax()]
    best_model = str(best["model"])
    best_split = int(best["split_index"])
    tau = float(t[best_split])
    X_best = transient_design(t, best_model, tau)
    posterior = _nig_evidence_and_posterior(
        y, X_best, rho=rho, prior_scale=prior_scale
    )
    rng = np.random.default_rng(seed)
    sigma2 = float(posterior["b"]) / rng.gamma(
        shape=float(posterior["a"]), scale=1.0, size=n_samples
    )
    covariance_base = np.linalg.inv(np.asarray(posterior["precision"], float))
    chol = np.linalg.cholesky(covariance_base)
    beta_samples = np.asarray(posterior["beta"], float)[:, None] + (
        chol @ rng.normal(size=(X_best.shape[1], n_samples))
    ) * np.sqrt(sigma2)[None, :]
    Xs, _, scales = _standardize_design(X_best)
    fitted_samples = y_mean + y_scale * (Xs @ beta_samples)

    trend_derivative = np.zeros_like(X_best)
    trend_derivative[:, 1] = 1.0
    acceleration_derivative = np.zeros_like(X_best)
    if best_model in {"hinge", "acceleration"}:
        after = (t >= tau).astype(float)
        trend_derivative[:, 4] = after
    if best_model == "acceleration":
        hinge = np.maximum(0.0, t - tau)
        trend_derivative[:, 5] = hinge
        acceleration_derivative[:, 5] = after
    derivative_standardized = trend_derivative / scales[None, :]
    acceleration_standardized = acceleration_derivative / scales[None, :]
    velocity_samples = y_scale * (derivative_standardized @ beta_samples)
    acceleration_samples = y_scale * (
        acceleration_standardized @ beta_samples
    )

    transient_columns = np.zeros_like(X_best)
    transient_columns[:, 4:] = X_best[:, 4:]
    transient_standardized = transient_columns / scales[None, :]
    transient_displacement_samples = y_scale * (
        transient_standardized @ beta_samples
    )

    raw_hinge_samples = y_scale * beta_samples[4] / scales[4]
    if best_model == "acceleration":
        raw_acceleration_samples = y_scale * beta_samples[5] / scales[5]
        p_primary_positive = float(np.mean(raw_acceleration_samples > 0.0))
        p_primary_negative = float(np.mean(raw_acceleration_samples < 0.0))
    else:
        raw_acceleration_samples = np.zeros(n_samples)
        p_primary_positive = float(np.mean(raw_hinge_samples > 0.0))
        p_primary_negative = float(np.mean(raw_hinge_samples < 0.0))

    log_transient_mixture = float(
        logsumexp(
            [
                class_logs["hinge"] + np.log(0.5),
                class_logs["acceleration"] + np.log(0.5),
            ]
        )
    )
    summary: dict[str, object] = {
        "p_baseline": float(probability_by_class["baseline"]),
        "p_hinge": float(probability_by_class["hinge"]),
        "p_acceleration": float(probability_by_class["acceleration"]),
        "p_transient": float(
            probability_by_class["hinge"]
            + probability_by_class["acceleration"]
        ),
        "bayes_factor_transient_vs_baseline": float(
            np.exp(log_transient_mixture - class_logs["baseline"])
        ),
        "ar1_rho": rho,
        "best_model": best_model,
        "best_onset_date": pd.Timestamp(best["onset_date"]),
        "onset_ci95_start": pd.Timestamp(onset_posterior.iloc[lower]["onset_date"]),
        "onset_ci95_end": pd.Timestamp(onset_posterior.iloc[upper]["onset_date"]),
        "p_primary_coefficient_positive": p_primary_positive,
        "p_primary_coefficient_negative": p_primary_negative,
        "endpoint_transient_median_mm": float(
            np.median(transient_displacement_samples[-1])
        ),
        "endpoint_transient_ci95_low_mm": float(
            np.quantile(transient_displacement_samples[-1], 0.025)
        ),
        "endpoint_transient_ci95_high_mm": float(
            np.quantile(transient_displacement_samples[-1], 0.975)
        ),
    }
    table = (
        table.merge(
            transient_rows[
                ["model", "onset_date", "row_probability_given_transient"]
            ],
            on=["model", "onset_date"],
            how="left",
        )
        .merge(onset_posterior, on="onset_date", how="left")
        .sort_values("log_evidence", ascending=False, ignore_index=True)
    )
    samples = {
        "dates": dates.to_numpy(),
        "fitted": fitted_samples,
        "velocity": velocity_samples,
        "acceleration": acceleration_samples,
        "transient_displacement": transient_displacement_samples,
        "hinge_coefficient": raw_hinge_samples,
        "acceleration_coefficient": raw_acceleration_samples,
    }
    return summary, table, samples


def load_spatial_patch_series(
    h5_file: str | Path,
    *,
    center: tuple[float, float] = (35.74, -117.55),
    radius_km: float = 35.0,
    patch_size_km: float = 5.0,
    reference_box: tuple[int, int, int, int] = (497, 518, 50, 71),
    coherence_min: float = 0.30,
    residual_rms_max_mm: float = 2.0,
    max_gaps: float = 0.0,
    max_loop_errors: float = 5.0,
    min_pixels: int = 50,
    strike_deg: float = 140.0,
) -> PatchSeries:
    """Load fixed, quality-masked square-cell medians from a regional disk."""
    h5_file = Path(h5_file)
    with h5py.File(h5_file, "r") as h5:
        dates = pd.to_datetime(
            np.asarray(h5["imdates"][:], dtype=np.int64).astype(str),
            format="%Y%m%d",
        )
        _, ny, nx = h5["cum"].shape
        rows, cols = np.indices((ny, nx))
        latitude = float(h5["corner_lat"][()]) + rows * float(h5["post_lat"][()])
        longitude = float(h5["corner_lon"][()]) + cols * float(h5["post_lon"][()])
        lat0, lon0 = center
        north_km = (latitude - lat0) * 111.195
        east_km = (longitude - lon0) * 111.195 * np.cos(np.deg2rad(lat0))
        distance = np.hypot(east_km, north_km)
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
        target = quality & (distance <= radius_km)
        x1, x2, y1, y2 = reference_box
        reference_mask = np.zeros((ny, nx), dtype=bool)
        reference_mask[y1:y2, x1:x2] = quality[y1:y2, x1:x2]
        if np.count_nonzero(reference_mask) < 25:
            raise ValueError("Fewer than 25 valid pixels remain in reference box")

        east_bin = np.floor((east_km + radius_km) / patch_size_km).astype(int)
        north_bin = np.floor((north_km + radius_km) / patch_size_km).astype(int)
        nbin = int(np.ceil(2.0 * radius_km / patch_size_km))
        patch_id_grid = north_bin * nbin + east_bin
        selected_ids, counts = np.unique(patch_id_grid[target], return_counts=True)
        selected_ids = selected_ids[counts >= min_pixels]
        masks = [target & (patch_id_grid == patch_id) for patch_id in selected_ids]
        if not masks:
            raise ValueError("No spatial patches meet the requested pixel count")

        azimuth = np.deg2rad(strike_deg)
        metadata_rows: list[dict[str, float | int]] = []
        for patch_id, mask in zip(selected_ids, masks):
            east = float(np.mean(east_km[mask]))
            north = float(np.mean(north_km[mask]))
            metadata_rows.append(
                {
                    "patch_id": int(patch_id),
                    "latitude": float(np.mean(latitude[mask])),
                    "longitude": float(np.mean(longitude[mask])),
                    "east_km": east,
                    "north_km": north,
                    "distance_km": float(np.hypot(east, north)),
                    "along_strike_km": float(
                        east * np.sin(azimuth) + north * np.cos(azimuth)
                    ),
                    "cross_strike_km": float(
                        east * np.cos(azimuth) - north * np.sin(azimuth)
                    ),
                    "pixel_count": int(np.count_nonzero(mask)),
                }
            )

        values = np.empty((len(dates), len(masks)), dtype=float)
        reference = np.empty(len(dates), dtype=float)
        for i in range(len(dates)):
            epoch = np.asarray(h5["cum"][i], dtype=np.float32)
            reference[i] = float(np.nanmedian(epoch[reference_mask]))
            for j, mask in enumerate(masks):
                values[i, j] = float(np.nanmedian(epoch[mask])) - reference[i]

    return PatchSeries(
        dates=pd.DatetimeIndex(dates),
        values=values,
        metadata=pd.DataFrame(metadata_rows),
        reference=reference,
        quality_mask=quality,
        target_mask=target,
    )


def _residualize(X: np.ndarray, values: np.ndarray) -> np.ndarray:
    beta = np.linalg.lstsq(X, values, rcond=None)[0]
    return values - X @ beta


def _matched_filter_statistics(
    residual: np.ndarray,
    templates: np.ndarray,
    *,
    degrees_of_freedom: int,
) -> np.ndarray:
    """Return F-like one-degree-of-freedom template statistics."""
    scores = templates @ residual
    sse0 = np.sum(residual**2, axis=0)[None, :]
    sse1 = np.maximum(sse0 - scores**2, np.finfo(float).eps)
    return scores**2 / (sse1 / degrees_of_freedom)


def spatial_transient_matched_filter(
    patch_series: PatchSeries,
    *,
    min_before: int = 20,
    min_after: int = 4,
    candidate_start_date: pd.Timestamp | str | None = None,
    candidate_end_date: pd.Timestamp | str | None = None,
    n_boot: int = 499,
    block_length: int = 4,
    seed: int = 20260727,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    """Scan fixed patches with family-wise moving-block-bootstrap correction.

    Both a continuous velocity-change template and a quadratic acceleration
    template are searched.  The null maximum is taken over every patch, onset
    date, and template, so the reported corrected p-values account for the
    complete spatial and temporal search.
    """
    dates = patch_series.dates
    values = np.asarray(patch_series.values, dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("Patch series must be finite for spatial inference")
    t = time_years(dates)
    baseline = transient_design(t, "baseline")
    residual = _residualize(baseline, values)

    template_rows: list[np.ndarray] = []
    template_metadata: list[dict[str, object]] = []
    candidate_splits = list(range(min_before, len(dates) - min_after + 1))
    if candidate_start_date is not None:
        start_date = pd.Timestamp(candidate_start_date)
        candidate_splits = [
            split for split in candidate_splits if dates[split] >= start_date
        ]
    if candidate_end_date is not None:
        end_date = pd.Timestamp(candidate_end_date)
        candidate_splits = [
            split for split in candidate_splits if dates[split] <= end_date
        ]
    if not candidate_splits:
        raise ValueError("No onset dates remain inside the requested candidate window")

    for split in candidate_splits:
        tau = float(t[split])
        hinge = np.maximum(0.0, t - tau)
        for model, raw in (
            ("hinge", hinge),
            ("acceleration", 0.5 * hinge**2),
        ):
            projected = _residualize(baseline, raw[:, None]).ravel()
            norm = float(np.linalg.norm(projected))
            if norm <= np.finfo(float).eps:
                continue
            template_rows.append(projected / norm)
            template_metadata.append(
                {
                    "model": model,
                    "onset_date": pd.Timestamp(dates[split]),
                    "split_index": split,
                    "norm": norm,
                    "endpoint_value": float(raw[-1]),
                }
            )
    templates = np.vstack(template_rows)
    template_table = pd.DataFrame(template_metadata)
    degrees_of_freedom = len(dates) - baseline.shape[1] - 1
    observed = _matched_filter_statistics(
        residual, templates, degrees_of_freedom=degrees_of_freedom
    )
    best_template_index = np.argmax(observed, axis=0)
    patch_indices = np.arange(values.shape[1])
    best_statistic = observed[best_template_index, patch_indices]
    scores = templates @ residual
    best_score = scores[best_template_index, patch_indices]

    rng = np.random.default_rng(seed)
    starts = np.arange(max(1, len(dates) - block_length + 1))
    null_maxima = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled: list[int] = []
        while len(sampled) < len(dates):
            start = int(rng.choice(starts))
            sampled.extend(range(start, start + block_length))
        synthetic = residual[np.asarray(sampled[: len(dates)])]
        synthetic = _residualize(baseline, synthetic)
        statistics = _matched_filter_statistics(
            synthetic, templates, degrees_of_freedom=degrees_of_freedom
        )
        null_maxima[b] = float(np.max(statistics))

    corrected_p = np.array(
        [
            (1.0 + np.sum(null_maxima >= statistic)) / (n_boot + 1.0)
            for statistic in best_statistic
        ],
        dtype=float,
    )
    result = patch_series.metadata.copy()
    result["best_model"] = template_table.iloc[best_template_index][
        "model"
    ].to_numpy()
    result["best_onset_date"] = pd.to_datetime(
        template_table.iloc[best_template_index]["onset_date"].to_numpy()
    )
    result["max_statistic"] = best_statistic
    result["corrected_pvalue"] = corrected_p
    result["sign"] = np.sign(best_score)
    norms = template_table.iloc[best_template_index]["norm"].to_numpy(float)
    endpoints = template_table.iloc[best_template_index][
        "endpoint_value"
    ].to_numpy(float)
    coefficients = best_score / norms
    result["coefficient"] = coefficients
    result["endpoint_transient_mm"] = coefficients * endpoints
    result["fwer_significant_0_05"] = corrected_p < 0.05

    auxiliary = {
        "residual": residual,
        "templates": templates,
        "null_maxima": null_maxima,
        "observed_statistics": observed,
        "template_table": template_table,
    }
    return result, null_maxima, auxiliary


def quantile_band(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=float)
    return (
        np.quantile(samples, 0.025, axis=1),
        np.quantile(samples, 0.50, axis=1),
        np.quantile(samples, 0.975, axis=1),
    )
