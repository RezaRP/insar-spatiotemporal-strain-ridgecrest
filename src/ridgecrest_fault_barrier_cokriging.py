"""Fixed local fault-barrier cokriging for two-track horizontal InSAR.

The model treats east and north displacement as a latent bivariate spatial
field.  Ascending and descending horizontal LOS observations are linear
projections of that field.  Each target is conditioned on paired ascending and
descending observations in a fixed local neighbourhood and on the ascending
observation collocated with the target.  The resulting weights are built once
and can be reused for every epoch in a cumulative time series.

The residual covariance is a Matérn-3/2 correlation multiplied by a 2 x 2
east/north coregionalization matrix.  An affine east/north drift makes the
estimator universal cokriging.  Samples whose straight connection to a target
intersects a supplied *finite* fault segment are excluded.  This prevents
borrowing directly across a mapped rupture while still allowing support around
the ends of a finite segment.

This is an interpolation model, not a strain estimator.  In particular, its
filled descending LOS values should be validated by spatial holdout before
they are used to calculate near-fault displacement gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.spatial import cKDTree


PREDICTION_NAMES = ("east_mm", "north_mm", "descending_los_mm")


@dataclass(frozen=True)
class FixedFaultBarrierCokriging:
    """A build-once, evaluate-many local universal-cokriging operator.

    ``weights[target]`` has three rows.  Local observations are ordered as
    paired sample ascending/descending LOS, optionally followed by the target
    ascending LOS.  Prediction rows correspond to east, north, and target
    descending LOS.
    """

    sample_xy_km: np.ndarray
    target_xy_km: np.ndarray
    target_ascending_look_en: np.ndarray
    target_descending_look_en: np.ndarray
    neighbour_indices: tuple[np.ndarray, ...]
    excluded_sample_indices: tuple[np.ndarray, ...]
    buffer_excluded_indices: tuple[np.ndarray, ...]
    barrier_excluded_indices: tuple[np.ndarray, ...]
    weights: tuple[np.ndarray, ...]
    posterior_covariance: np.ndarray
    candidate_count: np.ndarray
    buffer_excluded_count: np.ndarray
    blocked_count: np.ndarray
    eligible_count: np.ndarray
    used_count: np.ndarray
    covariance_condition_number: np.ndarray
    drift_condition_number: np.ndarray
    drift_rank: np.ndarray
    unbiasedness_error: np.ndarray
    posterior_min_eigenvalue_raw: np.ndarray
    applied_jitter_mm2: np.ndarray
    look_determinant_median: np.ndarray
    valid: np.ndarray
    support_radius_km: float
    length_scale_km: float
    drift_scale_km: float
    minimum_paired_samples: int
    maximum_paired_samples: int | None
    target_exclusion_radius_km: float
    condition_on_target_ascending: bool

    @property
    def prediction_names(self) -> tuple[str, str, str]:
        """Names matching the first axis of each local weight matrix."""

        return PREDICTION_NAMES


def matern32_correlation(distance_km: np.ndarray, length_scale_km: float) -> np.ndarray:
    """Return the Matérn-3/2 correlation at the supplied distances."""

    if not math.isfinite(float(length_scale_km)) or length_scale_km <= 0.0:
        raise ValueError("length_scale_km must be finite and positive")
    distance = np.asarray(distance_km, dtype=float)
    if np.any(~np.isfinite(distance)) or np.any(distance < 0.0):
        raise ValueError("Distances must be finite and non-negative")
    scaled = math.sqrt(3.0) * distance / float(length_scale_km)
    return (1.0 + scaled) * np.exp(-scaled)


def _cross_2d(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def finite_segment_crossing_mask(
    start_xy_km: np.ndarray,
    end_xy_km: np.ndarray,
    fault_segments_xy_km: np.ndarray | None,
    *,
    orientation_tolerance_km2: float = 1.0e-10,
) -> np.ndarray:
    """Identify links that intersect at least one finite fault segment.

    Parameters
    ----------
    start_xy_km, end_xy_km
        Either one point ``(2,)`` or matched arrays ``(link, 2)``.  A single
        start point is broadcast over all end points (and conversely).
    fault_segments_xy_km
        Array with shape ``(segment, 2, 2)``.  The endpoint geometry is
        respected: a link passing beyond a segment endpoint is not blocked.
        Touching or collinear overlap counts as an intersection.
    orientation_tolerance_km2
        Numerical zero tolerance for two-dimensional cross products.
    """

    start = np.asarray(start_xy_km, dtype=float)
    end = np.asarray(end_xy_km, dtype=float)
    if start.shape == (2,):
        start = start[None, :]
    if end.shape == (2,):
        end = end[None, :]
    if start.ndim != 2 or start.shape[1] != 2:
        raise ValueError("start_xy_km must have shape (2,) or (link, 2)")
    if end.ndim != 2 or end.shape[1] != 2:
        raise ValueError("end_xy_km must have shape (2,) or (link, 2)")
    if len(start) == 1 and len(end) != 1:
        start = np.repeat(start, len(end), axis=0)
    elif len(end) == 1 and len(start) != 1:
        end = np.repeat(end, len(start), axis=0)
    elif len(start) != len(end):
        raise ValueError("Start and end arrays must be broadcastable by rows")
    if not (np.isfinite(start).all() and np.isfinite(end).all()):
        raise ValueError("Link coordinates must be finite")
    if orientation_tolerance_km2 < 0.0:
        raise ValueError("orientation_tolerance_km2 must be non-negative")

    if fault_segments_xy_km is None:
        return np.zeros(len(start), dtype=bool)
    segments = np.asarray(fault_segments_xy_km, dtype=float)
    if segments.size == 0:
        return np.zeros(len(start), dtype=bool)
    if segments.ndim != 3 or segments.shape[1:] != (2, 2):
        raise ValueError("fault_segments_xy_km must have shape (segment, 2, 2)")
    if not np.isfinite(segments).all():
        raise ValueError("Fault-segment coordinates must be finite")
    if np.any(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) == 0.0):
        raise ValueError("Fault segments must have distinct endpoints")

    # Dimensions are link x fault x coordinate.  The bounding-box check avoids
    # interpreting line intersections outside the finite segment extents.
    p = start[:, None, :]
    q = end[:, None, :]
    a = segments[None, :, 0, :]
    b = segments[None, :, 1, :]
    tolerance = float(orientation_tolerance_km2)

    min_pq = np.minimum(p, q)
    max_pq = np.maximum(p, q)
    min_ab = np.minimum(a, b)
    max_ab = np.maximum(a, b)
    coordinate_tolerance = math.sqrt(tolerance)
    boxes_overlap = np.all(
        (max_pq + coordinate_tolerance >= min_ab)
        & (max_ab + coordinate_tolerance >= min_pq),
        axis=2,
    )

    pq = q - p
    ab = b - a
    orientation_1 = _cross_2d(pq, a - p)
    orientation_2 = _cross_2d(pq, b - p)
    orientation_3 = _cross_2d(ab, p - a)
    orientation_4 = _cross_2d(ab, q - a)

    def signs(values: np.ndarray) -> np.ndarray:
        output = np.zeros(values.shape, dtype=np.int8)
        output[values > tolerance] = 1
        output[values < -tolerance] = -1
        return output

    sign_1 = signs(orientation_1)
    sign_2 = signs(orientation_2)
    sign_3 = signs(orientation_3)
    sign_4 = signs(orientation_4)
    intersects = (
        boxes_overlap
        & (sign_1 * sign_2 <= 0)
        & (sign_3 * sign_4 <= 0)
    )
    return np.any(intersects, axis=1)


def _broadcast_look_vectors(
    values: np.ndarray,
    count: int,
    name: str,
) -> np.ndarray:
    look = np.asarray(values, dtype=float)
    if look.shape == (2,):
        look = np.repeat(look[None, :], count, axis=0)
    if look.shape != (count, 2):
        raise ValueError(f"{name} must have shape (2,) or ({count}, 2)")
    if not np.isfinite(look).all():
        raise ValueError(f"{name} must be finite")
    if np.any(np.linalg.norm(look, axis=1) <= 0.0):
        raise ValueError(f"{name} must not contain zero vectors")
    return look


def _broadcast_variance(
    values: float | np.ndarray,
    count: int,
    name: str,
) -> np.ndarray:
    variance = np.asarray(values, dtype=float)
    if variance.ndim == 0:
        variance = np.full(count, float(variance), dtype=float)
    if variance.shape != (count,):
        raise ValueError(f"{name} must be scalar or have shape ({count},)")
    if np.any(~np.isfinite(variance)) or np.any(variance < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    return variance


def _positive_definite_coregionalization(values: np.ndarray) -> np.ndarray:
    covariance = np.asarray(values, dtype=float)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        raise ValueError("latent_covariance_mm2 must be a finite 2 x 2 matrix")
    covariance = 0.5 * (covariance + covariance.T)
    if float(np.min(np.linalg.eigvalsh(covariance))) <= 0.0:
        raise ValueError("latent_covariance_mm2 must be positive definite")
    return covariance


def _factor_with_adaptive_jitter(
    covariance: np.ndarray,
    *,
    jitter_relative: float,
    jitter_absolute_mm2: float,
    maximum_attempts: int,
) -> tuple[tuple[np.ndarray, bool], np.ndarray, float] | None:
    """Cholesky-factor a covariance matrix, adding only required jitter."""

    diagonal_scale = max(float(np.max(np.diag(covariance))), 1.0)
    identity = np.eye(len(covariance), dtype=float)
    for attempt in range(maximum_attempts):
        if attempt == 0:
            jitter = 0.0
        else:
            jitter = max(
                float(jitter_absolute_mm2),
                float(jitter_relative) * diagonal_scale * (10.0 ** (attempt - 1)),
            )
        regularized = covariance + jitter * identity
        try:
            factor = cho_factor(regularized, lower=True, check_finite=False)
        except LinAlgError:
            continue
        return factor, regularized, jitter
    return None


def _local_drift_design(
    sample_offset_scaled: np.ndarray,
    sample_ascending_look_en: np.ndarray,
    sample_descending_look_en: np.ndarray,
    target_ascending_look_en: np.ndarray | None,
) -> np.ndarray:
    sample_count = len(sample_offset_scaled)
    target_row_count = int(target_ascending_look_en is not None)
    design = np.zeros((2 * sample_count + target_row_count, 6), dtype=float)
    dx = sample_offset_scaled[:, 0]
    dy = sample_offset_scaled[:, 1]
    for observation_offset, look in (
        (0, sample_ascending_look_en),
        (1, sample_descending_look_en),
    ):
        rows = np.arange(sample_count) * 2 + observation_offset
        design[rows, 0] = look[:, 0]
        design[rows, 1] = look[:, 0] * dx
        design[rows, 2] = look[:, 0] * dy
        design[rows, 3] = look[:, 1]
        design[rows, 4] = look[:, 1] * dx
        design[rows, 5] = look[:, 1] * dy
    if target_ascending_look_en is not None:
        design[-1, 0] = target_ascending_look_en[0]
        design[-1, 3] = target_ascending_look_en[1]
    return design


def _target_drift_design(target_descending_look_en: np.ndarray) -> np.ndarray:
    look_east, look_north = target_descending_look_en
    return np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [look_east, 0.0, 0.0, look_north, 0.0, 0.0],
        ],
        dtype=float,
    )


def build_fixed_fault_barrier_cokriging(
    sample_xy_km: np.ndarray,
    target_xy_km: np.ndarray,
    *,
    sample_ascending_look_en: np.ndarray,
    sample_descending_look_en: np.ndarray,
    target_ascending_look_en: np.ndarray,
    target_descending_look_en: np.ndarray,
    fault_segments_xy_km: np.ndarray | None = None,
    latent_covariance_mm2: np.ndarray | None = None,
    length_scale_km: float = 8.0,
    support_radius_km: float = 24.0,
    drift_scale_km: float | None = None,
    sample_ascending_noise_variance_mm2: float | np.ndarray = 4.0,
    sample_descending_noise_variance_mm2: float | np.ndarray = 4.0,
    target_ascending_noise_variance_mm2: float | np.ndarray = 4.0,
    condition_on_target_ascending: bool = True,
    target_exclusion_radius_km: float = 0.0,
    minimum_paired_samples: int = 8,
    maximum_paired_samples: int | None = 64,
    maximum_covariance_condition_number: float = 1.0e12,
    maximum_drift_condition_number: float = 1.0e10,
    jitter_relative: float = 1.0e-10,
    jitter_absolute_mm2: float = 1.0e-10,
    maximum_jitter_attempts: int = 8,
    orientation_tolerance_km2: float = 1.0e-10,
) -> FixedFaultBarrierCokriging:
    """Build fixed local cokriging weights for all target pixels.

    The latent mean is affine in local east/north coordinates:

    ``E = E0 + E_x dx + E_y dy`` and
    ``N = N0 + N_x dx + N_y dy``.

    LOS look vectors contain only their east and north coefficients.  They
    should retain the physical horizontal magnitudes from the full LOS unit
    vector; they are not normalized within this function.

    Set ``condition_on_target_ascending=False`` to construct a paired-sample
    baseline for buffered cross-validation.  ``target_exclusion_radius_km``
    removes paired samples inside a target-centred buffer in both modes.  The
    exact per-target exclusions are retained in ``excluded_sample_indices``.
    """

    samples = np.asarray(sample_xy_km, dtype=float)
    targets = np.asarray(target_xy_km, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("sample_xy_km must have shape (sample, 2)")
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("target_xy_km must have shape (target, 2)")
    if not (np.isfinite(samples).all() and np.isfinite(targets).all()):
        raise ValueError("Sample and target coordinates must be finite")
    if support_radius_km <= 0.0 or length_scale_km <= 0.0:
        raise ValueError("Support radius and length scale must be positive")
    if (
        not math.isfinite(float(target_exclusion_radius_km))
        or target_exclusion_radius_km < 0.0
        or target_exclusion_radius_km >= support_radius_km
    ):
        raise ValueError(
            "target_exclusion_radius_km must be finite, non-negative, "
            "and smaller than support_radius_km"
        )
    if drift_scale_km is None:
        drift_scale_km = float(length_scale_km)
    if drift_scale_km <= 0.0 or not math.isfinite(float(drift_scale_km)):
        raise ValueError("drift_scale_km must be finite and positive")
    if minimum_paired_samples < 3:
        raise ValueError("minimum_paired_samples must be at least 3")
    if maximum_paired_samples is not None:
        if maximum_paired_samples < minimum_paired_samples:
            raise ValueError(
                "maximum_paired_samples must not be smaller than the minimum"
            )
    if maximum_jitter_attempts < 1:
        raise ValueError("maximum_jitter_attempts must be positive")

    sample_count_total = len(samples)
    target_count = len(targets)
    asc_sample_look = _broadcast_look_vectors(
        sample_ascending_look_en,
        sample_count_total,
        "sample_ascending_look_en",
    )
    desc_sample_look = _broadcast_look_vectors(
        sample_descending_look_en,
        sample_count_total,
        "sample_descending_look_en",
    )
    asc_target_look = _broadcast_look_vectors(
        target_ascending_look_en,
        target_count,
        "target_ascending_look_en",
    )
    desc_target_look = _broadcast_look_vectors(
        target_descending_look_en,
        target_count,
        "target_descending_look_en",
    )
    asc_sample_noise = _broadcast_variance(
        sample_ascending_noise_variance_mm2,
        sample_count_total,
        "sample_ascending_noise_variance_mm2",
    )
    desc_sample_noise = _broadcast_variance(
        sample_descending_noise_variance_mm2,
        sample_count_total,
        "sample_descending_noise_variance_mm2",
    )
    asc_target_noise = _broadcast_variance(
        target_ascending_noise_variance_mm2,
        target_count,
        "target_ascending_noise_variance_mm2",
    )
    if latent_covariance_mm2 is None:
        latent_covariance_mm2 = np.eye(2, dtype=float) * 400.0
    latent_covariance = _positive_definite_coregionalization(
        latent_covariance_mm2
    )

    if fault_segments_xy_km is not None:
        fault_segments = np.asarray(fault_segments_xy_km, dtype=float)
        # Reuse the public validator even if no target has neighbours.
        finite_segment_crossing_mask(
            np.zeros((0, 2), dtype=float),
            np.zeros((0, 2), dtype=float),
            fault_segments,
            orientation_tolerance_km2=orientation_tolerance_km2,
        )
    else:
        fault_segments = None

    tree = cKDTree(samples)
    neighbours: list[np.ndarray] = []
    excluded_indices: list[np.ndarray] = []
    buffer_excluded_indices: list[np.ndarray] = []
    barrier_excluded_indices: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    posterior_covariance = np.full((target_count, 3, 3), np.nan, dtype=float)
    candidate_count = np.zeros(target_count, dtype=int)
    buffer_excluded_count = np.zeros(target_count, dtype=int)
    blocked_count = np.zeros(target_count, dtype=int)
    eligible_count = np.zeros(target_count, dtype=int)
    used_count = np.zeros(target_count, dtype=int)
    covariance_condition = np.full(target_count, np.nan, dtype=float)
    drift_condition = np.full(target_count, np.nan, dtype=float)
    drift_rank = np.zeros(target_count, dtype=int)
    unbiasedness_error = np.full(target_count, np.nan, dtype=float)
    posterior_min_eigenvalue_raw = np.full(target_count, np.nan, dtype=float)
    applied_jitter = np.full(target_count, np.nan, dtype=float)
    look_determinant_median = np.full(target_count, np.nan, dtype=float)
    valid = np.zeros(target_count, dtype=bool)

    for target_index, target in enumerate(targets):
        candidate = np.asarray(
            tree.query_ball_point(target, r=float(support_radius_km)),
            dtype=int,
        )
        if candidate.size:
            distance = np.linalg.norm(samples[candidate] - target[None, :], axis=1)
            order = np.lexsort((candidate, distance))
            candidate = candidate[order]
        candidate_count[target_index] = len(candidate)

        if target_exclusion_radius_km > 0.0 and candidate.size:
            candidate_distance = np.linalg.norm(
                samples[candidate] - target[None, :],
                axis=1,
            )
            inside_buffer = (
                candidate_distance <= float(target_exclusion_radius_km)
            )
        else:
            inside_buffer = np.zeros(len(candidate), dtype=bool)
        buffer_excluded = candidate[inside_buffer]
        candidate_after_buffer = candidate[~inside_buffer]
        buffer_excluded_indices.append(buffer_excluded)
        buffer_excluded_count[target_index] = len(buffer_excluded)

        crossing = finite_segment_crossing_mask(
            target,
            samples[candidate_after_buffer],
            fault_segments,
            orientation_tolerance_km2=orientation_tolerance_km2,
        )
        blocked_count[target_index] = int(crossing.sum())
        barrier_excluded = candidate_after_buffer[crossing]
        barrier_excluded_indices.append(barrier_excluded)
        excluded_indices.append(
            np.sort(
                np.concatenate([buffer_excluded, barrier_excluded])
            )
        )
        eligible = candidate_after_buffer[~crossing]
        eligible_count[target_index] = len(eligible)
        if maximum_paired_samples is not None:
            used = eligible[: int(maximum_paired_samples)]
        else:
            used = eligible
        used_count[target_index] = len(used)
        neighbours.append(used)
        if len(used) < int(minimum_paired_samples):
            weights.append(np.empty((3, 0), dtype=float))
            continue

        local_asc_look = asc_sample_look[used]
        local_desc_look = desc_sample_look[used]
        determinants = (
            local_asc_look[:, 0] * local_desc_look[:, 1]
            - local_asc_look[:, 1] * local_desc_look[:, 0]
        )
        look_determinant_median[target_index] = float(
            np.median(np.abs(determinants))
        )

        sample_offset = (samples[used] - target[None, :]) / float(drift_scale_km)
        drift_observation = _local_drift_design(
            sample_offset,
            local_asc_look,
            local_desc_look,
            (
                asc_target_look[target_index]
                if condition_on_target_ascending
                else None
            ),
        )
        rank = int(np.linalg.matrix_rank(drift_observation))
        drift_rank[target_index] = rank
        if rank < 6:
            weights.append(np.empty((3, 0), dtype=float))
            continue

        target_observation_count = int(condition_on_target_ascending)
        observation_xy = np.empty(
            (2 * len(used) + target_observation_count, 2),
            dtype=float,
        )
        observation_xy[: 2 * len(used) : 2] = samples[used]
        observation_xy[1 : 2 * len(used) : 2] = samples[used]
        observation_look = np.empty_like(observation_xy)
        observation_look[: 2 * len(used) : 2] = local_asc_look
        observation_look[1 : 2 * len(used) : 2] = local_desc_look
        observation_noise = np.empty(len(observation_xy), dtype=float)
        observation_noise[: 2 * len(used) : 2] = asc_sample_noise[used]
        observation_noise[1 : 2 * len(used) : 2] = desc_sample_noise[used]
        if condition_on_target_ascending:
            observation_xy[-1] = target
            observation_look[-1] = asc_target_look[target_index]
            observation_noise[-1] = asc_target_noise[target_index]

        pairwise_distance = np.linalg.norm(
            observation_xy[:, None, :] - observation_xy[None, :, :],
            axis=2,
        )
        residual_projection = (
            observation_look @ latent_covariance @ observation_look.T
        )
        observation_covariance = (
            matern32_correlation(pairwise_distance, length_scale_km)
            * residual_projection
        )
        observation_covariance.flat[:: len(observation_covariance) + 1] += (
            observation_noise
        )

        factor_result = _factor_with_adaptive_jitter(
            observation_covariance,
            jitter_relative=jitter_relative,
            jitter_absolute_mm2=jitter_absolute_mm2,
            maximum_attempts=maximum_jitter_attempts,
        )
        if factor_result is None:
            weights.append(np.empty((3, 0), dtype=float))
            continue
        factor, regularized_covariance, jitter = factor_result
        applied_jitter[target_index] = jitter
        eigenvalue = np.linalg.eigvalsh(regularized_covariance)
        covariance_condition[target_index] = float(
            np.max(eigenvalue) / np.min(eigenvalue)
        )
        if covariance_condition[target_index] > float(
            maximum_covariance_condition_number
        ):
            weights.append(np.empty((3, 0), dtype=float))
            continue

        target_projection = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                desc_target_look[target_index],
            ],
            dtype=float,
        )
        observation_target_distance = np.linalg.norm(
            observation_xy - target[None, :],
            axis=1,
        )
        covariance_observation_target = (
            matern32_correlation(
                observation_target_distance,
                length_scale_km,
            )[:, None]
            * (
                observation_look
                @ latent_covariance
                @ target_projection.T
            )
        )
        covariance_target = (
            target_projection @ latent_covariance @ target_projection.T
        )
        drift_target = _target_drift_design(
            desc_target_look[target_index]
        )

        inverse_covariance_drift = cho_solve(
            factor,
            drift_observation,
            check_finite=False,
        )
        inverse_covariance_cross = cho_solve(
            factor,
            covariance_observation_target,
            check_finite=False,
        )
        drift_normal = drift_observation.T @ inverse_covariance_drift
        drift_condition[target_index] = float(np.linalg.cond(drift_normal))
        if (
            not math.isfinite(drift_condition[target_index])
            or drift_condition[target_index]
            > float(maximum_drift_condition_number)
        ):
            weights.append(np.empty((3, 0), dtype=float))
            continue
        inverse_drift_normal = np.linalg.inv(drift_normal)
        universal_residual = (
            drift_target.T
            - drift_observation.T @ inverse_covariance_cross
        )
        lambda_matrix = (
            inverse_covariance_cross
            + inverse_covariance_drift
            @ inverse_drift_normal
            @ universal_residual
        )
        local_weights = lambda_matrix.T
        weights.append(local_weights)

        unbiasedness_error[target_index] = float(
            np.max(
                np.abs(
                    local_weights @ drift_observation
                    - drift_target
                )
            )
        )
        posterior_raw = (
            covariance_target
            - covariance_observation_target.T
            @ inverse_covariance_cross
            + universal_residual.T
            @ inverse_drift_normal
            @ universal_residual
        )
        posterior_raw = 0.5 * (posterior_raw + posterior_raw.T)
        posterior_eigenvalue, posterior_eigenvector = np.linalg.eigh(posterior_raw)
        posterior_min_eigenvalue_raw[target_index] = float(
            np.min(posterior_eigenvalue)
        )
        clipped_eigenvalue = np.maximum(posterior_eigenvalue, 0.0)
        posterior_covariance[target_index] = (
            posterior_eigenvector
            * clipped_eigenvalue[None, :]
        ) @ posterior_eigenvector.T
        valid[target_index] = True

    return FixedFaultBarrierCokriging(
        sample_xy_km=samples,
        target_xy_km=targets,
        target_ascending_look_en=asc_target_look,
        target_descending_look_en=desc_target_look,
        neighbour_indices=tuple(neighbours),
        excluded_sample_indices=tuple(excluded_indices),
        buffer_excluded_indices=tuple(buffer_excluded_indices),
        barrier_excluded_indices=tuple(barrier_excluded_indices),
        weights=tuple(weights),
        posterior_covariance=posterior_covariance,
        candidate_count=candidate_count,
        buffer_excluded_count=buffer_excluded_count,
        blocked_count=blocked_count,
        eligible_count=eligible_count,
        used_count=used_count,
        covariance_condition_number=covariance_condition,
        drift_condition_number=drift_condition,
        drift_rank=drift_rank,
        unbiasedness_error=unbiasedness_error,
        posterior_min_eigenvalue_raw=posterior_min_eigenvalue_raw,
        applied_jitter_mm2=applied_jitter,
        look_determinant_median=look_determinant_median,
        valid=valid,
        support_radius_km=float(support_radius_km),
        length_scale_km=float(length_scale_km),
        drift_scale_km=float(drift_scale_km),
        minimum_paired_samples=int(minimum_paired_samples),
        maximum_paired_samples=(
            None
            if maximum_paired_samples is None
            else int(maximum_paired_samples)
        ),
        target_exclusion_radius_km=float(target_exclusion_radius_km),
        condition_on_target_ascending=bool(condition_on_target_ascending),
    )


def _as_epoch_by_location(
    values: np.ndarray,
    location_count: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != location_count:
        raise ValueError(
            f"{name} must have shape (epoch, {location_count}) "
            f"or ({location_count},)"
        )
    return array


def evaluate_fixed_fault_barrier_cokriging(
    model: FixedFaultBarrierCokriging,
    sample_ascending_los_mm: np.ndarray,
    sample_descending_los_mm: np.ndarray,
    target_ascending_los_mm: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Apply fixed cokriging weights to one or many cumulative epochs.

    An output is left ``NaN`` for an epoch/target whenever any observation
    required by that target's fixed operator is missing at that epoch.  Target
    ascending LOS is optional for an unconditioned baseline; if supplied, it
    is used only to report the reconstruction residual.
    """

    sample_count = len(model.sample_xy_km)
    target_count = len(model.target_xy_km)
    sample_ascending = _as_epoch_by_location(
        sample_ascending_los_mm,
        sample_count,
        "sample_ascending_los_mm",
    )
    sample_descending = _as_epoch_by_location(
        sample_descending_los_mm,
        sample_count,
        "sample_descending_los_mm",
    )
    if target_ascending_los_mm is None:
        if model.condition_on_target_ascending:
            raise ValueError(
                "target_ascending_los_mm is required because the model "
                "conditions on the collocated ascending observation"
            )
        target_ascending = np.full(
            (sample_ascending.shape[0], target_count),
            np.nan,
            dtype=float,
        )
    else:
        target_ascending = _as_epoch_by_location(
            target_ascending_los_mm,
            target_count,
            "target_ascending_los_mm",
        )
    if sample_ascending.shape != sample_descending.shape:
        raise ValueError("Paired sample LOS arrays must have identical shapes")
    if target_ascending.shape[0] != sample_ascending.shape[0]:
        raise ValueError("Sample and target LOS arrays must share the epoch axis")

    epoch_count = sample_ascending.shape[0]
    prediction = np.full((epoch_count, target_count, 3), np.nan, dtype=float)
    for target_index in np.flatnonzero(model.valid):
        neighbour = model.neighbour_indices[target_index]
        target_observation_count = int(model.condition_on_target_ascending)
        observation = np.empty(
            (epoch_count, 2 * len(neighbour) + target_observation_count),
            dtype=float,
        )
        observation[:, : 2 * len(neighbour) : 2] = (
            sample_ascending[:, neighbour]
        )
        observation[:, 1 : 2 * len(neighbour) : 2] = (
            sample_descending[:, neighbour]
        )
        if model.condition_on_target_ascending:
            observation[:, -1] = target_ascending[:, target_index]
        finite_epoch = np.isfinite(observation).all(axis=1)
        prediction[finite_epoch, target_index, :] = (
            observation[finite_epoch] @ model.weights[target_index].T
        )

    east = prediction[:, :, 0]
    north = prediction[:, :, 1]
    descending = prediction[:, :, 2]
    ascending_reconstructed = (
        east * model.target_ascending_look_en[None, :, 0]
        + north * model.target_ascending_look_en[None, :, 1]
    )
    ascending_residual = target_ascending - ascending_reconstructed
    return {
        "east_mm": east,
        "north_mm": north,
        "descending_los_mm": descending,
        "ascending_reconstructed_los_mm": ascending_reconstructed,
        "ascending_conditioning_residual_mm": ascending_residual,
    }


def cokriging_diagnostics(
    model: FixedFaultBarrierCokriging,
) -> dict[str, np.ndarray]:
    """Return target-level diagnostics without a pandas dependency."""

    posterior_sigma = np.sqrt(
        np.maximum(
            np.diagonal(model.posterior_covariance, axis1=1, axis2=2),
            0.0,
        )
    )
    return {
        "valid": model.valid.copy(),
        "candidate_count": model.candidate_count.copy(),
        "buffer_excluded_count": model.buffer_excluded_count.copy(),
        "blocked_count": model.blocked_count.copy(),
        "eligible_count": model.eligible_count.copy(),
        "used_count": model.used_count.copy(),
        "covariance_condition_number": (
            model.covariance_condition_number.copy()
        ),
        "drift_condition_number": model.drift_condition_number.copy(),
        "drift_rank": model.drift_rank.copy(),
        "unbiasedness_error": model.unbiasedness_error.copy(),
        "applied_jitter_mm2": model.applied_jitter_mm2.copy(),
        "look_determinant_median": model.look_determinant_median.copy(),
        "posterior_sigma_east_mm": posterior_sigma[:, 0],
        "posterior_sigma_north_mm": posterior_sigma[:, 1],
        "posterior_sigma_descending_los_mm": posterior_sigma[:, 2],
    }
