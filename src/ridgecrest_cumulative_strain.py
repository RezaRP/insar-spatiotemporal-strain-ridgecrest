"""Fixed-operator cumulative horizontal-strain estimation.

The routines in this module estimate a local affine displacement gradient from
paired cumulative east/north displacement fields.  The neighbourhoods,
spatial kernel, and generalized least-squares operators are constructed once
and then reused at every epoch.  This is important for a time series: changes
in the reported strain must not be caused by changing derivative geometry.

Without supplied fault barriers, the estimator assumes a continuous
displacement field inside each local support.  With finite fault barriers,
target-to-sample links crossing those barriers are removed; values evaluated
inside a mapped rupture zone are therefore finite-resolution, locally
regularized displacement gradients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class FixedJointMLS:
    """Precomputed joint E/N moving-least-squares derivative operators."""

    target_xy_km: np.ndarray
    neighbour_indices: tuple[np.ndarray, ...]
    operators: tuple[np.ndarray, ...]
    parameter_covariance: np.ndarray
    sample_count: np.ndarray
    effective_sample_count: np.ndarray
    condition_number: np.ndarray
    barrier_excluded_count: np.ndarray
    valid: np.ndarray
    support_radius_km: float
    bandwidth_km: float


def _connection_crosses_finite_segments(
    target_xy_km: np.ndarray,
    sample_xy_km: np.ndarray,
    segments_xy_km: np.ndarray,
    *,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Return samples whose connection to a target crosses a finite barrier.

    Intersections at target or sample endpoints are ignored so targets may be
    evaluated throughout the requested grid. Intersections at either endpoint
    of a barrier segment are retained so paths cannot leak around the end of a
    finite source-model trace by numerical accident.
    """

    samples = np.asarray(sample_xy_km, dtype=float)
    segments = np.asarray(segments_xy_km, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("sample_xy_km must have shape (sample, 2)")
    if segments.ndim != 3 or segments.shape[1:] != (2, 2):
        raise ValueError("segments_xy_km must have shape (segment, 2, 2)")
    target = np.asarray(target_xy_km, dtype=float)
    connection = samples - target[None, :]
    crossing = np.zeros(len(samples), dtype=bool)
    for segment in segments:
        barrier = segment[1] - segment[0]
        relative = segment[0] - target
        denominator = (
            connection[:, 0] * barrier[1]
            - connection[:, 1] * barrier[0]
        )
        nonparallel = np.abs(denominator) > float(tolerance)
        numerator_t = relative[0] * barrier[1] - relative[1] * barrier[0]
        numerator_u = (
            relative[0] * connection[:, 1]
            - relative[1] * connection[:, 0]
        )
        parameter_t = np.divide(
            numerator_t,
            denominator,
            out=np.full_like(denominator, np.nan, dtype=float),
            where=nonparallel,
        )
        parameter_u = np.divide(
            numerator_u,
            denominator,
            out=np.full_like(denominator, np.nan, dtype=float),
            where=nonparallel,
        )
        crossing |= (
            nonparallel
            & (parameter_t > float(tolerance))
            & (parameter_t < 1.0 - float(tolerance))
            & (parameter_u >= -float(tolerance))
            & (parameter_u <= 1.0 + float(tolerance))
        )
    return crossing


def _regularized_precision(
    covariance: np.ndarray,
    *,
    relative_floor: float,
    absolute_floor_mm2: float,
) -> np.ndarray:
    """Return positive-definite 2x2 precision matrices."""

    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 3 or cov.shape[1:] != (2, 2):
        raise ValueError("covariance must have shape (sample, 2, 2)")
    output = np.full_like(cov, np.nan, dtype=float)
    for index, matrix in enumerate(cov):
        if not np.isfinite(matrix).all():
            continue
        symmetric = 0.5 * (matrix + matrix.T)
        eigenvalue, eigenvector = np.linalg.eigh(symmetric)
        upper = max(float(np.max(eigenvalue)), float(absolute_floor_mm2))
        floor = max(
            float(absolute_floor_mm2),
            float(relative_floor) * upper,
        )
        clipped = np.maximum(eigenvalue, floor)
        regularized = (eigenvector * clipped[None, :]) @ eigenvector.T
        output[index] = np.linalg.inv(regularized)
    return output


def build_fixed_joint_mls(
    sample_xy_km: np.ndarray,
    target_xy_km: np.ndarray,
    *,
    covariance_en_mm2: np.ndarray | None = None,
    support_radius_km: float = 8.0,
    bandwidth_km: float = 4.0,
    min_samples: int = 16,
    max_condition_number: float = 1.0e8,
    covariance_relative_floor: float = 1.0e-6,
    covariance_absolute_floor_mm2: float = 1.0,
    fault_segments_xy_km: np.ndarray | None = None,
) -> FixedJointMLS:
    """Build one joint E/N local affine operator for every target.

    The local model is

    ``E = E0 + E_x dx + E_y dy`` and
    ``N = N0 + N_x dx + N_y dy``.

    Distances in the design matrix are divided by ``bandwidth_km`` for
    numerical stability.  ``evaluate_fixed_joint_mls`` reverses that scaling
    before returning physical derivatives.
    """

    samples = np.asarray(sample_xy_km, dtype=float)
    targets = np.asarray(target_xy_km, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("sample_xy_km must have shape (sample, 2)")
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("target_xy_km must have shape (target, 2)")
    if not np.isfinite(samples).all() or not np.isfinite(targets).all():
        raise ValueError("Sample and target coordinates must be finite")
    if support_radius_km <= 0.0 or bandwidth_km <= 0.0:
        raise ValueError("Support radius and bandwidth must be positive")
    if min_samples < 3:
        raise ValueError("At least three samples are required")
    if fault_segments_xy_km is None:
        fault_segments = None
    else:
        fault_segments = np.asarray(fault_segments_xy_km, dtype=float)
        if fault_segments.ndim != 3 or fault_segments.shape[1:] != (2, 2):
            raise ValueError(
                "fault_segments_xy_km must have shape (segment, 2, 2)"
            )
        if not np.isfinite(fault_segments).all():
            raise ValueError("Fault-segment coordinates must be finite")

    if covariance_en_mm2 is None:
        covariance = np.repeat(np.eye(2, dtype=float)[None, :, :], len(samples), axis=0)
    else:
        covariance = np.asarray(covariance_en_mm2, dtype=float)
        if covariance.shape != (len(samples), 2, 2):
            raise ValueError(
                "covariance_en_mm2 must have shape (number of samples, 2, 2)"
            )
    precision = _regularized_precision(
        covariance,
        relative_floor=covariance_relative_floor,
        absolute_floor_mm2=covariance_absolute_floor_mm2,
    )
    finite_precision = np.isfinite(precision).all(axis=(1, 2))

    tree = cKDTree(samples)
    neighbour_indices: list[np.ndarray] = []
    operators: list[np.ndarray] = []
    parameter_covariance = np.full((len(targets), 6, 6), np.nan, dtype=float)
    sample_count = np.zeros(len(targets), dtype=int)
    effective_sample_count = np.full(len(targets), np.nan, dtype=float)
    condition_number = np.full(len(targets), np.nan, dtype=float)
    barrier_excluded_count = np.zeros(len(targets), dtype=int)
    valid = np.zeros(len(targets), dtype=bool)

    for target_index, target in enumerate(targets):
        neighbour = np.asarray(
            tree.query_ball_point(target, r=float(support_radius_km)),
            dtype=int,
        )
        if neighbour.size:
            neighbour = neighbour[finite_precision[neighbour]]
        if neighbour.size and fault_segments is not None:
            crosses = _connection_crosses_finite_segments(
                target,
                samples[neighbour],
                fault_segments,
            )
            barrier_excluded_count[target_index] = int(crosses.sum())
            neighbour = neighbour[~crosses]
        neighbour_indices.append(neighbour)
        sample_count[target_index] = int(neighbour.size)
        if neighbour.size < int(min_samples):
            operators.append(np.empty((6, 0), dtype=float))
            continue

        offset = (samples[neighbour] - target[None, :]) / float(bandwidth_km)
        distance = np.linalg.norm(samples[neighbour] - target[None, :], axis=1)
        kernel = np.exp(-0.5 * np.square(distance / float(bandwidth_km)))
        effective_sample_count[target_index] = float(
            np.square(kernel.sum()) / np.square(kernel).sum()
        )

        normal = np.zeros((6, 6), dtype=float)
        rhs_blocks: list[np.ndarray] = []
        for local_index, (dx, dy) in enumerate(offset):
            design = np.asarray(
                [
                    [1.0, dx, dy, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, dx, dy],
                ],
                dtype=float,
            )
            weighted_precision = kernel[local_index] * precision[neighbour[local_index]]
            normal += design.T @ weighted_precision @ design
            rhs_blocks.append(design.T @ weighted_precision)

        condition = float(np.linalg.cond(normal))
        condition_number[target_index] = condition
        if not math.isfinite(condition) or condition > float(max_condition_number):
            operators.append(np.empty((6, 0), dtype=float))
            continue
        covariance_beta = np.linalg.pinv(normal)
        operator = covariance_beta @ np.concatenate(rhs_blocks, axis=1)
        parameter_covariance[target_index] = covariance_beta
        operators.append(operator)
        valid[target_index] = True

    return FixedJointMLS(
        target_xy_km=targets,
        neighbour_indices=tuple(neighbour_indices),
        operators=tuple(operators),
        parameter_covariance=parameter_covariance,
        sample_count=sample_count,
        effective_sample_count=effective_sample_count,
        condition_number=condition_number,
        barrier_excluded_count=barrier_excluded_count,
        valid=valid,
        support_radius_km=float(support_radius_km),
        bandwidth_km=float(bandwidth_km),
    )


def evaluate_fixed_joint_mls(
    model: FixedJointMLS,
    cumulative_east_mm: np.ndarray,
    cumulative_north_mm: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate cumulative strain at all epochs using fixed MLS operators.

    Inputs may have shape ``(epoch, sample)`` or ``(sample,)``.  Returned
    component arrays always have shape ``(epoch, target)``.  Numeric
    displacement gradients in mm/km are equal to microstrain; rotation has
    the same numeric conversion to microradians.
    """

    east = np.asarray(cumulative_east_mm, dtype=float)
    north = np.asarray(cumulative_north_mm, dtype=float)
    if east.ndim == 1:
        east = east[None, :]
    if north.ndim == 1:
        north = north[None, :]
    if east.shape != north.shape or east.ndim != 2:
        raise ValueError("East and north must share epoch x sample shape")

    largest_index = max(
        (int(np.max(index)) for index in model.neighbour_indices if index.size),
        default=-1,
    )
    if east.shape[1] <= largest_index:
        raise ValueError("Displacement sample axis does not match the MLS model")

    epoch_count = east.shape[0]
    target_count = len(model.target_xy_km)
    beta = np.full((epoch_count, target_count, 6), np.nan, dtype=float)
    for target_index in np.flatnonzero(model.valid):
        neighbour = model.neighbour_indices[target_index]
        local_east = east[:, neighbour]
        local_north = north[:, neighbour]
        if not (np.isfinite(local_east).all() and np.isfinite(local_north).all()):
            continue
        observation = np.empty((2 * len(neighbour), epoch_count), dtype=float)
        observation[0::2] = local_east.T
        observation[1::2] = local_north.T
        beta[:, target_index, :] = (
            model.operators[target_index] @ observation
        ).T

    scale = float(model.bandwidth_km)
    east_x = beta[:, :, 1] / scale
    east_y = beta[:, :, 2] / scale
    north_x = beta[:, :, 4] / scale
    north_y = beta[:, :, 5] / scale
    epsilon_ee = east_x
    epsilon_nn = north_y
    epsilon_en = 0.5 * (east_y + north_x)
    gamma_en = east_y + north_x
    dilatation = epsilon_ee + epsilon_nn
    rotation = 0.5 * (north_x - east_y)
    mean = 0.5 * dilatation
    radius = np.hypot(0.5 * (epsilon_ee - epsilon_nn), epsilon_en)
    principal_max = mean + radius
    principal_min = mean - radius
    principal_azimuth = (
        np.rad2deg(
            0.5 * np.arctan2(2.0 * epsilon_en, epsilon_ee - epsilon_nn)
        )
        % 180.0
    )

    return {
        "east_fitted_mm": beta[:, :, 0],
        "north_fitted_mm": beta[:, :, 3],
        "epsilon_EE_microstrain": epsilon_ee,
        "epsilon_NN_microstrain": epsilon_nn,
        "epsilon_EN_microstrain": epsilon_en,
        "gamma_EN_microstrain": gamma_en,
        "dilatation_microstrain": dilatation,
        "rotation_microradian": rotation,
        "principal_max_microstrain": principal_max,
        "principal_min_microstrain": principal_min,
        "principal_azimuth_deg": principal_azimuth,
    }


def fixed_joint_mls_component_sigma(
    model: FixedJointMLS,
) -> dict[str, np.ndarray]:
    """Return conditional 1-sigma component uncertainties at every target.

    These uncertainties propagate the covariance supplied when the fixed
    operator was built.  They remain conditional on the assumed local
    covariance and do not account for long-range spatial correlation in InSAR.
    """

    target_count = len(model.target_xy_km)
    output = {
        "epsilon_EE_microstrain": np.full(target_count, np.nan, dtype=float),
        "epsilon_NN_microstrain": np.full(target_count, np.nan, dtype=float),
        "epsilon_EN_microstrain": np.full(target_count, np.nan, dtype=float),
        "gamma_EN_microstrain": np.full(target_count, np.nan, dtype=float),
        "dilatation_microstrain": np.full(target_count, np.nan, dtype=float),
        "rotation_microradian": np.full(target_count, np.nan, dtype=float),
    }
    contrasts = {
        "epsilon_EE_microstrain": np.array([0, 1, 0, 0, 0, 0], dtype=float),
        "epsilon_NN_microstrain": np.array([0, 0, 0, 0, 0, 1], dtype=float),
        "epsilon_EN_microstrain": np.array([0, 0, 0.5, 0, 0.5, 0], dtype=float),
        "gamma_EN_microstrain": np.array([0, 0, 1, 0, 1, 0], dtype=float),
        "dilatation_microstrain": np.array([0, 1, 0, 0, 0, 1], dtype=float),
        "rotation_microradian": np.array([0, 0, -0.5, 0, 0.5, 0], dtype=float),
    }
    scale = float(model.bandwidth_km)
    for target_index in np.flatnonzero(model.valid):
        covariance = model.parameter_covariance[target_index]
        for name, contrast in contrasts.items():
            variance = float(contrast @ covariance @ contrast) / scale**2
            output[name][target_index] = math.sqrt(max(variance, 0.0))
    return output


def target_values_to_grid(
    values: np.ndarray,
    target_rows: np.ndarray,
    target_columns: np.ndarray,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize target values without interpolating unsupported cells."""

    array = np.asarray(values)
    row = np.asarray(target_rows, dtype=int)
    column = np.asarray(target_columns, dtype=int)
    if row.shape != column.shape or array.shape[-1] != row.size:
        raise ValueError("Target indices must match the final values axis")
    output = np.full((*array.shape[:-1], *grid_shape), np.nan, dtype=float)
    output[..., row, column] = array
    return output
