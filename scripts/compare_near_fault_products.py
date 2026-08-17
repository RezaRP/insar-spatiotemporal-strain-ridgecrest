"""Compare near-fault transient results across full and pre-event inversions."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ridgecrest_fault_points import (  # noqa: E402
    build_fault_sampling_points,
    extract_fault_point_series,
    paired_fault_differences,
)
from ridgecrest_transient import (  # noqa: E402
    PatchSeries,
    spatial_transient_matched_filter,
)


FAULT_FILE = ROOT / "data" / "cgs_2019_ridgecrest_fault_ruptures.geojson"
OUTPUT_DIR = ROOT / "outputs" / "full_scene_near_fault_points"
PRODUCTS = {
    "GACOS full inversion": ROOT / "data" / "cum_GACOS_full_scene.h5",
    "No-GACOS full inversion": ROOT / "data" / "cum_full_scene_no_GACOS.h5",
    "GACOS pre-event-only inversion": (
        ROOT / "data" / "cum_GACOS_pre_earthquake.h5"
    ),
    "No-GACOS pre-event-only inversion": (
        ROOT / "data" / "cum_noGACOS_onlyPre_set_reference.h5"
    ),
}


nodes, points, _ = build_fault_sampling_points(FAULT_FILE)
series = {}
pairs = {}
for name, path in PRODUCTS.items():
    print("Extracting", name)
    series[name] = extract_fault_point_series(
        path,
        points,
        sample_radius_km=1.25,
        min_pixels=20,
    )
    pairs[name] = paired_fault_differences(series[name], nodes)


def truncated(patch_series, last_date):
    mask = patch_series.dates <= pd.Timestamp(last_date)
    return PatchSeries(
        dates=patch_series.dates[mask],
        values=patch_series.values[mask],
        metadata=patch_series.metadata.copy(),
        reference=patch_series.reference[mask],
        quality_mask=patch_series.quality_mask,
        target_mask=patch_series.target_mask,
    )


rows = []
for last_date, candidate_end, min_after, label in [
    ("2019-06-22", "2019-05-17", 4, "common_through_22_June"),
    ("2019-07-04", "2019-05-29", 4, "through_4_July"),
]:
    for product, pair_series in pairs.items():
        if pair_series.dates[-1] < pd.Timestamp(last_date):
            continue
        subset = truncated(pair_series, last_date)
        result, _, _ = spatial_transient_matched_filter(
            subset,
            min_before=20,
            min_after=min_after,
            candidate_start_date="2019-01-01",
            candidate_end_date=candidate_end,
            n_boot=499,
            block_length=4,
            seed=20260810 + len(rows),
        )
        safe_product = (
            product.lower()
            .replace("-", "_")
            .replace(" ", "_")
        )
        result.to_csv(
            OUTPUT_DIR / f"{label}__{safe_product}.csv",
            index=False,
        )
        rows.append(
            {
                "comparison": label,
                "product": product,
                "last_epoch": subset.dates[-1],
                "epochs": len(subset.dates),
                "minimum_corrected_p": result["corrected_pvalue"].min(),
                "significant_pairs": int(
                    result["fwer_significant_0_05"].sum()
                ),
                "strongest_node": result.loc[
                    result["max_statistic"].idxmax(), "node_id"
                ],
                "strongest_endpoint_transient_mm": result.loc[
                    result["max_statistic"].idxmax(), "endpoint_transient_mm"
                ],
            }
        )

summary = pd.DataFrame(rows)
summary.to_csv(OUTPUT_DIR / "cross_product_transient_comparison.csv", index=False)
print(summary.to_string(index=False))

diagnostics = []
for full_name, pre_name, last_date in [
    (
        "GACOS full inversion",
        "GACOS pre-event-only inversion",
        "2019-06-22",
    ),
    (
        "No-GACOS full inversion",
        "No-GACOS pre-event-only inversion",
        "2019-07-04",
    ),
]:
    full = pairs[full_name]
    pre = pairs[pre_name]
    common = full.dates.intersection(pre.dates)
    common = common[common <= pd.Timestamp(last_date)]
    full_index = full.dates.get_indexer(common)
    pre_index = pre.dates.get_indexer(common)
    difference = full.values[full_index] - pre.values[pre_index]
    diagnostics.append(
        {
            "full_product": full_name,
            "pre_only_product": pre_name,
            "last_common_date": common[-1],
            "common_epochs": len(common),
            "pair_difference_rms_mm": float(
                np.sqrt(np.mean(difference**2))
            ),
            "pair_difference_max_abs_mm": float(
                np.max(np.abs(difference))
            ),
            "last_epoch_pair_difference_max_abs_mm": float(
                np.max(np.abs(difference[-1]))
            ),
        }
    )
diagnostic_table = pd.DataFrame(diagnostics)
diagnostic_table.to_csv(
    OUTPUT_DIR / "full_vs_preonly_pair_diagnostics.csv", index=False
)
print()
print(diagnostic_table.to_string(index=False))
