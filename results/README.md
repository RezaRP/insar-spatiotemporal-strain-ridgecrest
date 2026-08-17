# Results

Small, human-readable artefacts that let a reader check the manuscript's numbers without
re-running the analysis. Large arrays (`.npz`, `.h5`) are **not** here — they are archived
on Zenodo, DOI in the top-level README.

```text
results/
├── figures/    Manuscript figures as published (PNG, and PDF where vector matters)
└── tables/     Detection summaries, component time series, provenance manifests
```

## Key tables

| File | Contents |
|---|---|
| `tables/cumulative_strain_change_detection_summary.csv` | **Manuscript Table 3.** Both statistics, both intervals, observed values and empirical p-values |
| `tables/strain_component_regional_timeseries.csv` | Spatial median and IQR of all five descriptors at all 80 epochs |
| `tables/strain_cluster_interval_summary.csv` | Per-interval maximum-cluster statistics |
| `tables/strain_cluster_sensitivity.csv` | Sensitivity to the z-threshold and minimum cluster area |
| `tables/cumulative_two_track_strain_manifest.json` | Frozen parameter choices for the off-fault branch |
| `tables/track64_guided_near_fault_manifest.json` | Frozen parameter choices and cell accounting for the near-fault branch |

## Reading the numbers

Two sanity checks a reviewer can run in under a minute:

1. **Cell accounting.** The near-fault manifest must give 2,979 + 502 + 13 = **3,494**
   cells, including **305** within 1 km of the mapped rupture.
2. **Calibration boundary.** The detection manifest must show the calibration window
   ending on **2019-05-29**. No earthquake or post-earthquake epoch may enter baseline
   fitting or threshold calibration.

## Interpretation boundary

Figures 7–11 and every near-fault table are **retrospective, model-assisted sensitivity
products**, not validated measurements. Buffered spatial cross-validation error for the
underlying Track-71 reconstruction grows from ≈2.6 mm at calibration to ≈28 mm at the
16 July event-control interval. See [`../docs/methods.md`](../docs/methods.md) §4.

Figures 3–6 and 12–13 rest on the directly observed off-fault domain and are the primary
evidence of the study.
