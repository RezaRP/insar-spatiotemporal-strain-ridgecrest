# Reproducing the analysis

Run all commands from the repository root. Paths follow the layout in
[`../data/README.md`](../data/README.md).

## 0. Environment

```bash
conda env create -f environment.yml
conda activate insar-strain
pip install -e ".[test]"
pytest -q
```

Every test must pass before you trust any downstream product.

## 1. Arrange external inputs

Nothing large is redistributed here. Fetch the inputs listed in
[`data_sources.md`](data_sources.md) and place them as:

```text
data/
├── cum_full_scene_no_GACOS.h5          # T71 descending cumulative
├── cum_GACOS_full_scene.h5             # T71 descending, GACOS branch
├── cgs_2019_ridgecrest_fault_ruptures.geojson
├── track64/                            # T64 ascending per-epoch grids, YYYYMMDD.txt
└── external/
    ├── GEOC_asc/064A_05410_131313.geo.{E,N,U}.tif
    ├── GEOC_desc/071D_05377_131313.geo.{E,N,U}.tif
    └── GNSS/*.tenv3                    # 24 NGL stations
```

`data/external/` is git-ignored. Do not commit third-party rasters.

Do not alter displacement sign conventions, raster grids, reference definitions, or
acquisition dates without recording the change — the sign audit in Step 3 depends on them.

## 2. Rebuild the manifests (optional, verifies your inputs)

```bash
python scripts/build_manifests.py --data-dir data --output-dir data/manifests
```

This regenerates `insar_epochs.csv`, `gnss_stations.csv`, and refreshes checksums. If the
epoch count is not 80, or the station count is not 24, stop and reconcile before continuing
— every downstream number in the manuscript assumes those two totals.

## 3. Vertical field, LOS correction, and E–N inversion

```bash
export PYTHONPATH=src        # Windows PowerShell: $env:PYTHONPATH = "src"
python notebooks/13_validate_and_interpolate_gnss_vertical.py
python notebooks/10_validate_all_station_local_vertical.py
python notebooks/07_gnss_vertical_to_los_phase1.py
```

Step 13 selects and scores the vertical interpolator (7 candidate families, temporal and
station holdouts). Step 10 runs leave-one-station-out validation. Step 07 projects the
vertical field into each track's LOS, subtracts it, and performs the sign audit.

**Gate:** the pre-event temporal holdout must pass for both tracks. The 4–16 July
full-scene spatial gate is expected to *fail* (near-fault vertical displacement at P595 is
not recoverable when that station is withheld) — this is why the near-fault branch is
labelled a sensitivity product.

## 4. Validate the corrected horizontal LOS against GNSS

```bash
python notebooks/12_validate_p595_cccc_los_projection.py
python notebooks/17_validate_post_correction_hlos_against_gnss.py
```

Expected: *r* = 0.966 / RMSE 19.00 mm (T64, 8 stations) and *r* = 0.926 / RMSE 15.37 mm
(T71, 7 stations). **No offset, scale factor, or ramp is fitted during this comparison.**
If you find yourself fitting one to reach these numbers, something upstream is wrong.

## 5. Off-fault cumulative strain — the primary branch

```bash
python notebooks/15_cumulative_two_track_strain.py
```

Builds the fixed joint E–N GLS derivative operator once and applies it at all 80 epochs.
Produces the 5,410 supported off-fault targets, the cumulative component maps, and the
component time series.

Outputs → `outputs/cumulative_two_track_strain/`

## 6. Near-fault reconstruction — the sensitivity branch

```bash
python notebooks/16_track64_guided_near_fault_strain.py
```

Runs the fault-barrier-aware cokriging, the buffered spatial cross-validation, the fixed
20th/80th-percentile lobe partition, and the cross-fault displacement jump.

Outputs → `outputs/track64_guided_near_fault_strain/`

> Read `docs/methods.md` §4 before interpreting anything from this step. The CV MAE grows
> from ≈2.6 mm to ≈28 mm across the evaluation intervals.

## 7. Formal change detection

```bash
python notebooks/14_two_track_strain_change_detection.py
```

Calibrates robust baselines through 29 May 2019 only — earthquake and post-earthquake data
are never used for baseline fitting or threshold calibration — then evaluates the
maximum signed cluster-mass statistic and the Page CUSUM against the empirical block null.

**Authoritative numbers.** Manuscript Table 3 reports the detection run on the cumulative
strain cube from Step 5 (cluster *p* = 0.0377, CUSUM *p* = 0.0185). Any earlier
un-gap-filled run in this directory is superseded; check
`cumulative_strain_change_detection_summary.csv` for the values that match the paper.

## 8. Sensitivity and supporting analyses

```bash
python notebooks/09_forced_kriging_two_track_strain_sensitivity.py
python notebooks/08_gnss_fault_aware_strain_phase2.py
python notebooks/11_vertical_corrected_two_track_en_strain_timeseries.py
```

## 9. Manuscript figures

```bash
python scripts/make_manuscript_figures.py --output-dir results/figures
```

See [`figure_map.md`](figure_map.md) for which notebook and which output array feeds each
numbered figure in the paper.

## Runtime and hardware

The full chain is dominated by Step 5 (80 epochs × dense GLS operator) and Step 6
(per-target cokriging solves). Expect hours, not minutes, on a workstation. Steps 3–4 are
comparatively cheap and are the right place to verify your inputs before committing to the
long runs.

## Reproducibility boundary

A completed command is not evidence of a correct result. Before using any product:

- confirm the vertical holdout RMSE and the HLOS correlations match Step 3–4 above;
- confirm the near-fault cell accounting sums to 2,979 + 502 + 13 = 3,494;
- confirm the detection calibration window ends on 29 May 2019.

Statistical detection timing alone does not establish fault slip, earthquake preparation,
or precursory behaviour.
