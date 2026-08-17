# Manuscript figure and table map

Which code produced which numbered item in the paper. Referees and reusers should be able
to go from any figure to the exact array that generated it.

## Main text figures

| Fig. | Content | Produced by | Source array / table |
|---|---|---|---|
| 1 | Study area, S1 acquisition geometry, GNSS network, rupture traces | `scripts/make_manuscript_figures.py` | `data/cgs_2019_ridgecrest_fault_ruptures.geojson`, `data/manifests/gnss_stations.csv` |
| 2 | Geodetic analysis and validation workflow diagram | `scripts/make_manuscript_figures.py` | schematic — no data array |
| 3 | GNSS vertical interpolator temporal holdout validation | `notebooks/13_validate_and_interpolate_gnss_vertical.py` | `outputs/gnss_vertical_interpolation_gate/` model-selection CSVs |
| 4 | Post-correction HLOS vs GNSS, both tracks + residual evolution | `notebooks/17_validate_post_correction_hlos_against_gnss.py` | `outputs/cumulative_two_track_strain/07_post_correction_hlos_gnss_validation.*` |
| 5 | Direct off-fault dilatation, four key dates | `notebooks/15_cumulative_two_track_strain.py` | `dense_cumulative_strain_1km.npz` |
| 6 | Off-fault cumulative strain component time series (5 panels) | `notebooks/15_cumulative_two_track_strain.py` | `dense_cumulative_strain_1km.npz`, `03_cumulative_strain_component_timeseries.*` |
| 7 | Near-fault E–N coverage map, 4 July 2019 | `notebooks/16_track64_guided_near_fault_strain.py` | `01_track64_fill_coverage_20190704.*` |
| 8 | Pre-event spatial cross-validation of the Track-71 reconstruction | `notebooks/16_track64_guided_near_fault_strain.py` | buffered-CV summary CSV |
| 9 | Reconstructed near-fault dilatation, 3 key dates | `notebooks/16_track64_guided_near_fault_strain.py` | `near_fault_cumulative_strain_full_area_1km.npz` |
| 10 | Near-fault fixed-lobe strain time series (5 panels) | `notebooks/16_track64_guided_near_fault_strain.py` | `06b_pre_event_cumulative_strain_lobes.*` |
| 11 | Cross-fault displacement discontinuity, both segments | `notebooks/16_track64_guided_near_fault_strain.py` | `track64_guided_cumulative_en.npz` |
| 12 | Cluster statistic and map-level innovation energy vs time | `notebooks/14_two_track_strain_change_detection.py` | `cumulative_strain_change_detection_arrays.npz` |
| 13 | Standardised 12-day strain innovation maps, 4–16 July | `notebooks/14_two_track_strain_change_detection.py` | `cumulative_strain_change_detection_arrays.npz` |

## Main text tables

| Table | Content | Source |
|---|---|---|
| 1 | Geodetic and structural datasets | `data/manifests/dataset_manifest.csv` |
| 2 | Sentinel-1A/B interferometric parameters | `data/manifests/insar_epochs.csv` header block |
| 3 | Formal change-detection results | `outputs/cumulative_two_track_strain/cumulative_strain_change_detection_summary.csv` |

## Supplementary

| Item | Content | Produced by |
|---|---|---|
| Eq. S1–S2 | Grid resampling and LOS referencing | `src/ridgecrest_los_projection.py` |
| Eq. S3–S5 | GNSS vertical field estimation and GP kernel | `src/ridgecrest_local_vertical.py` |
| Eq. S6–S7 | Vertical-to-LOS correction and horizontal inversion | `src/ridgecrest_vertical_los.py`, `src/ridgecrest_two_track.py` |
| Fig. S1 | Effect of near-fault reconstruction on T71 HLOS | `notebooks/16_...py` |
| Fig. S2 | Propagated pointwise 1σ near-fault strain uncertainty | `09_cumulative_strain_uncertainty.*` |
| Fig. S3–S7 | Near-fault key-date maps: E–N (S3), ε_EE (S4), ε_NN (S5), γ_EN (S6), ω (S7) | `10_cumulative_strain_components_2019*.{png,pdf}` |

> **Numbering note.** Supplementary equations and supplementary figures both use an `S`
> prefix. Always write "Supplementary Equation S5" or "Supplementary Figure S5" in full —
> the bare form is ambiguous.

## Module → manuscript section

| Module | Section |
|---|---|
| `ridgecrest_los_projection.py` | 4.1 Data preparation and acquisition geometry |
| `ridgecrest_local_vertical.py` | 4.2 GNSS vertical field |
| `ridgecrest_gnss_strain.py` | 4.2 (bandwidth selection, holdout scoring) |
| `ridgecrest_vertical_los.py` | 4.3 Vertical-to-LOS correction |
| `ridgecrest_two_track.py` | 4.3 Two-track horizontal inversion |
| `ridgecrest_cumulative_strain.py` | 4.4 Direct off-fault strain, Eq. (1)–(2) |
| `ridgecrest_fault_barrier_cokriging.py` | 4.4 Near-fault reconstruction |
| `ridgecrest_strain_change.py` | 4.5 Change detection, Eq. (3)–(6) |
| `ridgecrest_fault_points.py` | 4.6 Lobe partition, 4.7 fault geometry |
| `ridgecrest_jump.py` | 4.7 Finite-aperture cross-fault jump |

Equation numbers above assume the manuscript's equations have been renumbered to start at
(1); the submitted draft began at (2).
