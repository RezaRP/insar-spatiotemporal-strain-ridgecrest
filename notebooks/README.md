# Analysis notebooks

Each notebook is committed twice: as a `.ipynb` with **all outputs stripped**, and as a
jupytext `.py` percent-format twin. The `.py` file is authoritative — it is what CI, `git
diff`, and code review see. Regenerate the notebook from it with:

```bash
jupytext --to notebook 15_cumulative_two_track_strain.py
```

Run order and full context: [`../docs/reproduction.md`](../docs/reproduction.md).

| Notebook | Stage | Manuscript section |
|---|---|---|
| `13_validate_and_interpolate_gnss_vertical` | Compare 7 vertical interpolator families; select the adaptive local GP | 4.2, Fig. 3 |
| `10_validate_all_station_local_vertical` | Leave-one-station-out validation of the selected model | 4.2 |
| `07_gnss_vertical_to_los_phase1` | Project vertical into each LOS, subtract, sign audit | 4.3 |
| `12_validate_p595_cccc_los_projection` | Station-level LOS projection check (P595, CCCC) | 4.3 |
| `17_validate_post_correction_hlos_against_gnss` | Post-correction HLOS vs independent GNSS | 5.1, Fig. 4 |
| `11_vertical_corrected_two_track_en_strain_timeseries` | Corrected E–N time series assembly | 4.3 |
| `09_build_track64_text_timeseries` | Build the Track-64 per-epoch grid series | 3.1 |
| `15_cumulative_two_track_strain` | **Primary branch**: off-fault cumulative strain, 80 epochs | 4.4, 5.2, Figs. 5–6 |
| `16_track64_guided_near_fault_strain` | **Sensitivity branch**: near-fault cokriging, lobes, fault jump | 4.4/4.6/4.7, 5.3–5.4, Figs. 7–11 |
| `09_forced_kriging_two_track_strain_sensitivity` | Sensitivity of strain to forced kriging choices | 5.3 |
| `08_gnss_fault_aware_strain_phase2` | Fault-aware GNSS strain comparison | 5.3 |
| `14_two_track_strain_change_detection` | Cluster-mass and Page CUSUM detection | 4.5, 5.5, Table 3, Figs. 12–13 |

> Two notebooks share the `09_` prefix for historical reasons. They are unrelated stages.
> Consider renaming `09_forced_kriging_two_track_strain_sensitivity` to `09b_` before
> release.

## Which numbers are authoritative

Manuscript Table 3 reports the change detection run on the **cumulative strain cube** from
notebook 15 (cluster *p* = 0.0377, CUSUM *p* = 0.0185). An earlier un-gap-filled run
produced different values (0.0690 / 0.0517) and is superseded — do not cite it.

## Notebooks 01–06

They belong to the companion Bayesian slip-inversion study and live in
[`ridgecrest-insar-change-detection-slip-inversion`](https://github.com/RezaRP/ridgecrest-insar-change-detection-slip-inversion).
