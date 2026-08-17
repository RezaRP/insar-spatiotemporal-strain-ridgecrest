# Tests

```bash
pytest -q
```

Unit tests for the numerical core — not integration tests. They run in seconds and require
no external data.

| Test | Covers |
|---|---|
| `test_ridgecrest_los_projection.py` | Mask-aware resampling weights, the `s ≥ 0.999` retention rule, look-vector renormalisation |
| `test_ridgecrest_local_vertical.py` | GP kernel, adaptive support selection, azimuthal-sector and convex-hull rules |
| `test_ridgecrest_gnss_strain.py` | Bandwidth selection, one-standard-error rule, holdout scoring |
| `test_ridgecrest_two_track.py` | Closed-form E–N inversion, condition-number rejection at 8, covariance propagation |
| `test_ridgecrest_cumulative_strain.py` | Local affine operator, Gaussian weighting, minimum-sample guard, tensor components |
| `test_ridgecrest_fault_barrier_cokriging.py` | Matérn-3/2 correlation, barrier crossing exclusion, positive-definite coregionalisation |
| `test_ridgecrest_local_kriging.py` | Local kriging solve path |
| `test_ridgecrest_strain_change.py` | Innovation construction, robust MAD scaling, cluster mass, Page CUSUM recursion |
| `test_ridgecrest_fault_points.py` | Segment geometry, centreline sampling, normal-vector convention |
| `test_ridgecrest_jump.py` | Finite-aperture cross-fault jump decomposition |

## Known gap

`src/ridgecrest_vertical_los.py` is the largest module in the package and has **no
dedicated test file**. Adding `test_ridgecrest_vertical_los.py` — covering at minimum the
sign convention (`s₆₄ = s₇₁ = +1`) and the referenced-subtraction identity — should be done
before the archived release.
