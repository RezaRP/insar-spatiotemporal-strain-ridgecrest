# Spatiotemporal InSAR Strain — 2019 Ridgecrest Sequence

## Cumulative 2-D horizontal strain from dual-track InSAR displacement time series

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Tests](https://github.com/RezaRP/insar-spatiotemporal-strain-ridgecrest/actions/workflows/tests.yml/badge.svg)](https://github.com/RezaRP/insar-spatiotemporal-strain-ridgecrest/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![Research software](https://img.shields.io/badge/type-research%20software-6f42c1.svg)](#)

Reproducible analysis workflow accompanying the manuscript:

> **Resolving near-fault spatiotemporal strain evolution from InSAR displacement time series: application to the 2019 Ridgecrest earthquake sequence**
> R. Rahimipour, H. Mehrabi, A. Abolghasem, H. Karshenas — submitted to *Tectonophysics*.

> [!NOTE]
> This repository covers the **cumulative 2-D horizontal strain and change-detection** study.
> A separate repository, [`ridgecrest-insar-change-detection-slip-inversion`](https://github.com/RezaRP/ridgecrest-insar-change-detection-slip-inversion),
> covers the companion Bayesian fault-slip inversion study. The two share a Sentinel-1
> and GNSS input base but are independent analyses with independent software archives.

---

## What this workflow does

It converts ascending Track 64 and descending Track 71 cumulative InSAR line-of-sight
(LOS) displacement into vertically corrected horizontal LOS, cumulative east–north
displacement, and the full cumulative 2-D horizontal strain tensor across **80 acquisition
epochs spanning 27 May 2017 to 25 November 2019**. It then tests for temporal change using
independently calibrated 12- and 24-day strain innovations.

Three components distinguish it from a conventional co-seismic InSAR strain analysis:

1. **Vertical constraint.** GNSS vertical displacement from 24 continuous stations is
   interpolated to every pixel by an adaptive local Gaussian process (15-km length scale,
   zero nugget, adaptive 35–120 km support) and removed from each track's LOS, with the
   interpolation uncertainty propagated through the displacement-inversion covariance.
   The support topology is frozen before the event period so that temporal change in the
   corrected fields cannot arise from changing station geometry.
2. **Fault-aware near-fault completion.** Where SBAS coherence masking removed Track 71,
   missing descending horizontal LOS is reconstructed by fault-barrier-aware, Track-64-
   conditioned universal cokriging (Matérn-3/2, 8-km length scale, 24-km donor support)
   that refuses donor samples across a mapped rupture trace, preserving displacement
   discontinuities.
3. **Calibrated change detection.** Temporal change is tested — not eyeballed — using a
   maximum signed spatial-cluster mass statistic with family-wise error control and a Page
   CUSUM on map-level innovation energy, both evaluated against an empirical null built
   from consecutive pre-event baseline blocks of matching duration.

## Headline results

| Quantity | Track 64 (asc) | Track 71 (desc) |
|---|---:|---:|
| GNSS vertical interpolation holdout RMSE | 4.52 mm | 3.13 mm |
| 90 % interval coverage | 0.81 | 0.92 |
| Post-correction HLOS vs GNSS, *r* | 0.966 | 0.926 |
| Post-correction HLOS vs GNSS, RMSE | 19.00 mm | 15.37 mm |
| Median residual bias | +3.25 mm (8 stn) | −6.44 mm (7 stn) |

Formal detection on the **directly observed off-fault domain** (5,410 supported 1-km
targets more than 18 km from mapped rupture):

| Interval | Max-cluster mass | *p* | Page CUSUM | *p* |
|---|---:|---:|---:|---:|
| Pre-event surveillance (29 May – 4 Jul 2019) | 0.000 | 1.000 | 0.075 | 0.481 |
| Event control (4 – 16 Jul 2019) | 5.804 | 0.0377 | 3.759 | 0.0185 |

No statistically supported pre-event departure; the event-control interval is detected by
both tests, serving as a positive control on detector sensitivity.

## Interpretation boundary

Please read this before reusing any product in this repository.

- **The off-fault domain is the primary evidence.** It uses two directly observed tracks
  with no reconstruction, and both the vertical correction and the change detection are
  independently validated there.
- **The near-fault domain is not validated recovery.** Buffered spatial cross-validation
  error for the Track-71 reconstruction grows from ≈2.6 mm during calibration to ≈28 mm at
  the 16 July event-control interval, and Track-64 conditioning gives no measurable gain
  over a simpler paired-track model (−0.1 %). Near-fault strain fields are published as
  **retrospective, model-assisted sensitivity maps**, not as primary evidence, and must not
  be cited as measured near-fault strain.
- **Cumulative maps are descriptive.** Statistical significance is established only from
  duration-matched temporal innovations, never from visual inspection of cumulative levels.
- **Temporal coincidence is not earthquake preparation.** Nothing here is offered as
  evidence of precursory or predictive behaviour.

## Repository structure

```text
.
├── data/           Manifests and external-input documentation (no bulk data)
├── docs/           Methods, reproduction order, data provenance, figure map
├── notebooks/      Documented analysis notebooks (07–17) and jupytext .py twins
├── results/        Selected tables and manuscript figures
├── scripts/        Manifest builders and figure-generation programs
├── src/            Reusable Python analysis modules
└── tests/          Unit tests for core numerical routines
```

Large LiCSAR, LiCSBAS, GACOS, GeoTIFF, HDF5, and NPZ products are **not** stored in this
repository. Their provenance, expected filenames, and processing roles are documented in
`data/manifests/`. Frozen derived products required for numerical reproduction are archived
separately (see [Data availability](#data-availability)).

## Installation

```bash
git clone https://github.com/RezaRP/insar-spatiotemporal-strain-ridgecrest.git
cd insar-spatiotemporal-strain-ridgecrest
conda env create -f environment.yml
conda activate insar-strain
pip install -e ".[test]"
```

Or with pip alone:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[analysis,notebooks,test]"
```

## Testing

```bash
pytest -q
```

The suite covers the strain operator, the two-track inversion and its rejection criteria,
the cokriging barrier logic, the vertical interpolation and its holdout scoring, the LOS
projection sign convention, and the change-detection statistics.

## Reproducing the analysis

Full order and required inputs are in [`docs/reproduction.md`](docs/reproduction.md). The
principal chain is:

```text
LiCSBAS cumulative LOS (T64 asc, T71 desc)  +  NGL GNSS daily solutions
                          ↓
        acquisition-time GNSS estimation and GP vertical interpolation
                          ↓
     vertical-to-LOS projection and subtraction  →  horizontal LOS (HLOS)
                          ↓
              two-track pixelwise E–N displacement inversion
                          ↓
        ┌─────────────────┴─────────────────┐
   off-fault strain                 near-fault cokriging
   (>18 km, validated)              (<18 km, sensitivity only)
        └─────────────────┬─────────────────┘
                          ↓
   calibrated 12/24-day innovations → cluster-mass + Page CUSUM detection
                          ↓
                 manuscript tables and figures
```

Absolute development paths are not part of the public workflow; input locations are
supplied through command-line arguments or configuration.

## Data availability

Original observations are public and are **not** redistributed here:

| Product | Provider | Access |
|---|---|---|
| Sentinel-1 SLC | EU Copernicus / ESA | <https://dataspace.copernicus.eu/> |
| LiCSAR / LiCSBAS interferometric products | COMET | <https://comet.nerc.ac.uk/comet-lics-portal/> |
| GACOS tropospheric delay | Newcastle University | <http://www.gacos.net/> |
| GNSS daily solutions (`.tenv3`) | Nevada Geodetic Laboratory | <https://geodesy.unr.edu/> |
| 2019 Ridgecrest surface ruptures | California Geological Survey | [CGS feature service](https://gis.conservation.ca.gov/server/rest/services/CGS/2019_Ridgecrest_Earthquakes_Rupture_Mapping/FeatureServer/0) |

Users must observe each provider's licensing and citation terms. Third-party datasets are
not relicensed by this repository.

Derived products required to reproduce the manuscript figures and tables without
re-running the full processing chain are archived on Zenodo at
**[10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)**.

## Citation

Cite both the article and the archived software release. Machine-readable metadata is in
[`CITATION.cff`](CITATION.cff) and [`.zenodo.json`](.zenodo.json).

```bibtex
@software{rahimipour_insar_strain_ridgecrest,
  author  = {Rahimipour, Reza and Mehrabi, Hamid and
             Abolghasem, Amir and Karshenas, Hossein},
  title   = {Spatiotemporal InSAR Strain --- 2019 Ridgecrest Sequence},
  version = {1.0.0},
  doi     = {10.5281/zenodo.XXXXXXX},
  url     = {https://github.com/RezaRP/insar-spatiotemporal-strain-ridgecrest},
  year    = {2026}
}
```

The article DOI will be added on publication.

## Authors

| | |
|---|---|
| **Reza Rahimipour** | Geomatics Engineering, University of Isfahan, Iran |
| **Hamid Mehrabi** (corresponding) | Geomatics Engineering, University of Isfahan, Iran |
| **Amir Abolghasem** | Earth and Environmental Sciences, LMU München, Germany |
| **Hossein Karshenas** | Artificial Intelligence, University of Isfahan, Iran |

## Acknowledgements

Sentinel-1 observations are provided by the European Union's Copernicus programme and the
European Space Agency. LiCSAR contains modified Copernicus Sentinel data (2015–2019)
analysed by the Centre for the Observation and Modelling of Earthquakes, Volcanoes and
Tectonics (COMET), and uses JASMIN, the UK's collaborative data-analysis environment. We
thank the LiCSAR and LiCSBAS development teams, GACOS, the Nevada Geodetic Laboratory, and
the California Geological Survey.

## License

Original source code is distributed under the [MIT License](LICENSE). Sentinel-1, LiCSAR,
LiCSBAS, GACOS, GNSS, fault-trace and other third-party products remain subject to their
respective providers' terms.

## Contact

Use the repository [Issues](https://github.com/RezaRP/insar-spatiotemporal-strain-ridgecrest/issues)
page, or contact the corresponding author identified in the manuscript.
