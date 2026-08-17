# Manuscript statements — ready to paste

Replacement text for the manuscript's back matter, consistent with this repository.
This file is a working aid; delete it before publishing the repository if you prefer.

---

## Data availability

> **Data availability**
>
> All original observations used in this study are publicly available. Sentinel-1
> interferometric products are distributed through the COMET–LiCSAR portal
> (https://comet.nerc.ac.uk/comet-lics-portal/); tropospheric delay products through GACOS
> (http://www.gacos.net/); processed GNSS daily solutions through the Nevada Geodetic
> Laboratory (https://geodesy.unr.edu/); and mapped surface-rupture traces through the
> California Geological Survey 2019 Ridgecrest Earthquakes Rupture Mapping feature service.
> Users must observe each provider's licensing and citation terms.
>
> All Python source code, analysis notebooks, automated tests, acquisition and station
> manifests, and figure-generation workflows are openly available at
> https://github.com/RezaRP/insar-spatiotemporal-strain-ridgecrest. The exact software
> version used for this study, v1.0.0, is permanently archived on Zenodo at
> https://doi.org/10.5281/zenodo.XXXXXXX.
>
> The derived products required to reproduce every figure and table in this paper —
> cumulative vertically corrected east–north displacement, the cumulative 1-km strain cube,
> the near-fault reconstruction arrays, and the change-detection statistics — are archived
> in the same Zenodo record. Large LiCSAR, LiCSBAS, GACOS, GeoTIFF and raw GNSS products
> are not duplicated there because of their size and third-party provenance; the repository
> documents the required frames, acquisition dates, GNSS stations, expected directory
> structure, and retrieval instructions needed to regenerate them from the original
> services.

### Why this replaces the submitted text

The submitted Data Availability statement cites Zenodo DOI **10.5281/zenodo.21759030**.
That record archives *Ridgecrest InSAR Change Detection and Bayesian Slip Inversion* — the
software for the companion, unpublished study. A referee following that DOI would land on a
different codebase. This study needs its own archive.

The submitted text also ends with "available from the corresponding author upon reasonable
request" for the processed HDF5 products. Elsevier geoscience journals increasingly refuse
that formulation. Depositing the derived arrays in the Zenodo record removes the objection
and costs nothing.

---

## Software citation for the reference list

Elsevier/Tectonophysics format:

> Rahimipour, R., Mehrabi, H., Abolghasem, A., Karshenas, H., 2026.
> insar-spatiotemporal-strain-ridgecrest: cumulative 2-D horizontal strain and change
> detection from dual-track InSAR displacement time series (v1.0.0). Zenodo.
> https://doi.org/10.5281/zenodo.XXXXXXX

Cite it in the text at first mention of the workflow, not only in Data Availability.

---

## Third-party citations the repository obliges you to make

| Product | Required citation |
|---|---|
| LiCSAR | Lazecký et al. (2020), *Remote Sensing* 12, 2430 |
| LiCSBAS | Morishita et al. (2020), *Remote Sensing* 12, 424 |
| GACOS | Yu et al. (2018), *JGR Solid Earth* 123, 9202–9222 |
| NGL GNSS | Blewitt, Hammond & Kreemer (2018), *Eos* 99 — **currently missing from the manuscript** |
| CGS rupture mapping | Ponti et al. (2020), *Seismol. Res. Lett.* 91, 2942–2959 |

---

## Release checklist

- [ ] Run `scripts/populate_repo.py --dry-run`, review, then run for real
- [ ] Resolve the three items flagged as ambiguous by that script
- [ ] Add `tests/test_ridgecrest_vertical_los.py` (largest module, currently untested)
- [ ] Regenerate `data/manifests/insar_epochs.csv` with `scripts/build_manifests.py`
- [ ] Confirm the 24-station list in `gnss_stations.csv` against your actual input set
- [ ] `pytest -q` and `ruff check src tests scripts` both green
- [ ] Push to GitHub; confirm the Tests badge goes green
- [ ] Enable the GitHub–Zenodo integration for the new repository
- [ ] Tag and publish `v1.0.0`; Zenodo mints the DOI automatically
- [ ] Replace every `10.5281/zenodo.XXXXXXX` in README.md, CITATION.cff, .zenodo.json and
      this file with the real DOI, then push (and re-release if you want the archive to
      contain its own DOI)
- [ ] Upload the derived `.npz` products to the same Zenodo record
- [ ] Paste the Data availability text above into the manuscript
- [ ] Add ORCIDs to `CITATION.cff` and `.zenodo.json`
