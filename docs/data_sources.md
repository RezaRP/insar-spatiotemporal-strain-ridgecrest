# Data sources and provenance

No bulk third-party data is redistributed in this repository. This document records what
was used, where it came from, and what each product does in the workflow.

## Sentinel-1 and LiCSAR / LiCSBAS

Interferometric products originate from Sentinel-1A/B observations supplied through the
European Union's Copernicus programme and the European Space Agency, processed by COMET
into LiCSAR products and time series by LiCSBAS.

Portal: <https://comet.nerc.ac.uk/comet-lics-portal/>

| | Ascending | Descending |
|---|---|---|
| Track | 64 | 71 |
| Frame | `064A_05410_131313` | `071D_05377_131313` |
| Frame master | 2019-04-17 | 2017-03-04 |
| Heading | −10.146887° | −169.84503° |
| Mean incidence | 39.6181° | 33.7677° |

Both stacks share the SBAS reference epoch **2017-05-27** and the acquisition period
**27 May 2017 – 25 November 2019** (80 common epochs).

Used: LiCSBAS cumulative LOS displacement; pixel-specific E, N, U look-vector components
taken directly from LiCSAR metadata.

> Track 71 carries additional quality screening (average coherence ≥ 0.30, residual RMS
> ≤ 5 mm, unwrapping-gap count ≤ 2, phase-closure error count ≤ 10). This asymmetry is the
> direct cause of the near-fault descending gaps that the cokriging step reconstructs, and
> it should be stated whenever the reconstruction is discussed.

Cite: Lazecký et al. (2020), *Remote Sensing* 12, 2430; Morishita et al. (2020),
*Remote Sensing* 12, 424.

## GACOS

Tropospheric delay products from <http://www.gacos.net/>, applied as the atmospheric phase
correction to both stacks. GACOS is a model-based correction, not an independent
acquisition.

Cite: Yu et al. (2018), *JGR Solid Earth* 123, 9202–9222.

## GNSS

Daily `.tenv3` position time series from the Nevada Geodetic Laboratory,
<https://geodesy.unr.edu/>. **24 continuous stations** spanning approximately 34.5–37.0 °N
and 118.8–116.0 °W, drawn from the Plate Boundary Observatory network and regional sites.

Station list: [`../data/manifests/gnss_stations.csv`](../data/manifests/gnss_stations.csv)

Positions are estimated at each track-specific Sentinel-1 acquisition time — linearly
interpolated within uninterrupted daily segments, and predicted from a 30-day weighted
pre-event trend for the two 4 July 2019 acquisitions, whose ordinary interpolation window
would otherwise cross the co-seismic discontinuity. **Acquisition-time GNSS values are
therefore temporally estimated quantities, not independent observations.**

NGL data use requires citing: Blewitt, G., Hammond, W.C., Kreemer, C. (2018). Harnessing
the GPS data explosion for interdisciplinary science. *Eos* 99.
<https://doi.org/10.1029/2018EO104623>

## Surface rupture traces

Mapped 2019 Ridgecrest surface ruptures from the California Geological Survey feature
service:

<https://gis.conservation.ca.gov/server/rest/services/CGS/2019_Ridgecrest_Earthquakes_Rupture_Mapping/FeatureServer/0>

Query metadata are recorded in `data/cgs_2019_ridgecrest_fault_ruptures_source.json`. The
retrieved GeoJSON is not relicensed by this repository.

Two finite source-model segments representing the first-order **Paxton Ranch** and **Salt
Wells Valley** geometries determine fault-side connectivity in the near-fault interpolation
and in the strain operators.

Companion documentation of the field mapping: Ponti et al. (2020), *Seismological Research
Letters* 91(5), 2942–2959.

## Redistribution policy

Large or externally licensed source rasters and GNSS files are not duplicated on GitHub.
This repository records their provenance, expected filenames, roles, and processing
relationships. Derived products needed to reproduce the manuscript figures and tables
without re-running the full chain are archived on Zenodo (DOI in the README).

Users must observe each provider's licensing and citation requirements independently.
