# Data directory

This repository does **not** redistribute the Sentinel-1, LiCSAR, LiCSBAS, GACOS, or GNSS
source products used in the study — they are large, third-party licensed, and publicly
available from their original providers. What is committed here is enough to reconstruct
the exact input set: manifests naming every acquisition, station, and file, with provider
and expected local path.

See [`../docs/data_sources.md`](../docs/data_sources.md) for provenance and
[`../docs/reproduction.md`](../docs/reproduction.md) for the analysis order.

## Expected local layout

```text
data/
├── cum_full_scene_no_GACOS.h5              # T71 descending cumulative LOS
├── cum_GACOS_full_scene.h5                 # T71 descending, GACOS branch
├── cgs_2019_ridgecrest_fault_ruptures.geojson
├── cgs_2019_ridgecrest_fault_ruptures_source.json
├── track64/
│   └── YYYYMMDD.txt                        # T64 ascending per-epoch cumulative grids
├── external/
│   ├── GEOC_asc/
│   │   ├── 064A_05410_131313.geo.E.tif
│   │   ├── 064A_05410_131313.geo.N.tif
│   │   └── 064A_05410_131313.geo.U.tif
│   ├── GEOC_desc/
│   │   ├── 071D_05377_131313.geo.E.tif
│   │   ├── 071D_05377_131313.geo.N.tif
│   │   └── 071D_05377_131313.geo.U.tif
│   └── GNSS/
│       └── *.tenv3                         # 24 NGL station files
└── manifests/                              # committed
    ├── dataset_manifest.csv
    ├── gnss_stations.csv
    └── insar_epochs.csv
```

`data/external/`, `data/track64/`, and all `*.h5` / `*.tif` / `*.tenv3` files are
git-ignored. Do not commit them.

## Manifests

| File | Contents |
|---|---|
| `dataset_manifest.csv` | One row per input dataset: role, provider, public source, expected local path, whether it is redistributed here |
| `gnss_stations.csv` | The 24 NGL stations, with the role each plays (constraint vs. validation) |
| `insar_epochs.csv` | The 80 common acquisition epochs with track-specific acquisition times |

Regenerate them against your own copy of the inputs with:

```bash
python scripts/build_manifests.py --data-dir data --output-dir data/manifests
```

If the regenerated epoch count is not **80**, or the station count is not **24**, reconcile
before running anything downstream — every number in the manuscript assumes those totals.

## Units and conventions

- LiCSBAS cumulative displacement is analysed as line-of-sight (LOS) displacement in **mm**.
- Positive LOS is motion **toward** the satellite. The sign audit confirmed
  `s₆₄ = s₇₁ = +1`; neither track is negated anywhere in the workflow.
- All grids are **UTM Zone 11N, EPSG:32611**, 1-km spacing, on the footprint intersection
  of the two tracks.
- The common temporal reference epoch is **27 May 2017**.
- The common spatial reference is the median displacement within **1.5 km of GNSS station
  P463**, applied *before* resampling so the datum is resolution-independent.
- Acquisition and interferogram identifiers use `YYYYMMDD`.
- Track 64 is ascending; Track 71 is descending.
- Strain is reported in **microstrain (µstrain)**; vertical-axis rotation in **microradians
  (µrad)**.

## A note on the two Ridgecrest repositories

The GNSS station file in the companion slip-inversion repository lists **25** stations, of
which 21 constrain the source geometry and 4 are withheld for validation. This study uses
**24** stations for the vertical field. The two sets are not interchangeable — use
`gnss_stations.csv` in *this* repository for *this* analysis.
