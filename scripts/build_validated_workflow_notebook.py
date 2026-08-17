"""Build the review/reproduction notebook for the validated Ridgecrest workflow."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(r"D:\Thises\Paper\JD")
OUTPUT = ROOT / "notebooks" / "06_full_scene_spatiotemporal_interval_inversion.ipynb"


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3"}
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Full-scene spatiotemporal detection and interval slip inversion

This notebook reviews the saved outputs from the complete Ridgecrest workflow.
Candidate pixels were selected over the **entire valid scene**, not around a
fault or a hand-picked point. Temporal innovation and spatial cluster tests are
both required. A direct pre-event interferogram then provides an independent
check outside the cumulative SBAS network.

The source section reviews every raw GEOC interferogram: eight ascending and
five descending. A transferred joint-event geometry is used because geometry
and slip cannot be independently identified from a single LOS interval."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import subprocess
import sys
import pandas as pd
from IPython.display import Image, display

ROOT = Path(r"D:\\Thises\\Paper\\JD")
DETECTION = ROOT / "outputs" / "full_scene_detection"
INVERSION = ROOT / "outputs" / "interval_inversions"
pd.set_option("display.max_columns", 100)"""
        ),
        nbf.v4.new_markdown_cell("## 1. Full-scene temporal and spatial detection"),
        nbf.v4.new_code_cell(
            """summary = json.loads((DETECTION / "summary.json").read_text())
summary"""
        ),
        nbf.v4.new_code_cell(
            """clusters = pd.read_csv(DETECTION / "full_scene_clusters.csv")
clusters.sort_values(["cluster_fwer_p", "cluster_mass"], ascending=[True, False])[
    ["cluster_id", "centroid_latitude", "centroid_longitude",
     "median_change_mm", "direct_ifg_median_mm", "cluster_fwer_p",
     "direct_ifg_spatial_p_holm", "research_status"]
].head(10)"""
        ),
        nbf.v4.new_code_cell(
            """display(Image(filename=str(DETECTION / "01_full_scene_change_detection.png")))
display(Image(filename=str(
    INVERSION / "manuscript_figures" / "01_detection_inversion_summary.png"
)))"""
        ),
        nbf.v4.new_markdown_cell(
            """**Decision:** the July 4 acquisition is before the earthquake, but
none of the cumulative-series clusters passes the independent
direct-interferogram spatial test. The result is an apparent, unvalidated
time-series anomaly; no predictive or causal interpretation is made."""
        ),
        nbf.v4.new_markdown_cell("## 2. Source support for every GEOC interval"),
        nbf.v4.new_code_cell(
            """intervals = pd.read_csv(INVERSION / "interval_inversion_summary.csv")
intervals[
    ["track", "pair", "distributed_cv_gain_over_ramp", "distributed_support",
     "distributed_equivalent_Mw_median", "distributed_peak_slip_median_m",
     "median_resolution_diagonal", "checkerboard_correlation"]
]"""
        ),
        nbf.v4.new_code_cell(
            """source = pd.read_csv(INVERSION / "interval_source_parameters.csv")
source[source.source_support == "supported"][
    ["track", "pair", "fault", "strike_deg", "dip_deg", "length_km",
     "width_km", "top_depth_km", "strike_slip_median_m",
     "dip_slip_median_m", "equivalent_Mw_median",
     "source_cv_gain_over_ramp"]
]"""
        ),
        nbf.v4.new_code_cell(
            """display(Image(filename=str(INVERSION / "01_interval_source_comparison.png")))
display(Image(filename=str(
    INVERSION / "manuscript_figures" / "02_paxton_ranch_compact_atlas.png"
)))
display(Image(filename=str(
    INVERSION / "manuscript_figures" / "03_salt_wells_compact_atlas.png"
)))"""
        ),
        nbf.v4.new_markdown_cell(
            """**Decision:** only earthquake-spanning interferograms have
spatial-block predictive support for a fault source. Numerical slip posteriors
for unsupported pre-event and post-event-only intervals are retained for
transparency but are not interpreted as physical sources. Descending windows
overlap and their moments must not be summed."""
        ),
        nbf.v4.new_markdown_cell("## 3. Optional full reproduction"),
        nbf.v4.new_code_cell(
            """RUN_FULL_ANALYSIS = False  # set True deliberately; this overwrites saved outputs

if RUN_FULL_ANALYSIS:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "detect_full_scene_pre_event_change.py")],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "invert_all_geoc_intervals.py")],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "plot_interval_manuscript_figures.py")],
        check=True,
    )
else:
    print("Review mode: using the saved, completed outputs.")"""
        ),
        nbf.v4.new_markdown_cell(
            """## 4. Files used by the manuscript

- `outputs/full_scene_detection/full_scene_clusters.csv`
- `outputs/interval_inversions/interval_inversion_summary.csv`
- `outputs/interval_inversions/interval_source_parameters.csv`
- `outputs/interval_inversions/all_interval_patch_slip.csv`
- `outputs/interval_inversions/intervals/<track>_<pair>/`
- `manuscript/Manuscript_Ridgecrest_validated_revision.docx`"""
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
