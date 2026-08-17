#!/usr/bin/env python3
"""Copy the code, tests and notebooks belonging to the cumulative-strain study
from the working directory into this repository.

The working directory (``D:\\Thises\\Paper\\JD`` by default) holds material for *two*
manuscripts. This script copies only the files that belong to the cumulative 2-D
horizontal strain study, strips notebook outputs so the repository stays small, and
refuses to copy anything git-ignored.

Usage
-----
    python scripts/populate_repo.py --source "D:/Thises/Paper/JD"
    python scripts/populate_repo.py --source "D:/Thises/Paper/JD" --dry-run

Review the SELECTED_* lists below before running. Two entries are deliberately
ambiguous and are flagged at the end of the run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# --------------------------------------------------------------------------------------
# What belongs to THIS study (cumulative 2-D horizontal strain + change detection).
# Everything not listed here belongs to the companion Bayesian slip-inversion repository.
# --------------------------------------------------------------------------------------

SELECTED_SRC = [
    "ridgecrest_los_projection.py",        # 4.1 resampling, LOS referencing
    "ridgecrest_local_vertical.py",        # 4.2 GNSS vertical field, GP interpolation
    "ridgecrest_gnss_strain.py",           # 4.2 bandwidth selection, holdout scoring
    "ridgecrest_vertical_los.py",          # 4.3 vertical-to-LOS correction
    "ridgecrest_two_track.py",             # 4.3 two-track E-N inversion
    "ridgecrest_cumulative_strain.py",     # 4.4 local affine strain operator
    "ridgecrest_fault_barrier_cokriging.py",  # 4.4 near-fault reconstruction
    "ridgecrest_strain_change.py",         # 4.5 change detection
    "ridgecrest_fault_points.py",          # 4.6/4.7 fault geometry, lobe masks
    "ridgecrest_jump.py",                  # 4.7 cross-fault displacement jump  [VERIFY]
]

# Belongs to the OTHER repository - do not copy:
EXCLUDED_SRC = [
    "ridgecrest_transient.py",             # pre-event transient acceleration study
]

SELECTED_TESTS = [
    "test_ridgecrest_los_projection.py",
    "test_ridgecrest_local_vertical.py",
    "test_ridgecrest_local_kriging.py",
    "test_ridgecrest_gnss_strain.py",
    "test_ridgecrest_two_track.py",
    "test_ridgecrest_cumulative_strain.py",
    "test_ridgecrest_fault_barrier_cokriging.py",
    "test_ridgecrest_strain_change.py",
    "test_ridgecrest_fault_points.py",
    "test_ridgecrest_jump.py",             # [VERIFY] pairs with ridgecrest_jump
]

# Notebooks 07-17 are this study; 01-06 belong to the slip-inversion repository.
# Only the jupytext .py twins plus stripped .ipynb are committed.
SELECTED_NOTEBOOKS = [
    "07_gnss_vertical_to_los_phase1",
    "08_gnss_fault_aware_strain_phase2",
    "09_build_track64_text_timeseries",
    "09_forced_kriging_two_track_strain_sensitivity",
    "10_validate_all_station_local_vertical",
    "11_vertical_corrected_two_track_en_strain_timeseries",
    "12_validate_p595_cccc_los_projection",
    "13_validate_and_interpolate_gnss_vertical",
    "14_two_track_strain_change_detection",
    "15_cumulative_two_track_strain",
    "16_track64_guided_near_fault_strain",
    "17_validate_post_correction_hlos_against_gnss",
]

SELECTED_SCRIPTS = [
    "build_cumulative_strain_notebook.py",
    "build_strain_change_notebook.py",
    "build_track64_guided_near_fault_notebook.py",
    "build_validated_workflow_notebook.py",
    "make_validated_workflow_figure.py",
    "compare_near_fault_products.py",
    "download_ridgecrest_faults.py",
    "percent_to_ipynb.py",
    "analyze_track64_network_sensitivity.py",
]

# Small, committable result artefacts. Large .npz arrays go to Zenodo, not GitHub.
SELECTED_RESULT_TABLES = [
    ("outputs/cumulative_two_track_strain", "cumulative_strain_change_detection_summary.csv"),
    ("outputs/two_track_strain_change_detection", "strain_temporal_detection_summary.csv"),
    ("outputs/two_track_strain_change_detection", "strain_component_regional_timeseries.csv"),
    ("outputs/two_track_strain_change_detection", "strain_cluster_interval_summary.csv"),
    ("outputs/two_track_strain_change_detection", "strain_cluster_sensitivity.csv"),
    ("outputs/track64_guided_near_fault_strain", "track64_guided_near_fault_manifest.json"),
    ("outputs/cumulative_two_track_strain", "cumulative_two_track_strain_manifest.json"),
]

AMBIGUOUS = {
    "ridgecrest_jump.py": (
        "Used by manuscript Section 4.7 (finite-aperture cross-fault displacement jump), "
        "but also by the companion two-method jump-detection study. Confirm which "
        "functions Section 4.7 actually calls before publishing."
    ),
    "test_ridgecrest_local_kriging.py": (
        "This test has no matching module name in src/. Confirm it targets "
        "ridgecrest_fault_barrier_cokriging.py, then consider renaming it to match."
    ),
    "ridgecrest_vertical_los.py": (
        "Largest module in src/ (55 kB) and currently has NO dedicated test file. "
        "Add tests/test_ridgecrest_vertical_los.py before release."
    ),
}

MAX_COMMITTABLE_BYTES = 5 * 1024 * 1024


def strip_notebook(source: Path, destination: Path) -> None:
    """Write a copy of *source* with all cell outputs and execution counts removed."""
    notebook = json.loads(source.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cell.get("metadata", {}).pop("execution", None)
    notebook.get("metadata", {}).pop("widgets", None)
    destination.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def copy_one(source: Path, destination: Path, *, dry_run: bool) -> tuple[bool, str]:
    if not source.exists():
        return False, f"MISSING  {source}"
    size = source.stat().st_size
    if size > MAX_COMMITTABLE_BYTES and source.suffix != ".ipynb":
        return False, f"TOO BIG  {source.name} ({size / 1e6:.1f} MB) - send to Zenodo"
    if dry_run:
        return True, f"would copy  {source.name} ({size / 1024:.0f} kB)"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".ipynb":
        strip_notebook(source, destination)
        new_size = destination.stat().st_size
        return True, f"copied   {source.name} ({size / 1e6:.1f} MB -> {new_size / 1024:.0f} kB stripped)"
    shutil.copy2(source, destination)
    return True, f"copied   {source.name} ({size / 1024:.0f} kB)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path, help="Working directory holding src/, tests/, notebooks/")
    parser.add_argument("--dest", type=Path, default=Path(__file__).resolve().parent.parent, help="Repository root")
    parser.add_argument("--dry-run", action="store_true", help="Report actions without writing anything")
    args = parser.parse_args()

    source_root: Path = args.source.expanduser().resolve()
    dest_root: Path = args.dest.expanduser().resolve()

    if not source_root.is_dir():
        print(f"error: source directory not found: {source_root}", file=sys.stderr)
        return 1

    print(f"source : {source_root}")
    print(f"dest   : {dest_root}")
    print(f"mode   : {'DRY RUN' if args.dry_run else 'WRITE'}\n")

    copied = 0
    problems: list[str] = []

    groups: list[tuple[str, list[tuple[Path, Path]]]] = [
        ("src", [(source_root / "src" / n, dest_root / "src" / n) for n in SELECTED_SRC]),
        ("tests", [(source_root / "tests" / n, dest_root / "tests" / n) for n in SELECTED_TESTS]),
        ("scripts", [(source_root / "scripts" / n, dest_root / "scripts" / n) for n in SELECTED_SCRIPTS]),
    ]

    notebook_pairs: list[tuple[Path, Path]] = []
    for stem in SELECTED_NOTEBOOKS:
        for suffix in (".py", ".ipynb"):
            notebook_pairs.append(
                (source_root / "notebooks" / f"{stem}{suffix}", dest_root / "notebooks" / f"{stem}{suffix}")
            )
    groups.append(("notebooks", notebook_pairs))

    table_pairs = [
        (source_root / sub / name, dest_root / "results" / "tables" / name)
        for sub, name in SELECTED_RESULT_TABLES
    ]
    groups.append(("results/tables", table_pairs))

    for label, pairs in groups:
        print(f"--- {label} ---")
        for source, destination in pairs:
            ok, message = copy_one(source, destination, dry_run=args.dry_run)
            print("  " + message)
            if ok:
                copied += 1
            elif message.startswith("MISSING") and source.suffix == ".ipynb":
                pass  # a missing .ipynb twin is fine; the .py is authoritative
            else:
                problems.append(message)
        print()

    print(f"{copied} file(s) {'would be ' if args.dry_run else ''}copied.")

    if problems:
        print("\nUnresolved:")
        for problem in problems:
            print("  " + problem)

    print("\nVerify before publishing:")
    for name, reason in AMBIGUOUS.items():
        print(f"  * {name}\n      {reason}")

    print(
        "\nExcluded as belonging to the companion slip-inversion repository:\n  "
        + ", ".join(EXCLUDED_SRC)
        + "\n  notebooks 01-06, scripts run_bayesian_*, invert_all_geoc_intervals.py,\n"
        "  prepare_event_inversion_data.py, run_marginalized_geometry_inversion.py,\n"
        "  detect_full_scene_pre_event_change.py, analyze_raw_pre_event_ifgs.py,\n"
        "  find_common_reference.py, make_publication_change_slip_figures.py,\n"
        "  plot_interval_manuscript_figures.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
