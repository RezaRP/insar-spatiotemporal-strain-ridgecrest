#!/usr/bin/env python3
"""Rebuild the data manifests from a local copy of the study inputs.

Regenerates ``insar_epochs.csv`` from the acquisition files actually present on disk,
and verifies ``gnss_stations.csv`` against the GNSS files actually present. This makes
the committed manifests auditable rather than hand-maintained.

Usage
-----
    python scripts/build_manifests.py --data-dir data --output-dir data/manifests
    python scripts/build_manifests.py --data-dir data --output-dir data/manifests --check
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from datetime import date
from pathlib import Path

EXPECTED_EPOCHS = 80
EXPECTED_STATIONS = 24
REFERENCE_EPOCH = date(2017, 5, 27)
LAST_EPOCH = date(2019, 11, 25)

# Track-specific Sentinel-1 acquisition times (UTC) for the frames used in this study.
# The 4 July 2019 descending pass at 13:51:41 UTC is 3 h 42 min before the Mw 6.4
# foreshock origin time of 17:33:49 UTC.
ACQUISITION_TIME_UTC = {
    "T064A": "13:52:00",   # nominal; refine from LiCSAR metadata if available
    "T071D": "13:51:41",
}

DATE_PATTERN = re.compile(r"^(\d{8})\.txt$")


def parse_yyyymmdd(token: str) -> date:
    return date(int(token[:4]), int(token[4:6]), int(token[6:8]))


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def discover_epochs(data_dir: Path) -> list[date]:
    """Collect acquisition dates from the ascending per-epoch grid files."""
    candidates = [data_dir / "track64", data_dir]
    dates: set[date] = set()
    for directory in candidates:
        if not directory.is_dir():
            continue
        for entry in directory.iterdir():
            match = DATE_PATTERN.match(entry.name)
            if match:
                dates.add(parse_yyyymmdd(match.group(1)))
        if dates:
            break
    return sorted(dates)


def discover_stations(data_dir: Path) -> list[str]:
    gnss_dir = data_dir / "external" / "GNSS"
    if not gnss_dir.is_dir():
        return []
    return sorted({p.stem.split("_")[0].upper() for p in gnss_dir.glob("*.tenv3")})


def write_epoch_manifest(epochs: list[date], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch_index",
                "acquisition_date",
                "days_from_reference",
                "asc_t64_time_utc",
                "desc_t71_time_utc",
                "is_reference_epoch",
                "is_last_pre_event",
            ]
        )
        for index, day in enumerate(epochs):
            writer.writerow(
                [
                    index,
                    day.isoformat(),
                    (day - REFERENCE_EPOCH).days,
                    ACQUISITION_TIME_UTC["T064A"],
                    ACQUISITION_TIME_UTC["T071D"],
                    "yes" if day == REFERENCE_EPOCH else "no",
                    "yes" if day == date(2019, 7, 4) else "no",
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--check", action="store_true", help="Validate only; do not write")
    parser.add_argument("--checksums", action="store_true", help="Also emit SHA-256 for committed manifests")
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not data_dir.is_dir():
        print(f"error: data directory not found: {data_dir}", file=sys.stderr)
        return 1

    failures: list[str] = []

    epochs = discover_epochs(data_dir)
    if not epochs:
        print(f"warning: no YYYYMMDD.txt acquisition grids found under {data_dir}")
    else:
        print(f"epochs found : {len(epochs)}  ({epochs[0]} .. {epochs[-1]})")
        if len(epochs) != EXPECTED_EPOCHS:
            failures.append(
                f"epoch count is {len(epochs)}, expected {EXPECTED_EPOCHS}. "
                "The manuscript's 80-epoch common series is the intersection of the two "
                "tracks - a raw single-track directory will legitimately hold more. "
                "Reconcile before running the analysis."
            )
        if epochs[0] != REFERENCE_EPOCH:
            failures.append(f"first epoch {epochs[0]} is not the reference epoch {REFERENCE_EPOCH}")
        if epochs[-1] != LAST_EPOCH:
            failures.append(f"last epoch {epochs[-1]} is not {LAST_EPOCH}")

    stations = discover_stations(data_dir)
    if stations:
        print(f"stations found: {len(stations)}  {', '.join(stations)}")
        if len(stations) != EXPECTED_STATIONS:
            failures.append(f"station count is {len(stations)}, expected {EXPECTED_STATIONS}")
        committed = output_dir / "gnss_stations.csv"
        if committed.exists():
            with committed.open(encoding="utf-8") as handle:
                listed = {row["station_id"].strip().upper() for row in csv.DictReader(handle)}
            missing = listed - set(stations)
            extra = set(stations) - listed
            if missing:
                failures.append(f"listed in manifest but absent on disk: {sorted(missing)}")
            if extra:
                failures.append(f"present on disk but absent from manifest: {sorted(extra)}")
    else:
        print(f"warning: no .tenv3 files found under {data_dir / 'external' / 'GNSS'}")

    if epochs and not args.check:
        target = output_dir / "insar_epochs.csv"
        write_epoch_manifest(epochs, target)
        print(f"wrote {target}")

    if args.checksums:
        print("\nSHA-256 of committed manifests:")
        for path in sorted(output_dir.glob("*.csv")):
            print(f"  {sha256_of(path)}  {path.name}")

    if failures:
        print("\nFAILED CHECKS:")
        for failure in failures:
            print(f"  * {failure}")
        return 2

    print("\nAll manifest checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
