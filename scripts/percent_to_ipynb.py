"""Convert a VS Code # %% notebook-style Python file into a Jupyter notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4


def as_markdown(lines: list[str]) -> str:
    output: list[str] = []
    for line in lines:
        if line.startswith("# "):
            output.append(line[2:])
        elif line.startswith("#"):
            output.append(line[1:])
        else:
            output.append(line)
    return "".join(output)


def build_notebook(source: Path) -> dict[str, object]:
    cells: list[dict[str, object]] = []
    kind: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer, kind
        if kind is None or not buffer:
            buffer = []
            return
        if kind == "markdown":
            cells.append(
                {
                    "id": uuid4().hex,
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": as_markdown(buffer).splitlines(keepends=True),
                }
            )
        else:
            cells.append(
                {
                    "id": uuid4().hex,
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": buffer,
                }
            )
        buffer = []

    for line in source.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line.lower() else "code"
        else:
            buffer.append(line)
    flush()

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    args.target.parent.mkdir(parents=True, exist_ok=True)
    args.target.write_text(
        json.dumps(build_notebook(args.source), indent=1),
        encoding="utf-8",
    )
    print(args.target)


if __name__ == "__main__":
    main()
