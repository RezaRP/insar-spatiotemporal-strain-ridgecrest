"""Build Notebook 16 from its percent-format Python source."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "16_track64_guided_near_fault_strain.py"
OUTPUT = ROOT / "notebooks" / "16_track64_guided_near_fault_strain.ipynb"


def markdown_source(lines: list[str]) -> str:
    converted: list[str] = []
    for line in lines:
        if line.startswith("# "):
            converted.append(line[2:])
        elif line == "#":
            converted.append("")
        elif line.startswith("#"):
            converted.append(line[1:].lstrip())
        else:
            converted.append(line)
    return "\n".join(converted).rstrip() + "\n"


def build() -> None:
    cells = []
    kind: str | None = None
    buffer: list[str] = []
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        if line in {"# %% [markdown]", "# %%"}:
            if kind == "markdown":
                cells.append(
                    nbf.v4.new_markdown_cell(markdown_source(buffer))
                )
            elif kind == "code":
                cells.append(
                    nbf.v4.new_code_cell(
                        "\n".join(buffer).rstrip() + "\n"
                    )
                )
            kind = "markdown" if line.endswith("[markdown]") else "code"
            buffer = []
        else:
            buffer.append(line)
    if kind == "markdown":
        cells.append(nbf.v4.new_markdown_cell(markdown_source(buffer)))
    elif kind == "code":
        cells.append(
            nbf.v4.new_code_cell("\n".join(buffer).rstrip() + "\n")
        )

    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
    )
    nbf.write(notebook, OUTPUT)
    print(f"Wrote {OUTPUT} with {len(cells)} cells")


if __name__ == "__main__":
    build()
