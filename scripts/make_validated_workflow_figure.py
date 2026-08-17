"""Create the validated Ridgecrest analysis workflow for the manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "publication_figures"


COLORS = {
    "data_fill": "#DCEEFF",
    "data_edge": "#2C6F9E",
    "prep_fill": "#E4F4E8",
    "prep_edge": "#3B8C5A",
    "analysis_fill": "#FFF0D2",
    "analysis_edge": "#C87800",
    "validation_fill": "#F1E5F5",
    "validation_edge": "#8E4C97",
    "output_fill": "#ECEFF1",
    "output_edge": "#4F5963",
    "lane_fill": "#F7F8FA",
    "lane_edge": "#CBD0D6",
    "text": "#202428",
    "muted": "#5C6670",
}


def rounded_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    fill: str,
    edge: str,
    fontsize: float = 8.4,
    weight: str = "normal",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.010,rounding_size=0.012",
        linewidth=1.5,
        edgecolor=edge,
        facecolor=fill,
        transform=axis.transAxes,
        zorder=2,
    )
    axis.add_patch(patch)
    if "\n\n" in text:
        title, body = text.split("\n\n", 1)
        axis.text(
            x + width / 2,
            y + height * 0.82,
            title,
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=edge,
            zorder=3,
        )
        axis.text(
            x + width / 2,
            y + height * 0.43,
            body,
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=weight,
            color=COLORS["text"],
            linespacing=1.15,
            zorder=3,
        )
    else:
        axis.text(
            x + width / 2,
            y + height / 2,
            text,
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=weight,
            color=COLORS["text"],
            linespacing=1.15,
            zorder=3,
        )


def arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#606A73",
    connectionstyle: str = "arc3,rad=0",
) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=axis.transAxes,
        textcoords=axis.transAxes,
        arrowprops={
            "arrowstyle": "-|>",
            "linewidth": 1.35,
            "color": color,
            "shrinkA": 2,
            "shrinkB": 2,
            "connectionstyle": connectionstyle,
        },
        zorder=4,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # Final two-column journal width: 190 mm (7.48 in).  Designing directly
    # at insertion size prevents Word/LaTeX from halving the effective type.
    figure, axis = plt.subplots(figsize=(7.48, 4.70))
    figure.patch.set_facecolor("white")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    lane_x, lane_w, lane_h = 0.025, 0.95, 0.365
    top_y, bottom_y = 0.555, 0.090
    for y in (top_y, bottom_y):
        lane = FancyBboxPatch(
            (lane_x, y),
            lane_w,
            lane_h,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=1.0,
            edgecolor=COLORS["lane_edge"],
            facecolor=COLORS["lane_fill"],
            transform=axis.transAxes,
            zorder=0,
        )
        axis.add_patch(lane)

    axis.text(
        0.044,
        top_y + lane_h / 2,
        "A",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=COLORS["data_edge"],
    )
    axis.text(
        0.065,
        top_y + lane_h - 0.032,
        "Full-scene pre-event change assessment",
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=9.6,
        fontweight="bold",
        color=COLORS["text"],
    )
    axis.text(
        0.044,
        bottom_y + lane_h / 2,
        "B",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=COLORS["analysis_edge"],
    )
    axis.text(
        0.065,
        bottom_y + lane_h - 0.032,
        "Earthquake-sequence source inference",
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=9.6,
        fontweight="bold",
        color=COLORS["text"],
    )

    xs = [0.075, 0.305, 0.535, 0.765]
    width = 0.185
    height = 0.245
    top_box_y = top_y + 0.045
    bottom_box_y = bottom_y + 0.045

    top_texts = [
        "DATA\n\nTrack 71\ncumulative LOS\nNo GACOS / GACOS",
        "DETECTION\n\nShared QC:\n1.10 million pixels\n12/24-day changes\nRobust innovations\nClusters + FWER",
        "REPLICATION\n\nDirect 22 Jun–4 Jul\ninterferogram\nAscending/descending\ntest",
        "EVIDENCE\n\nApparent change\nor supported\ndeformation",
    ]
    bottom_texts = [
        "DATA\n\nEvent-spanning InSAR\nT64: 4–10 Jul\nT71: 4–16 Jul\n25 GNSS stations",
        "PREPROCESSING\n\nCoherence mask +\nfar-field ramp\nQuadtree sampling\nRobust GNSS offsets",
        "INFERENCE\n\nBayesian two-fault\ngeometry\nDistributed +\ninterval slip\nTransferred geometry",
        "VALIDATION\n\nSpatial-block CV\nResolution +\nuncertainty\n4 GNSS withheld",
    ]

    top_styles = [
        ("data_fill", "data_edge"),
        ("analysis_fill", "analysis_edge"),
        ("validation_fill", "validation_edge"),
        ("output_fill", "output_edge"),
    ]
    bottom_styles = [
        ("data_fill", "data_edge"),
        ("prep_fill", "prep_edge"),
        ("analysis_fill", "analysis_edge"),
        ("output_fill", "output_edge"),
    ]
    for x, text, (fill_key, edge_key) in zip(xs, top_texts, top_styles):
        rounded_box(
            axis,
            x,
            top_box_y,
            width,
            height,
            text,
            fill=COLORS[fill_key],
            edge=COLORS[edge_key],
            fontsize=8.4,
        )
    for x, text, (fill_key, edge_key) in zip(xs, bottom_texts, bottom_styles):
        rounded_box(
            axis,
            x,
            bottom_box_y,
            width,
            height,
            text,
            fill=COLORS[fill_key],
            edge=COLORS[edge_key],
            fontsize=8.4,
        )

    for y in (top_box_y, bottom_box_y):
        for left_x, right_x in zip(xs[:-1], xs[1:]):
            arrow(
                axis,
                (left_x + width, y + height / 2),
                (right_x, y + height / 2),
            )

    axis.text(
        0.5,
        0.512,
        "Detection timing alone does not establish fault slip or earthquake preparation.",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=COLORS["output_edge"],
    )

    caption = (
        "Pre-event intervals are tested on the transferred geometry, but unsupported "
        "source estimates are not interpreted as physical fault slip."
    )
    axis.text(
        0.5,
        0.035,
        caption,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=7.8,
        color=COLORS["muted"],
        style="italic",
    )

    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            OUT / f"figure2_validated_workflow.{suffix}",
            dpi=600 if suffix == "png" else None,
            facecolor="white",
            edgecolor="none",
        )
    # Screen-scale QA copy: 190 mm wide at 144 dpi. At native display size,
    # this approximates a 100% page-width inspection on a high-density screen.
    figure.savefig(
        OUT / "figure2_validated_workflow_100pct.png",
        dpi=144,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
