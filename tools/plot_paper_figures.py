from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, str(Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"))
from audit_panel_alignment import require_matplotlib_panel_alignment


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_ieee_draft" / "3" / "figures"
DATA = ROOT / "datasets" / "coco_indoor" / "val"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams.update(
    {
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


ABLATION = [
    ("baseline-yolo11n", "Baseline", 0.3219, 0.0666, 0.3036),
    ("ab1-orb", "+ORB", 0.3227, 0.0632, 0.3050),
    ("ab2-mrfe", "+MRFE", 0.3293, 0.0551, 0.2935),
    ("ab3-lcr", "+LCR", 0.3274, 0.0703, 0.3114),
    ("ab3b-lcrbase", "+LCRBase", 0.3256, 0.0578, 0.3011),
    ("ab4-rcr-fb", "ORB+MRFE+LCRBase", 0.3343, 0.0683, 0.3143),
    ("ab5-orbmrfe", "ORB+MRFE", 0.3317, 0.0595, 0.2999),
    ("mrfelcrb", "MRFE+LCRBase", 0.3299, 0.0591, 0.2999),
    ("orblcrb", "ORB+LCRBase", 0.3282, 0.0593, 0.2944),
    ("rcr-full", "ORB+MRFE+LCR", 0.3342, 0.0669, 0.3176),
]
BASE = ABLATION[0]
METRICS = [
    ("Hard AP50–95", 2),
    ("Hard AP50-small", 3),
    ("Crowd-heavy AP50", 4),
]


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def export_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_ablation() -> None:
    deltas = np.array(
        [[(row[index] - BASE[index]) * 100 for _, index in METRICS] for row in ABLATION]
    )
    raw_rows = []
    for i, row in enumerate(ABLATION):
        raw_rows.append(
            [
                row[0],
                row[1],
                *[row[index] for _, index in METRICS],
                *deltas[i].tolist(),
                "seed 0",
            ]
        )
    write_csv(
        OUT / "ablation_delta_source.csv",
        [
            "model",
            "display_name",
            "hard_ap50_95",
            "hard_ap50_small",
            "crowd_heavy_ap50",
            "delta_hard_ap50_95_pp",
            "delta_hard_ap50_small_pp",
            "delta_crowd_heavy_ap50_pp",
            "seed",
        ],
        raw_rows,
    )

    cmap = LinearSegmentedColormap.from_list(
        "rcr_delta", ["#b65e55", "#f4f2ed", "#388b8a"], N=256
    )
    fig, ax = plt.subplots(figsize=(7.16, 3.52), layout="constrained")
    norm = TwoSlopeNorm(vmin=-1.5, vcenter=0.0, vmax=1.5)
    im = ax.imshow(deltas, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(METRICS)), [name for name, _ in METRICS])
    ax.set_yticks(np.arange(len(ABLATION)), [row[1] for row in ABLATION])
    ax.tick_params(axis="both", length=0, pad=4)
    ax.set_xticks(np.arange(-0.5, len(METRICS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ABLATION), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("Change from baseline (percentage points; seed 0)", loc="left", pad=8)

    for row_i in range(len(ABLATION)):
        for col_i in range(len(METRICS)):
            value = deltas[row_i, col_i]
            color = "white" if abs(value) >= 1.0 else "#242424"
            ax.text(
                col_i,
                row_i,
                f"{value:+.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=7,
                fontweight="bold" if row_i in (5, 9) else "normal",
            )

    for label in ax.get_yticklabels():
        if label.get_text() in ("ORB+MRFE+LCRBase", "ORB+MRFE+LCR"):
            label.set_fontweight("bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025, ticks=[-1.5, 0, 1.5])
    cbar.set_label("ΔAP (percentage points)", labelpad=5)
    cbar.ax.tick_params(length=0, labelsize=6.5)
    ax.set_xlabel("")

    require_matplotlib_panel_alignment(
        fig,
        json_out=OUT / "ablation_delta.alignment.json",
        overlay_svg=OUT / "ablation_delta.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        exclude_axes=[cbar.ax],
        strict=True,
    )
    export_figure(fig, "ablation_delta")


def add_ground_truth_boxes(ax: plt.Axes, image_id: str) -> None:
    image_path = DATA / "images" / f"{image_id}.jpg"
    label_path = DATA / "labels" / f"{image_id}.txt"
    canvas_width, canvas_height = 640, 427
    with Image.open(image_path) as source:
        width, height = source.size
        scale = min(canvas_width / width, canvas_height / height)
        rendered_width = round(width * scale)
        rendered_height = round(height * scale)
        offset_x = (canvas_width - rendered_width) / 2
        offset_y = (canvas_height - rendered_height) / 2
        resized = source.convert("RGB").resize(
            (rendered_width, rendered_height), Image.Resampling.LANCZOS
        )
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        canvas.paste(resized, (round(offset_x), round(offset_y)))
        ax.imshow(canvas)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        _, xc, yc, bw, bh = values[:5]
        x = offset_x + (float(xc) - float(bw) / 2) * rendered_width
        y = offset_y + (float(yc) - float(bh) / 2) * rendered_height
        w = float(bw) * rendered_width
        h = float(bh) * rendered_height
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#161616", linewidth=1.2))
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#f2c14e", linewidth=0.6))
    ax.set_xlim(0, canvas_width)
    ax.set_ylim(canvas_height, 0)
    ax.set_aspect("equal")
    ax.set_axis_off()


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.035,
        1.035,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        clip_on=False,
    )


def plot_dataset_overview() -> None:
    size_rows = [
        ("Small", 7846, "A < 32²"),
        ("Medium", 9431, "32² ≤ A < 96²"),
        ("Large", 8705, "A ≥ 96²"),
    ]
    crowd_rows = [
        ("Crowd-free", 8449, "overlap < 0.1"),
        ("Partial", 6207, "0.1–0.5"),
        ("Crowd-heavy", 11326, "overlap ≥ 0.5"),
    ]
    summary_rows = [
        ["object_size", name, count, definition]
        for name, count, definition in size_rows
    ] + [
        ["box_overlap", name, count, definition]
        for name, count, definition in crowd_rows
    ]
    write_csv(
        OUT / "dataset_summary_source.csv",
        ["stratification", "group", "ground_truth_boxes", "definition"],
        summary_rows,
    )

    fig = plt.figure(figsize=(7.16, 3.75), layout="constrained")
    outer = fig.add_gridspec(2, 1, height_ratios=[0.68, 1.0], hspace=0.12)
    image_grid = outer[0].subgridspec(1, 3, wspace=0.08)
    examples = [
        ("000000000139", "Typical room"),
        ("000000018380", "Crowded dining"),
        ("000000435081", "Food and small objects"),
    ]
    image_axes = []
    for i, (image_id, title) in enumerate(examples):
        ax = fig.add_subplot(image_grid[0, i])
        add_ground_truth_boxes(ax, image_id)
        ax.set_title(title, pad=4)
        add_panel_label(ax, chr(ord("a") + i))
        image_axes.append(ax)

    chart_grid = outer[1].subgridspec(1, 2, wspace=0.27)
    size_ax = fig.add_subplot(chart_grid[0, 0])
    crowd_ax = fig.add_subplot(chart_grid[0, 1])
    categories = ["#397b83", "#7ea7a2", "#b7c9c1"]
    for ax, rows, panel, title in (
        (size_ax, size_rows, "d", "Object-size groups"),
        (crowd_ax, crowd_rows, "e", "Ground-truth box overlap"),
    ):
        labels = [row[0] for row in rows]
        counts = [row[1] for row in rows]
        bars = ax.barh(labels, np.array(counts) / 1000, color=categories, height=0.58)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", pad=7)
        ax.set_xlabel("Ground-truth boxes (thousands)", labelpad=3)
        ax.set_xlim(0, 13.2)
        ax.set_xticks([0, 4, 8, 12])
        ax.tick_params(axis="both", length=0, pad=3)
        ax.grid(False)
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_width() + 0.18,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,}",
                ha="left",
                va="center",
                fontsize=6.5,
                color="#333333",
            )
        add_panel_label(ax, panel)

    require_matplotlib_panel_alignment(
        fig,
        json_out=OUT / "dataset_overview.alignment.json",
        overlay_svg=OUT / "dataset_overview.alignment.svg",
        row_groups=[["a", "b", "c"], ["d", "e"]],
        strict=True,
    )
    export_figure(fig, "dataset_overview")


def main() -> None:
    plot_ablation()
    plot_dataset_overview()
    (OUT / "figure_notes.md").write_text(
        "# Figure notes\n\n"
        "- Ablation values are seed-0 eval_hard v2 AP values from the 300-epoch runs. "
        "Cells show percentage-point differences from baseline; no uncertainty is implied.\n"
        "- Dataset overview uses the 3,970-image validation split. Size buckets follow "
        "COCO area thresholds. Crowd groups are based on the project box-overlap proxy, "
        "not physical occlusion.\n"
        "- Dataset examples: COCO image IDs 000000000139, 000000018380, and 000000435081. "
        "Only ground-truth rectangles are overlaid; source images are otherwise unaltered.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
