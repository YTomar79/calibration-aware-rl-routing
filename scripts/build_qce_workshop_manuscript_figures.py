#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "paper_artifacts" / "qce_workshop_2026"
DATA_DIR = ARTIFACT_ROOT / "data"
FIG_DIR = ARTIFACT_ROOT / "figures"


BLUE = "#2F6DB3"
ORANGE = "#EE8A0A"
GREEN = "#4F9D69"
GRAY = "#5A5A5A"
LIGHT = "#F4F6F8"


def save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def build_protocol_overview() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.02, 0.25, 0.17, 0.5, "Benchmark\nMQT Bench\nfamilies"),
        (0.225, 0.25, 0.17, 0.5, "IBM calibration\nsnapshots\n(Fez/Kingston/\nMarrakesh)"),
        (0.43, 0.25, 0.17, 0.5, "Fine-tuned RL\ncheckpoints\n(10 seeds)"),
        (0.635, 0.25, 0.17, 0.5, "Noisy exact-\nfidelity\nsimulation"),
        (0.84, 0.25, 0.14, 0.5, "Agent vs\nSABRE-best20\nvs target-aware\nSABRE"),
    ]

    for x, y, w, h, text in boxes:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.4,
            edgecolor=GRAY,
            facecolor=LIGHT,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=10,
            color="#222222",
        )

    arrow_y = 0.5
    arrow_specs = [(0.19, 0.225), (0.395, 0.43), (0.60, 0.635), (0.805, 0.84)]
    for x1, x2 in arrow_specs:
        ax.annotate(
            "",
            xy=(x2, arrow_y),
            xytext=(x1, arrow_y),
            arrowprops=dict(arrowstyle="-|>", lw=1.5, color=GRAY, shrinkA=0, shrinkB=0),
        )

    ax.text(
        0.505,
        0.08,
        "Metrics: completion, exact fidelity, two-qubit count, depth",
        ha="center",
        va="center",
        fontsize=10,
        color="#222222",
        fontweight="semibold",
    )
    save(fig, "figure00_protocol_overview")


def build_compact_family_breakdown() -> None:
    rows = []
    with open(DATA_DIR / "table04_per_class.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    method_order = ["Agent", "SABRE-best20", "Target-aware SABRE"]
    color_map = {
        "Agent": BLUE,
        "SABRE-best20": ORANGE,
        "Target-aware SABRE": GREEN,
    }
    marker_map = {
        "Agent": "o",
        "SABRE-best20": "s",
        "Target-aware SABRE": "^",
    }

    grouped: dict[str, dict[str, dict[str, float]]] = {}
    completion: dict[str, float] = {}
    counts: dict[str, int] = {}

    for row in rows:
        label = row["class_label"]
        grouped.setdefault(label, {})
        grouped[label][row["method_label"]] = {
            "fidelity": float(row["exact_fidelity"]),
        }
        if row["agent_completion_rate"]:
            completion[label] = float(row["agent_completion_rate"])
        counts[label] = int(row["num_episodes"])

    labels = [
        "DJ-5q",
        "GHZ-5q",
        "QFT-5q",
        "DJ-8q",
        "GHZ-8q",
        "QFT-8q",
        "DJ-10q",
        "GHZ-10q",
        "QFT-10q",
    ]
    display_labels = [f"{label} (n={counts[label]})" for label in labels]
    ys = list(range(len(labels)))[::-1]

    fig, ax = plt.subplots(figsize=(3.4, 4.7))
    ax.set_facecolor("white")
    ax.grid(axis="x", alpha=0.22)
    ax.set_xlim(0.0, 1.06)
    ax.set_ylim(-0.75, len(labels) - 0.25)

    for y, label, display in zip(ys, labels, display_labels):
        ax.hlines(y, 0.0, 1.0, color="#B0B0B0", lw=0.8, zorder=0)
        for method in method_order:
            x = grouped[label][method]["fidelity"]
            ax.scatter(
                x,
                y,
                s=54 if method == "Agent" else 48,
                marker=marker_map[method],
                color=color_map[method],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        ax.text(
            1.02,
            y,
            f"c={completion[label]:.3f}",
            va="center",
            ha="left",
            fontsize=8.2,
            color=BLUE,
        )

    ax.set_yticks(ys)
    ax.set_yticklabels(display_labels, fontsize=8.6)
    ax.set_xlabel("Exact fidelity", fontsize=10)
    ax.set_ylabel("Circuit family", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D([0], [0], marker=marker_map[m], color="w", markerfacecolor=color_map[m],
               markeredgecolor="white", markeredgewidth=0.5, markersize=8.5, label=m)
        for m in method_order
    ]
    ax.legend(
        handles=handles,
        ncol=1,
        frameon=False,
        fontsize=8.5,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.02),
    )

    save(fig, "figure02_family_breakdown_compact")


if __name__ == "__main__":
    build_protocol_overview()
    build_compact_family_breakdown()
