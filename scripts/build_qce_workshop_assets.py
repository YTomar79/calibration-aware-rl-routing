#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


METHOD_ORDER = ["agent", "sabre", "qiskit_noise_aware_vf2"]
METHOD_LABELS = {
    "agent": "Agent",
    "sabre": "SABRE-best20",
    # Display label for the calibration-derived Qiskit Target + SABRE baseline.
    "qiskit_noise_aware_vf2": "Target-aware SABRE",
}
METHOD_COLORS = {
    "agent": "#2C6DB2",
    "sabre": "#E68613",
    "qiskit_noise_aware_vf2": "#4C9F70",
}
METHOD_MARKERS = {
    "agent": "o",
    "sabre": "s",
    "qiskit_noise_aware_vf2": "^",
}
CALIBRATION_LABELS = {
    "ibm_fez": "Fez",
    "ibm_kingston": "Kingston",
    "ibm_marrakesh": "Marrakesh",
}
CLASS_FAMILY_ORDER = {"dj": 0, "ghz": 1, "qft": 2}
DEFAULT_SEED = 20260525

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
    }
)


@dataclass
class ClusterRecord:
    calibration: str
    seed_label: str
    run_dir: str
    metrics: Dict[str, float]


def _load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "--"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def _fmt_p(value) -> str:
    if value is None:
        return "--"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return "--"
    if value == 0.0:
        return "0"
    if value < 1e-3:
        return f"{value:.2e}"
    return f"{value:.3f}"


def _bootstrap_mean_ci(values: Iterable[float], n_boot: int = 6000, seed: int = DEFAULT_SEED) -> Tuple[float, float, float]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(np.mean(arr))
    if arr.size == 1:
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return (mean, float(lo), float(hi))


def _finite_values(values: Iterable[float]) -> List[float]:
    finite = []
    for value in values:
        try:
            value = float(value)
        except Exception:
            continue
        if math.isfinite(value):
            finite.append(value)
    return finite


def _is_finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _ci_fields(values: Iterable[float], seed: int = DEFAULT_SEED) -> Dict[str, float]:
    all_values = _finite_values(values)
    _, lo, hi = _bootstrap_mean_ci(all_values, seed=seed)
    return {
        "ci95_low": lo,
        "ci95_high": hi,
        "record_n": len(all_values),
        "unique_value_n": len({round(value, 15) for value in all_values}),
    }


def _parse_seed_label(run_dir: str) -> str:
    name = os.path.basename(run_dir.rstrip("/"))
    match = re.search(r"(seed\d+)", name)
    if match:
        return match.group(1)
    return name


def _parse_calibration_label(calibration_file: str) -> str:
    base = os.path.basename(calibration_file)
    return base.replace("_calibration.json", "")


def _parse_class_key(class_key: str) -> Tuple[str, str, int]:
    tail = class_key.split(":")[-1]
    match = re.match(r"([a-z]+)_alg_(\d+)q", tail)
    if not match:
        return (tail, tail.upper(), 0)
    family = match.group(1)
    qubits = int(match.group(2))
    pretty = f"{family.upper()}-{qubits}q"
    return (family, pretty, qubits)


def _family_tick_label(row: Dict) -> str:
    return f"{row['class_label']}\n(n={row['num_episodes']})"


def _pareto_label_offset(label: str) -> Tuple[int, int]:
    custom_offsets = {
        "DJ-5q": (8, 5),
        "GHZ-5q": (10, 7),
        "QFT-5q": (12, 10),
        "DJ-8q": (8, 6),
        "GHZ-8q": (10, -8),
        "QFT-8q": (10, 7),
        "DJ-10q": (10, 6),
        "GHZ-10q": (10, 7),
        "QFT-10q": (10, 7),
    }
    return custom_offsets.get(label, (6, 4))


def _class_sort_key(class_key: str) -> Tuple[int, int, str]:
    family, _, qubits = _parse_class_key(class_key)
    return (qubits, CLASS_FAMILY_ORDER.get(family, 99), family)


def _series_from_summary(cluster: ClusterRecord, method: str, metric: str) -> float:
    return cluster.metrics.get(f"{method}_{metric}", float("nan"))


def _write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        cleaned_rows = []
        for row in rows:
            cleaned = {}
            for key in fieldnames:
                value = row.get(key)
                if value is None:
                    cleaned[key] = ""
                elif isinstance(value, (float, np.floating)):
                    cleaned[key] = "" if not math.isfinite(float(value)) else float(value)
                else:
                    cleaned[key] = value
            cleaned_rows.append(cleaned)
        writer.writerows(cleaned_rows)


def _write_text(path: str, text: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(text)


def _save_figure(fig, base_path: str) -> None:
    fig.savefig(base_path + ".pdf", bbox_inches="tight")
    fig.savefig(base_path + ".png", bbox_inches="tight")
    plt.close(fig)


def _collect_clusters(shard_dir: str) -> List[ClusterRecord]:
    records: List[ClusterRecord] = []
    for path in sorted(glob.glob(os.path.join(shard_dir, "*", "reviewer_benchmark_summary.json"))):
        payload = _load_json(path)
        if not payload.get("per_run"):
            continue
        per_run = payload["per_run"][0]
        summary = per_run.get("summary", {})
        flat_metrics = {}
        for name, item in summary.items():
            if isinstance(item, dict) and "mean" in item:
                flat_metrics[name] = item["mean"]
        records.append(
            ClusterRecord(
                calibration=_parse_calibration_label(str(per_run.get("calibration_file", ""))),
                seed_label=_parse_seed_label(str(per_run.get("run_dir", ""))),
                run_dir=str(per_run.get("run_dir", "")),
                metrics=flat_metrics,
            )
        )
    if not records:
        raise FileNotFoundError(f"No shard summaries found under {shard_dir}")
    return records


def _aggregate_overall_summary(summary: Dict, clusters: List[ClusterRecord]) -> List[Dict]:
    rows = []
    aggregate = summary["aggregate"]
    completion = float(summary.get("agent_completion_rate", float("nan")))
    for idx, method in enumerate(METHOD_ORDER):
        values = [_series_from_summary(cluster, method, "fidelity") for cluster in clusters]
        fidelity_ci = _ci_fields(values, seed=DEFAULT_SEED + 1700 + idx)
        rows.append(
            {
                "method_key": method,
                "method_label": METHOD_LABELS[method],
                "completion_rate": completion if method == "agent" else None,
                "exact_fidelity": aggregate.get(f"{method}_fidelity", {}).get("mean_across_runs"),
                "exact_fidelity_ci95_low": fidelity_ci["ci95_low"],
                "exact_fidelity_ci95_high": fidelity_ci["ci95_high"],
                "exact_fidelity_record_n": fidelity_ci["record_n"],
                "exact_fidelity_unique_value_n": fidelity_ci["unique_value_n"],
                "two_qubit_count": aggregate.get(f"{method}_twoq", {}).get("mean_across_runs"),
                "depth": aggregate.get(f"{method}_depth", {}).get("mean_across_runs"),
                "wall_seconds": aggregate.get(f"{method}_wall_seconds", {}).get("mean_across_runs"),
            }
        )
    return rows


def _aggregate_calibration_summary(clusters: List[ClusterRecord]) -> List[Dict]:
    grouped: Dict[str, List[ClusterRecord]] = defaultdict(list)
    for cluster in clusters:
        grouped[cluster.calibration].append(cluster)
    rows = []
    for calibration in sorted(grouped):
        bucket = grouped[calibration]
        row = {
            "calibration_key": calibration,
            "calibration_label": CALIBRATION_LABELS.get(calibration, calibration),
            "num_runs": len(bucket),
            "agent_completion_rate": np.mean([_series_from_summary(item, "agent", "completed") for item in bucket]),
        }
        for method_idx, method in enumerate(METHOD_ORDER):
            for metric in ("fidelity", "twoq", "depth", "wall_seconds"):
                row[f"{method}_{metric}"] = float(
                    np.mean([_series_from_summary(item, method, metric) for item in bucket])
                )
            fidelity_ci = _ci_fields(
                [_series_from_summary(item, method, "fidelity") for item in bucket],
                seed=DEFAULT_SEED + 1800 + method_idx + 10 * len(rows),
            )
            row[f"{method}_fidelity_ci95_low"] = fidelity_ci["ci95_low"]
            row[f"{method}_fidelity_ci95_high"] = fidelity_ci["ci95_high"]
            row[f"{method}_fidelity_record_n"] = fidelity_ci["record_n"]
            row[f"{method}_fidelity_unique_value_n"] = fidelity_ci["unique_value_n"]
        rows.append(row)
    return rows


def _annotate_agent_cost_telemetry(rows: List[Dict]) -> List[Dict]:
    for row in rows:
        row["agent_twoq_source"] = "reported_summary" if _is_finite(row.get("agent_twoq")) else "missing_in_summary"
        row["agent_depth_source"] = "reported_summary" if _is_finite(row.get("agent_depth")) else "missing_in_summary"
    return rows


def _aggregate_class_summary(summary: Dict) -> List[Dict]:
    rows = []
    completion = summary.get("completion", {}).get("per_circuit_class", {})
    for class_key in sorted(summary.get("per_circuit_class", {}), key=_class_sort_key):
        item = summary["per_circuit_class"][class_key]
        family, pretty_label, qubits = _parse_class_key(class_key)
        class_completion = completion.get(class_key, {})
        agent_twoq = item.get("agent_twoq_mean")
        agent_depth = item.get("agent_depth_mean")
        rows.append(
            {
                "class_key": class_key,
                "class_label": pretty_label,
                "family": family.upper(),
                "qubits": qubits,
                "num_episodes": int(item.get("count", 0)),
                "agent_completion_rate": class_completion.get("agent_completion_rate"),
                "agent_timeout_rate": class_completion.get("agent_timeout_rate"),
                "agent_fidelity": item.get("agent_fidelity_mean"),
                "sabre_fidelity": item.get("sabre_fidelity_mean"),
                "qiskit_noise_aware_vf2_fidelity": item.get("qiskit_noise_aware_vf2_fidelity_mean"),
                "agent_twoq": agent_twoq,
                "sabre_twoq": item.get("sabre_twoq_mean"),
                "qiskit_noise_aware_vf2_twoq": item.get("qiskit_noise_aware_vf2_twoq_mean"),
                "agent_depth": agent_depth,
                "sabre_depth": item.get("sabre_depth_mean"),
                "qiskit_noise_aware_vf2_depth": item.get("qiskit_noise_aware_vf2_depth_mean"),
                "agent_minus_sabre_fidelity": float(item.get("agent_fidelity_mean", float("nan")))
                - float(item.get("sabre_fidelity_mean", float("nan"))),
                "agent_minus_target_aware_fidelity": float(item.get("agent_fidelity_mean", float("nan")))
                - float(item.get("qiskit_noise_aware_vf2_fidelity_mean", float("nan"))),
            }
        )
    return _annotate_agent_cost_telemetry(rows)


def _aggregate_paired_effects(summary: Dict) -> List[Dict]:
    comparisons = [
        ("agent_vs_sabre_fidelity", "Exact fidelity", "Agent vs SABRE-best20"),
        (
            "agent_vs_qiskit_noise_aware_vf2_fidelity",
            "Exact fidelity",
            f"Agent vs {METHOD_LABELS['qiskit_noise_aware_vf2']}",
        ),
        ("agent_vs_sabre_twoq", "Two-qubit count", "Agent vs SABRE-best20"),
        ("agent_vs_sabre_depth", "Depth", "Agent vs SABRE-best20"),
        ("agent_vs_sabre_wall_seconds", "Wall-clock time (s)", "Agent vs SABRE-best20"),
    ]
    rows = []
    for key, metric_label, comparison_label in comparisons:
        effect = summary.get("effect_sizes", {}).get(key, {})
        paired = summary.get("paired_tests", {}).get(key, {})
        rows.append(
            {
                "comparison_key": key,
                "comparison_label": comparison_label,
                "metric": metric_label,
                "mean_improvement_positive_favors_agent": effect.get("mean_improvement"),
                "ci95_low": effect.get("ci95_low"),
                "ci95_high": effect.get("ci95_high"),
                "mean_relative_improvement": effect.get("mean_relative_improvement"),
                "rank_biserial": effect.get("rank_biserial"),
                "holm_adjusted_pvalue": paired.get("pvalue_holm"),
                "num_cells": paired.get("num_clusters"),
            }
        )
    return rows


def _aggregate_cluster_gain_rows(clusters: List[ClusterRecord]) -> List[Dict]:
    rows = []
    for cluster in clusters:
        for baseline_key in ("sabre", "qiskit_noise_aware_vf2"):
            rows.append(
                {
                    "calibration_key": cluster.calibration,
                    "calibration_label": CALIBRATION_LABELS.get(cluster.calibration, cluster.calibration),
                    "seed_label": cluster.seed_label,
                    "baseline_key": baseline_key,
                    "baseline_label": METHOD_LABELS[baseline_key],
                    "agent_minus_baseline_fidelity": _series_from_summary(cluster, "agent", "fidelity")
                    - _series_from_summary(cluster, baseline_key, "fidelity"),
                }
            )
    return rows


def _write_uncertainty_summary(out_dir: str, clusters: List[ClusterRecord], cluster_gain_rows: List[Dict]) -> List[Dict]:
    rows = []
    for idx, method in enumerate(METHOD_ORDER):
        values = [_series_from_summary(cluster, method, "fidelity") for cluster in clusters]
        pooled_mean = float(np.mean(_finite_values(values)))
            ci = _ci_fields(values, seed=DEFAULT_SEED + 1700 + idx)
            summary_type = "bootstrap_95_run_cells"
        rows.append(
            {
                "scope": "overall",
                "calibration_key": "all",
                "calibration_label": "All",
                "quantity": f"{method}_fidelity",
                "label": METHOD_LABELS[method],
                "pooled_mean": pooled_mean,
                "summary_type": summary_type,
                **ci,
            }
        )
    for calibration_idx, calibration in enumerate(sorted({cluster.calibration for cluster in clusters})):
        bucket = [cluster for cluster in clusters if cluster.calibration == calibration]
        for idx, method in enumerate(METHOD_ORDER):
            values = [_series_from_summary(cluster, method, "fidelity") for cluster in bucket]
            pooled_mean = float(np.mean(_finite_values(values)))
            ci = _ci_fields(values, seed=DEFAULT_SEED + 1800 + idx + 10 * calibration_idx)
            summary_type = "bootstrap_95_run_cells"
            rows.append(
                {
                    "scope": "calibration",
                    "calibration_key": calibration,
                    "calibration_label": CALIBRATION_LABELS.get(calibration, calibration),
                    "quantity": f"{method}_fidelity",
                    "label": METHOD_LABELS[method],
                    "pooled_mean": pooled_mean,
                    "summary_type": summary_type,
                    **ci,
                }
            )
    for idx, baseline_key in enumerate(("sabre", "qiskit_noise_aware_vf2")):
        values = [
            row["agent_minus_baseline_fidelity"]
            for row in cluster_gain_rows
            if row["baseline_key"] == baseline_key
        ]
        pooled_mean = float(np.mean(_finite_values(values)))
        ci = _ci_fields(values, seed=DEFAULT_SEED + 2400 + idx)
        rows.append(
            {
                "scope": "paired_delta",
                "calibration_key": "all",
                "calibration_label": "All",
                "quantity": f"agent_minus_{baseline_key}_fidelity",
                "label": f"Agent minus {METHOD_LABELS[baseline_key]}",
                "pooled_mean": pooled_mean,
                "summary_type": "bootstrap_95_run_cells",
                **ci,
            }
        )
    _write_csv(
        os.path.join(out_dir, "data", "table06_run_cell_uncertainty.csv"),
        rows,
        [
            "scope",
            "calibration_key",
            "calibration_label",
            "quantity",
            "label",
            "pooled_mean",
            "summary_type",
            "ci95_low",
            "ci95_high",
            "record_n",
            "unique_value_n",
        ],
    )
    return rows


def _write_latex_tables(out_dir: str, overall_rows: List[Dict], paired_rows: List[Dict], calibration_rows: List[Dict], class_rows: List[Dict]) -> None:
    tables_dir = _ensure_dir(os.path.join(out_dir, "tables"))

    overall_tex = []
    for row in overall_rows:
        overall_tex.append(
            " & ".join(
                [
                    row["method_label"],
                    _fmt(row["exact_fidelity"], 4),
                    _fmt(row["two_qubit_count"], 2),
                    _fmt(row["depth"], 2),
                ]
            )
            + r" \\"
        )
    _write_text(os.path.join(tables_dir, "table01_overall_summary.tex"), "\n".join(overall_tex) + "\n")

    paired_tex = []
    for row in paired_rows:
        paired_tex.append(
            " & ".join(
                [
                    row["comparison_label"],
                    row["metric"],
                    _fmt(row["mean_improvement_positive_favors_agent"], 4),
                    f"[{_fmt(row['ci95_low'], 4)}, {_fmt(row['ci95_high'], 4)}]",
                    _fmt_p(row["holm_adjusted_pvalue"]),
                ]
            )
            + r" \\"
        )
    _write_text(os.path.join(tables_dir, "table02_paired_effects.tex"), "\n".join(paired_tex) + "\n")

    calibration_tex = []
    for row in calibration_rows:
        calibration_tex.append(
            " & ".join(
                [
                    row["calibration_label"],
                    str(row["num_runs"]),
                    _fmt(row["agent_fidelity"], 4),
                    _fmt(row["sabre_fidelity"], 4),
                    _fmt(row["qiskit_noise_aware_vf2_fidelity"], 4),
                ]
            )
            + r" \\"
        )
    _write_text(os.path.join(tables_dir, "table03_per_calibration.tex"), "\n".join(calibration_tex) + "\n")

    class_tex = []
    for row in class_rows:
        class_tex.append(
            " & ".join(
                [
                    row["class_label"],
                    str(row["num_episodes"]),
                    _fmt(row["agent_fidelity"], 4),
                    _fmt(row["sabre_fidelity"], 4),
                    _fmt(row["qiskit_noise_aware_vf2_fidelity"], 4),
                ]
            )
            + r" \\"
        )
    _write_text(os.path.join(tables_dir, "table04_per_class.tex"), "\n".join(class_tex) + "\n")


def _figure_overall_fidelity(clusters: List[ClusterRecord], out_dir: str) -> List[Dict]:
    figures_dir = _ensure_dir(os.path.join(out_dir, "figures"))
    group_keys = ["ibm_fez", "ibm_kingston", "ibm_marrakesh", "overall"]
    group_labels = [CALIBRATION_LABELS.get(key, "Overall") if key != "overall" else "Overall" for key in group_keys]
    jitter_rng = np.random.default_rng(DEFAULT_SEED)
    x = np.arange(len(group_keys), dtype=float)
    width = 0.22
    offsets = np.linspace(-width, width, num=len(METHOD_ORDER))
    fig, ax = plt.subplots(figsize=(7.1, 3.6))
    data_rows = []
    for idx, method in enumerate(METHOD_ORDER):
        means = []
        lows = []
        highs = []
        for group_key in group_keys:
            if group_key == "overall":
                values = [_series_from_summary(cluster, method, "fidelity") for cluster in clusters]
            else:
                values = [
                    _series_from_summary(cluster, method, "fidelity")
                    for cluster in clusters
                    if cluster.calibration == group_key
                ]
            finite_values = _finite_values(values)
            mean = float(np.mean(finite_values))
            _, lo, hi = _bootstrap_mean_ci(finite_values, seed=DEFAULT_SEED + idx * 101 + len(values))
            lo = min(lo, mean)
            hi = max(hi, mean)
            means.append(mean)
            lows.append(mean - lo)
            highs.append(hi - mean)
            for value in values:
                data_rows.append(
                    {
                        "group_key": group_key,
                        "group_label": "Overall" if group_key == "overall" else CALIBRATION_LABELS.get(group_key, group_key),
                        "method_key": method,
                        "method_label": METHOD_LABELS[method],
                        "exact_fidelity": value,
                    }
                )
        bar_x = x + offsets[idx]
        ax.bar(
            bar_x,
            means,
            width=width * 0.95,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.8,
            label=METHOD_LABELS[method],
            zorder=2,
        )
        ax.errorbar(
            bar_x,
            means,
            yerr=[lows, highs],
            fmt="none",
            ecolor="#2A2A2A",
            elinewidth=1.0,
            capsize=3.0,
            zorder=3,
        )
        for group_idx, group_key in enumerate(group_keys):
            values = [
                row["exact_fidelity"]
                for row in data_rows
                if row["group_key"] == group_key and row["method_key"] == method
            ]
            if not values:
                continue
            jitter = jitter_rng.uniform(-width * 0.18, width * 0.18, size=len(values))
            ax.scatter(
                np.full(len(values), bar_x[group_idx]) + jitter,
                values,
                s=16,
                alpha=0.55,
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.35,
                zorder=4,
            )
    ax.set_ylabel("Benchmark exact fidelity")
    ax.set_xticks(x, group_labels)
    ax.set_ylim(0.0, 1.02)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    fig.tight_layout()
    _save_figure(fig, os.path.join(figures_dir, "figure01_fidelity_by_calibration"))
    return data_rows


def _figure_family_breakdown(summary: Dict, out_dir: str) -> List[Dict]:
    figures_dir = _ensure_dir(os.path.join(out_dir, "figures"))
    class_rows = _aggregate_class_summary(summary)
    y = np.arange(len(class_rows), dtype=float)
    offsets = np.linspace(-0.18, 0.18, num=len(METHOD_ORDER))
    fig, (ax_fid, ax_delta) = plt.subplots(
        1,
        2,
        figsize=(6.6, 4.9),
        gridspec_kw={"width_ratios": [2.0, 1.15]},
        sharey=True,
    )
    tick_fontsize = 9.5
    axis_fontsize = 11
    legend_fontsize = 9.5
    data_rows = []
    for idx, method in enumerate(METHOD_ORDER):
        values = [row[f"{method}_fidelity"] for row in class_rows]
        data_rows.extend(
            [
                {
                    "class_key": row["class_key"],
                    "class_label": row["class_label"],
                    "method_key": method,
                    "method_label": METHOD_LABELS[method],
                    "exact_fidelity": row[f"{method}_fidelity"],
                    "agent_minus_sabre": row["agent_fidelity"] - row["sabre_fidelity"],
                    "agent_minus_target_aware": row["agent_fidelity"] - row["qiskit_noise_aware_vf2_fidelity"],
                }
                for row in class_rows
            ]
        )
        ax_fid.scatter(
            values,
            y + offsets[idx],
            s=52,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            label=METHOD_LABELS[method],
            zorder=3,
        )
    for row_idx, row in enumerate(class_rows):
        band_values = [row[f"{method}_fidelity"] for method in METHOD_ORDER if math.isfinite(float(row[f"{method}_fidelity"]))]
        if band_values:
            ax_fid.hlines(
                y[row_idx],
                xmin=min(band_values),
                xmax=max(band_values),
                color="#A8A8A8",
                linewidth=0.8,
                alpha=0.7,
                zorder=1,
            )
    ax_fid.set_xlabel("Exact fidelity", fontsize=axis_fontsize)
    ax_fid.set_ylabel("Circuit family", fontsize=axis_fontsize)
    ax_fid.set_xlim(0.0, 1.02)
    ax_fid.set_yticks(y, [_family_tick_label(row) for row in class_rows], fontsize=tick_fontsize)
    ax_fid.tick_params(axis="x", labelsize=tick_fontsize)
    ax_fid.invert_yaxis()
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKERS[method],
            color="none",
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor=METHOD_COLORS[method],
            markersize=9,
            label=METHOD_LABELS[method],
        )
        for method in METHOD_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        fontsize=legend_fontsize,
        handletextpad=0.4,
        columnspacing=1.2,
    )

    delta_sabre = [row["agent_fidelity"] - row["sabre_fidelity"] for row in class_rows]
    delta_target = [row["agent_fidelity"] - row["qiskit_noise_aware_vf2_fidelity"] for row in class_rows]
    ax_delta.axvline(0.0, color="#4A4A4A", linewidth=0.8)
    ax_delta.scatter(
        delta_sabre,
        y - 0.08,
        s=44,
        color=METHOD_COLORS["sabre"],
        marker="o",
        label="vs SABRE-best20",
    )
    ax_delta.scatter(
        delta_target,
        y + 0.08,
        s=44,
        color=METHOD_COLORS["qiskit_noise_aware_vf2"],
        marker="s",
        label="vs Target-aware",
    )
    ax_delta.set_xlabel("Agent fidelity gain", fontsize=axis_fontsize)
    ax_delta.set_xlim(-0.16, 0.78)
    ax_delta.set_xticks([-0.1, 0.0, 0.3, 0.6])
    ax_delta.tick_params(axis="x", labelsize=tick_fontsize)
    ax_delta.legend(loc="lower right", frameon=False, fontsize=legend_fontsize)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), w_pad=1.1)
    _save_figure(fig, os.path.join(figures_dir, "figure02_family_breakdown"))
    return data_rows


def _figure_pareto(class_rows: List[Dict], out_dir: str) -> List[Dict]:
    figures_dir = _ensure_dir(os.path.join(out_dir, "figures"))
    fig, ax = plt.subplots(figsize=(6.2, 4.9))
    tick_fontsize = 10
    axis_fontsize = 11
    legend_fontsize = 10
    data_rows = []
    for method in METHOD_ORDER:
        xs = []
        ys = []
        labels = []
        for row in class_rows:
            twoq = row[f"{method}_twoq"]
            fidelity = row[f"{method}_fidelity"]
            if not (_is_finite(twoq) and _is_finite(fidelity)):
                continue
            xs.append(float(twoq))
            ys.append(float(fidelity))
            labels.append(row["class_label"])
            data_rows.append(
                {
                    "class_key": row["class_key"],
                    "class_label": row["class_label"],
                    "method_key": method,
                    "method_label": METHOD_LABELS[method],
                    "two_qubit_count": twoq,
                    "two_qubit_count_source": row.get(f"{method}_twoq_source", "observed"),
                    "exact_fidelity": fidelity,
                    "num_episodes": row["num_episodes"],
                }
            )
        sizes = [60 + 0.24 * next(item["num_episodes"] for item in class_rows if item["class_label"] == label) for label in labels]
        ax.scatter(
            xs,
            ys,
            s=sizes,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.6,
            label=METHOD_LABELS[method],
        )
        if method == "agent":
            for xx, yy, label in zip(xs, ys, labels):
                dx, dy = _pareto_label_offset(label)
                ax.annotate(
                    label,
                    (xx, yy),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8.8,
                    color="#2A2A2A",
                )
    ax.set_xlabel("Mean two-qubit gate count", fontsize=axis_fontsize)
    ax.set_ylabel("Mean exact fidelity", fontsize=axis_fontsize)
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.02)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.legend(loc="lower right", frameon=False, fontsize=legend_fontsize)
    fig.tight_layout()
    _save_figure(fig, os.path.join(figures_dir, "figure03_pareto_by_family"))
    return data_rows


def _figure_cluster_gains(cluster_gain_rows: List[Dict], summary: Dict, out_dir: str) -> None:
    figures_dir = _ensure_dir(os.path.join(out_dir, "figures"))
    order = ["sabre", "qiskit_noise_aware_vf2"]
    fig, ax = plt.subplots(figsize=(6.8, 3.1))
    y_positions = np.arange(len(order))
    rng = np.random.default_rng(DEFAULT_SEED + 77)
    for yy, baseline_key in zip(y_positions, order):
        values = [
            row["agent_minus_baseline_fidelity"]
            for row in cluster_gain_rows
            if row["baseline_key"] == baseline_key and math.isfinite(float(row["agent_minus_baseline_fidelity"]))
        ]
        bp = ax.boxplot(
            [values],
            positions=[yy],
            vert=False,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.2},
        )
        for patch in bp["boxes"]:
            patch.set(facecolor=METHOD_COLORS[baseline_key], alpha=0.25, edgecolor=METHOD_COLORS[baseline_key], linewidth=1.0)
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(
            values,
            np.full(len(values), yy) + jitter,
            s=24,
            color=METHOD_COLORS[baseline_key],
            alpha=0.72,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        effect_key = f"agent_vs_{baseline_key}_fidelity"
        effect = summary["effect_sizes"][effect_key]
        ax.text(
            max(values) + 0.01,
            yy,
            f"mean={_fmt(effect['mean_improvement'], 3)}\nrecords={len(values)}",
            va="center",
            ha="left",
            fontsize=8,
        )
    ax.axvline(0.0, color="#4F4F4F", linewidth=1.0, linestyle="--")
    ax.set_yticks(y_positions, [METHOD_LABELS[key] for key in order])
    ax.set_xlabel("Run-level fidelity gain (agent minus baseline)")
    ax.set_xlim(left=min(row["agent_minus_baseline_fidelity"] for row in cluster_gain_rows) - 0.02)
    fig.tight_layout()
    _save_figure(fig, os.path.join(figures_dir, "figure04_record_fidelity_gains"))


def _write_readme(out_dir: str, summary: Dict) -> None:
    text = f"""# QCE Workshop Asset Bundle

This bundle was generated from the merged benchmark at:

- `mqt_completion_finetune/reviewer_benchmark/reviewer_benchmark_summary.json`
- `mqt_completion_finetune/reviewer_benchmark_shards/*`

## Main-paper candidates

1. `figures/figure01_fidelity_by_calibration.pdf`
   - Main headline figure for the benchmark exact-fidelity win across Fez, Kingston, Marrakesh, and overall.
2. `figures/figure02_family_breakdown.pdf`
   - Shows where the fidelity gain comes from by circuit family and summarizes the agent-minus-baseline deltas.
3. `tables/table01_overall_summary.tex`
   - Compact aggregate metric table for exact fidelity, two-qubit count, and depth.
4. `tables/table02_paired_effects.tex`
   - Paired record-level fidelity/resource comparisons.
5. `data/table06_run_cell_uncertainty.csv`
   - Run-by-calibration-cell bootstrap intervals used by the workshop tables.

## Backup / appendix candidates

- `figures/figure03_pareto_by_family.pdf`
- `figures/figure04_record_fidelity_gains.pdf`
- `tables/table03_per_calibration.tex`
- `tables/table04_per_class.tex`

## Baseline naming note

This asset bundle uses the display label `{METHOD_LABELS['qiskit_noise_aware_vf2']}` for the
calibration-derived Qiskit `Target` baseline with SABRE layout/routing and optimization level 3
over the same best-of-20 seed sweep, selected first by proxy fidelity and then by the same cost
score. The underlying JSON keys still use `qiskit_noise_aware_vf2_*`.

## Top-line numbers

- Benchmark episode pairs: {summary['pooled_episode_pairs']}
- Episode pairs per run_dir x calibration_file cell: 50
- Agent mean progress: {_fmt(summary['agent_mean_progress'], 3)}
- Agent exact fidelity: {_fmt(summary['aggregate']['agent_fidelity']['mean_across_runs'], 4)}
- SABRE-best20 exact fidelity: {_fmt(summary['aggregate']['sabre_fidelity']['mean_across_runs'], 4)}
- {METHOD_LABELS['qiskit_noise_aware_vf2']} exact fidelity: {_fmt(summary['aggregate']['qiskit_noise_aware_vf2_fidelity']['mean_across_runs'], 4)}
"""
    _write_text(os.path.join(out_dir, "README.md"), text)


def _write_caption_notes(out_dir: str) -> None:
    text = """# Suggested manuscript captions and placement notes

## Figure 1
`figures/figure01_fidelity_by_calibration.pdf`

Benchmark exact fidelity across the three calibration snapshots and the pooled overall benchmark. Bars show mean fidelity across 10 seed-level benchmark records per calibration (30 records total); points show the individual record means. The agent outperforms both SABRE-best20 and the historical target-aware SABRE fallback on every calibration.

## Figure 2
`figures/figure02_family_breakdown.pdf`

Per-circuit-family exact fidelity and agent-minus-baseline fidelity deltas for all benchmark families. The figure highlights the 5q/8q gains and the narrower or reversed gains at 10q.

## Figure 3 (backup / appendix)
`figures/figure03_pareto_by_family.pdf`

Mean exact fidelity versus mean two-qubit gate count, stratified by circuit family. This figure supports the tradeoff discussion: the agent usually buys fidelity with additional entangling-gate count.

## Figure 4 (backup / appendix)
`figures/figure04_record_fidelity_gains.pdf`

Distribution of record-level fidelity gains (agent minus baseline) over the 30 run-by-calibration benchmark records. Every record shows a positive fidelity gain against both baselines, with coincident points reflecting repeated run-level means.

## Table 1
`tables/table01_overall_summary.tex`

Main summary table for the workshop paper. Keep this in the core results section.

## Table 2
`tables/table02_paired_effects.tex`

Run-by-calibration-cell paired effects. This table is compact enough for the main paper if space permits; otherwise move it to a short appendix / supplementary block.

## Table 3
`tables/table03_per_calibration.tex`

Useful if we want a calibration-specific paragraph, but probably not necessary in a 4-page workshop version.

## Table 4
`tables/table04_per_class.tex`

Good appendix material. If the main paper gets tight, Figure 2 carries the same story more efficiently.

## Table 6
`data/table06_run_cell_uncertainty.csv`

Use this for manuscript intervals and any reviewer response about the statistical unit. The intervals are descriptive bootstrap intervals over the 30 run-by-calibration cell means.

"""
    _write_text(os.path.join(out_dir, "captions.md"), text)


def _write_manifest(out_dir: str) -> None:
    manifest = {
        "figures": sorted(glob.glob(os.path.join(out_dir, "figures", "*"))),
        "tables": sorted(glob.glob(os.path.join(out_dir, "tables", "*"))),
        "data": sorted(glob.glob(os.path.join(out_dir, "data", "*"))),
    }
    _write_text(os.path.join(out_dir, "manifest.json"), json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build QCE workshop figures and tables from merged benchmark artifacts.")
    parser.add_argument(
        "--summary",
        default="mqt_completion_finetune/reviewer_benchmark/reviewer_benchmark_summary.json",
        help="Path to merged reviewer benchmark summary JSON.",
    )
    parser.add_argument(
        "--shard-dir",
        default="mqt_completion_finetune/reviewer_benchmark_shards",
        help="Directory containing per-cell reviewer benchmark shards.",
    )
    parser.add_argument(
        "--out-dir",
        default="paper_artifacts/qce_workshop_2026",
        help="Output directory for figures, tables, and export data.",
    )
    args = parser.parse_args()

    summary = _load_json(args.summary)
    clusters = _collect_clusters(args.shard_dir)
    overall_rows = _aggregate_overall_summary(summary, clusters)
    calibration_rows = _aggregate_calibration_summary(clusters)
    class_rows = _aggregate_class_summary(summary)
    paired_rows = _aggregate_paired_effects(summary)
    cluster_gain_rows = _aggregate_cluster_gain_rows(clusters)
    _write_uncertainty_summary(args.out_dir, clusters, cluster_gain_rows)

    out_dir = _ensure_dir(args.out_dir)
    data_dir = _ensure_dir(os.path.join(out_dir, "data"))
    for stale_path in [
        os.path.join(out_dir, "figures", "figure04_cluster_fidelity_gains.pdf"),
        os.path.join(out_dir, "figures", "figure04_cluster_fidelity_gains.png"),
        os.path.join(data_dir, "table05_cluster_fidelity_gains.csv"),
        os.path.join(data_dir, "table06_distinct_uncertainty.csv"),
    ]:
        if os.path.exists(stale_path):
            os.remove(stale_path)

    _write_csv(
        os.path.join(data_dir, "table01_overall_summary.csv"),
        overall_rows,
        [
            "method_key",
            "method_label",
            "exact_fidelity",
            "exact_fidelity_ci95_low",
            "exact_fidelity_ci95_high",
            "exact_fidelity_record_n",
            "exact_fidelity_unique_value_n",
            "two_qubit_count",
            "depth",
            "wall_seconds",
        ],
    )
    _write_csv(
        os.path.join(data_dir, "table02_paired_effects.csv"),
        paired_rows,
        [
            "comparison_key",
            "comparison_label",
            "metric",
            "mean_improvement_positive_favors_agent",
            "ci95_low",
            "ci95_high",
            "mean_relative_improvement",
            "rank_biserial",
            "holm_adjusted_pvalue",
            "num_cells",
        ],
    )
    _write_csv(
        os.path.join(data_dir, "table03_per_calibration.csv"),
        calibration_rows,
        [
            "calibration_key",
            "calibration_label",
            "num_runs",
            "agent_fidelity",
            "agent_fidelity_ci95_low",
            "agent_fidelity_ci95_high",
            "agent_fidelity_record_n",
            "agent_fidelity_unique_value_n",
            "sabre_fidelity",
            "sabre_fidelity_ci95_low",
            "sabre_fidelity_ci95_high",
            "sabre_fidelity_record_n",
            "sabre_fidelity_unique_value_n",
            "qiskit_noise_aware_vf2_fidelity",
            "qiskit_noise_aware_vf2_fidelity_ci95_low",
            "qiskit_noise_aware_vf2_fidelity_ci95_high",
            "qiskit_noise_aware_vf2_fidelity_record_n",
            "qiskit_noise_aware_vf2_fidelity_unique_value_n",
            "agent_twoq",
            "sabre_twoq",
            "qiskit_noise_aware_vf2_twoq",
            "agent_depth",
            "sabre_depth",
            "qiskit_noise_aware_vf2_depth",
            "agent_wall_seconds",
            "sabre_wall_seconds",
            "qiskit_noise_aware_vf2_wall_seconds",
        ],
    )
    _write_csv(
        os.path.join(data_dir, "table04_per_class.csv"),
        class_rows,
        [
            "class_key",
            "class_label",
            "family",
            "qubits",
            "num_episodes",
            "agent_completion_rate",
            "agent_timeout_rate",
            "agent_fidelity",
            "sabre_fidelity",
            "qiskit_noise_aware_vf2_fidelity",
            "agent_twoq",
            "agent_twoq_source",
            "sabre_twoq",
            "qiskit_noise_aware_vf2_twoq",
            "agent_depth",
            "agent_depth_source",
            "sabre_depth",
            "qiskit_noise_aware_vf2_depth",
            "agent_minus_sabre_fidelity",
            "agent_minus_target_aware_fidelity",
        ],
    )
    _write_csv(
        os.path.join(data_dir, "table05_record_fidelity_gains.csv"),
        cluster_gain_rows,
        ["calibration_key", "calibration_label", "seed_label", "baseline_key", "baseline_label", "agent_minus_baseline_fidelity"],
    )

    figure1_rows = _figure_overall_fidelity(clusters, out_dir)
    _write_csv(
        os.path.join(data_dir, "figure01_fidelity_by_calibration.csv"),
        figure1_rows,
        ["group_key", "group_label", "method_key", "method_label", "exact_fidelity"],
    )
    figure2_rows = _figure_family_breakdown(summary, out_dir)
    _write_csv(
        os.path.join(data_dir, "figure02_family_breakdown.csv"),
        figure2_rows,
        ["class_key", "class_label", "method_key", "method_label", "exact_fidelity"],
    )
    figure3_rows = _figure_pareto(class_rows, out_dir)
    _write_csv(
        os.path.join(data_dir, "figure03_pareto_by_family.csv"),
        figure3_rows,
        [
            "class_key",
            "class_label",
            "method_key",
            "method_label",
            "two_qubit_count",
            "two_qubit_count_source",
            "exact_fidelity",
            "num_episodes",
        ],
    )
    _figure_cluster_gains(cluster_gain_rows, summary, out_dir)

    _write_latex_tables(out_dir, overall_rows, paired_rows, calibration_rows, class_rows)
    _write_readme(out_dir, summary)
    _write_caption_notes(out_dir)
    _write_manifest(out_dir)

    print(f"Wrote QCE workshop assets to {out_dir}")


if __name__ == "__main__":
    main()
