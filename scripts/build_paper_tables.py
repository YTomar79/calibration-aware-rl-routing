import argparse
import json
import math
import os


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _fmt(value, digits=3):
    if value is None:
        return "--"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _latex_escape(value):
    return str(value).replace("_", "\\_")


def _metric(summary, name):
    item = summary.get("aggregate", {}).get(name, {})
    return item.get("mean_across_runs")


def build_baseline_table(summary):
    rows = []
    labels = [
        ("Agent", "agent"),
        ("SABRE-best20", "sabre"),
        ("Target-aware SABRE", "qiskit_noise_aware_vf2"),
    ]
    for label, key in labels:
        fid = _metric(summary, f"{key}_fidelity")
        twoq = _metric(summary, f"{key}_twoq")
        depth = _metric(summary, f"{key}_depth")
        wall = _metric(summary, f"{key}_wall_seconds")
        rows.append(f"{label} & {_fmt(fid, 4)} & {_fmt(twoq, 1)} & {_fmt(depth, 1)} & {_fmt(wall, 3)} \\\\")
    return "\n".join(rows) + "\n"


def build_stats_table(summary):
    rows = []
    for name, test in sorted(summary.get("paired_tests", {}).items()):
        effect = summary.get("effect_sizes", {}).get(name, {})
        rows.append(
            f"{_latex_escape(name)} & "
            f"{_fmt(test.get('pvalue_holm'), 4)} & "
            f"{_fmt(effect.get('mean_improvement'), 4)} & "
            f"{_fmt(effect.get('mean_relative_improvement'), 4)} & "
            f"{_fmt(effect.get('rank_biserial'), 3)} \\\\"
        )
    return "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--out", default="paper_tables")
    parser.add_argument("--calibration", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--matrix-dir", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    summary = _load_json(args.benchmark)

    _write(os.path.join(args.out, "baseline_table.tex"), build_baseline_table(summary))
    _write(os.path.join(args.out, "stats_table.tex"), build_stats_table(summary))
    print(f"Wrote RL benchmark LaTeX table inputs to {args.out}")


if __name__ == "__main__":
    main()
