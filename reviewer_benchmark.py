"""Held-out benchmark: evaluate the trained router against SABRE baselines under exact noisy simulation."""

import json
import os
import copy
import subprocess
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
try:
    from importlib import metadata as importlib_metadata
except ImportError:
    import importlib_metadata

from scalable_quantum import (
    PPOAgent,
    QuantumRoutingEnv,
    TrainingConfig,
    evaluate_agent_and_baselines_holdout,
    find_latest_checkpoint,
)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _package_version(name: str):
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None


def _git_revision(project_root: str):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _safe_wilcoxon(x: List[float], y: List[float]) -> Dict[str, float]:
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return {"statistic": float("nan"), "pvalue": float("nan")}

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    if x_arr.size < 2 or y_arr.size < 2 or x_arr.size != y_arr.size:
        return {"statistic": float("nan"), "pvalue": float("nan")}
    if np.allclose(x_arr - y_arr, 0.0):
        return {"statistic": 0.0, "pvalue": 1.0}

    stat, pvalue = wilcoxon(
        x_arr,
        y_arr,
        zero_method="wilcox",
        alternative="two-sided",
        method="approx",
    )
    return {"statistic": float(stat), "pvalue": float(pvalue)}


def _holm_bonferroni(pvalues_by_name: Dict[str, float]) -> Dict[str, float]:
    finite = [
        (name, float(p))
        for name, p in pvalues_by_name.items()
        if np.isfinite(float(p)) and 0.0 <= float(p) <= 1.0
    ]
    adjusted = {name: float("nan") for name in pvalues_by_name}
    if not finite:
        return adjusted

    finite.sort(key=lambda item: item[1])
    m = len(finite)
    running = 0.0
    for rank, (name, pvalue) in enumerate(finite, start=1):
        corrected = min(1.0, float(m - rank + 1) * pvalue)
        running = max(running, corrected)
        adjusted[name] = float(running)
    return adjusted


def _paired_improvement_summary(
    x: List[float],
    y: List[float],
    higher_is_better: bool,
    seed: int = 12345,
) -> Dict[str, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    if x_arr.size == 0 or x_arr.size != y_arr.size:
        return {
            "mean_improvement": float("nan"),
            "mean_relative_improvement": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "rank_biserial": float("nan"),
            "n": 0,
        }

    improvements = x_arr - y_arr if higher_is_better else y_arr - x_arr
    denom = np.maximum(np.abs(y_arr), 1e-12)
    relative_improvements = improvements / denom
    nonzero = improvements[np.abs(improvements) > 1e-12]
    if nonzero.size == 0:
        rank_biserial = 0.0
    else:
        try:
            from scipy.stats import rankdata

            ranks = rankdata(np.abs(nonzero))
            pos = float(np.sum(ranks[nonzero > 0.0]))
            neg = float(np.sum(ranks[nonzero < 0.0]))
            denom = pos + neg
            rank_biserial = float((pos - neg) / denom) if denom > 0.0 else 0.0
        except Exception:
            rank_biserial = float("nan")

    if improvements.size < 2:
        low = high = float(np.mean(improvements))
    else:
        rng = np.random.default_rng(seed)
        sample_means = []
        for _ in range(2000):
            sample = rng.choice(improvements, size=improvements.size, replace=True)
            sample_means.append(float(np.mean(sample)))
        low, high = np.percentile(sample_means, [2.5, 97.5])

    return {
        "mean_improvement": float(np.mean(improvements)),
        "mean_relative_improvement": float(np.mean(relative_improvements)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "rank_biserial": float(rank_biserial),
        "n": int(improvements.size),
    }


def _clustered_metric_pairs(pooled_episodes: List[Dict], algorithm: str, metric: str):
    clusters = {}
    for item in pooled_episodes:
        if "agent" not in item or algorithm not in item:
            continue
        if metric not in item["agent"] or metric not in item[algorithm]:
            continue
        cluster_key = (
            os.path.abspath(str(item.get("run_dir", ""))),
            os.path.abspath(str(item.get("calibration_file", ""))),
        )
        bucket = clusters.setdefault(cluster_key, {"agent": [], "baseline": []})
        bucket["agent"].append(float(item["agent"][metric]))
        bucket["baseline"].append(float(item[algorithm][metric]))

    xs = []
    ys = []
    cluster_summaries = []
    for (run_dir, calibration_file), bucket in sorted(clusters.items()):
        x = np.asarray(bucket["agent"], dtype=float)
        y = np.asarray(bucket["baseline"], dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if not valid.any():
            continue
        x_mean = float(np.mean(x[valid]))
        y_mean = float(np.mean(y[valid]))
        xs.append(x_mean)
        ys.append(y_mean)
        cluster_summaries.append(
            {
                "run_dir": run_dir,
                "calibration_file": calibration_file,
                "episode_pairs": int(np.sum(valid)),
                "agent_mean": x_mean,
                "baseline_mean": y_mean,
            }
        )
    return xs, ys, cluster_summaries


def _pooled_metric_pairs(pooled_episodes: List[Dict], algorithm: str, metric: str):
    xs = []
    ys = []
    for item in pooled_episodes:
        if "agent" not in item or algorithm not in item:
            continue
        if metric not in item["agent"] or metric not in item[algorithm]:
            continue
        xs.append(float(item["agent"][metric]))
        ys.append(float(item[algorithm][metric]))
    return xs, ys


def _paired_delta_summary(x: List[float], y: List[float]) -> Dict[str, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    deltas = x_arr[valid] - y_arr[valid]
    if deltas.size == 0:
        return {"mean_delta": float("nan"), "std_delta": float("nan"), "n": 0}
    return {
        "mean_delta": float(np.mean(deltas)),
        "std_delta": float(np.std(deltas, ddof=0)),
        "n": int(deltas.size),
    }


def _stouffer_combine(pvalues: List[float]) -> Dict[str, float]:
    finite = [float(p) for p in pvalues if np.isfinite(p) and 0.0 < float(p) <= 1.0]
    if not finite:
        return {"z": float("nan"), "pvalue": float("nan"), "n": 0}
    try:
        from scipy.stats import norm
    except Exception:
        return {"z": float("nan"), "pvalue": float("nan"), "n": len(finite)}
    z_scores = [norm.isf(p / 2.0) for p in finite]
    z = float(np.sum(z_scores) / np.sqrt(len(z_scores)))
    pvalue = float(2.0 * norm.sf(abs(z)))
    return {"z": z, "pvalue": pvalue, "n": len(finite)}


def _parse_iso_datetime(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _calibration_metadata(path: str, max_age_days: int) -> Dict:
    metadata = {
        "path": path,
        "backend_name": None,
        "last_update_date": None,
        "requested_datetime": None,
        "downloaded_at_utc": None,
        "age_days": float("nan"),
        "fresh_within_max_age": False,
    }
    try:
        with open(path, "r") as f:
            raw = json.load(f)
    except Exception as e:
        metadata["error"] = str(e)
        return metadata

    metadata.update(
        {
            "backend_name": raw.get("backend_name"),
            "last_update_date": raw.get("last_update_date"),
            "requested_datetime": raw.get("requested_datetime"),
            "downloaded_at_utc": raw.get("downloaded_at_utc"),
        }
    )
    effective_dt = (
        _parse_iso_datetime(raw.get("requested_datetime"))
        or _parse_iso_datetime(raw.get("last_update_date"))
        or _parse_iso_datetime(raw.get("downloaded_at_utc"))
    )
    if effective_dt is not None:
        if effective_dt.tzinfo is None:
            effective_dt = effective_dt.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - effective_dt.astimezone(timezone.utc)).total_seconds() / 86400.0
        metadata["age_days"] = float(age_days)
        metadata["fresh_within_max_age"] = bool(age_days <= float(max_age_days))
    return metadata


def _build_agent_for_run(config: TrainingConfig, checkpoint_dir: str):
    env = QuantumRoutingEnv(
        num_qubits=config.train_num_qubits,
        calibration_file=config.calibration_file,
        max_steps_per_episode=config.train_max_steps,
        max_steps_base=config.train_max_steps_base,
        max_steps_per_2q=config.train_max_steps_per_2q,
        max_steps_cap=config.train_max_steps_cap,
        routing_method="sabre",
        optimization_level=config.optimization_level,
        cost_lambda=config.cost_lambda,
        cost_w_twoq=config.cost_w_twoq,
        cost_w_depth=config.cost_w_depth,
        debug=False,
        use_proxy_reward=config.use_proxy_reward_phase1,
        reward_mode=config.reward_mode,
        fidelity_scale=config.fidelity_scale,
        invalid_action_penalty=config.invalid_action_penalty,
        swap_penalty=config.swap_penalty,
        distance_reduction_reward_scale=config.distance_reduction_reward_scale,
        progress_reward_scale=config.progress_reward_scale,
        executed_gate_reward_scale=config.executed_gate_reward_scale,
        timeout_penalty=config.timeout_penalty,
        incomplete_episode_penalty=config.incomplete_episode_penalty,
        random_2q_prob=config.random_2q_prob,
        qaoa_prob=config.qaoa_prob,
        quantum_volume_prob=config.quantum_volume_prob,
        vqe_prob=config.vqe_prob,
        clifford_prob=config.clifford_prob,
        positive_control_prob=config.positive_control_prob,
        zero_noise_features=config.zero_noise_features,
        calibration_feature_mask=config.calibration_feature_mask,
        benchmark_qasm_files=config.benchmark_qasm_files,
        benchmark_qasm_dir=config.benchmark_qasm_dir,
        benchmark_corpus_prob=config.benchmark_corpus_prob,
        benchmark_corpus_name=config.benchmark_corpus_name,
    )
    state = env.reset(episode=0)

    agent = PPOAgent(
        state_size=state.shape[0],
        action_size=env.get_action_size(),
        action_set_name=config.action_set_phase1,
        num_qubits=env.num_qubits,
        coupling_edges=env._physical_edges,
        **config.to_agent_kwargs(),
    )
    agent.set_action_set("routing_only")

    checkpoint_path = find_latest_checkpoint(os.path.join(checkpoint_dir, "ppo_checkpoint_*.pt"))
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    agent.load_checkpoint(checkpoint_path, env=env)
    return agent, checkpoint_path


def _benchmark_corpus_diagnostics(config: TrainingConfig) -> Dict:
    env = QuantumRoutingEnv(
        num_qubits=config.train_num_qubits,
        calibration_file=config.calibration_file,
        max_steps_per_episode=config.train_max_steps,
        max_steps_base=config.train_max_steps_base,
        max_steps_per_2q=config.train_max_steps_per_2q,
        max_steps_cap=config.train_max_steps_cap,
        routing_method="sabre",
        optimization_level=config.optimization_level,
        cost_lambda=config.cost_lambda,
        cost_w_twoq=config.cost_w_twoq,
        cost_w_depth=config.cost_w_depth,
        debug=False,
        use_proxy_reward=True,
        reward_mode=config.reward_mode,
        fidelity_scale=config.fidelity_scale,
        invalid_action_penalty=config.invalid_action_penalty,
        swap_penalty=config.swap_penalty,
        distance_reduction_reward_scale=config.distance_reduction_reward_scale,
        progress_reward_scale=config.progress_reward_scale,
        executed_gate_reward_scale=config.executed_gate_reward_scale,
        timeout_penalty=config.timeout_penalty,
        incomplete_episode_penalty=config.incomplete_episode_penalty,
        random_2q_prob=config.random_2q_prob,
        qaoa_prob=config.qaoa_prob,
        quantum_volume_prob=config.quantum_volume_prob,
        vqe_prob=config.vqe_prob,
        clifford_prob=config.clifford_prob,
        positive_control_prob=config.positive_control_prob,
        zero_noise_features=config.zero_noise_features,
        calibration_feature_mask=config.calibration_feature_mask,
        benchmark_qasm_files=config.benchmark_qasm_files,
        benchmark_qasm_dir=config.benchmark_qasm_dir,
        benchmark_corpus_prob=config.benchmark_corpus_prob,
        benchmark_corpus_name=config.benchmark_corpus_name,
    )
    records = list(getattr(env, "_benchmark_qasm_records", []) or [])
    return {
        "compatible_qasm_record_count": int(len(records)),
        "sample_records": records[:10],
    }


def _plot_summary(per_run: List[Dict], output_dir: str, sabre_label: str):
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    labels = [
        f"{os.path.basename(item['run_dir'].rstrip(os.sep)) or item['run_dir']} | "
        f"{os.path.splitext(os.path.basename(item['calibration_file']))[0]}"
        for item in per_run
    ]
    series = [
        ("Agent", "agent_fidelity"),
        (sabre_label, "sabre_fidelity"),
        ("SABRE trivial-layout", "sabre_trivial_layout_fidelity"),
        ("Qiskit VF2+SABRE target-aware", "qiskit_noise_aware_vf2_fidelity"),
        ("Greedy", "greedy_fidelity"),
        ("Lookahead", "lookahead_fidelity"),
        ("Random", "random_fidelity"),
    ]
    plotted = []
    for label, metric_name in series:
        values = np.asarray(
            [item["summary"].get(metric_name, {}).get("mean", float("nan")) for item in per_run],
            dtype=float,
        )
        if np.isfinite(values).any():
            plotted.append((label, values))

    x = np.arange(len(labels))
    width = min(0.16, 0.8 / max(1, len(plotted)))

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(labels)), 5.5))
    center = (len(plotted) - 1) / 2.0
    for idx, (label, values) in enumerate(plotted):
        ax.bar(x + (idx - center) * width, values, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Held-Out Exact Fidelity")
    ax.set_title("Reviewer Benchmark Across Training Runs")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    plot_path = os.path.join(output_dir, "reviewer_benchmark.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def _plot_pareto(pooled_episodes: List[Dict], output_dir: str, sabre_label: str):
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    valid_agent_points = [
        item
        for item in pooled_episodes
        if bool(item.get("agent", {}).get("completed", item.get("agent_completed", False)))
        and np.isfinite(float(item.get("agent", {}).get("twoq", float("nan"))))
        and np.isfinite(float(item.get("agent", {}).get("fidelity", float("nan"))))
    ]
    incomplete_agent_count = int(len(pooled_episodes) - len(valid_agent_points))

    agent_twoq = np.asarray([float(item["agent"]["twoq"]) for item in valid_agent_points], dtype=float)
    agent_fid = np.asarray([float(item["agent"]["fidelity"]) for item in valid_agent_points], dtype=float)
    sabre_points = [item for item in pooled_episodes if "sabre" in item]
    sabre_twoq = np.asarray([float(item["sabre"]["twoq"]) for item in sabre_points], dtype=float)
    sabre_fid = np.asarray([float(item["sabre"]["fidelity"]) for item in sabre_points], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(sabre_twoq, sabre_fid, alpha=0.45, label=sabre_label, color="#4C78A8")
    if agent_twoq.size:
        ax.scatter(agent_twoq, agent_fid, alpha=0.45, label="Agent completed", color="#E45756")
    ax.set_xlabel("Two-Qubit Gate Count")
    ax.set_ylabel("Exact Fidelity")
    ax.set_title(f"Fidelity vs Two-Qubit Cost (excluded incomplete agent: {incomplete_agent_count})")
    ax.grid(True, alpha=0.25)
    ax.legend()

    plot_path = os.path.join(output_dir, "reviewer_pareto_fidelity_twoq.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def _summarize_by_circuit_class(pooled_episodes: List[Dict]):
    out = {}
    algorithms = sorted(
        {
            key
            for episode in pooled_episodes
            for key, value in episode.items()
            if isinstance(value, dict) and "fidelity" in value
        }
    )
    metric_names = ["fidelity", "twoq", "depth"]
    for episode in pooled_episodes:
        cls = str(episode.get("target_circuit_type", "unknown"))
        bucket = out.setdefault(
            cls,
            {"count": 0},
        )
        bucket["count"] += 1
        for algorithm in algorithms:
            if algorithm not in episode:
                continue
            for metric in metric_names:
                key = f"{algorithm}_{metric}"
                bucket.setdefault(key, []).append(float(episode[algorithm][metric]))

    summary = {}
    for cls, bucket in out.items():
        item = {"count": int(bucket["count"])}
        for key, values in bucket.items():
            if key == "count":
                continue
            item[f"{key}_mean"] = float(np.mean(values)) if values else float("nan")
        summary[cls] = item
    return summary


def _agent_completion_value(episode: Dict) -> float:
    if "agent_completed" in episode:
        return 1.0 if bool(episode.get("agent_completed")) else 0.0
    agent = episode.get("agent", {})
    if isinstance(agent, dict) and "completed" in agent:
        return 1.0 if bool(agent.get("completed")) else 0.0
    return float("nan")


def _agent_timeout_value(episode: Dict) -> float:
    agent = episode.get("agent", {})
    if isinstance(agent, dict) and "timed_out" in agent:
        return 1.0 if bool(agent.get("timed_out")) else 0.0
    reason = str(episode.get("agent_terminal_reason", agent.get("terminal_reason", "")))
    if reason:
        return 1.0 if reason == "timeout" else 0.0
    return float("nan")


def _agent_progress_value(episode: Dict) -> float:
    if "agent_progress" in episode:
        return float(episode.get("agent_progress", float("nan")))
    agent = episode.get("agent", {})
    if isinstance(agent, dict):
        return float(agent.get("progress", float("nan")))
    return float("nan")


def _completion_summary(pooled_episodes: List[Dict]) -> Dict:
    completed = np.asarray([_agent_completion_value(ep) for ep in pooled_episodes], dtype=float)
    timed_out = np.asarray([_agent_timeout_value(ep) for ep in pooled_episodes], dtype=float)
    progress = np.asarray([_agent_progress_value(ep) for ep in pooled_episodes], dtype=float)

    def _nanmean(values):
        finite = values[np.isfinite(values)]
        return float(np.mean(finite)) if finite.size else float("nan")

    by_class = {}
    for episode in pooled_episodes:
        cls = str(episode.get("target_circuit_type", "unknown"))
        bucket = by_class.setdefault(cls, {"completed": [], "timed_out": [], "progress": []})
        bucket["completed"].append(_agent_completion_value(episode))
        bucket["timed_out"].append(_agent_timeout_value(episode))
        bucket["progress"].append(_agent_progress_value(episode))

    per_class = {}
    for cls, bucket in by_class.items():
        c = np.asarray(bucket["completed"], dtype=float)
        t = np.asarray(bucket["timed_out"], dtype=float)
        p = np.asarray(bucket["progress"], dtype=float)
        per_class[cls] = {
            "count": int(len(bucket["completed"])),
            "agent_completion_rate": _nanmean(c),
            "agent_timeout_rate": _nanmean(t),
            "agent_mean_progress": _nanmean(p),
        }

    return {
        "agent_completion_rate": _nanmean(completed),
        "agent_timeout_rate": _nanmean(timed_out),
        "agent_mean_progress": _nanmean(progress),
        "per_circuit_class": per_class,
    }


def _enforce_completion_threshold(completion: Dict, min_rate: float):
    rate = float(completion.get("agent_completion_rate", float("nan")))
    if not np.isfinite(rate) or rate < float(min_rate):
        raise RuntimeError(
            f"Reviewer benchmark failed completion gate: agent_completion_rate={rate:.4f} "
            f"< REVIEW_MIN_AGENT_COMPLETION_RATE={float(min_rate):.4f}. "
            "Discard old shortcut checkpoints or set REVIEW_REQUIRE_AGENT_COMPLETION=0 "
            "only for exploratory debugging."
        )


def _merge_external_baselines(pooled_episodes: List[Dict], paths: List[str]):
    merged = 0
    loaded = []
    for path in paths:
        try:
            with open(path, "r") as f:
                payload = json.load(f)
        except Exception as e:
            loaded.append({"path": path, "error": str(e), "records": 0})
            continue
        records = payload.get("episodes", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            loaded.append({"path": path, "error": "expected a list or {'episodes': [...]} object", "records": 0})
            continue
        count = 0
        for record in records:
            if not isinstance(record, dict) or "seed" not in record:
                continue
            algorithm = str(record.get("algorithm", "dqn_routing_baseline")).strip() or "dqn_routing_baseline"
            metrics = record.get("metrics", record)
            if not isinstance(metrics, dict) or "fidelity" not in metrics:
                continue
            for episode in pooled_episodes:
                if int(episode.get("seed", -1)) != int(record["seed"]):
                    continue
                if record.get("run_dir") and os.path.abspath(record["run_dir"]) != os.path.abspath(episode.get("run_dir", "")):
                    continue
                if record.get("calibration_file") and os.path.abspath(record["calibration_file"]) != os.path.abspath(episode.get("calibration_file", "")):
                    continue
                episode[algorithm] = {
                    "fidelity": float(metrics["fidelity"]),
                    "cost": float(metrics.get("cost", float("nan"))),
                    "twoq": float(metrics.get("twoq", float("nan"))),
                    "depth": float(metrics.get("depth", float("nan"))),
                    "wall_seconds": float(metrics.get("wall_seconds", float("nan"))),
                }
                merged += 1
                count += 1
        loaded.append({"path": path, "records": int(len(records)), "merged_episode_metrics": int(count)})
    return {"files": loaded, "merged_episode_metrics": int(merged)}


if __name__ == "__main__":
    run_dirs = [
        os.path.abspath(part.strip())
        for part in os.getenv("REVIEW_RUN_DIRS", ".").split(",")
        if part.strip()
    ]
    holdout_episodes = _env_int("REVIEW_HOLDOUT_EPISODES", 300)
    holdout_start_seed = _env_int("REVIEW_HOLDOUT_START_SEED", 70000)
    max_calibration_age_days = _env_int("REVIEW_MAX_CALIBRATION_AGE_DAYS", 45)
    require_fresh_calibrations = str(os.getenv("REVIEW_REQUIRE_FRESH_CALIBRATIONS", "1")).strip().lower() in {"1", "true", "yes", "y", "on"}
    output_dir = os.path.abspath(os.getenv("REVIEW_OUTPUT_DIR", "reviewer_benchmark"))
    output_path = os.path.join(output_dir, "reviewer_benchmark_summary.json")
    os.makedirs(output_dir, exist_ok=True)

    config = TrainingConfig(load_hyperparams=True)
    require_benchmark_corpus = _env_flag("REVIEW_REQUIRE_BENCHMARK_CORPUS", False)
    if require_benchmark_corpus and not (config.benchmark_qasm_files or config.benchmark_qasm_dir):
        raise RuntimeError(
            "Reviewer benchmark corpus guard is enabled, but BENCHMARK_QASM_DIR and "
            "BENCHMARK_QASM_FILES are empty. Set BENCHMARK_QASM_DIR for paper runs, "
            "or set REVIEW_REQUIRE_BENCHMARK_CORPUS=0 for synthetic smoke/debug runs."
        )
    corpus_diagnostics = _benchmark_corpus_diagnostics(config) if (config.benchmark_qasm_files or config.benchmark_qasm_dir) else {
        "compatible_qasm_record_count": 0,
        "sample_records": [],
    }
    requested_calibration_files = [
        os.path.abspath(part.strip())
        for part in os.getenv("REVIEW_CALIBRATION_FILES", config.calibration_file).split(",")
        if part.strip()
    ]
    calibration_files = [path for path in requested_calibration_files if os.path.exists(path)]
    missing_calibration_files = [path for path in requested_calibration_files if not os.path.exists(path)]
    if missing_calibration_files:
        print("Skipping missing calibration files:")
        for path in missing_calibration_files:
            print(f"  {path}")
    if not calibration_files:
        raise FileNotFoundError("No calibration files were found for reviewer benchmarking.")
    calibration_snapshots = [_calibration_metadata(path, max_calibration_age_days) for path in calibration_files]
    stale_calibrations = [
        item for item in calibration_snapshots
        if not bool(item.get("fresh_within_max_age", False))
    ]
    if stale_calibrations and require_fresh_calibrations:
        details = "\n".join(
            f"  {item['path']} last_update={item.get('last_update_date')} "
            f"requested={item.get('requested_datetime')} downloaded={item.get('downloaded_at_utc')}"
            for item in stale_calibrations
        )
        raise RuntimeError(
            "Reviewer benchmark requires fresh downloaded calibration snapshots. "
            f"Set REVIEW_REQUIRE_FRESH_CALIBRATIONS=0 only for exploratory local runs.\n{details}"
        )

    per_run = []
    pooled_episodes = []
    total_eval_cells = max(1, len(calibration_files) * len(run_dirs))
    eval_cell_idx = 0
    for calibration_file in calibration_files:
        config_for_device = copy.deepcopy(config)
        config_for_device.calibration_file = calibration_file
        for run_dir in run_dirs:
            eval_cell_idx += 1
            print(
                f"[review] cell {eval_cell_idx}/{total_eval_cells}: "
                f"calibration={os.path.basename(calibration_file)} run_dir={run_dir} "
                f"episodes={holdout_episodes}",
                flush=True,
            )
            agent, checkpoint_path = _build_agent_for_run(config_for_device, run_dir)
            holdout = evaluate_agent_and_baselines_holdout(
                agent,
                config_for_device,
                num_episodes=holdout_episodes,
                start_seed=holdout_start_seed,
                reward_mode=config_for_device.reward_mode,
                return_episode_records=True,
            )
            pooled_episodes.extend(
                [
                    {
                        **episode,
                        "run_dir": run_dir,
                        "calibration_file": calibration_file,
                    }
                    for episode in holdout["episodes"]
                ]
            )
            per_run.append(
                {
                    "run_dir": run_dir,
                    "calibration_file": calibration_file,
                    "checkpoint_path": checkpoint_path,
                    "summary": holdout["summary"],
                    "num_episode_pairs": len(holdout["episodes"]),
                }
            )
            print(
                f"[review] cell {eval_cell_idx}/{total_eval_cells} complete: "
                f"episode_pairs={len(holdout['episodes'])}",
                flush=True,
            )

    external_baseline_paths = [
        os.path.abspath(part.strip())
        for part in os.getenv("REVIEW_EXTERNAL_BASELINE_FILES", "").split(",")
        if part.strip()
    ]
    external_baselines = _merge_external_baselines(pooled_episodes, external_baseline_paths)

    aggregate = {}
    metric_names = list(per_run[0]["summary"].keys()) if per_run else []
    for metric_name in metric_names:
        values = np.asarray([item["summary"][metric_name]["mean"] for item in per_run], dtype=float)
        aggregate[metric_name] = {
            "mean_across_runs": float(np.nanmean(values)),
            "std_across_runs": float(np.nanstd(values, ddof=0)),
            "num_runs": int(values.size),
        }

    baseline_algorithms = sorted(
        {
            key
            for episode in pooled_episodes
            for key, value in episode.items()
            if key != "agent" and isinstance(value, dict) and "fidelity" in value
        }
    )
    comparison_metrics = {
        "fidelity": True,
        "twoq": False,
        "depth": False,
        "wall_seconds": False,
    }
    paired_tests = {}
    pooled_episode_paired_tests = {}
    effect_sizes = {}
    cluster_effect_sizes = {}
    cluster_pair_counts = {}
    for algorithm in baseline_algorithms:
        for metric, higher_is_better in comparison_metrics.items():
            xs, ys, clusters = _clustered_metric_pairs(pooled_episodes, algorithm, metric)
            if len(xs) == 0:
                continue
            key = f"agent_vs_{algorithm}_{metric}"
            paired_tests[key] = _safe_wilcoxon(xs, ys)
            paired_tests[key]["statistical_unit"] = "run_dir_x_calibration_file_cluster"
            paired_tests[key]["num_clusters"] = int(len(xs))
            cluster_pair_counts[key] = clusters
            effect_sizes[key] = _paired_improvement_summary(
                xs,
                ys,
                higher_is_better=higher_is_better,
                seed=holdout_start_seed + len(effect_sizes),
            )
            cluster_effect_sizes[key] = effect_sizes[key]
            pooled_xs, pooled_ys = _pooled_metric_pairs(pooled_episodes, algorithm, metric)
            pooled_episode_paired_tests[key] = _safe_wilcoxon(pooled_xs, pooled_ys)
            pooled_episode_paired_tests[key]["statistical_unit"] = "episode_pair_diagnostic_only"
            pooled_episode_paired_tests[key]["n_episode_pairs"] = int(len(pooled_xs))
    adjusted_pvalues = _holm_bonferroni({key: value["pvalue"] for key, value in paired_tests.items()})
    for key, adjusted in adjusted_pvalues.items():
        paired_tests[key]["pvalue_holm"] = float(adjusted)
    pooled_adjusted_pvalues = _holm_bonferroni({key: value["pvalue"] for key, value in pooled_episode_paired_tests.items()})
    for key, adjusted in pooled_adjusted_pvalues.items():
        pooled_episode_paired_tests[key]["pvalue_holm"] = float(adjusted)

    per_run_paired_tests = []
    per_run_pvalues = {"agent_vs_sabre_fidelity": [], "agent_vs_sabre_twoq": [], "agent_vs_sabre_depth": []}
    for item in per_run:
        run_episodes = [
            ep for ep in pooled_episodes
            if ep["run_dir"] == item["run_dir"] and ep["calibration_file"] == item["calibration_file"]
        ]
        def _run_pair(algo, metric):
            xs = []
            ys = []
            for ep in run_episodes:
                if "agent" not in ep or algo not in ep:
                    continue
                xs.append(float(ep["agent"][metric]))
                ys.append(float(ep[algo][metric]))
            return xs, ys
        run_tests = {}
        for metric in ("fidelity", "twoq", "depth"):
            xs, ys = _run_pair("sabre", metric)
            run_tests[f"agent_vs_sabre_{metric}"] = _safe_wilcoxon(xs, ys)
        for key in per_run_pvalues:
            p = run_tests[key]["pvalue"]
            if np.isfinite(p):
                per_run_pvalues[key].append(float(p))
        per_run_paired_tests.append(
            {
                "run_dir": item["run_dir"],
                "calibration_file": item["calibration_file"],
                "tests": run_tests,
            }
        )

    circuit_class_counts = {}
    for episode in pooled_episodes:
        cls = str(episode.get("target_circuit_type", "unknown"))
        circuit_class_counts[cls] = circuit_class_counts.get(cls, 0) + 1
    completion = _completion_summary(pooled_episodes)

    baseline_episode_counts = {
        algorithm: int(
            sum(
                1
                for episode in pooled_episodes
                if algorithm in episode
                and isinstance(episode.get(algorithm), dict)
                and np.isfinite(float(episode[algorithm].get("fidelity", float("nan"))))
            )
        )
        for algorithm in baseline_algorithms
    }
    required_baselines = [
        item.strip()
        for item in os.getenv(
            "REVIEW_REQUIRED_BASELINES",
            "sabre,sabre_trivial_layout,qiskit_noise_aware_vf2,greedy,lookahead,random",
        ).split(",")
        if item.strip()
    ]
    missing_required_baselines = [
        algorithm
        for algorithm in required_baselines
        if baseline_episode_counts.get(algorithm, 0) < len(pooled_episodes)
    ]
    if missing_required_baselines and str(os.getenv("REVIEW_ALLOW_MISSING_BASELINES", "0")).lower() not in {"1", "true", "yes"}:
        details = ", ".join(
            f"{algorithm}={baseline_episode_counts.get(algorithm, 0)}/{len(pooled_episodes)}"
            for algorithm in missing_required_baselines
        )
        raise RuntimeError(
            "Required reviewer baselines did not run for every paired episode. "
            "Set REVIEW_ALLOW_MISSING_BASELINES=1 only for exploratory local runs. "
            f"Missing/partial: {details}"
        )

    summary = {
        "protocol": {
            "holdout_episodes": holdout_episodes,
            "holdout_start_seed": holdout_start_seed,
            "optimization_level": config.optimization_level,
            "sabre_baseline_trials": config.sabre_baseline_trials,
            "policy_backbone": config.policy_backbone,
            "reward_mode": config.reward_mode,
            "zero_noise_features": config.zero_noise_features,
            "calibration_feature_mask": config.calibration_feature_mask,
            "train_num_qubits": config.train_num_qubits,
            "train_max_steps": config.train_max_steps,
            "train_max_steps_base": config.train_max_steps_base,
            "train_max_steps_per_2q": config.train_max_steps_per_2q,
            "train_max_steps_cap": config.train_max_steps_cap,
            "incomplete_episode_penalty": config.incomplete_episode_penalty,
            "calibration_files": calibration_files,
            "calibration_snapshots": calibration_snapshots,
            "max_calibration_age_days": max_calibration_age_days,
            "require_fresh_calibrations": require_fresh_calibrations,
            "missing_calibration_files": missing_calibration_files,
            "circuit_mix": {
                "random_2q_prob": config.random_2q_prob,
                "qaoa_prob": config.qaoa_prob,
                "quantum_volume_prob": config.quantum_volume_prob,
                "vqe_prob": config.vqe_prob,
                "clifford_prob": config.clifford_prob,
                "positive_control_prob": config.positive_control_prob,
            },
            "benchmark_corpus": {
                "name": config.benchmark_corpus_name,
                "qasm_files": config.benchmark_qasm_files,
                "qasm_dir": config.benchmark_qasm_dir,
                "probability": config.benchmark_corpus_prob,
                **corpus_diagnostics,
            },
            "external_baseline_files": external_baseline_paths,
            "required_baselines": required_baselines,
            "baseline_episode_counts": baseline_episode_counts,
            "baseline_missing_allowed": str(os.getenv("REVIEW_ALLOW_MISSING_BASELINES", "0")).lower() in {"1", "true", "yes"},
            "require_benchmark_corpus": require_benchmark_corpus,
            "require_agent_completion": _env_flag("REVIEW_REQUIRE_AGENT_COMPLETION", True),
            "min_agent_completion_rate": _env_float("REVIEW_MIN_AGENT_COMPLETION_RATE", 0.98),
            "secondary_router_comparison_statistical_unit": "run_dir x calibration_file cluster",
            "holm_family_size": int(len(paired_tests)),
        },
        "artifact_metadata": {
            "git_revision": _git_revision(os.path.dirname(os.path.abspath(__file__))),
            "python_version": os.sys.version,
            "package_versions": {
                "numpy": _package_version("numpy"),
                "scipy": _package_version("scipy"),
                "qiskit": _package_version("qiskit"),
                "qiskit-aer": _package_version("qiskit-aer"),
                "torch": _package_version("torch"),
            },
        },
        "per_run": per_run,
        "external_baselines": external_baselines,
        "pooled_episode_pairs": int(len(pooled_episodes)),
        "circuit_class_counts": circuit_class_counts,
        "per_circuit_class": _summarize_by_circuit_class(pooled_episodes),
        "completion": completion,
        "agent_completion_rate": completion["agent_completion_rate"],
        "agent_timeout_rate": completion["agent_timeout_rate"],
        "agent_mean_progress": completion["agent_mean_progress"],
        "aggregate": aggregate,
        "paired_tests": paired_tests,
        "pooled_episode_paired_tests_diagnostic_only": pooled_episode_paired_tests,
        "effect_sizes": effect_sizes,
        "cluster_pair_summaries": cluster_pair_counts,
        "per_run_paired_tests": per_run_paired_tests,
        "stouffer_combined_tests": {
            key: _stouffer_combine(values) for key, values in per_run_pvalues.items()
        },
        "paired_deltas": {
            "agent_minus_sabre_fidelity": _paired_delta_summary(*_clustered_metric_pairs(pooled_episodes, "sabre", "fidelity")[:2]),
            "agent_minus_sabre_twoq": _paired_delta_summary(*_clustered_metric_pairs(pooled_episodes, "sabre", "twoq")[:2]),
            "agent_minus_sabre_depth": _paired_delta_summary(*_clustered_metric_pairs(pooled_episodes, "sabre", "depth")[:2]),
        },
    }
    sabre_label = "SABRE-best20" if int(config.sabre_baseline_trials) >= 20 else f"SABRE-best{int(config.sabre_baseline_trials)}"
    summary["artifact_paths"] = {
        "json": output_path,
        "plot": _plot_summary(per_run, output_dir, sabre_label=sabre_label),
        "pareto_plot": _plot_pareto(pooled_episodes, output_dir, sabre_label=sabre_label),
    }
    save_episode_records = str(os.getenv("REVIEW_SAVE_EPISODE_RECORDS", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    if save_episode_records:
        episode_records_path = os.path.join(output_dir, "reviewer_benchmark_episodes.jsonl")
        with open(episode_records_path, "w") as f:
            for episode in pooled_episodes:
                f.write(json.dumps(episode, default=str) + "\n")
        summary["artifact_paths"]["episode_records"] = episode_records_path

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved reviewer benchmark summary:")
    print(f"  {output_path}")
    print(f"  {summary['artifact_paths']['plot']}")
    print(f"  {summary['artifact_paths']['pareto_plot']}")

    if summary["protocol"]["require_agent_completion"]:
        _enforce_completion_threshold(
            completion,
            min_rate=float(summary["protocol"]["min_agent_completion_rate"]),
        )
