"""Optuna-based hyperparameter search for the PPO routing policy."""

import os
import json
import copy
import optuna
import numpy as np
import torch
from collections import deque

# Restrict thread counts to avoid CPU oversubscription across parallel trials.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['RAYON_NUM_THREADS'] = '1'

# Shared training components.
from scalable_quantum import QuantumRoutingEnv, PPOAgent, Transition, TrainingConfig, _set_global_seeds


BASE_CONFIG = TrainingConfig(load_hyperparams=False)


def _env_int(name, default):
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_flag(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _build_optuna_storage():
    journal_path = os.getenv("OPTUNA_JOURNAL_PATH", "").strip()
    if journal_path:
        os.makedirs(os.path.dirname(os.path.abspath(journal_path)), exist_ok=True)
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend

        return JournalStorage(JournalFileBackend(journal_path))

    storage_url = os.getenv("OPTUNA_STORAGE", "").strip()
    if storage_url:
        return storage_url

    return None


def _completed_trials(study):
    return [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]


def sample_agent_hyperparameters(trial):
    return {
        "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
        "gnn_layers": trial.suggest_int("gnn_layers", 1, 4),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "k_epochs": trial.suggest_int("k_epochs", 3, 6),
        "eps_clip": trial.suggest_float("eps_clip", 0.10, 0.30),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.90, 0.99),
        "entropy_coefficient": trial.suggest_float("entropy_coefficient", 0.1, 0.8),
        "min_entropy_coeff": trial.suggest_float("min_entropy_coeff", 0.001, 0.05, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "entropy_decay": BASE_CONFIG.entropy_decay,
        "aux_fidelity_coef": BASE_CONFIG.aux_fidelity_coef,
        "aux_coherence_coef": BASE_CONFIG.aux_coherence_coef,
        "value_coef": BASE_CONFIG.value_coef,
    }


def save_reviewer_artifacts(study, output_dir):
    os.environ.setdefault("MPLBACKEND", "Agg")
    xdg_cache_dir = os.path.join(output_dir, ".cache")
    os.makedirs(xdg_cache_dir, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", xdg_cache_dir)
    mpl_cache_dir = os.path.join(output_dir, ".matplotlib")
    os.makedirs(mpl_cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_cache_dir)

    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    complete_trials = _completed_trials(study)
    if not complete_trials:
        return []

    complete_trials.sort(key=lambda t: t.number)
    saved_paths = []

    trial_numbers = [int(t.number) for t in complete_trials]
    exact_values = [float(t.value) for t in complete_trials]
    best_so_far = np.maximum.accumulate(exact_values)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(trial_numbers, exact_values, marker="o", linewidth=1.5, label="Exact Eval Fidelity")
    ax.plot(trial_numbers, best_so_far, linestyle="--", linewidth=2.0, label="Best So Far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Exact Fidelity")
    ax.set_title("Hyperparameter Search Progress")
    ax.grid(True, alpha=0.25)
    ax.legend()
    history_path = os.path.join(output_dir, "optimization_history.png")
    fig.tight_layout()
    fig.savefig(history_path, dpi=180)
    plt.close(fig)
    saved_paths.append(history_path)

    proxy_values = [t.user_attrs.get("proxy_train_fidelity", np.nan) for t in complete_trials]
    valid_proxy = [
        (float(proxy), float(exact), int(t.number))
        for t, proxy, exact in zip(complete_trials, proxy_values, exact_values)
        if np.isfinite(proxy) and np.isfinite(exact)
    ]
    if valid_proxy:
        px = np.asarray([row[0] for row in valid_proxy], dtype=float)
        py = np.asarray([row[1] for row in valid_proxy], dtype=float)
        lo = float(min(px.min(), py.min()))
        hi = float(max(px.max(), py.max()))

        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.scatter(px, py, alpha=0.8)
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", alpha=0.7, label="proxy = exact")
        best_trial = max(complete_trials, key=lambda t: float(t.value))
        best_proxy = float(best_trial.user_attrs.get("proxy_train_fidelity", np.nan))
        if np.isfinite(best_proxy):
            ax.scatter(
                [best_proxy],
                [float(best_trial.value)],
                color="crimson",
                s=90,
                label=f"Best Trial ({best_trial.number})",
                zorder=3,
            )
        ax.set_xlabel("Proxy Train Fidelity")
        ax.set_ylabel("Exact Eval Fidelity")
        ax.set_title("Proxy Metric vs Exact Holdout Metric")
        ax.grid(True, alpha=0.25)
        ax.legend()
        proxy_path = os.path.join(output_dir, "proxy_vs_exact.png")
        fig.tight_layout()
        fig.savefig(proxy_path, dpi=180)
        plt.close(fig)
        saved_paths.append(proxy_path)

    param_order = [
        "lr",
        "hidden_size",
        "gnn_layers",
        "gamma",
        "k_epochs",
        "eps_clip",
        "gae_lambda",
        "entropy_coefficient",
        "min_entropy_coeff",
        "batch_size",
    ]
    present_params = [name for name in param_order if any(name in t.params for t in complete_trials)]
    if present_params:
        ncols = 2
        nrows = int(np.ceil(len(present_params) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4 * nrows))
        axes = np.atleast_1d(axes).reshape(nrows, ncols)
        best_trial = max(complete_trials, key=lambda t: float(t.value))

        for idx, name in enumerate(present_params):
            ax = axes[idx // ncols, idx % ncols]
            xs = np.asarray([float(t.params[name]) for t in complete_trials if name in t.params], dtype=float)
            ys = np.asarray([float(t.value) for t in complete_trials if name in t.params], dtype=float)
            ax.scatter(xs, ys, alpha=0.75)
            if name == "lr" or name == "min_entropy_coeff":
                ax.set_xscale("log")
            if name in best_trial.params:
                ax.scatter(
                    [float(best_trial.params[name])],
                    [float(best_trial.value)],
                    color="crimson",
                    s=80,
                    zorder=3,
                )
            ax.set_title(f"{name} vs Exact Fidelity")
            ax.set_xlabel(name)
            ax.set_ylabel("Exact Fidelity")
            ax.grid(True, alpha=0.2)

        for idx in range(len(present_params), nrows * ncols):
            axes[idx // ncols, idx % ncols].axis("off")

        sensitivity_path = os.path.join(output_dir, "hyperparameter_sensitivity.png")
        fig.tight_layout()
        fig.savefig(sensitivity_path, dpi=180)
        plt.close(fig)
        saved_paths.append(sensitivity_path)

    ranked_trials = sorted(complete_trials, key=lambda t: float(t.value), reverse=True)
    reviewer_summary = {
        "evaluation_protocol": {
            "tune_num_qubits": _env_int("TUNE_NUM_QUBITS", BASE_CONFIG.train_num_qubits),
            "tune_max_steps": _env_int("TUNE_MAX_STEPS", BASE_CONFIG.train_max_steps),
            "exact_eval_episodes": _env_int("TUNE_EXACT_EVAL_EPISODES", 30),
            "exact_eval_start_seed": _env_int("TUNE_EXACT_EVAL_START_SEED", 20000),
            "optimization_level": BASE_CONFIG.optimization_level,
            "sabre_baseline_trials": BASE_CONFIG.sabre_baseline_trials,
            "policy_backbone": BASE_CONFIG.policy_backbone,
            "reward_mode": BASE_CONFIG.reward_mode,
            "circuit_mix": {
                "random_2q_prob": BASE_CONFIG.random_2q_prob,
                "qaoa_prob": BASE_CONFIG.qaoa_prob,
                "quantum_volume_prob": BASE_CONFIG.quantum_volume_prob,
                "vqe_prob": BASE_CONFIG.vqe_prob,
                "clifford_prob": BASE_CONFIG.clifford_prob,
            },
        },
        "best_trial": {
            "number": int(study.best_trial.number),
            "exact_eval_fidelity": float(study.best_trial.value),
            "proxy_train_fidelity": study.best_trial.user_attrs.get("proxy_train_fidelity"),
            "exact_eval_fidelity_delta": study.best_trial.user_attrs.get("exact_eval_fidelity_delta"),
            "params": dict(study.best_trial.params),
        },
        "top_trials": [
            {
                "number": int(t.number),
                "exact_eval_fidelity": float(t.value),
                "proxy_train_fidelity": t.user_attrs.get("proxy_train_fidelity"),
                "exact_eval_fidelity_delta": t.user_attrs.get("exact_eval_fidelity_delta"),
                "params": dict(t.params),
            }
            for t in ranked_trials[: min(10, len(ranked_trials))]
        ],
        "artifact_paths": saved_paths,
    }
    summary_path = os.path.join(output_dir, "reviewer_summary.json")
    with open(summary_path, "w") as f:
        json.dump(reviewer_summary, f, indent=2)
    saved_paths.append(summary_path)

    return saved_paths


def evaluate_agent_exact(agent, num_episodes, episode_seed_start=20000):
    if num_episodes <= 0:
        return {
            "mean_fidelity": float("nan"),
            "mean_fidelity_delta": float("nan"),
            "episodes": 0,
        }

    env = QuantumRoutingEnv(
        num_qubits=_env_int("TUNE_NUM_QUBITS", BASE_CONFIG.train_num_qubits),
        calibration_file=BASE_CONFIG.calibration_file,
        max_steps_per_episode=_env_int("TUNE_MAX_STEPS", BASE_CONFIG.train_max_steps),
        routing_method="sabre",
        optimization_level=BASE_CONFIG.optimization_level,
        cost_lambda=BASE_CONFIG.cost_lambda,
        cost_w_twoq=BASE_CONFIG.cost_w_twoq,
        cost_w_depth=BASE_CONFIG.cost_w_depth,
        debug=False,
        use_proxy_reward=False,
        reward_mode=BASE_CONFIG.reward_mode,
        fidelity_scale=BASE_CONFIG.fidelity_scale,
        invalid_action_penalty=BASE_CONFIG.invalid_action_penalty,
        swap_penalty=BASE_CONFIG.swap_penalty,
        distance_reduction_reward_scale=BASE_CONFIG.distance_reduction_reward_scale,
        progress_reward_scale=BASE_CONFIG.progress_reward_scale,
        executed_gate_reward_scale=BASE_CONFIG.executed_gate_reward_scale,
        timeout_penalty=BASE_CONFIG.timeout_penalty,
        random_2q_prob=BASE_CONFIG.random_2q_prob,
        qaoa_prob=BASE_CONFIG.qaoa_prob,
        quantum_volume_prob=BASE_CONFIG.quantum_volume_prob,
        vqe_prob=BASE_CONFIG.vqe_prob,
        clifford_prob=BASE_CONFIG.clifford_prob,
        positive_control_prob=BASE_CONFIG.positive_control_prob,
        zero_noise_features=BASE_CONFIG.zero_noise_features,
        calibration_feature_mask=BASE_CONFIG.calibration_feature_mask,
        benchmark_qasm_files=BASE_CONFIG.benchmark_qasm_files,
        benchmark_qasm_dir=BASE_CONFIG.benchmark_qasm_dir,
        benchmark_corpus_prob=BASE_CONFIG.benchmark_corpus_prob,
        benchmark_corpus_name=BASE_CONFIG.benchmark_corpus_name,
    )

    fidelities = []
    deltas = []
    for offset in range(int(num_episodes)):
        state = env.reset(episode=episode_seed_start + offset)
        done = False
        step = 0
        info = dict(env.last_metrics)

        while not done and step < env.max_steps_per_episode:
            action_mask = agent.compute_action_mask(env)
            action, _, _, _ = agent.select_greedy_action(state, action_mask=action_mask)
            state, _, done, info = env.step(action)
            step += 1

        fidelity = float(info.get("fidelity", env.last_metrics.get("fidelity", 0.0)))
        fidelities.append(fidelity)
        deltas.append(fidelity - float(env.baseline_fidelity))

    return {
        "mean_fidelity": float(np.mean(fidelities)),
        "mean_fidelity_delta": float(np.mean(deltas)),
        "episodes": int(num_episodes),
    }


def objective(trial):
    _set_global_seeds(_env_int("OPTUNA_TRIAL_BASE_SEED", 1000) + int(trial.number))
    agent_hparams = sample_agent_hyperparameters(trial)

    # Tune at the same scale as deployment by default to avoid train/test mismatch.
    env = QuantumRoutingEnv(
        num_qubits=_env_int("TUNE_NUM_QUBITS", BASE_CONFIG.train_num_qubits),
        max_steps_per_episode=_env_int("TUNE_MAX_STEPS", BASE_CONFIG.train_max_steps),
        calibration_file=BASE_CONFIG.calibration_file,
        routing_method="sabre",
        optimization_level=BASE_CONFIG.optimization_level,
        cost_lambda=BASE_CONFIG.cost_lambda,
        cost_w_twoq=BASE_CONFIG.cost_w_twoq,
        cost_w_depth=BASE_CONFIG.cost_w_depth,
        debug=False,
        use_proxy_reward=True, # Crucial for speed during tuning
        reward_mode=BASE_CONFIG.reward_mode,
        fidelity_scale=BASE_CONFIG.fidelity_scale,
        invalid_action_penalty=BASE_CONFIG.invalid_action_penalty,
        swap_penalty=BASE_CONFIG.swap_penalty,
        distance_reduction_reward_scale=BASE_CONFIG.distance_reduction_reward_scale,
        progress_reward_scale=BASE_CONFIG.progress_reward_scale,
        executed_gate_reward_scale=BASE_CONFIG.executed_gate_reward_scale,
        timeout_penalty=BASE_CONFIG.timeout_penalty,
        random_2q_prob=BASE_CONFIG.random_2q_prob,
        qaoa_prob=BASE_CONFIG.qaoa_prob,
        quantum_volume_prob=BASE_CONFIG.quantum_volume_prob,
        vqe_prob=BASE_CONFIG.vqe_prob,
        clifford_prob=BASE_CONFIG.clifford_prob,
        positive_control_prob=BASE_CONFIG.positive_control_prob,
        zero_noise_features=BASE_CONFIG.zero_noise_features,
        calibration_feature_mask=BASE_CONFIG.calibration_feature_mask,
        benchmark_qasm_files=BASE_CONFIG.benchmark_qasm_files,
        benchmark_qasm_dir=BASE_CONFIG.benchmark_qasm_dir,
        benchmark_corpus_prob=BASE_CONFIG.benchmark_corpus_prob,
        benchmark_corpus_name=BASE_CONFIG.benchmark_corpus_name,
    )

    state = env.reset(episode=0)
    
    # 3. Initialize Agent with Trial Hyperparameters
    agent = PPOAgent(
        state_size=state.shape[0],
        action_size=env.get_action_size(),
        action_set_name=BASE_CONFIG.action_set_phase1,
        num_qubits=env.num_qubits,
        coupling_edges=env._physical_edges,
        **agent_hparams,
    )

    # 4. Shortened Training Loop Setup
    tuning_episodes = _env_int("TUNE_EPISODES", 300)
    train_frequency = max(1, _env_int("TUNE_TRAIN_FREQUENCY", 10))
    progress_interval = max(1, _env_int("TUNE_PROGRESS_INTERVAL", 10))
    print(
        f"[trial {trial.number}] starting tuning: episodes={tuning_episodes}, "
        f"train_frequency={train_frequency}, progress_interval={progress_interval}",
        flush=True,
    )
    
    # RL is noisy. We use a rolling average of fidelity to decide if a run is good.
    recent_fidelities = deque(maxlen=20)
    
    for episode in range(tuning_episodes):
        state = env.reset(episode=episode)
        done = False
        step = 0
        info = dict(env.last_metrics)
        
        while not done and step < env.max_steps_per_episode:
            action_mask = agent.compute_action_mask(env)
            
            action, action_log_prob, value, fid_pred, coh_pred, entropy = agent.select_action(
                state, action_mask=action_mask
            )
            
            next_state, reward, done, info = env.step(action)
            
            # Record transition
            with torch.no_grad():
                ns = torch.tensor(next_state, dtype=torch.float32, device=agent.device).unsqueeze(0)
                next_mask = agent.compute_action_mask(env).unsqueeze(0)
                _, next_value, _, _ = agent.policy(ns, action_mask=next_mask)
                next_value = next_value.squeeze(0).detach()

            transition = Transition(
                state=torch.tensor(state, dtype=torch.float32, device=agent.device),
                action=action,
                reward=float(reward),
                log_prob=action_log_prob,
                value=value,
                next_value=next_value,
                done=float(done),
                true_fidelity=float(info.get("fidelity", 0.0)) if done else float("nan"),
                true_coherence=float("nan"),
                action_mask=action_mask.detach(),
            )
            agent.store_transition(transition)
            
            state = next_state
            step += 1

        # Track fidelity
        current_fidelity = float(info.get("fidelity", 0.0))
        recent_fidelities.append(current_fidelity)
        if episode == 0 or (episode + 1) % progress_interval == 0 or episode + 1 == tuning_episodes:
            avg = float(np.mean(recent_fidelities)) if recent_fidelities else float("nan")
            print(
                f"[trial {trial.number}] episode {episode + 1}/{tuning_episodes} "
                f"steps={step} fidelity={current_fidelity:.4f} rolling={avg:.4f}",
                flush=True,
            )
        
        # Train
        if (episode + 1) % train_frequency == 0:
            agent.train(writer=None, episode=episode) # No tensorboard during tuning to save I/O
            
        # 5. Pruning Logic
        # We only report and check for pruning every 10 episodes to allow the rolling avg to populate
        if (episode + 1) % 10 == 0:
            avg_fidelity = float(np.mean(recent_fidelities))
            trial.report(avg_fidelity, episode)
            
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    # The metric we want Optuna to maximize
    exact_eval_episodes = max(1, _env_int("TUNE_EXACT_EVAL_EPISODES", 30))
    exact_eval = evaluate_agent_exact(
        agent,
        num_episodes=exact_eval_episodes,
        episode_seed_start=_env_int("TUNE_EXACT_EVAL_START_SEED", 20000),
    )

    proxy_metric = float(np.mean(recent_fidelities))
    trial.set_user_attr("proxy_train_fidelity", proxy_metric)
    trial.set_user_attr("exact_eval_fidelity", exact_eval["mean_fidelity"])
    trial.set_user_attr("exact_eval_fidelity_delta", exact_eval["mean_fidelity_delta"])

    print(
        f"[trial {trial.number}] proxy_train_fidelity={proxy_metric:.4f} | "
        f"exact_eval_fidelity={exact_eval['mean_fidelity']:.4f} | "
        f"exact_eval_delta={exact_eval['mean_fidelity_delta']:.4f}",
        flush=True,
    )
    return float(exact_eval["mean_fidelity"])


def _train_trial_agent(agent_hparams, tuning_episodes, seed_base):
    env = QuantumRoutingEnv(
        num_qubits=_env_int("TUNE_NUM_QUBITS", BASE_CONFIG.train_num_qubits),
        max_steps_per_episode=_env_int("TUNE_MAX_STEPS", BASE_CONFIG.train_max_steps),
        calibration_file=BASE_CONFIG.calibration_file,
        routing_method="sabre",
        optimization_level=BASE_CONFIG.optimization_level,
        cost_lambda=BASE_CONFIG.cost_lambda,
        cost_w_twoq=BASE_CONFIG.cost_w_twoq,
        cost_w_depth=BASE_CONFIG.cost_w_depth,
        debug=False,
        use_proxy_reward=True,
        reward_mode=BASE_CONFIG.reward_mode,
        fidelity_scale=BASE_CONFIG.fidelity_scale,
        invalid_action_penalty=BASE_CONFIG.invalid_action_penalty,
        swap_penalty=BASE_CONFIG.swap_penalty,
        distance_reduction_reward_scale=BASE_CONFIG.distance_reduction_reward_scale,
        progress_reward_scale=BASE_CONFIG.progress_reward_scale,
        executed_gate_reward_scale=BASE_CONFIG.executed_gate_reward_scale,
        timeout_penalty=BASE_CONFIG.timeout_penalty,
        random_2q_prob=BASE_CONFIG.random_2q_prob,
        qaoa_prob=BASE_CONFIG.qaoa_prob,
        quantum_volume_prob=BASE_CONFIG.quantum_volume_prob,
        vqe_prob=BASE_CONFIG.vqe_prob,
        clifford_prob=BASE_CONFIG.clifford_prob,
        positive_control_prob=BASE_CONFIG.positive_control_prob,
        zero_noise_features=BASE_CONFIG.zero_noise_features,
        calibration_feature_mask=BASE_CONFIG.calibration_feature_mask,
        benchmark_qasm_files=BASE_CONFIG.benchmark_qasm_files,
        benchmark_qasm_dir=BASE_CONFIG.benchmark_qasm_dir,
        benchmark_corpus_prob=BASE_CONFIG.benchmark_corpus_prob,
        benchmark_corpus_name=BASE_CONFIG.benchmark_corpus_name,
    )
    state = env.reset(episode=seed_base)
    agent = PPOAgent(
        state_size=state.shape[0],
        action_size=env.get_action_size(),
        action_set_name=BASE_CONFIG.action_set_phase1,
        num_qubits=env.num_qubits,
        coupling_edges=env._physical_edges,
        **agent_hparams,
    )

    recent_fidelities = deque(maxlen=20)
    train_frequency = max(1, _env_int("TUNE_TRAIN_FREQUENCY", 10))
    for episode in range(int(tuning_episodes)):
        state = env.reset(episode=seed_base + episode)
        done = False
        step = 0
        info = dict(env.last_metrics)

        while not done and step < env.max_steps_per_episode:
            action_mask = agent.compute_action_mask(env)
            action, action_log_prob, value, _, _, _ = agent.select_action(state, action_mask=action_mask)
            next_state, reward, done, info = env.step(action)

            with torch.no_grad():
                ns = torch.tensor(next_state, dtype=torch.float32, device=agent.device).unsqueeze(0)
                next_mask = agent.compute_action_mask(env).unsqueeze(0)
                _, next_value, _, _ = agent.policy(ns, action_mask=next_mask)
                next_value = next_value.squeeze(0).detach()

            transition = Transition(
                state=torch.tensor(state, dtype=torch.float32, device=agent.device),
                action=action,
                reward=float(reward),
                log_prob=action_log_prob,
                value=value,
                next_value=next_value,
                done=float(done),
                true_fidelity=float(info.get("fidelity", 0.0)) if done else float("nan"),
                true_coherence=float("nan"),
                action_mask=action_mask.detach(),
            )
            agent.store_transition(transition)

            state = next_state
            step += 1

        recent_fidelities.append(float(info.get("fidelity", 0.0)))
        if (episode + 1) % train_frequency == 0:
            agent.train(writer=None, episode=episode)

    return agent, float(np.mean(recent_fidelities)) if recent_fidelities else float("nan")


def run_top_trial_verification(study, output_dir):
    verify_top_k = max(0, _env_int("TUNE_VERIFY_TOP_K", 0))
    verify_episodes = max(0, _env_int("TUNE_VERIFY_EPISODES", 0))
    if verify_top_k <= 0 or verify_episodes <= 0:
        return None

    complete_trials = _completed_trials(study)
    if not complete_trials:
        return None

    ranked_trials = sorted(complete_trials, key=lambda t: float(t.value), reverse=True)[:verify_top_k]
    verify_exact_eval_episodes = max(1, _env_int("TUNE_VERIFY_EXACT_EVAL_EPISODES", _env_int("TUNE_EXACT_EVAL_EPISODES", 30)))
    verify_seed_base = _env_int("TUNE_VERIFY_BASE_SEED", 200000)

    rows = []
    for rank, trial in enumerate(ranked_trials):
        agent_hparams = BASE_CONFIG.to_agent_kwargs()
        agent_hparams.update(trial.params)
        _set_global_seeds(verify_seed_base + rank)
        agent, proxy_metric = _train_trial_agent(
            agent_hparams=copy.deepcopy(agent_hparams),
            tuning_episodes=verify_episodes,
            seed_base=verify_seed_base + 1000 * rank,
        )
        exact_eval = evaluate_agent_exact(
            agent,
            num_episodes=verify_exact_eval_episodes,
            episode_seed_start=verify_seed_base + 50000 + 1000 * rank,
        )
        rows.append(
            {
                "trial_number": int(trial.number),
                "original_exact_eval_fidelity": float(trial.value),
                "original_proxy_train_fidelity": float(trial.user_attrs.get("proxy_train_fidelity", float("nan"))),
                "verification_proxy_train_fidelity": proxy_metric,
                "verification_exact_eval_fidelity": float(exact_eval["mean_fidelity"]),
                "verification_exact_eval_delta": float(exact_eval["mean_fidelity_delta"]),
                "params": dict(agent_hparams),
            }
        )

    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "verification_rerank.png")
    labels = [f"trial {row['trial_number']}" for row in rows]
    orig = np.asarray([row["original_exact_eval_fidelity"] for row in rows], dtype=float)
    verify = np.asarray([row["verification_exact_eval_fidelity"] for row in rows], dtype=float)
    x = np.arange(len(rows))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, 2 * len(rows)), 5))
    ax.bar(x - width / 2, orig, width=width, label="Original exact eval")
    ax.bar(x + width / 2, verify, width=width, label="Long-run verification")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Exact Fidelity")
    ax.set_title("Top-Trial Verification After Longer Training")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    summary = {
        "verification_protocol": {
            "top_k": verify_top_k,
            "verification_episodes": verify_episodes,
            "verification_exact_eval_episodes": verify_exact_eval_episodes,
        },
        "rows": rows,
        "artifact_paths": {"plot": plot_path},
    }
    summary_path = os.path.join(output_dir, "verification_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    summary["artifact_paths"]["json"] = summary_path
    return summary

if __name__ == "__main__":
    print("Starting Hyperparameter Optimization...")
    
    # We use a MedianPruner. It will let trials run for at least 60 episodes.
    # After that, if a trial's reported fidelity is worse than the median of past trials
    # at that same episode, it ruthlessly kills it to save CPU time.
    pruner = optuna.pruners.MedianPruner(
        n_warmup_steps=_env_int("TUNE_WARMUP_STEPS", 60),
        n_startup_trials=_env_int("TUNE_STARTUP_TRIALS", 5),
    )
    
    sampler = optuna.samplers.TPESampler(seed=_env_int("OPTUNA_SAMPLER_SEED", 0))
    storage = _build_optuna_storage()
    study_name = os.getenv("OPTUNA_STUDY_NAME", "quantum_routing_tuning")
    if storage is None:
        study = optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler)
    else:
        study = optuna.create_study(
            direction="maximize",
            pruner=pruner,
            sampler=sampler,
            storage=storage,
            study_name=study_name,
            load_if_exists=True,
        )

    n_trials = _env_int("OPTUNA_TRIALS", 50)
    if n_trials > 0:
        study.optimize(objective, n_trials=n_trials)

    complete_trials = _completed_trials(study)
    if not complete_trials:
        raise RuntimeError("No completed Optuna trials are available to export.")

    print("\n" + "="*40)
    print("Optimization Finished!")
    print(f"Best Trial Fidelity: {study.best_trial.value:.4f}")
    print("Best Hyperparameters:")
    resolved_best_params = BASE_CONFIG.to_agent_kwargs()
    resolved_best_params.update(study.best_trial.params)

    for key, value in resolved_best_params.items():
        print(f"  {key}: {value}")
        
    # Export to JSON
    if _env_flag("TUNING_EXPORT_RESULTS", default=True):
        export_path = os.getenv("OPTIMAL_HYPERPARAMS_PATH", "optimal_hyperparams.json")
        with open(export_path, "w") as f:
            json.dump(resolved_best_params, f, indent=4)

        print(f"\nSaved optimal configuration to {export_path}")
        artifact_dir = os.getenv("TUNING_ARTIFACT_DIR", "tuning_artifacts")
        artifact_paths = save_reviewer_artifacts(study, artifact_dir)
        if artifact_paths:
            print("Saved reviewer artifacts:")
            for path in artifact_paths:
                print(f"  {path}")
        verification = run_top_trial_verification(study, artifact_dir)
        if verification is not None:
            print("Saved longer-horizon verification artifacts:")
            for path in verification["artifact_paths"].values():
                print(f"  {path}")
    else:
        print("\nTUNING_EXPORT_RESULTS=0, so this worker did not write shared output files.")
    print("You can now safely run 'python scalable_quantum.py'.")
