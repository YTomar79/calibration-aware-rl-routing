"""DQN routing baseline, sharing the same environment as the PPO policy for a like-for-like comparison."""

import json
import os
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from scalable_quantum import QuantumRoutingEnv, TrainingConfig, _env_int, _set_global_seeds


class QNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x):
        return self.net(x)


def _make_env(config, use_proxy_reward=True):
    return QuantumRoutingEnv(
        num_qubits=config.train_num_qubits,
        calibration_file=config.calibration_file,
        max_steps_per_episode=config.train_max_steps,
        routing_method="sabre",
        optimization_level=config.optimization_level,
        cost_lambda=config.cost_lambda,
        cost_w_twoq=config.cost_w_twoq,
        cost_w_depth=config.cost_w_depth,
        debug=False,
        use_proxy_reward=use_proxy_reward,
        reward_mode=config.reward_mode,
        fidelity_scale=config.fidelity_scale,
        invalid_action_penalty=config.invalid_action_penalty,
        swap_penalty=config.swap_penalty,
        distance_reduction_reward_scale=config.distance_reduction_reward_scale,
        progress_reward_scale=config.progress_reward_scale,
        executed_gate_reward_scale=config.executed_gate_reward_scale,
        timeout_penalty=config.timeout_penalty,
        random_2q_prob=config.random_2q_prob,
        qaoa_prob=config.qaoa_prob,
        quantum_volume_prob=config.quantum_volume_prob,
        vqe_prob=config.vqe_prob,
        clifford_prob=config.clifford_prob,
        positive_control_prob=config.positive_control_prob,
        zero_noise_features=True,
        calibration_feature_mask="topology_only",
        benchmark_qasm_files=config.benchmark_qasm_files,
        benchmark_qasm_dir=config.benchmark_qasm_dir,
        benchmark_corpus_prob=config.benchmark_corpus_prob,
        benchmark_corpus_name=config.benchmark_corpus_name,
    )


def _masked_argmax(q_values, mask):
    masked = q_values.clone()
    mask_t = torch.tensor(mask, dtype=torch.bool, device=masked.device)
    masked[~mask_t] = -1e9
    return int(torch.argmax(masked).item())


def train_dqn(config):
    seed = _env_int("DQN_BASELINE_SEED", 1234)
    _set_global_seeds(seed)
    rng = np.random.default_rng(seed)

    env = _make_env(config, use_proxy_reward=True)
    state = env.reset(episode=0)
    state_size = int(state.shape[0])
    action_size = int(env.get_action_size())

    policy = QNet(state_size, action_size, hidden_size=_env_int("DQN_BASELINE_HIDDEN_SIZE", 256))
    target = QNet(state_size, action_size, hidden_size=_env_int("DQN_BASELINE_HIDDEN_SIZE", 256))
    target.load_state_dict(policy.state_dict())

    optimizer = optim.Adam(policy.parameters(), lr=float(os.getenv("DQN_BASELINE_LR", "0.0005")))
    replay = deque(maxlen=_env_int("DQN_BASELINE_REPLAY_SIZE", 20000))
    gamma = float(os.getenv("DQN_BASELINE_GAMMA", "0.99"))
    batch_size = _env_int("DQN_BASELINE_BATCH_SIZE", 64)
    train_episodes = _env_int("DQN_BASELINE_TRAIN_EPISODES", 1000)
    start_seed = _env_int("DQN_BASELINE_TRAIN_START_SEED", 1000000)
    epsilon = float(os.getenv("DQN_BASELINE_EPSILON_START", "1.0"))
    epsilon_min = float(os.getenv("DQN_BASELINE_EPSILON_MIN", "0.05"))
    epsilon_decay = float(os.getenv("DQN_BASELINE_EPSILON_DECAY", "0.995"))

    for episode in range(train_episodes):
        state = env.reset(episode=start_seed + episode)
        done = False
        steps = 0
        while not done and steps < env.max_steps_per_episode:
            mask = env.get_action_mask()
            valid = np.flatnonzero(mask > 0.0)
            if rng.random() < epsilon:
                action = int(rng.choice(valid))
            else:
                with torch.no_grad():
                    q = policy(torch.tensor(state, dtype=torch.float32))
                    action = _masked_argmax(q, mask)
            next_state, reward, done, _ = env.step(action)
            replay.append((state, action, float(reward), next_state, bool(done), mask))
            state = next_state
            steps += 1

            if len(replay) >= batch_size:
                idx = rng.choice(len(replay), size=batch_size, replace=False)
                batch = [replay[int(i)] for i in idx]
                states = torch.tensor(np.asarray([b[0] for b in batch]), dtype=torch.float32)
                actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
                rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                next_states = torch.tensor(np.asarray([b[3] for b in batch]), dtype=torch.float32)
                dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)

                q_sa = policy(states).gather(1, actions[:, None]).squeeze(1)
                with torch.no_grad():
                    next_q = target(next_states).max(dim=1).values
                    target_q = rewards + gamma * (1.0 - dones) * next_q
                loss = nn.functional.smooth_l1_loss(q_sa, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if episode % _env_int("DQN_BASELINE_TARGET_UPDATE", 25) == 0:
            target.load_state_dict(policy.state_dict())
    return policy


def evaluate(policy, config):
    eval_episodes = _env_int("REVIEW_HOLDOUT_EPISODES", 300)
    start_seed = _env_int("REVIEW_HOLDOUT_START_SEED", 70000)
    env = _make_env(config, use_proxy_reward=False)
    records = []
    for offset in range(eval_episodes):
        seed = start_seed + offset
        state = env.reset(episode=seed)
        done = False
        info = dict(env.last_metrics)
        steps = 0
        t0 = time.perf_counter()
        while not done and steps < env.max_steps_per_episode:
            with torch.no_grad():
                q = policy(torch.tensor(state, dtype=torch.float32))
                action = _masked_argmax(q, env.get_action_mask())
            state, _, done, info = env.step(action)
            steps += 1
        records.append(
            {
                "seed": int(seed),
                "algorithm": "topology_only_dqn",
                "metrics": {
                    "fidelity": float(info.get("fidelity", env.last_metrics.get("fidelity", 0.0))),
                    "proxy_fidelity": float(env._calculate_proxy_fidelity(env.compiled_circuit)) if env.compiled_circuit is not None else float("nan"),
                    "twoq": float(info.get("twoq", env.last_metrics.get("twoq", 0.0))),
                    "depth": float(info.get("depth", env.last_metrics.get("depth", 0.0))),
                    "cost": float(info.get("cost", env.last_metrics.get("cost", 0.0))),
                    "wall_seconds": float(time.perf_counter() - t0),
                },
            }
        )
    return records


if __name__ == "__main__":
    config = TrainingConfig(load_hyperparams=True)
    policy = train_dqn(config)
    records = evaluate(policy, config)
    output_path = os.path.abspath(os.getenv("DQN_BASELINE_OUTPUT", "dqn_routing_baseline.json"))
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": "In-harness topology-only DQN routing baseline",
        "note": "This is an in-harness DQN reproduction baseline, not author-provided code.",
        "episodes": records,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved DQN baseline records: {output_path}")
