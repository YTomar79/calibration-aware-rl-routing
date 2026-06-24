"""Calibration-aware PPO router: training environment, agent, and noisy-simulation evaluation."""

import torch
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer, AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, ReadoutError
from qiskit.quantum_info import state_fidelity, DensityMatrix
import hashlib
from qiskit.transpiler import CouplingMap
import json
import os
import glob
import collections
import time
from typing import List, Tuple, Optional


torch.set_default_dtype(torch.float32)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def add_histogram(self, *args, **kwargs):
            pass

        def flush(self):
            pass

        def close(self):
            pass


def _resolve_existing_path(*candidates):
    for candidate in candidates:
        if not candidate:
            continue
        path = os.path.expanduser(str(candidate))
        if os.path.exists(path):
            return path
    return None


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


def _set_global_seeds(seed: int):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def find_latest_checkpoint(pattern="ppo_checkpoint_*.pt"):
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None
    # robust: prefer highest episode number if present, else newest mtime
    def episode_num(p):
        base = os.path.basename(p)
        try:
            # "ppo_checkpoint_{episode}.pt"
            return int(base.split("_")[-1].split(".")[0])
        except Exception:
            return -1
    ckpts.sort(key=lambda p: (episode_num(p), os.path.getmtime(p)))
    return ckpts[-1]
    
    
# --- Circuit fingerprint helper (cache key for fidelity simulation) ---
def _circuit_fingerprint(qc):
    """
    Robustly hash a circuit across Qiskit versions.
    We try qasm3, then qasm2, then fallback to repr().
    """
    payload = None
    try:
        from qiskit import qasm3
        payload = qasm3.dumps(qc)
    except Exception:
        pass

    if payload is None:
        try:
            from qiskit import qasm2
            payload = qasm2.dumps(qc)
        except Exception:
            pass

    if payload is None:
        try:
            payload = qc.qasm()
        except Exception:
            payload = repr(qc)

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class QuantumRoutingEnv:
    """
    Routing-by-SWAP RL under realistic device noise.

    Environment contract:
      - A *logical* target circuit is frozen at reset().
      - The agent's action is to choose a *physical* SWAP on an allowed hardware edge.
      - The environment maintains a token-swapping mapping `logical_to_physical` that tracks where each
        logical qubit currently resides on hardware after the inserted SWAPs.
      - After each SWAP, the env greedily appends as many next logical operations as are currently
        executable:
          * 1Q gates are always executable (if the mapped physical qubit is active)
          * 2Q gates are executable only when the mapped physical qubits are adjacent in the coupling graph
      - The env never reorders, edits, or drops logical operations; it only inserts SWAPs.

    Compilation policy:
      - The *agent circuit* is already hardware-compliant, so we transpile it **without routing**
        (no extra SWAP insertion).
      - Baselines (trivial/sabre) are compiled by Qiskit's transpiler with routing enabled, under the
        same coupling constraints and the same episode noise snapshot.

    Notes:
      - Observation includes device noise features + current layout + a small "next-gate" hint (which
        is essential for routing to be learnable).
      - Reward is shaped with a per-SWAP penalty; the final step adds a baseline-delta reward based on
        noisy fidelity and a simple cost model.
    """

    COH_MIN_S = 1e-6
    COH_MAX_S = 300e-6

    def __init__(
        self,
        num_qubits: int = 10,
        device: str = "cpu",
        calibration_file: str = "downloaded_calibrations/ibm_torino_calibration.json",
        optimization_level: int = 3,
        routing_method: str = "sabre",
        cost_lambda: float = 1.0,
        cost_w_twoq: float = 1.0,
        cost_w_depth: float = 0.01,
        max_steps_per_episode: int = 200,
        max_steps_base: Optional[int] = None,
        max_steps_per_2q: Optional[float] = None,
        max_steps_cap: Optional[int] = None,
        debug: bool = False,
        use_proxy_reward: bool = True,
        reward_mode: str = "shaped",
        invalid_action_penalty: float = 0.2,
        swap_penalty: float = 0.02,
        distance_reduction_reward_scale: float = 0.05,
        progress_reward_scale: float = 2.0,
        executed_gate_reward_scale: float = 0.01,
        timeout_penalty: float = 0.5,
        incomplete_episode_penalty: Optional[float] = None,
        fidelity_scale: float = 10.0,
        readout_jitter_frac: float = 0.10,
        oneq_jitter_frac: float = 0.15,
        twoq_jitter_frac: float = 0.15,
        coherence_jitter_frac: float = 0.10,
        random_2q_prob: float = 0.45,
        qaoa_prob: float = 0.20,
        quantum_volume_prob: float = 0.20,
        vqe_prob: float = 0.15,
        clifford_prob: float = 0.0,
        positive_control_prob: float = 0.0,
        zero_noise_features: bool = False,
        calibration_feature_mask: Optional[str] = None,
        benchmark_qasm_files: Optional[str] = None,
        benchmark_qasm_dir: Optional[str] = None,
        benchmark_corpus_prob: float = 0.0,
        benchmark_corpus_name: str = "external_qasm",
    ):
        self.num_qubits = int(num_qubits)
        self.device = torch.device(device)
        self.debug = bool(debug)
        self.use_proxy_reward = bool(use_proxy_reward)
        self.reward_mode = str(reward_mode)
        self.zero_noise_features = bool(zero_noise_features)
        self.calibration_feature_mask = {
            item.strip().lower()
            for item in str(calibration_feature_mask or "").split(",")
            if item.strip()
        }
        self.max_exact_dm_qubits = _env_int("EXACT_DM_MAX_QUBITS", 16)

        self.optimization_level = int(optimization_level)
        self.routing_method = str(routing_method)
        self.sabre_baseline_trials = _env_int("SABRE_BASELINE_TRIALS", 20)

        self.cost_lambda = float(cost_lambda)
        self.cost_w_twoq = float(cost_w_twoq)
        self.cost_w_depth = float(cost_w_depth)

        self.base_max_steps_per_episode = int(max_steps_per_episode)
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.max_steps_base = int(max_steps_base if max_steps_base is not None else _env_int("TRAIN_MAX_STEPS_BASE", 100))
        self.max_steps_per_2q = float(max_steps_per_2q if max_steps_per_2q is not None else _env_float("TRAIN_MAX_STEPS_PER_2Q", 8.0))
        self.max_steps_cap = int(max_steps_cap if max_steps_cap is not None else _env_int("TRAIN_MAX_STEPS_CAP", 2000))
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.swap_penalty = float(swap_penalty)
        self.distance_reduction_reward_scale = float(distance_reduction_reward_scale)
        self.progress_reward_scale = float(progress_reward_scale)
        self.executed_gate_reward_scale = float(executed_gate_reward_scale)
        self.timeout_penalty = float(timeout_penalty)
        self.incomplete_episode_penalty = float(
            incomplete_episode_penalty
            if incomplete_episode_penalty is not None
            else _env_float("INCOMPLETE_EPISODE_PENALTY", 10.0)
        )
        self.fidelity_scale = float(fidelity_scale)

        self.readout_jitter_frac = float(readout_jitter_frac)
        self.oneq_jitter_frac = float(oneq_jitter_frac)
        self.twoq_jitter_frac = float(twoq_jitter_frac)
        self.coherence_jitter_frac = float(coherence_jitter_frac)

        self.random_2q_prob = float(random_2q_prob)
        self.qaoa_prob = float(qaoa_prob)
        self.quantum_volume_prob = float(quantum_volume_prob)
        self.vqe_prob = float(vqe_prob)
        self.clifford_prob = float(clifford_prob)
        self.positive_control_prob = float(positive_control_prob)
        self.benchmark_qasm_files = benchmark_qasm_files
        self.benchmark_qasm_dir = benchmark_qasm_dir
        self.benchmark_corpus_prob = float(benchmark_corpus_prob)
        self.benchmark_corpus_name = str(benchmark_corpus_name or "external_qasm")
        self._benchmark_qasm_records = self._discover_benchmark_qasm_records(
            benchmark_qasm_files=benchmark_qasm_files,
            benchmark_qasm_dir=benchmark_qasm_dir,
        )
        self.require_benchmark_qasm_records = _env_flag("REQUIRE_BENCHMARK_QASM_RECORDS", False) or _env_flag(
            "REVIEW_REQUIRE_BENCHMARK_CORPUS",
            False,
        )
        if self.require_benchmark_qasm_records and (self.benchmark_qasm_files or self.benchmark_qasm_dir) and not self._benchmark_qasm_records:
            raise RuntimeError(
                "Benchmark corpus was required, but no compatible QASM circuits were discovered. "
                f"qasm_dir={self.benchmark_qasm_dir!r}, qasm_files={self.benchmark_qasm_files!r}, "
                f"env_num_qubits={self.num_qubits}. Check that the directory exists, contains .qasm/.qasm2/.qasm3 files, "
                "and that at least one circuit has <= TRAIN_NUM_QUBITS qubits and can be parsed by Qiskit."
            )
        if self._benchmark_qasm_records and self.benchmark_corpus_prob <= 0.0:
            self.benchmark_corpus_prob = 1.0
        self._target_type_probs = np.asarray(
            [
                self.random_2q_prob,
                self.qaoa_prob,
                self.quantum_volume_prob,
                self.vqe_prob,
                self.clifford_prob,
            ],
            dtype=np.float64,
        )
        if np.any(self._target_type_probs < 0):
            raise ValueError("Circuit type probabilities must be non-negative.")
        if self.positive_control_prob < 0:
            raise ValueError("POSITIVE_CONTROL_PROB must be non-negative.")
        if self.benchmark_corpus_prob < 0:
            raise ValueError("BENCHMARK_CORPUS_PROB must be non-negative.")
        synthetic_prob_sum = float(self._target_type_probs.sum())
        corpus_only = bool(self._benchmark_qasm_records) and self.benchmark_corpus_prob >= 1.0
        positive_control_only = self.positive_control_prob >= 1.0
        if synthetic_prob_sum <= 0.0:
            if not (corpus_only or positive_control_only):
                raise ValueError(
                    "At least one synthetic circuit probability must be positive unless "
                    "BENCHMARK_CORPUS_PROB=1 with compatible QASM records or POSITIVE_CONTROL_PROB=1."
                )
            self._target_type_probs = np.full_like(self._target_type_probs, 1.0 / len(self._target_type_probs))
        else:
            self._target_type_probs = self._target_type_probs / synthetic_prob_sum
        if (not self.use_proxy_reward) and self.num_qubits > self.max_exact_dm_qubits:
            raise ValueError(
                f"Exact density-matrix fidelity is configured for <= {self.max_exact_dm_qubits} qubits; "
                "set USE_PROXY_REWARD=1 or provide a cheaper fidelity estimator for larger systems."
            )

        # RNG
        self.rng = np.random.default_rng(0)

        # Hardware/noise tensors (kept minimal but compatible with agent mask code)
        self.deactivated_qubits = torch.zeros(self.num_qubits, dtype=torch.bool, device=self.device)
        # For backwards compatibility with existing observation code: treat as readout error rates.
        self.error_rates = torch.zeros(self.num_qubits, dtype=torch.float32, device=self.device)
        self.coherence_times = torch.full((self.num_qubits,), 50e-6, dtype=torch.float32, device=self.device)

        # More detailed calibration-derived rates (used by the noise model)
        self.oneq_error_rates = torch.zeros(self.num_qubits, dtype=torch.float32, device=self.device)
        self.twoq_error_rates = {}  # (i,j) undirected local edge -> p_error

        # Backend simulator (only used for transpile target; not an IBM backend)
        self.backend = Aer.get_backend("qasm_simulator")

        # Load and process calibration snapshot (RESTORED: was missing in the original file)
        self._calib_snapshot = self._load_calibration_snapshot(calibration_file)

        # Initialize coupling map + candidate edges from calibration snapshot (or fall back safely)
        self._init_hardware_from_calibration()
        self._store_base_noise_snapshot()

        # Fixed action sizing: one op-type ("swap") × number of physical edges
        self.edge_action_types = ("swap",)
        self.max_candidate_edges = int(len(self._physical_edges))

        # Precompute adjacency for fast checks
        self._adjacency = set()
        self._neighbors = [[] for _ in range(self.num_qubits)]
        for a, b in self._physical_edges:
            a = int(a); b = int(b)
            self._adjacency.add((a, b))
            self._adjacency.add((b, a))
            self._neighbors[a].append(b)
            self._neighbors[b].append(a)


        # Episode state
        self.current_step = 0
        self.done = False
        self.invalid_gate_count = 0

        self.target_circuit = None
        self.target_circuit_type = None
        self.target_circuit_source = None
        self.target_circuit_source_sha256 = None
        self.target_circuit_sha256 = None
        self.current_circuit = None
        self.compiled_circuit = None

        self.logical_to_physical = [int(i) for i in range(self.num_qubits)]
        self.physical_to_logical = [int(i) for i in range(self.num_qubits)]

        self._op_cursor = 0  # next instruction index in target_circuit.data
        self.executed_ops_total = 0
        self.completed_target = False
        self.timed_out = False
        self.terminal_reason = "not_started"
        self.final_progress = 0.0
        self.effective_max_steps = int(self.max_steps_per_episode)
        self.target_twoq_count = 0
        self.target_depth = 0

        # Caches
        self._compile_cache = {}
        self._ideal_dm_cache = {}
        self._noisy_dm_cache = {}
        self._episode_seed = 0
        self._auto_episode = 0

        # Baseline stats
        self.baseline_fidelity = 0.0
        self.baseline_cost = 0.0
        self.baseline_compiled = None

        self.last_metrics = {}

        # Optional: allow external code to confirm env "mode"
        self.action_set_name = "routing_only"

        # Basis gates: include CZ/RZZ if present in calibration, but keep CX too for robustness.
        # (IBM Torino calibration commonly reports 'cz'/'rzz' instead of 'cx'.)
        self.basis_gates = ["id", "rz", "rx", "sx", "x", "h", "cz", "cx", "rzz", "swap"]

        # Candidate edges cache (unused in this minimal version, but kept for compatibility)
        self._candidate_edges_cache = None
        
        self.k_calib = 1.0
        self.k_calib_by_class = {}

    def _active_k_calib(self) -> float:
        cls = str(getattr(self, "target_circuit_type", ""))
        if cls and cls in self.k_calib_by_class:
            return float(self.k_calib_by_class[cls])
        return float(self.k_calib)

    def _discover_benchmark_qasm_records(self, benchmark_qasm_files=None, benchmark_qasm_dir=None):
        candidates = []
        for raw in str(benchmark_qasm_files or "").split(","):
            raw = raw.strip()
            if raw:
                candidates.append(raw)

        root = str(benchmark_qasm_dir or "").strip()
        if root:
            root = os.path.abspath(os.path.expanduser(root))
            for pattern in ("**/*.qasm", "**/*.qasm2", "**/*.qasm3"):
                candidates.extend(glob.glob(os.path.join(root, pattern), recursive=True))

        seen = set()
        records = []
        max_files = _env_int("BENCHMARK_QASM_MAX_FILES", 10000)
        for candidate in candidates:
            path = os.path.abspath(os.path.expanduser(candidate))
            if path in seen or not os.path.exists(path):
                continue
            seen.add(path)
            records.append(
                {
                    "path": path,
                    "name": os.path.splitext(os.path.basename(path))[0],
                    "source_sha256": self._file_sha256(path),
                }
            )
            if len(records) >= max_files:
                break
        records = self._attach_qasm_circuit_hashes(records)
        records = self._filter_excluded_qasm_hashes(records)
        self._write_qasm_hash_manifest(records)
        return records

    def _file_sha256(self, path: str):
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    def _load_hash_set(self, path: str):
        if not path:
            return set()
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(path):
            return set()
        try:
            with open(path, "r") as f:
                payload = json.load(f)
        except Exception:
            return set()
        hashes = set()
        if isinstance(payload, list):
            for item in payload:
                hashes.add(str(item))
        elif isinstance(payload, dict):
            for key in ("source_sha256", "circuit_sha256", "hashes"):
                value = payload.get(key)
                if isinstance(value, list):
                    hashes.update(str(item) for item in value)
            for record in payload.get("records", []):
                if isinstance(record, dict):
                    for key in ("source_sha256", "circuit_sha256"):
                        if record.get(key):
                            hashes.add(str(record[key]))
        return hashes

    def _attach_qasm_circuit_hashes(self, records):
        if not _env_flag("BENCHMARK_QASM_HASH_CIRCUITS", True):
            return records
        out = []
        for record in records:
            qc = self._normalize_external_circuit(self._load_qasm_circuit(record["path"]))
            if qc is None:
                continue
            record = dict(record)
            record["circuit_sha256"] = _circuit_fingerprint(qc)
            record["num_qubits"] = int(qc.num_qubits)
            record["depth"] = int(qc.depth())
            record["size"] = int(qc.size())
            out.append(record)
        return out

    def _filter_excluded_qasm_hashes(self, records):
        exclude_path = os.getenv("BENCHMARK_QASM_EXCLUDE_HASHES_FILE", "").strip()
        excluded = self._load_hash_set(exclude_path)
        if not excluded:
            return records
        kept = []
        dropped = []
        for record in records:
            source_hash = record.get("source_sha256")
            circuit_hash = record.get("circuit_sha256")
            if source_hash in excluded or circuit_hash in excluded:
                dropped.append(record)
            else:
                kept.append(record)
        if dropped and self.debug:
            print(f"[benchmark] excluded {len(dropped)} QASM circuits by hash from {exclude_path}")
        if not kept and records:
            raise RuntimeError("All benchmark QASM circuits were excluded by BENCHMARK_QASM_EXCLUDE_HASHES_FILE.")
        return kept

    def _write_qasm_hash_manifest(self, records):
        manifest_path = os.getenv("BENCHMARK_QASM_HASH_MANIFEST", "").strip()
        if not manifest_path:
            return
        manifest_path = os.path.abspath(os.path.expanduser(manifest_path))
        os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
        payload = {
            "corpus_name": self.benchmark_corpus_name,
            "qasm_files": self.benchmark_qasm_files,
            "qasm_dir": self.benchmark_qasm_dir,
            "source_sha256": [record.get("source_sha256") for record in records if record.get("source_sha256")],
            "circuit_sha256": [record.get("circuit_sha256") for record in records if record.get("circuit_sha256")],
            "records": records,
        }
        with open(manifest_path, "w") as f:
            json.dump(payload, f, indent=2)

    def _load_qasm_circuit(self, path: str):
        try:
            if path.endswith(".qasm3"):
                from qiskit import qasm3

                return qasm3.load(path)
            return QuantumCircuit.from_qasm_file(path)
        except Exception:
            try:
                from qiskit import qasm2

                return qasm2.load(path)
            except Exception as e:
                if self.debug:
                    print(f"[benchmark] failed to load QASM circuit {path}: {e}")
                return None

    def _normalize_external_circuit(self, qc: QuantumCircuit):
        if qc is None or qc.num_qubits <= 0 or qc.num_qubits > self.num_qubits:
            return None

        try:
            qc = qc.remove_final_measurements(inplace=False)
        except Exception:
            pass

        out = QuantumCircuit(self.num_qubits)
        for instr in qc.data:
            operation = instr.operation
            if operation.name in {"measure", "barrier", "delay"}:
                continue
            q_indices = [qc.find_bit(q).index for q in instr.qubits]
            if any(q >= self.num_qubits for q in q_indices):
                return None
            try:
                operation = operation.copy()
            except Exception:
                pass
            out.append(operation, q_indices, [])
        return out

    def _strip_non_unitary_ops(self, qc: QuantumCircuit) -> QuantumCircuit:
        out = QuantumCircuit(self.num_qubits)
        for instr in qc.data:
            operation = instr.operation
            if operation.name in {"measure", "barrier", "delay"}:
                continue
            q_indices = [qc.find_bit(q).index for q in instr.qubits]
            if any(q >= self.num_qubits for q in q_indices):
                raise ValueError(
                    f"Target circuit uses qubit index outside env width {self.num_qubits}: "
                    f"{operation.name}{tuple(q_indices)}"
                )
            try:
                operation = operation.copy()
            except Exception:
                pass
            out.append(operation, q_indices, [])
        return out

    @staticmethod
    def _wide_ops(qc: QuantumCircuit):
        wide = []
        for idx, instr in enumerate(qc.data):
            n_qubits = len(instr.qubits)
            if n_qubits > 2:
                wide.append((idx, instr.operation.name, n_qubits))
        return wide

    def _raise_unsupported_target_ops(self, qc: QuantumCircuit, context: str, cause: Optional[Exception] = None):
        wide = self._wide_ops(qc)
        if wide:
            details = ", ".join(
                f"#{idx}:{name}/{nq}q" for idx, name, nq in wide[:12]
            )
            if len(wide) > 12:
                details += f", ... (+{len(wide) - 12} more)"
        else:
            details = "no >2q ops found after fallback inspection"
        message = (
            f"Target circuit {context!r} could not be normalized to router-supported "
            f"0/1/2-qubit operations. Offending ops: {details}."
        )
        if cause is not None:
            message += f" Normalization error: {cause}"
        raise ValueError(message)

    def _normalize_target_circuit(self, qc: QuantumCircuit, context: Optional[str] = None) -> QuantumCircuit:
        if qc is None:
            raise ValueError("Target circuit generation returned None.")
        context = str(context or getattr(self, "target_circuit_type", "unknown"))
        if qc.num_qubits <= 0:
            raise ValueError(f"Target circuit {context!r} has no qubits.")
        if qc.num_qubits > self.num_qubits:
            raise ValueError(
                f"Target circuit {context!r} uses {qc.num_qubits} qubits, "
                f"but env width is {self.num_qubits}."
            )

        stripped = self._strip_non_unitary_ops(qc)
        if not stripped.data:
            return stripped

        try:
            normalized = transpile(
                stripped,
                basis_gates=self.basis_gates,
                optimization_level=0,
                seed_transpiler=int(self._episode_seed),
            )
            normalized = self._strip_non_unitary_ops(normalized)
        except Exception as exc:
            decomposed = stripped
            for _ in range(8):
                try:
                    decomposed = decomposed.decompose()
                except Exception:
                    break
                if not self._wide_ops(decomposed):
                    break
            decomposed = self._strip_non_unitary_ops(decomposed)
            if self._wide_ops(decomposed):
                self._raise_unsupported_target_ops(decomposed, context, cause=exc)
            normalized = decomposed

        if self._wide_ops(normalized):
            self._raise_unsupported_target_ops(normalized, context)
        return normalized

    def _count_target_twoq_ops(self) -> int:
        if self.target_circuit is None:
            return 0
        return int(sum(1 for instr in self.target_circuit.data if len(instr.qubits) == 2))

    def _compute_effective_max_steps(self) -> int:
        twoq = int(getattr(self, "target_twoq_count", 0))
        dynamic_budget = int(np.ceil(float(self.max_steps_base) + float(self.max_steps_per_2q) * float(twoq)))
        requested = int(max(1, self.base_max_steps_per_episode))
        cap = int(max(1, self.max_steps_cap))
        return int(min(cap, max(requested, dynamic_budget)))

    def _current_progress(self) -> float:
        if self.target_circuit is None or len(self.target_circuit.data) == 0:
            return 1.0
        return float(self._op_cursor) / float(len(self.target_circuit.data))

    def _terminal_status_fields(self):
        return {
            "completed_target": bool(self.completed_target),
            "timed_out": bool(self.timed_out),
            "terminal_reason": str(self.terminal_reason),
            "final_progress": float(self.final_progress),
            "progress": float(self.final_progress),
            "executed_ops_total": int(self.executed_ops_total),
            "effective_max_steps": int(self.effective_max_steps),
            "target_twoq": int(self.target_twoq_count),
            "target_depth": int(self.target_depth),
        }

    def _generate_benchmark_qasm_circuit(self):
        if not self._benchmark_qasm_records:
            return None
        start_idx = int(self.rng.integers(0, len(self._benchmark_qasm_records)))
        for offset in range(len(self._benchmark_qasm_records)):
            record = self._benchmark_qasm_records[(start_idx + offset) % len(self._benchmark_qasm_records)]
            qc = self._normalize_external_circuit(self._load_qasm_circuit(record["path"]))
            if qc is None:
                continue
            self.target_circuit_type = f"{self.benchmark_corpus_name}:{record['name']}"
            self.target_circuit_source = record["path"]
            self.target_circuit_source_sha256 = record.get("source_sha256")
            self.target_circuit_sha256 = record.get("circuit_sha256") or _circuit_fingerprint(qc)
            return qc
        return None

    def _generate_positive_control_circuit(self):
        qc = QuantumCircuit(self.num_qubits)
        for q in range(self.num_qubits):
            qc.rx(float(0.1 * (q + 1)), q)
        for q in range(self.num_qubits):
            qc.rz(float(0.05 * (q + 1)), q)
        self.target_circuit_type = "positive_control_singleq_depth1"
        self.target_circuit_source = "positive_control"
        self.target_circuit_source_sha256 = None
        self.target_circuit_sha256 = _circuit_fingerprint(qc)
        return qc

    # -------------------------
    # Calibration loading (RESTORED)
    # -------------------------
    def _load_calibration_snapshot(self, calibration_file: str):
        """
        Load an IBM-style calibration JSON and extract:
          - per-qubit: readout_error, T1, T2, avg 1Q gate_error
          - 2Q edges (cz/cx/rzz): gate_error (used as 2Q depolarizing strength proxy)
        """
        if calibration_file is None:
            return {}

        path = calibration_file
        if not os.path.exists(path):
            # Try relative to this file / cwd
            path = os.path.join(os.getcwd(), calibration_file)
        if not os.path.exists(path):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), calibration_file)
        if not os.path.exists(path):
            if self.debug:
                print(f"[calib] file not found: {calibration_file}")
            return {}

        try:
            with open(path, "r") as f:
                raw = json.load(f)
        except Exception as e:
            if self.debug:
                print(f"[calib] failed to load JSON: {e}")
            return {}

        # Helper to read per-qubit property values
        def _get_qubit_prop(qubit_props, name: str, default=None):
            try:
                for item in qubit_props:
                    if item.get("name") == name:
                        return item.get("value", default)
            except Exception:
                pass
            return default

        n_backend = len(raw.get("qubits", []))
        # Per-backend-qubit arrays
        readout = [0.02] * n_backend
        t1 = [50e-6] * n_backend
        t2 = [50e-6] * n_backend

        for q in range(n_backend):
            props = raw["qubits"][q]
            ro = _get_qubit_prop(props, "readout_error", 0.02)
            T1 = _get_qubit_prop(props, "T1", 50e-6)
            T2 = _get_qubit_prop(props, "T2", 50e-6)
            # Calibration typically reports in seconds
            try:
                readout[q] = float(ro) if ro is not None else 0.02
            except Exception:
                readout[q] = 0.02
            try:
                t1[q] = float(T1) if T1 is not None else 50e-6
            except Exception:
                t1[q] = 50e-6
            try:
                t2[q] = float(T2) if T2 is not None else t1[q]
            except Exception:
                t2[q] = t1[q]

        # Collect 1Q gate errors per backend qubit
        oneq_errs = [[] for _ in range(n_backend)]
        # Collect 2Q gate edges + errors
        edges_2q = []  # (qa, qb)
        edge_err = {}  # (min, max) -> min gate_error observed

        for g in raw.get("gates", []):
            qs = g.get("qubits", [])
            gate = g.get("gate", "")
            params = {p.get("name"): p.get("value") for p in g.get("parameters", [])}

            if not isinstance(qs, list):
                continue

            if len(qs) == 1 and gate in {"sx", "x", "rz", "rx", "id"}:
                q = int(qs[0])
                ge = params.get("gate_error", None)
                if ge is not None:
                    try:
                        oneq_errs[q].append(float(ge))
                    except Exception:
                        pass

            if len(qs) == 2 and gate in {"cz", "cx", "rzz", "ecr"}:
                a, b = int(qs[0]), int(qs[1])
                if a == b:
                    continue
                edges_2q.append((a, b))
                ge = params.get("gate_error", None)
                if ge is None:
                    continue
                key = (min(a, b), max(a, b))
                try:
                    ge = float(ge)
                except Exception:
                    continue
                if key not in edge_err:
                    edge_err[key] = ge
                else:
                    edge_err[key] = min(edge_err[key], ge)

        oneq_avg = [float(np.mean(v)) if len(v) else None for v in oneq_errs]

        # Build an undirected graph for connectivity
        undirected = set((min(a, b), max(a, b)) for (a, b) in edges_2q if a != b)

        # Choose a connected subgraph of size `self.num_qubits` to avoid disconnected action graphs.
        degrees = [0] * n_backend
        adj = [[] for _ in range(n_backend)]
        for a, b in undirected:
            degrees[a] += 1
            degrees[b] += 1
            adj[a].append(b)
            adj[b].append(a)

        start = None
        if n_backend > 0:
            max_deg = max(degrees) if degrees else 0
            cands = [i for i, d in enumerate(degrees) if d == max_deg]
            start = int(min(cands)) if cands else 0

        selected = []
        if start is not None:
            q = collections.deque([start])
            seen = set([start])
            while q and len(selected) < self.num_qubits:
                u = q.popleft()
                selected.append(u)
                for v in sorted(adj[u]):
                    if v not in seen:
                        seen.add(v)
                        q.append(v)

        # If still short (e.g., empty edges), just take the first N backend qubits
        if len(selected) < self.num_qubits:
            for u in range(n_backend):
                if u not in selected:
                    selected.append(u)
                if len(selected) >= self.num_qubits:
                    break

        selected = selected[: self.num_qubits]
        backend_to_local = {b: i for i, b in enumerate(selected)}

        # Local coupling edges and 2Q error map
        local_edges = set()
        local_edge_err = {}
        for (a, b) in undirected:
            if a in backend_to_local and b in backend_to_local:
                la, lb = backend_to_local[a], backend_to_local[b]
                if la == lb:
                    continue
                local_edges.add((la, lb))
                key = (min(a, b), max(a, b))
                if key in edge_err:
                    local_edge_err[(min(la, lb), max(la, lb))] = float(edge_err[key])

        # Local per-qubit values
        local_readout = []
        local_t1 = []
        local_t2 = []
        local_1q = []
        for b in selected:
            local_readout.append(float(readout[b]) if readout[b] is not None else 0.02)
            local_t1.append(float(t1[b]) if t1[b] is not None else 50e-6)
            local_t2.append(float(t2[b]) if t2[b] is not None else float(t1[b]) if t1[b] is not None else 50e-6)
            local_1q.append(float(oneq_avg[b]) if oneq_avg[b] is not None else None)

        return {
            "backend_name": raw.get("backend_name", None),
            "last_update_date": raw.get("last_update_date", None),
            "selected_backend_qubits": selected,  # local index -> backend qubit index
            "backend_to_local": backend_to_local,
            "edges_undirected_local": sorted(list({(min(a, b), max(a, b)) for (a, b) in local_edges})),
            "edge_error_undirected_local": local_edge_err,
            "readout_error_local": local_readout,
            "t1_local": local_t1,
            "t2_local": local_t2,
            "oneq_error_local": local_1q,
        }

    def _init_hardware_from_calibration(self):
        """
        Build coupling_map + candidate physical edges from the calibration snapshot and populate
        noise tensors from the same snapshot.
        """
        calib = getattr(self, "_calib_snapshot", None) or {}

        edges = list(calib.get("edges_undirected_local", []))

        # CouplingMap expects directed edges.
        directed = []
        for (a, b) in edges:
            a = int(a); b = int(b)
            if a == b:
                continue
            directed.append((a, b))
            directed.append((b, a))

        try:
            self.coupling_map = CouplingMap(directed) if directed else None
        except Exception:
            self.coupling_map = None

        # Candidate edges for SWAP actions: undirected unique pairs
        self._physical_edges = sorted(list({(min(int(a), int(b)), max(int(a), int(b))) for (a, b) in edges if int(a) != int(b)}))

        # If empty, fall back to a simple chain so action space is non-empty
        if not self._physical_edges:
            self._physical_edges = [(i, i + 1) for i in range(self.num_qubits - 1)]

        # Populate per-qubit noise tensors
        ro = calib.get("readout_error_local", None)
        t1 = calib.get("t1_local", None)
        t2 = calib.get("t2_local", None)
        oneq = calib.get("oneq_error_local", None)
        edge_err = calib.get("edge_error_undirected_local", {})

        for q in range(self.num_qubits):
            if ro and q < len(ro):
                self.error_rates[q] = float(ro[q])
            if t1 and q < len(t1):
                # Use min(T1, T2) as a simple coherence proxy
                t1q = float(t1[q])
                t2q = float(t2[q]) if (t2 and q < len(t2)) else t1q
                self.coherence_times[q] = float(max(self.COH_MIN_S, min(self.COH_MAX_S, min(t1q, t2q))))
            if oneq and q < len(oneq) and oneq[q] is not None:
                self.oneq_error_rates[q] = float(oneq[q])
            else:
                # fallback: small fraction of readout error
                self.oneq_error_rates[q] = float(min(0.02, max(0.0, 0.5 * float(self.error_rates[q].item()))))

        # 2Q edge errors (undirected local)
        self.twoq_error_rates = {}
        for (a, b), ge in (edge_err or {}).items():
            a = int(a); b = int(b)
            if 0 <= a < self.num_qubits and 0 <= b < self.num_qubits and a != b:
                self.twoq_error_rates[(min(a, b), max(a, b))] = float(ge)

    def _store_base_noise_snapshot(self):
        self._base_error_rates = self.error_rates.detach().cpu().clone()
        self._base_coherence_times = self.coherence_times.detach().cpu().clone()
        self._base_oneq_error_rates = self.oneq_error_rates.detach().cpu().clone()
        self._base_twoq_error_rates = {tuple(k): float(v) for k, v in self.twoq_error_rates.items()}

    def _jitter_positive_tensor(self, base_tensor: torch.Tensor, frac: float, floor: float, ceil: Optional[float] = None):
        if frac <= 0.0:
            out = base_tensor.clone()
        else:
            eps = self.rng.normal(loc=0.0, scale=frac, size=tuple(base_tensor.shape))
            scale = np.clip(1.0 + eps, 0.05, None)
            out = base_tensor.clone() * torch.tensor(scale, dtype=base_tensor.dtype)
        out = torch.clamp(out, min=float(floor))
        if ceil is not None:
            out = torch.clamp(out, max=float(ceil))
        return out.to(self.device)

    def _sample_episode_noise_state(self):
        self.error_rates = self._jitter_positive_tensor(
            self._base_error_rates,
            frac=self.readout_jitter_frac,
            floor=1e-5,
            ceil=0.25,
        )
        self.coherence_times = self._jitter_positive_tensor(
            self._base_coherence_times,
            frac=self.coherence_jitter_frac,
            floor=self.COH_MIN_S,
            ceil=self.COH_MAX_S,
        )
        self.oneq_error_rates = self._jitter_positive_tensor(
            self._base_oneq_error_rates,
            frac=self.oneq_jitter_frac,
            floor=1e-6,
            ceil=0.10,
        )

        self.twoq_error_rates = {}
        for key, base_val in self._base_twoq_error_rates.items():
            if self.twoq_jitter_frac <= 0.0:
                sample = float(base_val)
            else:
                sample = float(base_val) * float(max(0.05, 1.0 + self.rng.normal(0.0, self.twoq_jitter_frac)))
            self.twoq_error_rates[tuple(key)] = float(min(0.30, max(1e-6, sample)))

    # -------------------------
    # Action space helpers
    # -------------------------
    def set_action_set(self, action_set_name: str):
        if action_set_name != "routing_only":
            raise ValueError("This env only supports action_set_name='routing_only' (Routing-by-SWAP).")
        self.action_set_name = action_set_name

    def get_action_size(self):
        return int(len(self.edge_action_types) * self.max_candidate_edges)

    def _get_candidate_edges(self):
        """Return the fixed physical edge list used for action indexing."""
        return self._physical_edges

    def get_action_mask(self):
        """Binary action mask (1=allowed) matching the fixed edge ordering."""
        mask = np.zeros(self.get_action_size(), dtype=np.float32)
        for idx, (a, b) in enumerate(self._physical_edges):
            a = int(a); b = int(b)
            if (not bool(self.deactivated_qubits[a].item())) and (not bool(self.deactivated_qubits[b].item())):
                mask[idx] = 1.0
        return mask

    def _decode_action(self, action_id: int):
        try:
            action_id = int(action_id)
        except Exception:
            return "__invalid__", None

        if action_id < 0 or action_id >= self.get_action_size():
            return "__invalid__", None

        op_idx = action_id // self.max_candidate_edges
        edge_idx = action_id % self.max_candidate_edges

        if op_idx != 0:
            return "__invalid__", None

        pa, pb = self._physical_edges[edge_idx]
        return "swap", (int(pa), int(pb))

    def _is_action_currently_valid(self, action_id: int) -> bool:
        try:
            action_id = int(action_id)
        except Exception:
            return False

        if action_id < 0 or action_id >= self.get_action_size():
            return False

        try:
            mask = self.get_action_mask()
        except Exception:
            return False
        return bool(mask[action_id] > 0.0)

    # -------------------------
    # Routing mechanics
    # -------------------------
    def _apply_physical_swap(self, pa: int, pb: int):
        """
        Update the token-swapping mapping for a physical SWAP(pa, pb).
        """
        pa = int(pa); pb = int(pb)
        la = int(self.physical_to_logical[pa])
        lb = int(self.physical_to_logical[pb])

        # Swap the logical tokens at those physical positions
        self.physical_to_logical[pa], self.physical_to_logical[pb] = lb, la
        self.logical_to_physical[la], self.logical_to_physical[lb] = pb, pa

    def _is_adjacent(self, a: int, b: int) -> bool:
        return (int(a), int(b)) in self._adjacency
        
    def _shortest_path_length(self, src: int, dst: int) -> int:
        if src == dst:
            return 0
        from collections import deque
        q = deque([(int(src), 0)])
        seen = {int(src)}
        while q:
            u, d = q.popleft()
            for v in self._neighbors[u]:
                if v == int(dst):
                    return d + 1
                if v not in seen:
                    seen.add(v)
                    q.append((v, d + 1))
        return self.num_qubits  # disconnected fallback


    def _flush_executable_ops(self):
        """
        Append as many next target instructions as possible given the current mapping.

        Returns:
            executed (int): number of logical instructions appended.
        """
        if self.target_circuit is None or self.current_circuit is None:
            return 0

        executed = 0
        tgt = self.target_circuit

        while self._op_cursor < len(tgt.data):
            instr = tgt.data[self._op_cursor]
            op = instr.operation

            # Map target qubits -> physical qubits under current token mapping
            logical_idxs = [tgt.find_bit(q).index for q in instr.qubits]
            phys = [int(self.logical_to_physical[li]) for li in logical_idxs]

            # Safety: deactivated qubits cannot be used
            if any(bool(self.deactivated_qubits[p].item()) for p in phys):
                break

            if len(phys) <= 1:
                qargs = [self.current_circuit.qubits[phys[0]]] if len(phys) == 1 else []
                self.current_circuit.append(op, qargs, list(instr.clbits))
                self._op_cursor += 1
                executed += 1
                continue

            if len(phys) == 2:
                a, b = phys[0], phys[1]
                if self._is_adjacent(a, b):
                    qargs = [self.current_circuit.qubits[a], self.current_circuit.qubits[b]]
                    self.current_circuit.append(op, qargs, list(instr.clbits))
                    self._op_cursor += 1
                    executed += 1
                    continue
                break

            # >2Q ops are not supported in this simple env; stop.
            break

        return executed

    # -------------------------
    # Observation
    # -------------------------
    def _next_gate_features(self):
        """
        Return a small hint about the next *blocking* 2Q gate (if any).
        """
        onehot = torch.zeros(self.num_qubits, dtype=torch.float32, device=self.device)
        phys_pair = torch.zeros(2, dtype=torch.float32, device=self.device)
        distance = torch.tensor([1.0], dtype=torch.float32, device=self.device)

        if self.target_circuit is None:
            return onehot, phys_pair, distance

        # Find next *blocking* 2Q gate from cursor (since 1Q gates are auto-flushed)
        tgt = self.target_circuit
        idx = self._op_cursor

        # Only treat true 2-qubit operations as the "next routing target"
        while idx < len(tgt.data):
            instr = tgt.data[idx]
            if len(instr.qubits) == 2:
                break
            idx += 1

        if idx >= len(tgt.data):
            return onehot, phys_pair, torch.tensor([0.0], dtype=torch.float32, device=self.device)

        instr = tgt.data[idx]
        logical_idxs = [tgt.find_bit(q).index for q in instr.qubits]
        if len(logical_idxs) < 2:
            return onehot, phys_pair, distance

        a = int(self.logical_to_physical[int(logical_idxs[0])])
        b = int(self.logical_to_physical[int(logical_idxs[1])])
        onehot[a] = 1.0
        onehot[b] = 1.0


        denom = max(1.0, float(self.num_qubits - 1))
        phys_pair[0] = float(a) / denom
        phys_pair[1] = float(b) / denom

        # crude distance feature: 0 if adjacent else 1 (normalized)
        d = float(self._shortest_path_length(a, b))
        den = max(1.0, float(self.num_qubits - 1))
        distance = torch.tensor([d / den], dtype=torch.float32, device=self.device)

        return onehot, phys_pair, distance

    def _next_blocking_logical_pair(self):
        if self.target_circuit is None:
            return None

        idx = self._op_cursor
        while idx < len(self.target_circuit.data):
            instr = self.target_circuit.data[idx]
            if len(instr.qubits) == 2:
                logical_idxs = [self.target_circuit.find_bit(q).index for q in instr.qubits]
                if len(logical_idxs) == 2:
                    return int(logical_idxs[0]), int(logical_idxs[1])
                break
            idx += 1
        return None

    def select_greedy_swap_action(self):
        logical_pair = self._next_blocking_logical_pair()
        if logical_pair is None:
            return 0

        l1, l2 = logical_pair
        best_idx = None
        best_score = None

        for idx, (pa, pb) in enumerate(self._physical_edges):
            if not self._is_action_currently_valid(idx):
                continue

            p1 = int(self.logical_to_physical[l1])
            p2 = int(self.logical_to_physical[l2])
            if p1 == int(pa):
                p1 = int(pb)
            elif p1 == int(pb):
                p1 = int(pa)
            if p2 == int(pa):
                p2 = int(pb)
            elif p2 == int(pb):
                p2 = int(pa)

            dist = self._shortest_path_length(p1, p2)
            edge_err = float(self.twoq_error_rates.get((min(int(pa), int(pb)), max(int(pa), int(pb))), 0.01))
            score = (int(dist), float(edge_err), int(idx))
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            return 0
        return int(best_idx)

    def _lookahead_demand_distance_per_physical(self, front_k: int = 6, ext_k: int = 24) -> torch.Tensor:
        """
        Per-physical-node localized 'demand distance' feature.
        """
        Q = int(self.num_qubits)
        out = torch.zeros(Q, dtype=torch.float32, device=self.device)

        if self.target_circuit is None:
            return out

        tgt = self.target_circuit
        start = int(self._op_cursor)

        pairs = []
        idx = start
        while idx < len(tgt.data) and len(pairs) < ext_k:
            instr = tgt.data[idx]
            if len(instr.qubits) == 2:
                logical_idxs = [tgt.find_bit(q).index for q in instr.qubits]
                if len(logical_idxs) == 2:
                    pairs.append((int(logical_idxs[0]), int(logical_idxs[1])))
            idx += 1

        if not pairs:
            return out

        best_partner = [-1] * Q
        best_rank = [10**9] * Q

        for k, (l1, l2) in enumerate(pairs):
            rank = k if k < front_k else (k + front_k)
            if rank < best_rank[l1]:
                best_rank[l1] = rank
                best_partner[l1] = l2
            if rank < best_rank[l2]:
                best_rank[l2] = rank
                best_partner[l2] = l1

        denom = max(1.0, float(Q - 1))
        for p in range(Q):
            l = int(self.physical_to_logical[p])
            partner_l = int(best_partner[l]) if 0 <= l < Q else -1
            if partner_l < 0:
                out[p] = 0.0
                continue
            partner_p = int(self.logical_to_physical[partner_l])
            d = float(self._shortest_path_length(int(p), int(partner_p)))
            out[p] = float(d / denom)

        return out

    def _get_observation(self):
        er_max = torch.clamp(self.error_rates.max(), min=1e-8)
        coh_max = torch.clamp(self.coherence_times.max(), min=1e-12)

        error_norm = (self.error_rates / er_max).to(torch.float32)
        coh_norm = (self.coherence_times / coh_max).to(torch.float32)

        Q = int(self.num_qubits)
        twoq_by_qubit = [[] for _ in range(Q)]
        for (a, b), err in self.twoq_error_rates.items():
            a = int(a)
            b = int(b)
            if 0 <= a < Q and 0 <= b < Q:
                twoq_by_qubit[a].append(float(err))
                twoq_by_qubit[b].append(float(err))
        twoq_feature = torch.tensor(
            [float(np.mean(vals)) if vals else 0.0 for vals in twoq_by_qubit],
            dtype=torch.float32,
            device=self.device,
        )
        twoq_max = torch.clamp(twoq_feature.max(), min=1e-8)
        twoq_norm = (twoq_feature / twoq_max).to(torch.float32)

        mask = set(self.calibration_feature_mask)
        if self.zero_noise_features or "topology_only" in mask or "all" in mask:
            error_norm = torch.zeros_like(error_norm)
            coh_norm = torch.zeros_like(coh_norm)
            twoq_norm = torch.zeros_like(twoq_norm)
        else:
            if "no_readout" in mask or "readout" in mask:
                error_norm = torch.zeros_like(error_norm)
            if "no_coherence" in mask or "coherence" in mask:
                coh_norm = torch.zeros_like(coh_norm)
            if "no_twoq" in mask or "twoq" in mask:
                twoq_norm = torch.zeros_like(twoq_norm)

        denom = max(1.0, float(Q - 1))
        hosted_logical_norm = torch.tensor(
            [float(int(self.physical_to_logical[p])) / denom for p in range(Q)],
            dtype=torch.float32,
            device=self.device,
        )

        demand_dist = self._lookahead_demand_distance_per_physical(front_k=6, ext_k=24)

        next_onehot, next_phys_pair, next_dist = self._next_gate_features()

        prog = 0.0
        if self.target_circuit is not None and len(self.target_circuit.data) > 0:
            prog = float(self._op_cursor) / float(len(self.target_circuit.data))
        prog_t = torch.tensor([prog], dtype=torch.float32, device=self.device)

        obs = torch.cat(
            [error_norm, coh_norm, twoq_norm, hosted_logical_norm, demand_dist, next_onehot, next_phys_pair, next_dist, prog_t],
            dim=0,
        )
        return obs.cpu().numpy()


    # -------------------------
    # Target generation
    # -------------------------
    def _generate_target_circuit(self):
        self.target_circuit_source = "synthetic"
        self.target_circuit_source_sha256 = None
        self.target_circuit_sha256 = None
        if self.positive_control_prob > 0.0 and self.rng.uniform() < self.positive_control_prob:
            return self._generate_positive_control_circuit()
        if self._benchmark_qasm_records and self.rng.uniform() < self.benchmark_corpus_prob:
            benchmark_qc = self._generate_benchmark_qasm_circuit()
            if benchmark_qc is not None:
                return benchmark_qc
        elif self.benchmark_corpus_prob >= 1.0 and (self.benchmark_qasm_files or self.benchmark_qasm_dir):
            raise RuntimeError(
                "BENCHMARK_CORPUS_PROB=1 requested an external QASM circuit, but no compatible "
                "QASM records are available. Refusing to fall back to synthetic circuits."
            )

        qc = QuantumCircuit(self.num_qubits)
        Q = self.num_qubits

        target_type = self.rng.choice(
            ["random_2q", "qaoa", "quantum_volume", "vqe", "clifford"],
            p=self._target_type_probs,
        )
        self.target_circuit_type = str(target_type)

        if target_type == "random_2q":
            layers = int(3 * Q)
            for _ in range(layers):
                for q in range(Q):
                    if self.rng.uniform() < 0.5:
                        qc.rz(float(self.rng.uniform(0, 2*np.pi)), q)
                    if self.rng.uniform() < 0.5:
                        qc.rx(float(self.rng.uniform(0, 2*np.pi)), q)

                m = max(1, Q // 2)
                for _ in range(m):
                    a, b = self.rng.choice(Q, size=2, replace=False)
                    if self.rng.uniform() < 0.5:
                        qc.cx(int(a), int(b))
                    else:
                        qc.rzz(float(self.rng.uniform(0, 2*np.pi)), int(a), int(b))

        elif target_type == "qaoa":
            p = 3
            for _ in range(p):
                gamma = float(self.rng.uniform(0, 2*np.pi))
                beta = float(self.rng.uniform(0, 2*np.pi))

                pairs = []
                idxs = list(range(Q))
                self.rng.shuffle(idxs)
                for i in range(0, Q - 1, 2):
                    pairs.append((idxs[i], idxs[i+1]))
                for a, b in pairs:
                    qc.rzz(gamma, int(a), int(b))

                for q in range(Q):
                    qc.rx(beta, q)

        elif target_type == "quantum_volume":
            try:
                from qiskit.circuit.library import QuantumVolume

                qc = QuantumVolume(num_qubits=Q, depth=max(1, Q), seed=int(self._episode_seed))
            except Exception:
                layers = int(2 * Q)
                for _ in range(layers):
                    idxs = np.arange(Q)
                    self.rng.shuffle(idxs)
                    for i in range(0, Q - 1, 2):
                        a = int(idxs[i])
                        b = int(idxs[i + 1])
                        qc.sx(a)
                        qc.rz(float(self.rng.uniform(0, 2 * np.pi)), a)
                        qc.sx(b)
                        qc.rz(float(self.rng.uniform(0, 2 * np.pi)), b)
                        qc.cx(a, b)

        elif target_type == "vqe":
            layers = 3
            for _ in range(layers):
                for q in range(Q):
                    qc.ry(float(self.rng.uniform(0, 2 * np.pi)), q)
                    qc.rz(float(self.rng.uniform(0, 2 * np.pi)), q)
                for q in range(Q - 1):
                    qc.cx(q, q + 1)
                if Q > 2:
                    qc.cx(Q - 1, 0)

        else:
            layers = int(3 * Q)
            for _ in range(layers):
                for q in range(Q):
                    r = self.rng.integers(0, 3)
                    if r == 0:
                        qc.h(q)
                    elif r == 1:
                        qc.s(q)
                    else:
                        qc.sdg(q)

                m = max(1, Q // 2)
                for _ in range(m):
                    a, b = self.rng.choice(Q, size=2, replace=False)
                    qc.cx(int(a), int(b))

        return qc


    # -------------------------
    # Noise model
    # -------------------------
    def _build_noise_model_from_state(self):
        noise_model = NoiseModel()
        t_1q = 50e-9
        t_2q = 300e-9

        for q in range(self.num_qubits):
            p_ro = float(torch.clamp(self.error_rates[q], 0.0, 0.25).detach().cpu().item())
            tcoh = float(torch.clamp(self.coherence_times[q], self.COH_MIN_S, self.COH_MAX_S).detach().cpu().item())
            t1 = tcoh
            t2 = tcoh

            ro = ReadoutError([[1.0 - p_ro, p_ro], [p_ro, 1.0 - p_ro]])
            noise_model.add_readout_error(ro, [q])

            p_1q = float(torch.clamp(self.oneq_error_rates[q], 0.0, 0.10).detach().cpu().item())
            if p_1q > 0:
                de = depolarizing_error(p_1q, 1)
                for g in ["x", "y", "z", "h", "rx", "ry", "rz", "sx", "u", "u1", "u2", "u3"]:
                    try:
                        noise_model.add_quantum_error(de, g, [q])
                    except Exception:
                        pass

            try:
                te = thermal_relaxation_error(t1, t2, t_1q)
                noise_model.add_quantum_error(te, "id", [q])
            except Exception:
                pass

        for (a, b) in self._physical_edges:
            a = int(a); b = int(b)
            key = (min(a, b), max(a, b))
            p_2q = float(self.twoq_error_rates.get(key, None) or 0.01)
            p_2q = float(max(0.0, min(0.30, p_2q)))

            combined_2q_error = None
            if p_2q > 0:
                try:
                    combined_2q_error = depolarizing_error(p_2q, 2)
                except Exception:
                    combined_2q_error = None

            try:
                t1_a = float(torch.clamp(self.coherence_times[a], self.COH_MIN_S, self.COH_MAX_S).cpu().item())
                t2_a = t1_a
                t1_b = float(torch.clamp(self.coherence_times[b], self.COH_MIN_S, self.COH_MAX_S).cpu().item())
                t2_b = t1_b
                te_a = thermal_relaxation_error(t1_a, t2_a, t_2q)
                te_b = thermal_relaxation_error(t1_b, t2_b, t_2q)
                try:
                    te2 = te_a.tensor(te_b)
                except Exception:
                    te2 = te_a.expand(te_b)
                combined_2q_error = te2 if combined_2q_error is None else combined_2q_error.compose(te2)
            except Exception:
                pass

            if combined_2q_error is not None:
                for g in ["cx", "cz", "rzz", "swap"]:
                    try:
                        noise_model.add_quantum_error(combined_2q_error, g, [a, b])
                        noise_model.add_quantum_error(combined_2q_error, g, [b, a])
                    except Exception:
                        pass

        return noise_model

    # -------------------------
    # Compilation + metrics
    # -------------------------
    def _compiled_metrics(self, compiled_circuit):
        depth = int(compiled_circuit.depth()) if compiled_circuit is not None else 0
        ops = compiled_circuit.count_ops() if compiled_circuit is not None else {}

        twoq = 0
        for g in ("cx", "cz", "ecr", "swap", "iswap", "rzz"):
            twoq += int(ops.get(g, 0))

        cost = (self.cost_w_twoq * float(twoq)) + (self.cost_w_depth * float(depth))
        return {"depth": depth, "twoq": twoq, "cost": float(cost)}

    def _compile_agent_circuit(self, qc: QuantumCircuit):
        key = ("agent", _circuit_fingerprint(qc), int(self._episode_seed), int(self.optimization_level))
        cached = self._compile_cache.get(key)
        if cached is not None:
            return cached

        compiled = transpile(
            qc,
            coupling_map=self.coupling_map,
            initial_layout=list(range(self.num_qubits)),
            layout_method="trivial",
            routing_method=None,
            basis_gates=self.basis_gates,
            optimization_level=self.optimization_level,
            seed_transpiler=int(self._episode_seed),
        )

        self._compile_cache[key] = compiled
        return compiled

    def _compile_baseline(self, layout_method: str = "sabre", routing_method: str = "sabre", initial_layout=None):
        target_fp = _circuit_fingerprint(self.target_circuit) if self.target_circuit is not None else None
        key = (
            "baseline",
            target_fp,
            layout_method,
            routing_method,
            tuple(initial_layout) if initial_layout is not None else None,
            int(self._episode_seed),
            int(self.optimization_level),
        )
        cached = self._compile_cache.get(key)
        if cached is not None:
            return cached

        if layout_method == "sabre" and routing_method == "sabre":
            compiled = self._compile_sabre_baseline_best_of_n(initial_layout=initial_layout)
            self._compile_cache[key] = compiled
            return compiled

        if layout_method == "qiskit_noise_aware_vf2" and routing_method == "sabre":
            compiled = self._compile_qiskit_noise_aware_vf2_best_of_n(initial_layout=initial_layout)
            self._compile_cache[key] = compiled
            return compiled

        compiled = transpile(
            self.target_circuit,
            coupling_map=self.coupling_map,
            initial_layout=initial_layout,
            layout_method=layout_method,
            routing_method=routing_method,
            basis_gates=self.basis_gates,
            optimization_level=self.optimization_level,
            seed_transpiler=int(self._episode_seed),
        )
        self._compile_cache[key] = compiled
        return compiled

    def _compile_sabre_baseline_best_of_n(self, initial_layout=None):
        best_compiled = None
        best_score = None
        num_trials = max(1, int(self.sabre_baseline_trials))

        for trial_idx in range(num_trials):
            compiled = transpile(
                self.target_circuit,
                coupling_map=self.coupling_map,
                initial_layout=initial_layout,
                layout_method="sabre",
                routing_method="sabre",
                basis_gates=self.basis_gates,
                optimization_level=self.optimization_level,
                seed_transpiler=int(self._episode_seed) + int(trial_idx),
            )
            metrics = self._compiled_metrics(compiled)
            score = (
                float(metrics["cost"]),
                int(metrics["twoq"]),
                int(metrics["depth"]),
                compiled.size(),
            )
            if best_score is None or score < best_score:
                best_compiled = compiled
                best_score = score

        if best_compiled is None:
            raise RuntimeError("SABRE baseline compilation failed for all trials.")
        return best_compiled

    def _build_qiskit_noise_aware_target(self):
        try:
            from qiskit.circuit.library import (
                CXGate,
                CZGate,
                HGate,
                IGate,
                RXGate,
                RZGate,
                RZZGate,
                SXGate,
                SwapGate,
                XGate,
            )
            from qiskit.transpiler import InstructionProperties, Target
        except Exception:
            return None

        try:
            target = Target(num_qubits=self.num_qubits)

            def _add_1q(gate, error_scale=1.0):
                props = {}
                for q in range(self.num_qubits):
                    err = float(self.oneq_error_rates[q].detach().cpu().item()) * float(error_scale)
                    props[(q,)] = InstructionProperties(error=float(max(0.0, min(1.0, err))))
                target.add_instruction(gate, props)

            _add_1q(IGate(), error_scale=0.1)
            _add_1q(RZGate(0.0), error_scale=0.0)
            _add_1q(RXGate(0.0), error_scale=1.0)
            _add_1q(SXGate(), error_scale=1.0)
            _add_1q(XGate(), error_scale=1.0)
            _add_1q(HGate(), error_scale=1.5)

            twoq_props = {}
            swap_props = {}
            for a, b in self._physical_edges:
                a = int(a)
                b = int(b)
                err = float(self.twoq_error_rates.get((min(a, b), max(a, b)), 0.01))
                err = float(max(0.0, min(1.0, err)))
                twoq_props[(a, b)] = InstructionProperties(error=err)
                twoq_props[(b, a)] = InstructionProperties(error=err)
                swap_err = float(max(0.0, min(1.0, 3.0 * err)))
                swap_props[(a, b)] = InstructionProperties(error=swap_err)
                swap_props[(b, a)] = InstructionProperties(error=swap_err)

            if twoq_props:
                target.add_instruction(CXGate(), twoq_props)
                target.add_instruction(CZGate(), twoq_props)
                target.add_instruction(RZZGate(0.0), twoq_props)
                target.add_instruction(SwapGate(), swap_props)
            return target
        except Exception as e:
            if self.debug:
                print(f"[baseline] could not build noise-aware Qiskit target: {e}")
            return None

    def _find_qiskit_noise_aware_vf2_layout(self, target, seed: int):
        """Try VF2Layout directly against the noise-aware Target.

        Newer Qiskit versions expose VF2 as a pass, not a built-in
        `layout_method="vf2"` stage plugin.
        """
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import VF2Layout

        pm = PassManager(
            [
                VF2Layout(
                    target=target,
                    seed=int(seed),
                    max_trials=1,
                )
            ]
        )
        pm.run(self.target_circuit)
        return pm.property_set.get("layout"), pm.property_set.get("VF2Layout_stop_reason")

    def _compile_qiskit_noise_aware_vf2_best_of_n(self, initial_layout=None):
        target = self._build_qiskit_noise_aware_target()
        if target is None:
            raise RuntimeError("Qiskit Target with calibration error rates is unavailable.")

        best_compiled = None
        best_score = None
        num_trials = max(1, int(self.sabre_baseline_trials))
        vf2_pass_available = True

        for trial_idx in range(num_trials):
            seed = int(self._episode_seed) + int(trial_idx)
            chosen_layout = initial_layout
            if initial_layout is None and vf2_pass_available:
                try:
                    chosen_layout, stop_reason = self._find_qiskit_noise_aware_vf2_layout(
                        target=target,
                        seed=seed,
                    )
                except Exception as e:
                    vf2_pass_available = False
                    if self.debug or _env_flag("BASELINE_PROGRESS", False):
                        print(
                            "[baseline] qiskit_noise_aware_vf2 VF2Layout unavailable; "
                            "falling back to Target+sabre best-of-N.",
                            flush=True,
                        )
                else:
                    if chosen_layout is None and (self.debug or _env_flag("BASELINE_PROGRESS", False)):
                        print(
                            f"[baseline] qiskit_noise_aware_vf2 no perfect VF2 layout "
                            f"(stop_reason={stop_reason}); falling back to Target+sabre.",
                            flush=True,
                        )
            if chosen_layout is None:
                compiled = transpile(
                    self.target_circuit,
                    target=target,
                    initial_layout=initial_layout,
                    layout_method="sabre",
                    routing_method="sabre",
                    optimization_level=max(3, int(self.optimization_level)),
                    seed_transpiler=seed,
                )
            else:
                compiled = transpile(
                    self.target_circuit,
                    target=target,
                    initial_layout=chosen_layout,
                    routing_method="sabre",
                    optimization_level=max(3, int(self.optimization_level)),
                    seed_transpiler=seed,
                )
            metrics = self._compiled_metrics(compiled)
            proxy_fidelity = float(self._calculate_proxy_fidelity(compiled))
            score = (
                -proxy_fidelity,
                float(metrics["cost"]),
                int(metrics["twoq"]),
                int(metrics["depth"]),
                compiled.size(),
            )
            if best_score is None or score < best_score:
                best_compiled = compiled
                best_score = score

        if best_compiled is None:
            raise RuntimeError("Qiskit noise-aware VF2/SABRE baseline failed for all trials.")
        return best_compiled
        
    def _compute_ideal_target_dm(self):
        if self.target_circuit is None:
            return None

        qc_t = transpile(
            self.target_circuit,
            basis_gates=self.basis_gates,
            optimization_level=0,
            seed_transpiler=int(self._episode_seed),
            routing_method=None,
            layout_method="trivial",
            initial_layout=list(range(self.num_qubits)),
        )

        qc_t = qc_t.copy()
        qc_t.save_density_matrix()

        tr_t = transpile(qc_t, self._ideal_sim, optimization_level=0, seed_transpiler=int(self._episode_seed))
        res_t = self._ideal_sim.run(tr_t).result()
        return DensityMatrix(res_t.data(tr_t)["density_matrix"])

    # --- ADDED: Proxy Fidelity Calculation ---
    def _calculate_proxy_fidelity(self, qc: QuantumCircuit):
        """
        Fast O(N) estimation of circuit fidelity using Estimated Success Probability (ESP).
        """
        if qc is None:
            return 0.0
        
        log_success_prob = 0.0
        for instr in qc.data:
            q_indices = [qc.find_bit(q).index for q in instr.qubits]
            
            # 1-Qubit Gate Error
            if len(q_indices) == 1:
                err = float(self.oneq_error_rates[q_indices[0]].cpu().item())
                success = max(1e-10, 1.0 - err)
                log_success_prob += np.log(success)
                
            # 2-Qubit Gate Error
            elif len(q_indices) == 2:
                q1, q2 = min(q_indices), max(q_indices)
                err = self.twoq_error_rates.get((q1, q2), 0.01) # Default to 1% if missing
                success = max(1e-10, 1.0 - err)
                log_success_prob += np.log(success)
                
        return float(np.exp(log_success_prob))

    def get_current_fidelity(self, compiled_circuit: QuantumCircuit):
        """
        Routes the fidelity request with Automated Power-Law alignment.
        """
        if compiled_circuit is None: return 0.0

        if self.use_proxy_reward:
            raw_proxy = self._calculate_proxy_fidelity(compiled_circuit)
            n_gates = compiled_circuit.size()
            
            if n_gates == 0 or raw_proxy <= 1e-10:
                return float(raw_proxy)

            # POWER-LAW ALIGNMENT
            p_proxy = 1 - np.power(raw_proxy, 1/n_gates)
            p_aligned = np.clip(p_proxy * self._active_k_calib(), 0.0, 1.0) # clamp
            aligned_proxy = np.power(1 - p_aligned, n_gates)
            
            return float(np.clip(aligned_proxy, 0.0, 1.0))
            
        return self._calculate_fidelity(compiled_circuit)


    def _calculate_fidelity(self, compiled_circuit: QuantumCircuit):
        """
        Exact Fidelity Density Matrix Simulation.
        """
        if compiled_circuit is None:
            return 0.0

        ideal_dm = getattr(self, "_ideal_dm_target", None)
        if ideal_dm is None:
            ideal_dm = self._compute_ideal_target_dm()
            self._ideal_dm_target = ideal_dm

        fp = _circuit_fingerprint(compiled_circuit)

        noisy_dm = self._noisy_dm_cache.get(fp)
        if noisy_dm is None:
            qc_n = compiled_circuit.copy()
            qc_n.save_density_matrix()
            tr_n = transpile(qc_n, self._noisy_sim, optimization_level=0, seed_transpiler=int(self._episode_seed))
            res_n = self._noisy_sim.run(tr_n).result()
            noisy_dm = DensityMatrix(res_n.data(tr_n)["density_matrix"])
            self._noisy_dm_cache[fp] = noisy_dm

        try:
            return float(state_fidelity(noisy_dm, ideal_dm))
        except Exception:
            return float(state_fidelity(noisy_dm, ideal_dm, validate=False))


    # -------------------------
    # Reset/Step
    # -------------------------
    def reset(self, episode: Optional[int] = None):
        self.current_step = 0
        self.done = False
        self.invalid_gate_count = 0
        self.executed_ops_total = 0
        self.completed_target = False
        self.timed_out = False
        self.terminal_reason = "running"
        self.final_progress = 0.0

        if episode is None:
            episode = self._auto_episode
            self._auto_episode += 1
        else:
            episode = int(episode)
            self._auto_episode = max(self._auto_episode, episode + 1)

        self._episode_seed = int(episode)
        self.rng = np.random.default_rng(self._episode_seed)

        self._compile_cache = {}
        self._ideal_dm_cache = {}
        self._noisy_dm_cache = {}
        self._ideal_dm_target = None

        self._sample_episode_noise_state()
        generated_target = self._generate_target_circuit()
        self.target_circuit = self._normalize_target_circuit(
            generated_target,
            context=str(getattr(self, "target_circuit_type", "unknown")),
        )
        self.target_circuit_sha256 = _circuit_fingerprint(self.target_circuit)
        self.target_twoq_count = self._count_target_twoq_ops()
        self.target_depth = int(self.target_circuit.depth()) if self.target_circuit is not None else 0
        self.effective_max_steps = self._compute_effective_max_steps()
        self.max_steps_per_episode = int(self.effective_max_steps)

        self.logical_to_physical = [int(i) for i in range(self.num_qubits)]
        self.physical_to_logical = [int(i) for i in range(self.num_qubits)]

        self.current_circuit = QuantumCircuit(self.num_qubits, self.target_circuit.num_clbits)
        self.compiled_circuit = None
        self._op_cursor = 0

        self.noise_model = self._build_noise_model_from_state()
        self._noisy_sim = AerSimulator(method="density_matrix", noise_model=self.noise_model)
        self._ideal_sim = AerSimulator(method="density_matrix")

        self.executed_ops_total += int(self._flush_executable_ops())
        self.completed_target = bool(self.target_circuit is not None and self._op_cursor >= len(self.target_circuit.data))
        self.done = bool(self.completed_target)
        self.terminal_reason = "completed" if self.completed_target else "running"
        self.final_progress = self._current_progress()

        self._ideal_dm_target = self._compute_ideal_target_dm()

        self.baseline_compiled = self._compile_baseline(layout_method="sabre", routing_method="sabre")
        base_m = self._compiled_metrics(self.baseline_compiled)
        # REPLACED _calculate_fidelity WITH get_current_fidelity
        self.baseline_fidelity = float(self.get_current_fidelity(self.baseline_compiled))
        self.baseline_cost = float(base_m["cost"])

        self.last_metrics = {
            "fidelity": float(self.baseline_fidelity),
            "proxy_fidelity": float(self._calculate_proxy_fidelity(self.baseline_compiled)),
            "twoq": int(base_m["twoq"]),
            "depth": int(base_m["depth"]),
            "cost": float(base_m["cost"]),
            "raw_partial_twoq": float("nan"),
            "raw_partial_depth": float("nan"),
            "raw_partial_cost": float("nan"),
            **self._terminal_status_fields(),
        }

        if self.done:
            self.compiled_circuit = self._compile_agent_circuit(self.current_circuit)
            final_m = self._compiled_metrics(self.compiled_circuit)
            final_proxy = float(self._calculate_proxy_fidelity(self.compiled_circuit))
            self.last_metrics = {
                "fidelity": float(self.get_current_fidelity(self.compiled_circuit)),
                "proxy_fidelity": final_proxy,
                "twoq": int(final_m["twoq"]),
                "depth": int(final_m["depth"]),
                "cost": float(final_m["cost"]),
                "raw_partial_twoq": float("nan"),
                "raw_partial_depth": float("nan"),
                "raw_partial_cost": float("nan"),
                **self._terminal_status_fields(),
            }

        return self._get_observation()

    def step(self, action):
        if self.done:
            return self._get_observation(), 0.0, True, dict(self.last_metrics)

        self.current_step += 1

        action_type, details = self._decode_action(action)
        reward = 0.0
        executed = 0

        total_ops = max(1.0, float(len(self.target_circuit.data) if self.target_circuit is not None else 1))
        prev_progress = self._current_progress()
        prev_dist = float(self._next_gate_features()[2].detach().cpu().item())

        if action_type == "__invalid__" or details is None or (not self._is_action_currently_valid(action)):
            self.invalid_gate_count += 1
            reward -= self.invalid_action_penalty
        else:
            pa, pb = details
            self.current_circuit.swap(int(pa), int(pb))
            self._apply_physical_swap(pa, pb)

            executed = int(self._flush_executable_ops())
            self.executed_ops_total += int(executed)
            new_progress = self._current_progress()
            
            if self.reward_mode == "shaped":
                new_dist = float(self._next_gate_features()[2].detach().cpu().item())
                reward += self.distance_reduction_reward_scale * (prev_dist - new_dist)
                reward += self.progress_reward_scale * max(0.0, new_progress - prev_progress)
                reward += self.executed_gate_reward_scale * float(executed)
                reward -= self.swap_penalty * self.cost_w_twoq

        if self.target_circuit is not None and self._op_cursor >= len(self.target_circuit.data):
            self.completed_target = True
            self.terminal_reason = "completed"
            self.done = True

        if self.current_step >= self.max_steps_per_episode and not self.done:
            self.completed_target = False
            self.timed_out = True
            self.terminal_reason = "timeout"
            self.done = True
            if self.reward_mode == "shaped":
                reward -= self.timeout_penalty

        self.final_progress = self._current_progress()
        info = {
            "invalid": int(self.invalid_gate_count),
            "executed_ops": int(executed),
            "progress": float(self.final_progress),
            "fidelity": float(self.last_metrics.get("fidelity", 0.0)),
            "proxy_fidelity": float(self.last_metrics.get("proxy_fidelity", 0.0)),
            **self._terminal_status_fields(),
        }

        if self.done:
            if self.completed_target:
                self.compiled_circuit = self._compile_agent_circuit(self.current_circuit)
                m = self._compiled_metrics(self.compiled_circuit)
                fid = float(self.get_current_fidelity(self.compiled_circuit))
                proxy_fid = float(self._calculate_proxy_fidelity(self.compiled_circuit))

                final_delta = (
                    self.fidelity_scale * (fid - float(self.baseline_fidelity))
                    - self.cost_lambda * (float(m["cost"]) - float(self.baseline_cost))
                )
                reward += float(final_delta)

                self.final_progress = 1.0
                self.last_metrics = {
                    "fidelity": float(fid),
                    "proxy_fidelity": proxy_fid,
                    "twoq": int(m["twoq"]),
                    "depth": int(m["depth"]),
                    "cost": float(m["cost"]),
                    "raw_partial_twoq": float("nan"),
                    "raw_partial_depth": float("nan"),
                    "raw_partial_cost": float("nan"),
                    **self._terminal_status_fields(),
                }
            else:
                self.compiled_circuit = None
                raw_m = {"twoq": float("nan"), "depth": float("nan"), "cost": float("nan")}
                try:
                    partial = self._compile_agent_circuit(self.current_circuit)
                    raw_m = self._compiled_metrics(partial)
                except Exception:
                    pass
                missing = max(0.0, 1.0 - float(self.final_progress))
                if self.reward_mode == "shaped":
                    reward -= float(self.incomplete_episode_penalty) * missing

                self.last_metrics = {
                    "fidelity": 0.0,
                    "proxy_fidelity": 0.0,
                    "twoq": float("nan"),
                    "depth": float("nan"),
                    "cost": float("nan"),
                    "raw_partial_twoq": float(raw_m.get("twoq", float("nan"))),
                    "raw_partial_depth": float(raw_m.get("depth", float("nan"))),
                    "raw_partial_cost": float(raw_m.get("cost", float("nan"))),
                    **self._terminal_status_fields(),
                }
            info.update(self.last_metrics)

        return self._get_observation(), float(reward), bool(self.done), info

    # -------------------------
    # Baseline reporting
    # -------------------------
    def evaluate_baselines(self, enabled_baselines=None):
        if self.target_circuit is None:
            raise RuntimeError("Call reset() before evaluate_baselines().")
        enabled = None
        if enabled_baselines is not None:
            enabled = {str(item).strip() for item in enabled_baselines if str(item).strip()}

        def _should_run(name: str) -> bool:
            return enabled is None or name in enabled

        progress = _env_flag("BASELINE_PROGRESS", False)

        def _log(message: str):
            if progress:
                print(f"[baseline] {message}", flush=True)

        agent_compiled = self.compiled_circuit if self.compiled_circuit is not None else None
        if agent_compiled is None:
            agent_f = float(self.last_metrics.get("fidelity", self.baseline_fidelity))
            agent_m = {
                "twoq": float(self.last_metrics.get("twoq", float("nan"))),
                "depth": float(self.last_metrics.get("depth", float("nan"))),
                "cost": float(self.last_metrics.get("cost", self.baseline_cost)),
            }
        else:
            agent_m = self._compiled_metrics(agent_compiled)
            # REPLACED _calculate_fidelity WITH get_current_fidelity
            agent_f = float(self.get_current_fidelity(agent_compiled))
        agent_proxy_f = (
            float(self._calculate_proxy_fidelity(agent_compiled))
            if agent_compiled is not None
            else float(self.last_metrics.get("proxy_fidelity", float("nan")))
        )

        results = {
            "agent": {
                "fidelity": agent_f,
                "proxy_fidelity": agent_proxy_f,
                "twoq": agent_m["twoq"],
                "depth": agent_m["depth"],
                "cost": agent_m["cost"],
                "completed": bool(self.last_metrics.get("completed_target", False)),
                "progress": float(self.last_metrics.get("progress", float("nan"))),
                "terminal_reason": str(self.last_metrics.get("terminal_reason", "unknown")),
            },
        }

        if _should_run("trivial"):
            _log("trivial start")
            t0 = time.perf_counter()
            trivial = self._compile_baseline(
                initial_layout=list(range(self.num_qubits)),
                layout_method="trivial",
                routing_method="basic",
            )
            trivial_m = self._compiled_metrics(trivial)
            trivial_f = float(self.get_current_fidelity(trivial))
            trivial_proxy_f = float(self._calculate_proxy_fidelity(trivial))
            trivial_seconds = float(time.perf_counter() - t0)
            results["trivial"] = {"fidelity": trivial_f, "proxy_fidelity": trivial_proxy_f, "twoq": trivial_m["twoq"], "depth": trivial_m["depth"], "cost": trivial_m["cost"], "wall_seconds": trivial_seconds}
            _log(f"trivial done wall_seconds={trivial_seconds:.3f}")

        if _should_run("sabre"):
            _log("sabre start")
            t0 = time.perf_counter()
            sabre = self._compile_baseline(layout_method="sabre", routing_method="sabre")
            sabre_m = self._compiled_metrics(sabre)
            sabre_f = float(self.get_current_fidelity(sabre))
            sabre_proxy_f = float(self._calculate_proxy_fidelity(sabre))
            sabre_seconds = float(time.perf_counter() - t0)
            results["sabre"] = {"fidelity": sabre_f, "proxy_fidelity": sabre_proxy_f, "twoq": sabre_m["twoq"], "depth": sabre_m["depth"], "cost": sabre_m["cost"], "wall_seconds": sabre_seconds}
            _log(f"sabre done wall_seconds={sabre_seconds:.3f}")

        if _should_run("sabre_trivial_layout"):
            _log("sabre_trivial_layout start")
            t0 = time.perf_counter()
            sabre_trivial_layout = self._compile_baseline(
                initial_layout=list(range(self.num_qubits)),
                layout_method="trivial",
                routing_method="sabre",
            )
            sabre_trivial_m = self._compiled_metrics(sabre_trivial_layout)
            sabre_trivial_f = float(self.get_current_fidelity(sabre_trivial_layout))
            sabre_trivial_proxy_f = float(self._calculate_proxy_fidelity(sabre_trivial_layout))
            sabre_trivial_seconds = float(time.perf_counter() - t0)
            results["sabre_trivial_layout"] = {
                "fidelity": sabre_trivial_f,
                "proxy_fidelity": sabre_trivial_proxy_f,
                "twoq": sabre_trivial_m["twoq"],
                "depth": sabre_trivial_m["depth"],
                "cost": sabre_trivial_m["cost"],
                "wall_seconds": sabre_trivial_seconds,
            }
            _log(f"sabre_trivial_layout done wall_seconds={sabre_trivial_seconds:.3f}")

        if _should_run("lookahead"):
            try:
                _log("lookahead start")
                t0 = time.perf_counter()
                lookahead = self._compile_baseline(layout_method="sabre", routing_method="lookahead")
                lookahead_m = self._compiled_metrics(lookahead)
                lookahead_f = float(self.get_current_fidelity(lookahead))
                lookahead_seconds = float(time.perf_counter() - t0)
                results["lookahead"] = {
                    "fidelity": lookahead_f,
                    "proxy_fidelity": float(self._calculate_proxy_fidelity(lookahead)),
                    "twoq": lookahead_m["twoq"],
                    "depth": lookahead_m["depth"],
                    "cost": lookahead_m["cost"],
                    "wall_seconds": lookahead_seconds,
                }
                _log(f"lookahead done wall_seconds={lookahead_seconds:.3f}")
            except Exception as e:
                if self.debug:
                    print(f"[baseline] lookahead unavailable: {e}")
                _log(f"lookahead unavailable: {e}")

        if _should_run("qiskit_noise_aware_vf2"):
            try:
                _log("qiskit_noise_aware_vf2 start")
                t0 = time.perf_counter()
                noise_aware = self._compile_baseline(layout_method="qiskit_noise_aware_vf2", routing_method="sabre")
                noise_aware_m = self._compiled_metrics(noise_aware)
                noise_aware_f = float(self.get_current_fidelity(noise_aware))
                noise_aware_seconds = float(time.perf_counter() - t0)
                results["qiskit_noise_aware_vf2"] = {
                    "fidelity": noise_aware_f,
                    "proxy_fidelity": float(self._calculate_proxy_fidelity(noise_aware)),
                    "twoq": noise_aware_m["twoq"],
                    "depth": noise_aware_m["depth"],
                    "cost": noise_aware_m["cost"],
                    "wall_seconds": noise_aware_seconds,
                }
                _log(f"qiskit_noise_aware_vf2 done wall_seconds={noise_aware_seconds:.3f}")
            except Exception as e:
                if self.debug:
                    print(f"[baseline] qiskit_noise_aware_vf2 unavailable: {e}")
                _log(f"qiskit_noise_aware_vf2 unavailable: {e}")

        return results


import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
from torch.distributions import Categorical
from torch.optim.lr_scheduler import CosineAnnealingLR

# define a tuple for storing transitions
Transition = namedtuple(
    'Transition',
    (
        'state',
        'action',
        'reward',
        'log_prob',
        'value',
        'next_value',
        'done',
        'true_fidelity',
        'true_coherence',
        'action_mask',
    ),
)


class GraphMessagePassingLayer(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.10):
        super().__init__()
        self.msg = nn.Linear(hidden_size, hidden_size, bias=False)
        self.upd = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, edge_src: torch.Tensor, edge_dst: torch.Tensor) -> torch.Tensor:
        B, Q, H = x.shape
        x_flat = x.reshape(B * Q, H)

        device = x.device
        E_dir = edge_src.numel()
        offsets = (torch.arange(B, device=device) * Q).unsqueeze(1)
        src = (edge_src.unsqueeze(0) + offsets).reshape(B * E_dir)
        dst = (edge_dst.unsqueeze(0) + offsets).reshape(B * E_dir)

        m = self.msg(x_flat[src])
        agg = torch.zeros_like(x_flat)
        agg.index_add_(0, dst, m)

        agg = agg.reshape(B, Q, H)
        out = self.upd(torch.cat([x, agg], dim=-1))
        return self.ln(x + out)


class GraphPPOActorCritic(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        num_qubits: int,
        undirected_edges: List[Tuple[int, int]],
        hidden_size: int = 256,
        gnn_layers: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.num_qubits = int(num_qubits)

        if self.action_size != len(undirected_edges):
            raise ValueError(
                f"GraphPPOActorCritic expects action_size == #undirected_edges. "
                f"Got action_size={self.action_size}, edges={len(undirected_edges)}."
            )

        src = []
        dst = []
        for (u, v) in undirected_edges:
            u = int(u); v = int(v)
            if u == v:
                continue
            src += [u, v]
            dst += [v, u]
        self.register_buffer("edge_src", torch.tensor(src, dtype=torch.long))
        self.register_buffer("edge_dst", torch.tensor(dst, dtype=torch.long))

        u_src = [int(u) for (u, v) in undirected_edges]
        u_dst = [int(v) for (u, v) in undirected_edges]
        self.register_buffer("u_edge_src", torch.tensor(u_src, dtype=torch.long))
        self.register_buffer("u_edge_dst", torch.tensor(u_dst, dtype=torch.long))

        expected = (6 * self.num_qubits) + 4
        if self.state_size != expected:
            raise ValueError(
                f"Expected state_size = 6*Q + 4 = {expected}, got {self.state_size}. "
            )

        node_in = 10

        self.node_embed = nn.Sequential(
            nn.Linear(node_in, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.gnn = nn.ModuleList([GraphMessagePassingLayer(hidden_size, dropout=dropout) for _ in range(gnn_layers)])

        self.edge_mlp = nn.Sequential(
            nn.Linear(4 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        self.critic = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))
        self.fidelity_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))
        self.coherence_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def _parse_state(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = state.size(0)
        Q = self.num_qubits
        off = 0

        err = state[:, off : off + Q]; off += Q
        coh = state[:, off : off + Q]; off += Q
        twoq = state[:, off : off + Q]; off += Q
        hosted = state[:, off : off + Q]; off += Q
        demand = state[:, off : off + Q]; off += Q
        next_onehot = state[:, off : off + Q]; off += Q

        pair = state[:, off : off + 2]; off += 2
        next_dist = state[:, off : off + 1]; off += 1
        prog = state[:, off : off + 1]; off += 1

        global_feats = torch.cat([pair, next_dist, prog], dim=-1)
        global_b = global_feats.unsqueeze(1).expand(B, Q, 4)

        base = torch.stack([err, coh, twoq, hosted, demand, next_onehot], dim=-1)
        node_feats = torch.cat([base, global_b], dim=-1)
        return node_feats, global_feats

    def forward(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None):
        if state.dim() == 1:
            state = state.unsqueeze(0)

        node_feats, _ = self._parse_state(state)
        x = self.node_embed(node_feats)

        for layer in self.gnn:
            x = layer(x, self.edge_src, self.edge_dst)

        u = self.u_edge_src
        v = self.u_edge_dst

        hu = x[:, u, :]
        hv = x[:, v, :]
        e_feat = torch.cat([hu, hv, (hu - hv).abs(), (hu * hv)], dim=-1)
        logits = self.edge_mlp(e_feat).squeeze(-1)

        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            if torch.any(action_mask > 1e-4):
                raise ValueError("action_mask must be an additive logit mask.")
            logits = logits + action_mask.to(logits.device)

        action_probs = torch.softmax(logits, dim=-1)

        g = x.mean(dim=1)
        value = self.critic(g).squeeze(-1)
        fidelity = torch.sigmoid(self.fidelity_head(g).squeeze(-1))
        coherence = torch.sigmoid(self.coherence_head(g).squeeze(-1))

        return action_probs, value, fidelity, coherence


class PPOAgent:
    def __init__(
        self,
        state_size,
        action_size,
        lr=1e-4,                      # Changed default
        hidden_size=256,              # ADDED
        gnn_layers=3,                 # ADDED
        gamma=0.99,
        k_epochs=4,
        eps_clip=0.15,
        gae_lambda=0.95,
        entropy_coefficient=0.5,
        entropy_decay=0.9998,
        batch_size=64,
        min_entropy_coeff=0.01,
        max_entropy_coeff=1.0,
        action_set_name="routing_only",
        aux_fidelity_coef=0.1,
        aux_coherence_coef=0.1,
        value_coef=0.5,
        debug: bool = False,
        num_qubits=None,
        coupling_edges=None,
        policy_backbone: str = "gnn",
    ):
        self.state_size = int(state_size)
        self.action_size = int(action_size)

        self.gamma = float(gamma)
        self.k_epochs = int(k_epochs)
        self.eps_clip = float(eps_clip)
        self.gae_lambda = float(gae_lambda)

        self.entropy_coefficient = float(entropy_coefficient)
        self.entropy_decay = float(entropy_decay)
        self.min_entropy_coeff = float(min_entropy_coeff)
        self.max_entropy_coeff = float(max_entropy_coeff)
        self.entropy_schedule_mode = "adaptive"

        self.batch_size = int(batch_size)
        self.device = torch.device("cpu") # Update if tuning on GPU
        self.debug = bool(debug)
        self.policy_backbone = str(policy_backbone).strip().lower()

        self.aux_fidelity_coef = float(aux_fidelity_coef)
        self.aux_coherence_coef = float(aux_coherence_coef)
        self.value_coef = float(value_coef)

        if self.policy_backbone != "gnn":
            raise ValueError("policy_backbone must be 'gnn'")
        if num_qubits is None or coupling_edges is None:
            raise ValueError("For GNN policy, pass num_qubits=env.num_qubits and coupling_edges=env._physical_edges")
        self.policy = GraphPPOActorCritic(
            state_size=self.state_size,
            action_size=self.action_size,
            num_qubits=int(num_qubits),
            undirected_edges=list(coupling_edges),
            hidden_size=int(hidden_size),
            gnn_layers=int(gnn_layers),
        ).to(self.device)

        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-6)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-6)

        self.rollout_buffer = []
        self.entropy_moving_avg = None

        self.action_sets = {
            "routing_only": {"edge_ops": {"swap"}, "single_ops": set(), "global_ops": set()},
            "expanded": {"edge_ops": {"swap", "topology_aware_swap", "multi_qubit_gate"}, "single_ops": {"error_correction", "rotation_x", "rotation_z"}, "global_ops": {"gate_cancellation", "noise_aware_scheduling"}},
            "full": {"edge_ops": None, "single_ops": None, "global_ops": None},
        }

        self.action_set_name = "routing_only"
        if action_set_name != "routing_only" and self.debug:
            print("[agent] Only 'routing_only' is supported by the current edge-policy architecture; falling back.")

    def set_action_set(self, action_set_name: str):
        if action_set_name != "routing_only":
            raise ValueError("This PPOAgent only supports action_set_name='routing_only' with the current policy head.")
        self.action_set_name = "routing_only"

    def select_action(self, state, action_mask=None):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.as_tensor(action_mask, dtype=torch.float32, device=self.device)
            action_mask = action_mask.to(self.device).unsqueeze(0)

        with torch.no_grad():
            action_probs, value, fidelity_pred, coherence_pred = self.policy(state_t, action_mask=action_mask)

        action_probs = torch.clamp(action_probs, min=1e-12, max=1.0)
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        return (
            int(action.item()),
            action_log_prob.detach(),
            value.squeeze(0).detach(),
            fidelity_pred.squeeze(0).detach(),
            coherence_pred.squeeze(0).detach(),
            dist.entropy().mean().detach(),
        )

    def select_greedy_action(self, state, action_mask=None):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.as_tensor(action_mask, dtype=torch.float32, device=self.device)
            action_mask = action_mask.to(self.device).unsqueeze(0)

        with torch.no_grad():
            action_probs, value, fidelity_pred, coherence_pred = self.policy(state_t, action_mask=action_mask)

        action_probs = torch.clamp(action_probs, min=1e-12, max=1.0)
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        action = torch.argmax(action_probs, dim=-1)

        return (
            int(action.item()),
            value.squeeze(0).detach(),
            fidelity_pred.squeeze(0).detach(),
            coherence_pred.squeeze(0).detach(),
        )

    def store_transition(self, transition: "Transition"):
        self.rollout_buffer.append(transition)

    def clear_rollout(self):
        self.rollout_buffer.clear()

    def _compute_gae_from_transitions(self, rewards, dones, values, next_values):
        advantages = torch.zeros_like(rewards, device=self.device)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values[t] * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def adjust_entropy_coefficient(self, policy_entropy, writer=None, episode=None):
        policy_entropy = float(policy_entropy)

        if self.entropy_moving_avg is None:
            self.entropy_moving_avg = policy_entropy

        self.entropy_moving_avg = 0.8 * float(self.entropy_moving_avg) + 0.2 * policy_entropy

        if self.entropy_schedule_mode == "decay" and episode is not None:
            self.entropy_coefficient = max(self.entropy_coefficient * self.entropy_decay, self.min_entropy_coeff)
        else:
            entropy_ratio = policy_entropy / max(float(self.entropy_moving_avg), 1e-8)
            if entropy_ratio < 0.9:
                self.entropy_coefficient = min(self.entropy_coefficient * 1.5, self.max_entropy_coeff)
            elif entropy_ratio > 1.1:
                self.entropy_coefficient = max(self.entropy_coefficient * 0.8, self.min_entropy_coeff)

        if writer and episode is not None:
            writer.add_scalar("Policy/Entropy", policy_entropy, episode)
            writer.add_scalar("Entropy/Coefficient", float(self.entropy_coefficient), episode)
            writer.add_scalar("Policy/Entropy_Moving_Avg", float(self.entropy_moving_avg), episode)

    def train(self, writer=None, episode=None):
        if len(self.rollout_buffer) < max(8, self.batch_size // 2):
            print("Not enough on-policy rollout samples to train.")
            return

        batch = Transition(*zip(*self.rollout_buffer))

        states = torch.stack(batch.state).to(self.device)
        actions = torch.as_tensor(batch.action, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch.reward, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.done, dtype=torch.float32, device=self.device)

        old_log_probs = torch.stack([lp.to(self.device) for lp in batch.log_prob]).detach()
        values = torch.stack([v.to(self.device) for v in batch.value]).detach()
        next_values = torch.stack([nv.to(self.device) for nv in batch.next_value]).detach()

        true_fidelity = torch.as_tensor(batch.true_fidelity, dtype=torch.float32, device=self.device)
        true_coherence = torch.as_tensor(batch.true_coherence, dtype=torch.float32, device=self.device)
        action_masks = torch.stack([am.to(self.device) for am in batch.action_mask]).detach()

        advantages, returns = self._compute_gae_from_transitions(rewards, dones, values, next_values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.detach()
        returns = returns.detach()

        n = states.size(0)
        indices = torch.arange(n, device=self.device)

        avg_actor_loss = 0.0
        avg_value_loss = 0.0
        avg_aux_loss = 0.0
        avg_total_loss = 0.0
        avg_entropy = 0.0
        updates = 0

        for _ in range(self.k_epochs):
            perm = indices[torch.randperm(n)]
            for start in range(0, n, self.batch_size):
                mb_idx = perm[start : start + self.batch_size]
                if mb_idx.numel() == 0:
                    continue

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_true_fidelity = true_fidelity[mb_idx]
                mb_true_coherence = true_coherence[mb_idx]
                mb_action_masks = action_masks[mb_idx]

                action_probs, value_pred, fidelity_pred, coherence_pred = self.policy(
                    mb_states, action_mask=mb_action_masks
                )

                if value_pred.shape != mb_returns.shape:
                    raise RuntimeError(f"value_pred shape {tuple(value_pred.shape)} != returns shape {tuple(mb_returns.shape)}")

                action_probs = torch.clamp(action_probs, min=1e-12, max=1.0)
                action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(mb_actions)

                ratios = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(value_pred, mb_returns)
                fidelity_loss = torch.zeros((), device=self.device)
                valid_fidelity = ~torch.isnan(mb_true_fidelity)
                if valid_fidelity.any():
                    fidelity_loss = nn.MSELoss()(fidelity_pred[valid_fidelity], mb_true_fidelity[valid_fidelity])

                coherence_loss = torch.zeros((), device=self.device)
                if self.aux_coherence_coef > 0.0:
                    valid_coherence = ~torch.isnan(mb_true_coherence)
                    if valid_coherence.any():
                        coherence_loss = nn.MSELoss()(coherence_pred[valid_coherence], mb_true_coherence[valid_coherence])

                aux_loss = (self.aux_fidelity_coef * fidelity_loss) + (self.aux_coherence_coef * coherence_loss)
                entropy = dist.entropy().mean()
                total_loss = actor_loss + self.value_coef * value_loss + aux_loss - self.entropy_coefficient * entropy

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                avg_actor_loss += float(actor_loss.detach().cpu().item())
                avg_value_loss += float(value_loss.detach().cpu().item())
                avg_aux_loss += float(aux_loss.detach().cpu().item())
                avg_total_loss += float(total_loss.detach().cpu().item())
                avg_entropy += float(entropy.detach().cpu().item())
                updates += 1

        denom = max(1, updates)
        avg_actor_loss /= denom
        avg_value_loss /= denom
        avg_aux_loss /= denom
        avg_total_loss /= denom
        avg_entropy /= denom

        self.adjust_entropy_coefficient(avg_entropy, writer=writer, episode=episode)
        self.scheduler.step()

        if writer and episode is not None:
            writer.add_scalar("Loss/Actor_Loss", avg_actor_loss, episode)
            writer.add_scalar("Loss/Value_Loss", avg_value_loss, episode)
            writer.add_scalar("Loss/Aux_Loss", avg_aux_loss, episode)
            writer.add_scalar("Loss/Total_Loss", avg_total_loss, episode)
            writer.add_scalar("Optimizer/Learning_Rate", float(self.optimizer.param_groups[0]["lr"]), episode)

        self.clear_rollout()

    def _action_set_allows(self, action_type: str) -> bool:
        cfg = self.action_sets.get(self.action_set_name, self.action_sets["routing_only"])
        if action_type in {"swap", "topology_aware_swap", "multi_qubit_gate"}:
            allowed = cfg["edge_ops"]
        elif action_type in {"h", "x", "y", "z", "rotation_x", "rotation_z", "error_correction", "reset"}:
            allowed = cfg["single_ops"]
        else:
            allowed = cfg["global_ops"]
        return (allowed is None) or (action_type in allowed)

    def compute_action_mask(self, env):
        mask = torch.zeros(self.action_size, dtype=torch.float32, device=self.device)
        edge_types = tuple(getattr(env, "edge_action_types", ()))
        single_types = tuple(getattr(env, "single_action_types", ()))
        global_types = tuple(getattr(env, "global_action_types", ()))
        env_edge_mask = None
        if hasattr(env, "get_action_mask"):
            try:
                env_edge_mask = torch.as_tensor(env.get_action_mask(), dtype=torch.float32, device=self.device)
            except Exception:
                env_edge_mask = None

        num_qubits = int(getattr(env, "num_qubits"))
        max_edges = int(getattr(env, "max_candidate_edges"))
        edge_block = len(edge_types) * max_edges
        single_block = edge_block + len(single_types) * num_qubits

        candidate_edges = env._get_candidate_edges()
        valid_edges = int(len(candidate_edges))

        for op_idx, action_type in enumerate(edge_types):
            start = op_idx * max_edges
            end = start + max_edges

            if not self._action_set_allows(action_type):
                mask[start:end].fill_(-1e9)
                continue
            if env_edge_mask is not None and env_edge_mask.numel() >= end:
                mask[start:end] = torch.where(
                    env_edge_mask[start:end] > 0,
                    torch.zeros(end - start, dtype=torch.float32, device=self.device),
                    torch.full((end - start,), -1e9, dtype=torch.float32, device=self.device),
                )
            if valid_edges < max_edges:
                mask[start + valid_edges : end] = -1e9

        if len(single_types) > 0:
            coherence = env.coherence_times.to(self.device)
            active = (~env.deactivated_qubits).to(self.device)
            healthy = (coherence > 10e-6) & active

            single_health_checked = {"rotation_x", "rotation_z", "error_correction", "reset", "h", "x", "y", "z"}
            for op_idx, action_type in enumerate(single_types):
                start = edge_block + op_idx * num_qubits
                end = start + num_qubits

                if not self._action_set_allows(action_type):
                    mask[start:end].fill_(-1e9)
                    continue

                if action_type in single_health_checked:
                    mask[start:end].masked_fill_(~healthy, -1e9)

        for op_idx, action_type in enumerate(global_types):
            idx = single_block + op_idx
            if not self._action_set_allows(action_type):
                mask[idx] = -1e9

        if torch.all(mask < 0):
            mask.fill_(-1e9)
            if ("gate_cancellation" in global_types) and self._action_set_allows("gate_cancellation"):
                mask[single_block + global_types.index("gate_cancellation")] = 0.0
            else:
                if self.debug:
                    print("[mask] WARNING: all actions masked; falling back to a single arbitrary action_id=0.")
                mask[0] = 0.0

        return mask

    def load_checkpoint(self, filename, env=None):
        load_kwargs = {"map_location": self.device}
        try:
            checkpoint = torch.load(filename, weights_only=False, **load_kwargs)
        except TypeError:
            checkpoint = torch.load(filename, **load_kwargs)
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.entropy_coefficient = checkpoint.get("entropy_coefficient", self.entropy_coefficient)
        ckpt_backbone = str(checkpoint.get("policy_backbone", self.policy_backbone)).strip().lower()
        if ckpt_backbone != self.policy_backbone:
            raise ValueError(
                f"Checkpoint backbone '{ckpt_backbone}' does not match current policy_backbone '{self.policy_backbone}'."
            )
        loaded_action_set = checkpoint.get("action_set_name", self.action_set_name)
        if loaded_action_set != "routing_only":
            print(f"Warning: checkpoint action_set '{loaded_action_set}' is unsupported by this policy. Falling back to 'routing_only'.")
            loaded_action_set = "routing_only"
        self.action_set_name = loaded_action_set

        if env is not None and hasattr(env, "set_action_set"):
            try:
                env.set_action_set(self.action_set_name)
            except Exception as e:
                print(f"Warning: could not set env action_set to '{self.action_set_name}': {e}")

        if env is not None:
            env_state = checkpoint.get("environment_state", {})
            if env_state:
                if "logical_to_physical" in env_state:
                    l2p = env_state["logical_to_physical"]
                    l2p = np.asarray(l2p, dtype=np.int32).tolist()
                    if len(l2p) != int(env.num_qubits):
                        raise ValueError(f"Checkpoint logical_to_physical len {len(l2p)} != env.num_qubits {env.num_qubits}")
                    env.logical_to_physical = [int(x) for x in l2p]

                    env.physical_to_logical = [0] * int(env.num_qubits)
                    for li, pi in enumerate(env.logical_to_physical):
                        env.physical_to_logical[int(pi)] = int(li)

                if "baseline_fidelity" in env_state:
                    env.baseline_fidelity = float(env_state["baseline_fidelity"])
                if "baseline_cost" in env_state:
                    env.baseline_cost = float(env_state["baseline_cost"])

        print(f"Checkpoint loaded from {filename}, resuming from episode {checkpoint.get('episode', 0)}")
        return int(checkpoint.get("episode", 0))


def evaluate_agent_and_baselines_holdout(
    agent,
    config,
    num_episodes: int = 10,
    start_seed: int = 50000,
    reward_mode: str = "shaped",
    return_episode_records: bool = False,
):
    def _summary(records):
        out = {}
        for name, values in records.items():
            arr = np.asarray(values, dtype=float)
            out[name] = {
                "mean": float(np.mean(arr)) if arr.size else float("nan"),
                "std": float(np.std(arr, ddof=0)) if arr.size else float("nan"),
            }
        return out

    metrics = {
        "agent_fidelity": [],
        "agent_cost": [],
        "agent_twoq": [],
        "agent_depth": [],
        "agent_wall_seconds": [],
        "agent_completed": [],
        "agent_timed_out": [],
        "agent_progress": [],
        "agent_effective_max_steps": [],
        "trivial_fidelity": [],
        "trivial_cost": [],
        "trivial_twoq": [],
        "trivial_depth": [],
        "sabre_fidelity": [],
        "sabre_cost": [],
        "sabre_twoq": [],
        "sabre_depth": [],
        "sabre_wall_seconds": [],
        "sabre_trivial_layout_fidelity": [],
        "sabre_trivial_layout_cost": [],
        "sabre_trivial_layout_twoq": [],
        "sabre_trivial_layout_depth": [],
        "sabre_trivial_layout_wall_seconds": [],
        "lookahead_fidelity": [],
        "lookahead_cost": [],
        "lookahead_twoq": [],
        "lookahead_depth": [],
        "qiskit_noise_aware_vf2_fidelity": [],
        "qiskit_noise_aware_vf2_cost": [],
        "qiskit_noise_aware_vf2_twoq": [],
        "qiskit_noise_aware_vf2_depth": [],
        "greedy_fidelity": [],
        "greedy_cost": [],
        "greedy_twoq": [],
        "greedy_depth": [],
        "random_fidelity": [],
        "random_cost": [],
        "random_twoq": [],
        "random_depth": [],
    }
    episode_records = []
    requested_baselines_raw = os.getenv(
        "EVAL_BASELINES",
        os.getenv(
            "REVIEW_BASELINES",
            "trivial,sabre,sabre_trivial_layout,qiskit_noise_aware_vf2,greedy,lookahead,random",
        ),
    )
    requested_baselines = {
        item.strip()
        for item in str(requested_baselines_raw).split(",")
        if item.strip()
    }
    progress_interval = _env_int("EVAL_PROGRESS_INTERVAL", 10)
    print(
        f"[eval] holdout start episodes={int(num_episodes)} start_seed={int(start_seed)} "
        f"baselines={','.join(sorted(requested_baselines)) or 'none'} progress_interval={progress_interval}",
        flush=True,
    )

    for offset in range(int(num_episodes)):
        seed = int(start_seed) + int(offset)
        if progress_interval > 0 and (offset == 0 or offset % progress_interval == 0):
            print(f"[eval] episode {offset + 1}/{int(num_episodes)} seed={seed} start", flush=True)

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
            use_proxy_reward=False,
            reward_mode=reward_mode,
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
        state = env.reset(episode=seed)
        done = bool(env.done)
        info = dict(env.last_metrics)
        steps = 0

        t0 = time.perf_counter()
        while not done and steps < env.max_steps_per_episode:
            action_mask = agent.compute_action_mask(env)
            action, _, _, _ = agent.select_greedy_action(state, action_mask=action_mask)
            state, _, done, info = env.step(action)
            steps += 1
        agent_wall_seconds = float(time.perf_counter() - t0)

        agent_fidelity = float(info.get("fidelity", env.last_metrics.get("fidelity", 0.0)))
        agent_proxy_fidelity = (
            float(env._calculate_proxy_fidelity(env.compiled_circuit))
            if env.compiled_circuit is not None
            else float(info.get("proxy_fidelity", env.last_metrics.get("proxy_fidelity", 0.0)))
        )
        agent_cost = float(info.get("cost", env.last_metrics.get("cost", 0.0)))
        agent_twoq = float(info.get("twoq", env.last_metrics.get("twoq", 0.0)))
        agent_depth = float(info.get("depth", env.last_metrics.get("depth", 0.0)))
        agent_completed = bool(info.get("completed_target", env.last_metrics.get("completed_target", False)))
        agent_timed_out = bool(info.get("timed_out", env.last_metrics.get("timed_out", False)))
        agent_progress = float(info.get("progress", env.last_metrics.get("progress", 0.0)))
        agent_effective_max_steps = float(info.get("effective_max_steps", env.max_steps_per_episode))
        metrics["agent_fidelity"].append(agent_fidelity)
        metrics["agent_cost"].append(agent_cost)
        metrics["agent_twoq"].append(agent_twoq)
        metrics["agent_depth"].append(agent_depth)
        metrics["agent_wall_seconds"].append(agent_wall_seconds)
        metrics["agent_completed"].append(1.0 if agent_completed else 0.0)
        metrics["agent_timed_out"].append(1.0 if agent_timed_out else 0.0)
        metrics["agent_progress"].append(agent_progress)
        metrics["agent_effective_max_steps"].append(agent_effective_max_steps)
        if progress_interval > 0 and (offset == 0 or offset % progress_interval == 0):
            print(
                f"[eval] episode {offset + 1}/{int(num_episodes)} seed={seed} "
                f"agent done completed={int(agent_completed)} progress={agent_progress:.3f} "
                f"reason={info.get('terminal_reason', 'unknown')} wall_seconds={agent_wall_seconds:.3f}",
                flush=True,
            )

        baselines = env.evaluate_baselines(enabled_baselines=requested_baselines)
        if progress_interval > 0 and (offset == 0 or offset % progress_interval == 0):
            print(
                f"[eval] episode {offset + 1}/{int(num_episodes)} seed={seed} "
                f"compiler baselines done keys={','.join(sorted(k for k in baselines if k != 'agent'))}",
                flush=True,
            )
        if "trivial" in baselines:
            metrics["trivial_fidelity"].append(float(baselines["trivial"]["fidelity"]))
            metrics["trivial_cost"].append(float(baselines["trivial"]["cost"]))
            metrics["trivial_twoq"].append(float(baselines["trivial"]["twoq"]))
            metrics["trivial_depth"].append(float(baselines["trivial"]["depth"]))
        if "sabre" in baselines:
            metrics["sabre_fidelity"].append(float(baselines["sabre"]["fidelity"]))
            metrics["sabre_cost"].append(float(baselines["sabre"]["cost"]))
            metrics["sabre_twoq"].append(float(baselines["sabre"]["twoq"]))
            metrics["sabre_depth"].append(float(baselines["sabre"]["depth"]))
            metrics["sabre_wall_seconds"].append(float(baselines["sabre"].get("wall_seconds", float("nan"))))
        if "sabre_trivial_layout" in baselines:
            metrics["sabre_trivial_layout_fidelity"].append(float(baselines["sabre_trivial_layout"]["fidelity"]))
            metrics["sabre_trivial_layout_cost"].append(float(baselines["sabre_trivial_layout"]["cost"]))
            metrics["sabre_trivial_layout_twoq"].append(float(baselines["sabre_trivial_layout"]["twoq"]))
            metrics["sabre_trivial_layout_depth"].append(float(baselines["sabre_trivial_layout"]["depth"]))
            metrics["sabre_trivial_layout_wall_seconds"].append(float(baselines["sabre_trivial_layout"].get("wall_seconds", float("nan"))))
        if "lookahead" in baselines:
            metrics["lookahead_fidelity"].append(float(baselines["lookahead"]["fidelity"]))
            metrics["lookahead_cost"].append(float(baselines["lookahead"]["cost"]))
            metrics["lookahead_twoq"].append(float(baselines["lookahead"]["twoq"]))
            metrics["lookahead_depth"].append(float(baselines["lookahead"]["depth"]))
        if "qiskit_noise_aware_vf2" in baselines:
            metrics["qiskit_noise_aware_vf2_fidelity"].append(float(baselines["qiskit_noise_aware_vf2"]["fidelity"]))
            metrics["qiskit_noise_aware_vf2_cost"].append(float(baselines["qiskit_noise_aware_vf2"]["cost"]))
            metrics["qiskit_noise_aware_vf2_twoq"].append(float(baselines["qiskit_noise_aware_vf2"]["twoq"]))
            metrics["qiskit_noise_aware_vf2_depth"].append(float(baselines["qiskit_noise_aware_vf2"]["depth"]))

        greedy_fidelity = greedy_proxy_fidelity = greedy_cost = greedy_twoq = greedy_depth = float("nan")
        if "greedy" in requested_baselines:
            if progress_interval > 0 and (offset == 0 or offset % progress_interval == 0):
                print(f"[eval] episode {offset + 1}/{int(num_episodes)} seed={seed} greedy start", flush=True)
            greedy_env = QuantumRoutingEnv(
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
                use_proxy_reward=False,
                reward_mode=reward_mode,
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
            greedy_env.reset(episode=seed)
            greedy_done = bool(greedy_env.done)
            greedy_info = dict(greedy_env.last_metrics)
            greedy_steps = 0
            while not greedy_done and greedy_steps < greedy_env.max_steps_per_episode:
                greedy_action = greedy_env.select_greedy_swap_action()
                _, _, greedy_done, greedy_info = greedy_env.step(greedy_action)
                greedy_steps += 1

            greedy_fidelity = float(greedy_info.get("fidelity", greedy_env.last_metrics.get("fidelity", 0.0)))
            greedy_proxy_fidelity = (
                float(greedy_env._calculate_proxy_fidelity(greedy_env.compiled_circuit))
                if greedy_env.compiled_circuit is not None
                else float("nan")
            )
            greedy_cost = float(greedy_info.get("cost", greedy_env.last_metrics.get("cost", 0.0)))
            greedy_twoq = float(greedy_info.get("twoq", greedy_env.last_metrics.get("twoq", 0.0)))
            greedy_depth = float(greedy_info.get("depth", greedy_env.last_metrics.get("depth", 0.0)))
            metrics["greedy_fidelity"].append(greedy_fidelity)
            metrics["greedy_cost"].append(greedy_cost)
            metrics["greedy_twoq"].append(greedy_twoq)
            metrics["greedy_depth"].append(greedy_depth)
            if progress_interval > 0 and (offset == 0 or offset % progress_interval == 0):
                print(f"[eval] episode {offset + 1}/{int(num_episodes)} seed={seed} greedy done", flush=True)

        random_fidelity = random_proxy_fidelity = random_cost = random_twoq = random_depth = float("nan")
        if "random" in requested_baselines:
            if progress_interval > 0 and (offset == 0 or offset % progress_interval == 0):
                print(f"[eval] episode {offset + 1}/{int(num_episodes)} seed={seed} random start", flush=True)
            random_env = QuantumRoutingEnv(
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
                use_proxy_reward=False,
                reward_mode=reward_mode,
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
            random_env.reset(episode=seed)
            random_done = bool(random_env.done)
            random_info = dict(random_env.last_metrics)
            random_steps = 0
            while not random_done and random_steps < random_env.max_steps_per_episode:
                valid = np.flatnonzero(random_env.get_action_mask() > 0.0)
                if valid.size == 0:
                    break
                random_action = int(random_env.rng.choice(valid))
                _, _, random_done, random_info = random_env.step(random_action)
                random_steps += 1

            random_fidelity = float(random_info.get("fidelity", random_env.last_metrics.get("fidelity", 0.0)))
            random_proxy_fidelity = (
                float(random_env._calculate_proxy_fidelity(random_env.compiled_circuit))
                if random_env.compiled_circuit is not None
                else float("nan")
            )
            random_cost = float(random_info.get("cost", random_env.last_metrics.get("cost", 0.0)))
            random_twoq = float(random_info.get("twoq", random_env.last_metrics.get("twoq", 0.0)))
            random_depth = float(random_info.get("depth", random_env.last_metrics.get("depth", 0.0)))
            metrics["random_fidelity"].append(random_fidelity)
            metrics["random_cost"].append(random_cost)
            metrics["random_twoq"].append(random_twoq)
            metrics["random_depth"].append(random_depth)
            if progress_interval > 0 and (offset == 0 or offset % progress_interval == 0):
                print(f"[eval] episode {offset + 1}/{int(num_episodes)} seed={seed} random done", flush=True)

        if return_episode_records:
            record = {
                "seed": int(seed),
                "target_circuit_type": str(getattr(env, "target_circuit_type", "unknown")),
                "target_circuit_source": str(getattr(env, "target_circuit_source", "unknown")),
                "target_circuit_source_sha256": getattr(env, "target_circuit_source_sha256", None),
                "target_circuit_sha256": getattr(env, "target_circuit_sha256", None),
                "agent_completed": agent_completed,
                "agent_progress": agent_progress,
                "agent_terminal_reason": str(info.get("terminal_reason", env.last_metrics.get("terminal_reason", "unknown"))),
                "agent_effective_max_steps": int(agent_effective_max_steps),
                "agent_raw_partial_twoq": float(info.get("raw_partial_twoq", env.last_metrics.get("raw_partial_twoq", float("nan")))),
                "agent_raw_partial_depth": float(info.get("raw_partial_depth", env.last_metrics.get("raw_partial_depth", float("nan")))),
                "agent_raw_partial_cost": float(info.get("raw_partial_cost", env.last_metrics.get("raw_partial_cost", float("nan")))),
                "agent": {
                    "fidelity": agent_fidelity,
                    "cost": agent_cost,
                    "twoq": agent_twoq,
                    "depth": agent_depth,
                    "wall_seconds": agent_wall_seconds,
                    "completed": agent_completed,
                    "progress": agent_progress,
                    "terminal_reason": str(info.get("terminal_reason", env.last_metrics.get("terminal_reason", "unknown"))),
                    "effective_max_steps": int(agent_effective_max_steps),
                    "raw_partial_twoq": float(info.get("raw_partial_twoq", env.last_metrics.get("raw_partial_twoq", float("nan")))),
                    "raw_partial_depth": float(info.get("raw_partial_depth", env.last_metrics.get("raw_partial_depth", float("nan")))),
                    "raw_partial_cost": float(info.get("raw_partial_cost", env.last_metrics.get("raw_partial_cost", float("nan")))),
                    "timed_out": agent_timed_out,
                },
            }
            if "trivial" in baselines:
                record["trivial"] = {
                    "fidelity": float(baselines["trivial"]["fidelity"]),
                    "cost": float(baselines["trivial"]["cost"]),
                    "twoq": float(baselines["trivial"]["twoq"]),
                    "depth": float(baselines["trivial"]["depth"]),
                }
            if "sabre" in baselines:
                record["sabre"] = {
                    "fidelity": float(baselines["sabre"]["fidelity"]),
                    "cost": float(baselines["sabre"]["cost"]),
                    "twoq": float(baselines["sabre"]["twoq"]),
                    "depth": float(baselines["sabre"]["depth"]),
                    "wall_seconds": float(baselines["sabre"].get("wall_seconds", float("nan"))),
                }
            if "sabre_trivial_layout" in baselines:
                record["sabre_trivial_layout"] = {
                    "fidelity": float(baselines["sabre_trivial_layout"]["fidelity"]),
                    "cost": float(baselines["sabre_trivial_layout"]["cost"]),
                    "twoq": float(baselines["sabre_trivial_layout"]["twoq"]),
                    "depth": float(baselines["sabre_trivial_layout"]["depth"]),
                    "wall_seconds": float(baselines["sabre_trivial_layout"].get("wall_seconds", float("nan"))),
                }
            if "greedy" in requested_baselines:
                record["greedy"] = {
                    "fidelity": greedy_fidelity,
                    "cost": greedy_cost,
                    "twoq": greedy_twoq,
                    "depth": greedy_depth,
                }
            if "random" in requested_baselines:
                record["random"] = {
                    "fidelity": random_fidelity,
                    "cost": random_cost,
                    "twoq": random_twoq,
                    "depth": random_depth,
                }
            if "lookahead" in baselines:
                record["lookahead"] = {
                    "fidelity": float(baselines["lookahead"]["fidelity"]),
                    "cost": float(baselines["lookahead"]["cost"]),
                    "twoq": float(baselines["lookahead"]["twoq"]),
                    "depth": float(baselines["lookahead"]["depth"]),
                }
            if "qiskit_noise_aware_vf2" in baselines:
                record["qiskit_noise_aware_vf2"] = {
                    "fidelity": float(baselines["qiskit_noise_aware_vf2"]["fidelity"]),
                    "cost": float(baselines["qiskit_noise_aware_vf2"]["cost"]),
                    "twoq": float(baselines["qiskit_noise_aware_vf2"]["twoq"]),
                    "depth": float(baselines["qiskit_noise_aware_vf2"]["depth"]),
                }
            episode_records.append(record)
        if progress_interval > 0 and (offset == 0 or offset % progress_interval == 0):
            print(f"[eval] episode {offset + 1}/{int(num_episodes)} seed={seed} done", flush=True)

    summary = _summary(metrics)
    if return_episode_records:
        return {"summary": summary, "episodes": episode_records}
    return summary


class TrainingConfig:
    def __init__(self, load_hyperparams: bool = True):
        # Base settings
        self.train_frequency = _env_int("TRAIN_FREQUENCY", 10)
        self.checkpoint_frequency = _env_int("CHECKPOINT_FREQUENCY", 100)
        self.action_set_phase1 = "routing_only"
        self.policy_backbone = "gnn"
        self.log_action_histogram = _env_flag("LOG_ACTION_HISTOGRAM", True)
        self.eval_baselines_on_checkpoint = _env_flag("EVAL_BASELINES_ON_CHECKPOINT", True)
        self.use_proxy_reward_phase1 = _env_flag("USE_PROXY_REWARD", True)
        self.train_num_qubits = _env_int("TRAIN_NUM_QUBITS", 10)
        self.train_max_steps = _env_int("TRAIN_MAX_STEPS", 200)
        self.train_max_steps_base = _env_int("TRAIN_MAX_STEPS_BASE", 100)
        self.train_max_steps_per_2q = _env_float("TRAIN_MAX_STEPS_PER_2Q", 8.0)
        self.train_max_steps_cap = _env_int("TRAIN_MAX_STEPS_CAP", 2000)
        self.optimization_level = _env_int("TRANSPILE_OPTIMIZATION_LEVEL", 3)
        self.sabre_baseline_trials = _env_int("SABRE_BASELINE_TRIALS", 20)
        self.cost_lambda = _env_float("COST_LAMBDA", 0.01)
        self.cost_w_twoq = _env_float("COST_W_TWOQ", 1.0)
        self.cost_w_depth = _env_float("COST_W_DEPTH", 0.01)
        self.reward_mode = str(os.getenv("REWARD_MODE", "shaped")).strip().lower()
        self.fidelity_scale = _env_float("FIDELITY_SCALE", 10.0)
        self.invalid_action_penalty = _env_float("INVALID_ACTION_PENALTY", 0.2)
        self.swap_penalty = _env_float("SWAP_PENALTY", 0.02)
        self.distance_reduction_reward_scale = _env_float("DISTANCE_REDUCTION_REWARD_SCALE", 0.05)
        self.progress_reward_scale = _env_float("PROGRESS_REWARD_SCALE", 2.0)
        self.executed_gate_reward_scale = _env_float("EXECUTED_GATE_REWARD_SCALE", 0.01)
        self.timeout_penalty = _env_float("TIMEOUT_PENALTY", 0.5)
        self.incomplete_episode_penalty = _env_float("INCOMPLETE_EPISODE_PENALTY", 10.0)
        self.eval_holdout_episodes = _env_int("EVAL_HOLDOUT_EPISODES", 30)
        self.eval_holdout_start_seed = _env_int("EVAL_HOLDOUT_START_SEED", 50000)
        self.random_2q_prob = _env_float("RANDOM_2Q_PROB", 0.45)
        self.qaoa_prob = _env_float("QAOA_PROB", 0.20)
        self.quantum_volume_prob = _env_float("QUANTUM_VOLUME_PROB", 0.20)
        self.vqe_prob = _env_float("VQE_PROB", 0.15)
        self.clifford_prob = _env_float("CLIFFORD_PROB", 0.0)
        self.positive_control_prob = _env_float("POSITIVE_CONTROL_PROB", 0.0)
        self.zero_noise_features = _env_flag("ZERO_NOISE_FEATURES", False)
        self.calibration_feature_mask = os.getenv("CALIBRATION_FEATURE_MASK", "").strip()
        self.benchmark_qasm_files = os.getenv("BENCHMARK_QASM_FILES", "").strip() or None
        self.benchmark_qasm_dir = os.getenv("BENCHMARK_QASM_DIR", "").strip() or None
        benchmark_default_prob = 1.0 if (self.benchmark_qasm_files or self.benchmark_qasm_dir) else 0.0
        self.benchmark_corpus_prob = _env_float("BENCHMARK_CORPUS_PROB", benchmark_default_prob)
        self.benchmark_corpus_name = os.getenv("BENCHMARK_CORPUS_NAME", "external_qasm").strip() or "external_qasm"

        if self.reward_mode == "no_distance":
            self.reward_mode = "shaped"
            self.distance_reduction_reward_scale = 0.0
        elif self.reward_mode == "no_progress":
            self.reward_mode = "shaped"
            self.progress_reward_scale = 0.0
        elif self.reward_mode == "no_gate_reward":
            self.reward_mode = "shaped"
            self.executed_gate_reward_scale = 0.0

        # Default Hyperparameters (Will be overwritten by JSON if it exists)
        self.lr = 1e-4
        self.hidden_size = 256
        self.gnn_layers = 3
        self.gamma = 0.99
        self.k_epochs = 4
        self.eps_clip = 0.15
        self.gae_lambda = 0.95
        self.entropy_coefficient = 0.5
        self.entropy_decay = 0.9998
        self.min_entropy_coeff = 0.01
        self.batch_size = 64
        self.value_coef = 0.5
        self.aux_fidelity_coef = 0.1
        self.aux_coherence_coef = 0.0

        default_calibration_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "downloaded_calibrations",
            "ibm_torino_calibration.json",
        )
        self.calibration_file = _resolve_existing_path(
            os.getenv("QUANTUM_CALIBRATION_FILE"),
            os.path.join(os.getcwd(), "downloaded_calibrations", "ibm_torino_calibration.json"),
            default_calibration_file,
        ) or os.getenv("QUANTUM_CALIBRATION_FILE") or default_calibration_file

        if load_hyperparams:
            self._load_hyperparams("optimal_hyperparams.json")

    def to_agent_kwargs(self):
        return {
            "lr": self.lr,
            "hidden_size": self.hidden_size,
            "gnn_layers": self.gnn_layers,
            "policy_backbone": self.policy_backbone,
            "gamma": self.gamma,
            "k_epochs": self.k_epochs,
            "eps_clip": self.eps_clip,
            "gae_lambda": self.gae_lambda,
            "entropy_coefficient": self.entropy_coefficient,
            "entropy_decay": self.entropy_decay,
            "min_entropy_coeff": self.min_entropy_coeff,
            "batch_size": self.batch_size,
            "aux_fidelity_coef": self.aux_fidelity_coef,
            "aux_coherence_coef": self.aux_coherence_coef,
            "value_coef": self.value_coef,
        }

    def _load_hyperparams(self, filepath: str):
        resolved_path = _resolve_existing_path(
            filepath,
            os.path.join(os.getcwd(), filepath),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath),
        )
        if resolved_path and os.path.exists(resolved_path):
            try:
                with open(resolved_path, 'r') as f:
                    params = json.load(f)
                self.lr = params.get("lr", self.lr)
                self.hidden_size = params.get("hidden_size", self.hidden_size)
                self.gnn_layers = params.get("gnn_layers", self.gnn_layers)
                self.policy_backbone = "gnn"
                self.gamma = params.get("gamma", self.gamma)
                self.k_epochs = params.get("k_epochs", self.k_epochs)
                self.eps_clip = params.get("eps_clip", self.eps_clip)
                self.gae_lambda = params.get("gae_lambda", self.gae_lambda)
                self.entropy_coefficient = params.get("entropy_coefficient", self.entropy_coefficient)
                self.entropy_decay = params.get("entropy_decay", self.entropy_decay)
                self.min_entropy_coeff = params.get("min_entropy_coeff", self.min_entropy_coeff)
                self.batch_size = params.get("batch_size", self.batch_size)
                self.value_coef = params.get("value_coef", self.value_coef)
                self.aux_fidelity_coef = params.get("aux_fidelity_coef", self.aux_fidelity_coef)
                self.aux_coherence_coef = params.get("aux_coherence_coef", self.aux_coherence_coef)
                print(f"[Config] Successfully loaded hyperparams from {resolved_path}")
            except Exception as e:
                print(f"[Config] Failed to load {resolved_path}: {e}")
        else:
            print(f"[Config] No {filepath} found. Using default hyperparameters.")

if __name__ == "__main__":
    if _env_flag("REQUIRE_PIPELINE_ARTIFACTS", True):
        missing = []
        for artifact in ("optimal_hyperparams.json",):
            if _resolve_existing_path(
                artifact,
                os.path.join(os.getcwd(), artifact),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), artifact),
            ) is None:
                missing.append(artifact)
        if missing:
            raise FileNotFoundError(
                "Training requires finalized pipeline artifacts before launch: "
                + ", ".join(missing)
                + ". Run tune_hyperparameters.py first, "
                "or set REQUIRE_PIPELINE_ARTIFACTS=0 only for local debugging."
            )

    config = TrainingConfig()
    training_seed_offset = _env_int("TRAINING_SEED_OFFSET", 0)
    _set_global_seeds(training_seed_offset)

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
        use_proxy_reward=config.use_proxy_reward_phase1,  # Tied to config
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
    state_size = state.shape[0]
    action_size = env.get_action_size()

    ppo_agent = PPOAgent(
        state_size,
        action_size,
        action_set_name=config.action_set_phase1,
        num_qubits=env.num_qubits,
        coupling_edges=env._physical_edges,
        **config.to_agent_kwargs(),
    )

    ppo_agent.set_action_set("routing_only")
    env.set_action_set("routing_only")

    num_training_episodes = _env_int("NUM_TRAINING_EPISODES", 10001)
    checkpoint_dir = os.path.abspath(os.getenv("CHECKPOINT_DIR", "."))
    tensorboard_log_dir = os.path.abspath(os.getenv("TENSORBOARD_LOG_DIR", "final_ppo_optionA"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    checkpoint_path = find_latest_checkpoint(os.path.join(checkpoint_dir, "ppo_checkpoint_*.pt"))
    if checkpoint_path is None:
        print("No checkpoint found. Starting training from scratch.")
        start_episode = 0
    else:
        start_episode = ppo_agent.load_checkpoint(checkpoint_path, env=env)
        print(f"Resuming training from episode {start_episode} using {checkpoint_path}")

    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    for episode in range(start_episode, num_training_episodes):
        ppo_agent.set_action_set("routing_only")
        env.set_action_set("routing_only")

        state = env.reset(episode=training_seed_offset + episode)
        episode_reward = 0.0
        done = bool(env.done)
        step = 0
        info = dict(env.last_metrics)

        actions_this_episode = []
        valid_action_fracs = []

        fid_pred = torch.tensor(0.0)
        coh_pred = torch.tensor(0.0)

        while not done and step < env.max_steps_per_episode:
            action_mask = ppo_agent.compute_action_mask(env)
            valid_frac = float((action_mask == 0).float().mean().item())
            valid_action_fracs.append(valid_frac)

            action, action_log_prob, value, fid_pred, coh_pred, entropy = ppo_agent.select_action(
                state, action_mask=action_mask
            )

            next_state, reward, done, info = env.step(action)

            true_fidelity = float(info.get("fidelity", 0.0)) if done else float("nan")

            with torch.no_grad():
                ns = torch.tensor(next_state, dtype=torch.float32, device=ppo_agent.device).unsqueeze(0)
                next_mask = ppo_agent.compute_action_mask(env).unsqueeze(0)
                _, next_value, _, _ = ppo_agent.policy(ns, action_mask=next_mask)
                next_value = next_value.squeeze(0).detach()

            transition = Transition(
                state=torch.tensor(state, dtype=torch.float32, device=ppo_agent.device),
                action=action,
                reward=float(reward),
                log_prob=action_log_prob,
                value=value,
                next_value=next_value,
                done=float(done),
                true_fidelity=true_fidelity,
                true_coherence=float("nan"),
                action_mask=action_mask.detach(),
            )

            ppo_agent.store_transition(transition)

            actions_this_episode.append(action)
            state = next_state
            episode_reward += float(reward)
            step += 1

        episode_reward = episode_reward / max(1, step)

        if (episode + 1) % config.train_frequency == 0:
            print(f"Training agent at episode {episode + 1}...")
            ppo_agent.train(writer=writer, episode=episode)

        writer.add_scalar("Reward/Episode", episode_reward, episode)
        writer.add_scalar("Episode/Length", step, episode)
        writer.add_scalar("Env/Invalid_Action_Count", float(env.invalid_gate_count), episode)
        writer.add_scalar("Policy/Entropy_Coeff", float(ppo_agent.entropy_coefficient), episode)
        writer.add_scalar("Policy/Valid_Action_Fraction", float(np.mean(valid_action_fracs) if valid_action_fracs else 0.0), episode)
        writer.add_scalar("Fidelity/Predicted", float(fid_pred.item()), episode)
        writer.add_scalar("Fidelity/Actual", float(info.get("fidelity", env.last_metrics.get("fidelity", 0.0))), episode)
        writer.add_scalar("Cost/TwoQ", float(env.last_metrics.get("twoq", 0)), episode)
        writer.add_scalar("Cost/Depth", float(env.last_metrics.get("depth", 0)), episode)
        writer.add_scalar("Cost/Scalar", float(env.last_metrics.get("cost", 0.0)), episode)
        writer.add_scalar("Completion/Completed_Target", float(env.last_metrics.get("completed_target", 0.0)), episode)
        writer.add_scalar("Completion/Progress", float(env.last_metrics.get("progress", 0.0)), episode)
        writer.add_scalar("Completion/Effective_Max_Steps", float(env.max_steps_per_episode), episode)

        if config.log_action_histogram and actions_this_episode:
            writer.add_histogram("Policy/Action_ID", torch.tensor(actions_this_episode), episode)

        print(
            f"Episode {episode + 1} | Avg Reward/Step: {episode_reward:.4f} "
            f"| Completed: {int(bool(env.last_metrics.get('completed_target', False)))} "
            f"| Progress: {float(env.last_metrics.get('progress', 0.0)):.3f} "
            f"| Budget: {int(env.max_steps_per_episode)}"
        )

        if (episode + 1) % config.checkpoint_frequency == 0:
            checkpoint = {
                "model_state_dict": ppo_agent.policy.state_dict(),
                "optimizer_state_dict": ppo_agent.optimizer.state_dict(),
                "scheduler_state_dict": ppo_agent.scheduler.state_dict(),
                "entropy_coefficient": ppo_agent.entropy_coefficient,
                "policy_backbone": ppo_agent.policy_backbone,
                "action_set_name": ppo_agent.action_set_name,
                "episode": episode + 1,
                "environment_state": {
                    "logical_to_physical": [int(x) for x in env.logical_to_physical],
                    "baseline_fidelity": float(env.baseline_fidelity),
                    "baseline_cost": float(env.baseline_cost),
                },
            }
            checkpoint_file = os.path.join(checkpoint_dir, f"ppo_checkpoint_{episode + 1}.pt")
            torch.save(checkpoint, checkpoint_file)
            print(f"Checkpoint saved at episode {episode + 1}: {checkpoint_file}")

            if config.eval_baselines_on_checkpoint:
                try:
                    holdout = evaluate_agent_and_baselines_holdout(
                        ppo_agent,
                        config,
                        num_episodes=config.eval_holdout_episodes,
                        start_seed=config.eval_holdout_start_seed + training_seed_offset + (episode + 1),
                        reward_mode=config.reward_mode,
                    )
                    writer.add_scalar("Holdout/Agent_Fidelity_Mean", float(holdout["agent_fidelity"]["mean"]), episode)
                    writer.add_scalar("Holdout/Agent_Fidelity_Std", float(holdout["agent_fidelity"]["std"]), episode)
                    writer.add_scalar("Holdout/SABRE_Fidelity_Mean", float(holdout["sabre_fidelity"]["mean"]), episode)
                    writer.add_scalar("Holdout/Trivial_Fidelity_Mean", float(holdout["trivial_fidelity"]["mean"]), episode)
                    lookahead_fidelity_mean = float(holdout["lookahead_fidelity"]["mean"])
                    lookahead_cost_mean = float(holdout["lookahead_cost"]["mean"])
                    if np.isfinite(lookahead_fidelity_mean):
                        writer.add_scalar("Holdout/Lookahead_Fidelity_Mean", lookahead_fidelity_mean, episode)
                    writer.add_scalar("Holdout/Greedy_Fidelity_Mean", float(holdout["greedy_fidelity"]["mean"]), episode)
                    writer.add_scalar("Holdout/Agent_Cost_Mean", float(holdout["agent_cost"]["mean"]), episode)
                    writer.add_scalar("Holdout/SABRE_Cost_Mean", float(holdout["sabre_cost"]["mean"]), episode)
                    writer.add_scalar("Holdout/Trivial_Cost_Mean", float(holdout["trivial_cost"]["mean"]), episode)
                    if np.isfinite(lookahead_cost_mean):
                        writer.add_scalar("Holdout/Lookahead_Cost_Mean", lookahead_cost_mean, episode)
                    writer.add_scalar("Holdout/Greedy_Cost_Mean", float(holdout["greedy_cost"]["mean"]), episode)
                except Exception as e:
                    print(f"[holdout] eval failed: {e}")

    writer.flush()
    writer.close()
