import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


class ArtifactSchemaTests(unittest.TestCase):
    def test_reviewer_summary_baselines(self):
        data = json.loads((FIXTURES / "reviewer_benchmark_summary_fixture.json").read_text())
        aggregate = data["aggregate"]
        self.assertEqual(data["protocol"]["secondary_router_comparison_statistical_unit"], "run_dir x calibration_file cluster")
        self.assertIn("holm_family_size", data["protocol"])
        self.assertIn("agent_completion_rate", data)
        self.assertIn("agent_timeout_rate", data)
        self.assertIn("agent_mean_progress", data)
        self.assertIn("completion", data)
        self.assertGreaterEqual(data["agent_completion_rate"], data["protocol"]["min_agent_completion_rate"])
        for key in [
            "sabre_fidelity",
            "sabre_trivial_layout_fidelity",
            "qiskit_noise_aware_vf2_fidelity",
            "greedy_fidelity",
            "lookahead_fidelity",
            "random_fidelity",
        ]:
            self.assertIn(key, aggregate)
        for item in data["paired_tests"].values():
            self.assertEqual(item["statistical_unit"], "run_dir_x_calibration_file_cluster")

    def test_completion_summary_and_threshold_helpers(self):
        try:
            import qiskit  # noqa: F401
            from reviewer_benchmark import _completion_summary, _enforce_completion_threshold
        except Exception:
            self.skipTest("qiskit-backed benchmark module is not available in this environment")

        episodes = [
            {
                "target_circuit_type": "a",
                "agent_completed": True,
                "agent_progress": 1.0,
                "agent_terminal_reason": "completed",
                "agent": {"completed": True, "progress": 1.0, "terminal_reason": "completed"},
            },
            {
                "target_circuit_type": "a",
                "agent_completed": False,
                "agent_progress": 0.25,
                "agent_terminal_reason": "timeout",
                "agent": {"completed": False, "progress": 0.25, "terminal_reason": "timeout", "timed_out": True},
            },
        ]
        summary = _completion_summary(episodes)
        self.assertAlmostEqual(summary["agent_completion_rate"], 0.5)
        self.assertAlmostEqual(summary["agent_timeout_rate"], 0.5)
        self.assertAlmostEqual(summary["agent_mean_progress"], 0.625)
        with self.assertRaises(RuntimeError):
            _enforce_completion_threshold(summary, min_rate=0.98)

    def test_build_paper_tables_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.check_call(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_paper_tables.py"),
                    "--benchmark",
                    str(FIXTURES / "reviewer_benchmark_summary_fixture.json"),
                    "--out",
                    tmp,
                ],
                cwd=str(ROOT),
            )
            outputs = list(Path(tmp).glob("*.tex"))
            self.assertTrue(outputs)
            for path in outputs:
                self.assertGreater(path.stat().st_size, 0)

    def test_noise_aware_vf2_selection_does_not_use_exact_fidelity(self):
        try:
            import qiskit  # noqa: F401
            import scalable_quantum
            from scalable_quantum import QuantumRoutingEnv
        except Exception:
            self.skipTest("qiskit is not installed in this environment")

        class DummyCircuit:
            def __init__(self, idx):
                self.idx = idx

            def size(self):
                return 100 + self.idx

        calls = []

        def fake_transpile(*args, **kwargs):
            calls.append(kwargs)
            return DummyCircuit(len(calls) - 1)

        env = QuantumRoutingEnv.__new__(QuantumRoutingEnv)
        env.target_circuit = object()
        env.optimization_level = 3
        env._episode_seed = 123
        env.sabre_baseline_trials = 2
        env._build_qiskit_noise_aware_target = lambda: object()
        env._compiled_metrics = lambda compiled: {
            "cost": 10.0 + compiled.idx,
            "twoq": 5 + compiled.idx,
            "depth": 20 + compiled.idx,
        }
        env._calculate_proxy_fidelity = lambda compiled: 0.9 - (0.1 * compiled.idx)
        env.get_current_fidelity = lambda compiled: (_ for _ in ()).throw(
            AssertionError("candidate selection must not call exact fidelity")
        )

        old_transpile = scalable_quantum.transpile
        try:
            scalable_quantum.transpile = fake_transpile
            selected = env._compile_qiskit_noise_aware_vf2_best_of_n()
        finally:
            scalable_quantum.transpile = old_transpile

        self.assertEqual(selected.idx, 0)
        self.assertTrue(calls)
        self.assertTrue(all(call.get("layout_method") == "vf2" for call in calls))
        self.assertTrue(all(call.get("routing_method") == "sabre" for call in calls))

    def test_qasm_hash_exclusion_skips_duplicates(self):
        try:
            import qiskit  # noqa: F401
        except Exception:
            self.skipTest("qiskit is not installed in this environment")

        from scalable_quantum import QuantumRoutingEnv

        qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; cx q[0],q[1];'
        with tempfile.TemporaryDirectory() as tmp:
            train_dir = Path(tmp) / "train"
            eval_dir = Path(tmp) / "eval"
            train_dir.mkdir()
            eval_dir.mkdir()
            (train_dir / "dup.qasm").write_text(qasm)
            (eval_dir / "dup.qasm").write_text(qasm)
            manifest = Path(tmp) / "hashes.json"

            old_manifest = os.environ.get("BENCHMARK_QASM_HASH_MANIFEST")
            old_exclude = os.environ.get("BENCHMARK_QASM_EXCLUDE_HASHES_FILE")
            try:
                os.environ["BENCHMARK_QASM_HASH_MANIFEST"] = str(manifest)
                QuantumRoutingEnv(num_qubits=2, benchmark_qasm_dir=str(train_dir), benchmark_corpus_prob=1.0)
                os.environ.pop("BENCHMARK_QASM_HASH_MANIFEST", None)
                os.environ["BENCHMARK_QASM_EXCLUDE_HASHES_FILE"] = str(manifest)
                with self.assertRaises(RuntimeError):
                    QuantumRoutingEnv(num_qubits=2, benchmark_qasm_dir=str(eval_dir), benchmark_corpus_prob=1.0)
            finally:
                if old_manifest is None:
                    os.environ.pop("BENCHMARK_QASM_HASH_MANIFEST", None)
                else:
                    os.environ["BENCHMARK_QASM_HASH_MANIFEST"] = old_manifest
                if old_exclude is None:
                    os.environ.pop("BENCHMARK_QASM_EXCLUDE_HASHES_FILE", None)
                else:
                    os.environ["BENCHMARK_QASM_EXCLUDE_HASHES_FILE"] = old_exclude


class EnvCompletionInvariantTests(unittest.TestCase):
    def setUp(self):
        try:
            import qiskit  # noqa: F401
            import scalable_quantum
            from scalable_quantum import QuantumRoutingEnv
        except Exception:
            self.skipTest("qiskit is not installed in this environment")
        self.scalable_quantum = scalable_quantum
        self.QuantumRoutingEnv = QuantumRoutingEnv

    def test_quantum_volume_normalizes_to_router_supported_ops(self):
        from qiskit.circuit.library import QuantumVolume

        env = self.QuantumRoutingEnv(num_qubits=5, use_proxy_reward=True)
        env._episode_seed = 123
        qc = QuantumVolume(num_qubits=5, depth=5, seed=123)
        normalized = env._normalize_target_circuit(qc, context="quantum_volume_test")
        self.assertTrue(normalized.data)
        self.assertLessEqual(max(len(instr.qubits) for instr in normalized.data), 2)

    def test_unsupported_three_qubit_instruction_raises_clear_error(self):
        from qiskit import QuantumCircuit
        from qiskit.circuit import Gate

        env = self.QuantumRoutingEnv(num_qubits=3, use_proxy_reward=True)
        qc = QuantumCircuit(3)
        qc.append(Gate("opaque3", 3, []), [0, 1, 2])
        with self.assertRaisesRegex(ValueError, "0/1/2-qubit"):
            env._normalize_target_circuit(qc, context="opaque3_test")

    def test_normalized_target_hash_is_stable_for_same_seed(self):
        from scalable_quantum import _circuit_fingerprint

        env = self.QuantumRoutingEnv(num_qubits=5, use_proxy_reward=True)
        hashes = []
        for _ in range(2):
            env._episode_seed = 77
            env.rng = __import__("numpy").random.default_rng(77)
            env._target_type_probs = __import__("numpy").asarray([0.0, 0.0, 1.0, 0.0, 0.0])
            qc = env._generate_target_circuit()
            normalized = env._normalize_target_circuit(qc, context=env.target_circuit_type)
            hashes.append(_circuit_fingerprint(normalized))
        self.assertEqual(hashes[0], hashes[1])

    def test_positive_control_completes_at_reset(self):
        env = self.QuantumRoutingEnv(
            num_qubits=2,
            use_proxy_reward=True,
            positive_control_prob=1.0,
            random_2q_prob=0.0,
            qaoa_prob=0.0,
            quantum_volume_prob=0.0,
            vqe_prob=0.0,
            clifford_prob=1.0,
        )
        env.reset(episode=1)
        self.assertTrue(env.completed_target)
        self.assertEqual(env.last_metrics["terminal_reason"], "completed")
        self.assertAlmostEqual(env.last_metrics["progress"], 1.0)
        self.assertTrue(math.isfinite(float(env.last_metrics["cost"])))

    def test_timeout_marks_partial_solution_invalid_and_reward_negative(self):
        from qiskit import QuantumCircuit

        env = self.QuantumRoutingEnv(
            num_qubits=3,
            use_proxy_reward=True,
            max_steps_per_episode=1,
            max_steps_base=0,
            max_steps_per_2q=0.0,
            max_steps_cap=1,
            incomplete_episode_penalty=10.0,
            random_2q_prob=1.0,
            qaoa_prob=0.0,
            quantum_volume_prob=0.0,
            vqe_prob=0.0,
            clifford_prob=0.0,
        )

        def forced_target():
            qc = QuantumCircuit(3)
            qc.cx(0, 2)
            env.target_circuit_type = "forced_nonlocal"
            env.target_circuit_source = "test"
            env.target_circuit_source_sha256 = None
            return qc

        env._generate_target_circuit = forced_target
        env.reset(episode=2)
        _, reward, done, info = env.step(-1)
        self.assertTrue(done)
        self.assertFalse(info["completed_target"])
        self.assertTrue(info["timed_out"])
        self.assertEqual(info["fidelity"], 0.0)
        self.assertEqual(info["proxy_fidelity"], 0.0)
        self.assertTrue(math.isnan(float(info["cost"])))
        self.assertTrue(math.isnan(float(info["twoq"])))
        self.assertTrue(math.isnan(float(info["depth"])))
        self.assertTrue(math.isfinite(float(info["raw_partial_cost"])))
        self.assertLess(reward, 0.0)


if __name__ == "__main__":
    unittest.main()
