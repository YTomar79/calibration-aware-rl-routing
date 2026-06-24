"""Build the QASM benchmark corpus from MQT Bench used for training and evaluation."""

import json
import os
from datetime import datetime, timezone


def _env_int_list(name: str, default: str):
    raw = os.getenv(name, default)
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            out.append(int(item))
        except Exception:
            pass
    return out


if __name__ == "__main__":
    try:
        from mqt.bench import BenchmarkLevel, get_benchmark
    except Exception as e:
        raise RuntimeError(
            "Install the optional MQT Bench dependency first, for example: "
            ".venv/bin/pip install mqt-bench"
        ) from e

    try:
        from qiskit import qasm2
    except Exception:
        qasm2 = None
    try:
        from qiskit import qasm3
    except Exception:
        qasm3 = None

    output_dir = os.path.abspath(os.getenv("MQT_OUTPUT_DIR", "benchmark_corpora/mqt_bench"))
    algorithms = [
        item.strip()
        for item in os.getenv("MQT_BENCH_ALGORITHMS", "dj,qft,ghz").split(",")
        if item.strip()
    ]
    qubit_counts = _env_int_list("MQT_QUBIT_COUNTS", "5,8,10")
    level_name = os.getenv("MQT_BENCH_LEVEL", "ALG").strip().upper()
    level = getattr(BenchmarkLevel, level_name)

    os.makedirs(output_dir, exist_ok=True)
    generated = []
    failed = []

    for algorithm in algorithms:
        for num_qubits in qubit_counts:
            try:
                qc = get_benchmark(
                    benchmark=algorithm,
                    level=level,
                    circuit_size=int(num_qubits),
                    random_parameters=True,
                )
                if qasm2 is not None:
                    payload = qasm2.dumps(qc)
                    suffix = "qasm"
                elif qasm3 is not None:
                    payload = qasm3.dumps(qc)
                    suffix = "qasm3"
                else:
                    raise RuntimeError("Neither qiskit.qasm2 nor qiskit.qasm3 is available.")
                path = os.path.join(output_dir, f"{algorithm}_{level_name.lower()}_{num_qubits}q.{suffix}")
                with open(path, "w") as f:
                    f.write(payload)
                generated.append({"algorithm": algorithm, "num_qubits": int(num_qubits), "path": path})
                print(f"OK {algorithm} {num_qubits}q -> {path}")
            except Exception as e:
                failed.append({"algorithm": algorithm, "num_qubits": int(num_qubits), "error": str(e)})
                print(f"FAIL {algorithm} {num_qubits}q: {e}")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "level": level_name,
        "algorithms": algorithms,
        "qubit_counts": qubit_counts,
        "generated": generated,
        "failed": failed,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {manifest_path}")
    if not generated:
        raise RuntimeError("No MQT Bench circuits were generated.")
