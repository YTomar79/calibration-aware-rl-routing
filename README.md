# Graph Reinforcement Learning for Calibration-Aware Quantum Circuit Routing

This repo contains the code, raw CSVs, and results for "Graph Reinforcement Learning for Calibration-Aware Quantum Circuit Routing" (arXiv:2606.12816v3). Submitted to QCE AI4QC Workshop 2026. 


## Results

Held-out benchmark over 30 records (3 IBM calibration snapshots × 10 training
seeds), reporting exact density-matrix fidelity under noisy simulation.

| Router | Exact fidelity (95% CI) | Two-qubit gates | Depth |
| --- | --- | --- | --- |
| **Agent (this work)** | **0.727** (0.711–0.739) | 29.0 | 26.8 |
| Target-aware SABRE | 0.481 (0.464–0.496) | 20.3 | 23.4 |
| SABRE-best20 | 0.440 (0.437–0.443) | 20.2 | 22.2 |

Paired per-cell effects (positive favors the agent, Holm-adjusted):

| Comparison | Δ exact fidelity | Holm p | Cells |
| --- | --- | --- | --- |
| Agent vs SABRE-best20 | +0.287 (0.271–0.299) | 6.0e-6 | 30 |
| Agent vs Target-aware SABRE | +0.246 (0.225–0.267) | 6.0e-6 | 30 |

The agent improves fidelity on every calibration snapshot, trading a higher
two-qubit gate count for the gain. Full figures, tables, and per-cell records
are under [`results/`](results/).

## Repository layout

```text
scalable_quantum.py          Routing environment, PPO agent, noisy-simulation evaluation, training entry point
reviewer_benchmark.py        Held-out benchmark of the trained policy vs. SABRE baselines
run_dqn_routing_baseline.py  DQN routing baseline on the same environment
tune_hyperparameters.py      Optuna hyperparameter search
prepare_mqt_corpus.py        Build the QASM benchmark corpus from MQT Bench
download_calibrations.py     Fetch backend calibration snapshots from IBM Quantum
scripts/                     Paper-asset generation (tables and figures)
tests/                       Schema and invariant tests
downloaded_calibrations/     Calibration snapshots used in the reported runs
results/                     Benchmark summaries, per-seed shards, and paper figures/tables
```

## Installation

Requires Python 3.10+.

```bash
bash setup_env.sh          # creates .venv and installs pinned dependencies
# or, manually:
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

`requirements.lock.txt` pins the exact versions used to produce the reported
results.

## Quick check

Run a fast, small-scale pass through tuning → training → benchmark to confirm
the pipeline works locally:

```bash
bash smoke_test.sh
```

## Reproducing the results

Steps 1–2 require IBM Quantum credentials; the included `results/` and
`downloaded_calibrations/` let you inspect or rebuild paper assets (step 5)
without them.

**1. Credentials** (only for fresh calibration downloads)

```bash
cp quantum_credentials.example.sh quantum_credentials.sh   # fill in token + CRN
chmod 600 quantum_credentials.sh
source quantum_credentials.sh
```

**2. Calibration snapshots**

```bash
DOWNLOAD_BACKEND_NAMES=ibm_fez,ibm_kingston,ibm_marrakesh \
DOWNLOAD_CALIBRATION_DIR=downloaded_calibrations \
.venv/bin/python download_calibrations.py
```

**3. Benchmark corpus**

```bash
.venv/bin/pip install mqt-bench
MQT_BENCH_ALGORITHMS=dj,qft,ghz MQT_QUBIT_COUNTS=5,8,10 \
.venv/bin/python prepare_mqt_corpus.py
```

**4. Train and benchmark**

```bash
# Train (configure seeds/episodes/checkpoint dir via environment variables)
CHECKPOINT_DIR=checkpoints NUM_TRAINING_EPISODES=40001 \
BENCHMARK_QASM_DIR=benchmark_corpora/mqt_bench BENCHMARK_CORPUS_NAME=mqt_bench \
.venv/bin/python scalable_quantum.py

# Evaluate trained checkpoints against the baselines
REVIEW_RUN_DIRS=checkpoints \
REVIEW_CALIBRATION_FILES=downloaded_calibrations/ibm_fez_calibration.json,downloaded_calibrations/ibm_kingston_calibration.json,downloaded_calibrations/ibm_marrakesh_calibration.json \
REVIEW_OUTPUT_DIR=results/reviewer_benchmark \
REVIEW_SHARD_DIR=results/reviewer_benchmark_shards \
.venv/bin/python reviewer_benchmark.py
```

**5. Build figures and tables**

```bash
.venv/bin/python scripts/build_qce_workshop_assets.py \
  --summary results/reviewer_benchmark/reviewer_benchmark_summary.json \
  --shard-dir results/reviewer_benchmark_shards \
  --out-dir results/paper_assets
```

## Data and dependencies

Calibration snapshots are derived from IBM Quantum backends and are included for
reproducibility. Circuits come from [MQT Bench](https://www.cda.cit.tum.de/mqtbench/).
Core dependencies: PyTorch, Qiskit, Qiskit Aer, and Optuna (see
`requirements.txt`).

## License

Released under the MIT License. See [LICENSE](LICENSE).
