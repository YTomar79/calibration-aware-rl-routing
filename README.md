# Graph Reinforcement Learning for Calibration-Aware Quantum Circuit Routing

**Yash Vardhan Tomar**¹, **Dheeraj Peddireddy**¹

¹ Purdue University

[![arXiv](https://img.shields.io/badge/arXiv-2606.12816-b31b1b.svg)](https://arxiv.org/abs/2606.12816)

---

This repo contains the code, raw CSVs, and results for "Graph Reinforcement Learning for Calibration-Aware Quantum Circuit Routing" (arXiv:2606.12816v3). 

Submitted to IEEE Quantum Week International Workshop on AI for Circuit Synthesis, Optimization, and Discovery 2026.

# Abstract

Quantum circuit routing is a key step in compiling programs for noisy intermediate-scale quantum processors. Routes that appear efficient by standard overhead metrics can still lose fidelity when they pass through poorly calibrated couplers. We study a calibration-aware graph reinforcement-learning router that uses same-day IBM Heron r2 calibration data to choose hardware-edge SWAPs. We train the policy with proximal policy optimization and evaluate it with exact simulated fidelity across nine Munich Quantum Toolkit (MQT) Bench circuits and three calibration snapshots. Across these evaluations, pooled mean exact fidelity is 0.727, compared with 0.440 for SABRE-best20 and 0.481 for target-aware SABRE. We observed that fidelity gains came with higher routed two-qubit counts and were concentrated in 5 qubit and 8 qubit circuit families; under the fixed tree action graph, all 10 qubit families favored SABRE-best20. Overall, our results show that calibration-aware learned routing can improve fidelity beyond gate-count-driven compilation.

## Repository layout

```text
calibration-aware-rl-routing/
│
├── scalable_quantum.py            # Core library + training entry point:
│                                  #   routing environment, PPO agent, noise model,
│                                  #   exact density-matrix fidelity evaluation
├── reviewer_benchmark.py          # Held-out evaluation of a trained policy vs. SABRE baselines
├── run_dqn_routing_baseline.py    # DQN routing baseline on the same environment
├── tune_hyperparameters.py        # Optuna hyperparameter search over the PPO config
├── prepare_mqt_corpus.py          # Build the QASM benchmark corpus from MQT Bench
├── download_calibrations.py       # Fetch backend calibration snapshots from IBM Quantum
│
├── scripts/                       # Result-asset generation (no training logic)
│   ├── build_qce_workshop_assets.py        # Aggregate benchmark shards → figures + tables
│   ├── build_paper_tables.py               # Summary JSON → LaTeX tables
│   └── build_qce_workshop_manuscript_figures.py  # Method/flow diagrams
│
├── tests/                         # Unit tests for benchmark schema and env invariants
│   ├── test_artifacts.py
│   └── fixtures/                  # Small JSON fixture for schema/table tests
│
├── downloaded_calibrations/       # IBM calibration snapshots used in the reported runs
│   ├── ibm_fez_calibration.json
│   ├── ibm_kingston_calibration.json
│   ├── ibm_marrakesh_calibration.json
│   └── calibration_manifest.json  # Provenance (backend, version, snapshot time)
│
├── results/                       # All reported outputs
│   ├── reviewer_benchmark/        # Pooled benchmark: summary JSON + overview plots
│   ├── reviewer_benchmark_shards/ # Per-cell records, one dir per calibration × seed
│   │   └── ibm_<backend>_calibration_seed<N>/
│   │       ├── reviewer_benchmark_summary.json
│   │       ├── reviewer_benchmark_episodes.jsonl   # Per-episode raw records
│   │       └── *.png
│   └── paper_assets/              # Publication-ready artifacts
│       ├── figures/               # PNG + PDF figures
│       ├── tables/                # LaTeX result tables
│       ├── data/                  # CSV backing every figure and table
│       └── captions.md            # Figure/table captions
│
├── setup_env.sh                   # Create .venv and install pinned dependencies
├── smoke_test.sh                  # Fast end-to-end pipeline check on a small problem
├── quantum_credentials.example.sh # Template for IBM Quantum credentials
├── requirements.txt               # Direct dependencies
├── requirements.lock.txt          # Fully pinned versions used for the reported results
├── LICENSE
└── README.md
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

## Citation

If you use this code or find this work helpful, please cite:

```bibtex
@article{tomar2026calibration-aware,
  title={Graph Reinforcement Learning for Calibration-Aware Quantum Circuit Routing},
  author={Tomar, Yash Vardhan and Peddireddy, Dheeraj},
  journal={arXiv preprint arXiv:2606.12816},
  year={2026}
}
```

## License

Released under the [MIT License](LICENSE).
