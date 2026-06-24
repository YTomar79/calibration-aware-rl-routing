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
├── setup_env.sh                   # Create .venv and install declared dependencies
├── smoke_test.sh                  # Fast end-to-end pipeline check on a small problem
├── quantum_credentials.example.sh # Template for IBM Quantum credentials
├── requirements.txt               # Direct dependencies (portable install path)
├── requirements.lock.txt          # Provenance: exact Linux+CUDA 13.2 env behind the results
├── LICENSE
└── README.md
```

## Installation

Requires Python 3.10+.

```bash
bash setup_env.sh          # creates .venv and installs the declared dependencies
# or, manually:
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

`requirements.txt` lists the direct dependencies and is the portable install path
(macOS / Linux, CPU or GPU). `requirements.lock.txt` is a separate **provenance
record** of the exact environment behind the reported results — a `pip freeze`
from Linux x86-64 with CUDA 13.2. Because its NVIDIA/CUDA and `triton` pins are
Linux+GPU only, it is not installed by default. To reproduce that environment
bit-for-bit on matching hardware:

```bash
.venv/bin/pip install -r requirements.lock.txt
```

## Quick check

Run a fast, small-scale pass through tuning → training → benchmark to confirm
the pipeline works locally:

```bash
bash smoke_test.sh
```

## Reproducing the experiments

The full pipeline below reproduces the reported results end to end: calibration
snapshots → benchmark corpus → 10-seed training sweep → held-out benchmark →
figures and tables. Steps 2–3 require IBM Quantum credentials and a network
connection; the shipped `downloaded_calibrations/` and `results/` let you skip
straight to step 7 (asset generation) without them.

Assumes Python 3.10+ and that all commands run from the repository root.

### 1. Clone and install

```bash
git clone https://github.com/YTomar79/calibration-aware-rl-routing.git
cd calibration-aware-rl-routing
bash setup_env.sh          # creates .venv and installs the declared dependencies
```

### 2. Configure IBM Quantum credentials

Only needed to download fresh calibration snapshots (step 3).

```bash
cp quantum_credentials.example.sh quantum_credentials.sh   # fill in token + CRN
chmod 600 quantum_credentials.sh
source quantum_credentials.sh
```

### 3. Download calibration snapshots

Fetches per-backend calibration data and writes a provenance manifest.

```bash
DOWNLOAD_BACKEND_NAMES=ibm_fez,ibm_kingston,ibm_marrakesh \
DOWNLOAD_CALIBRATION_DIR=downloaded_calibrations \
.venv/bin/python download_calibrations.py
```

### 4. Build the benchmark corpus

Generates the QASM circuits (Deutsch–Jozsa, QFT, GHZ at 5/8/10 qubits) used for
training and evaluation.

```bash
.venv/bin/pip install mqt-bench
MQT_BENCH_ALGORITHMS=dj,qft,ghz MQT_QUBIT_COUNTS=5,8,10 \
MQT_OUTPUT_DIR=benchmark_corpora/mqt_bench \
.venv/bin/python prepare_mqt_corpus.py
```

### 5. (Optional) Generate your own hyperparameters

Training in step 6 loads `optimal_hyperparams.json` from the repo root. The repo
ships the tuned file used for the reported results, so this step is only needed
to produce your own. It runs an Optuna TPE search and, on completion, writes the
best configuration to that same file (plus search diagnostics under
`tuning_artifacts/`):

```bash
OPTUNA_TRIALS=50 \
OPTUNA_STUDY_NAME=quantum_routing_tuning \
.venv/bin/python tune_hyperparameters.py
```

The resulting `optimal_hyperparams.json` is the only file step 6 needs, so once
this finishes you can train directly. A few ways to control it:

- **Keep the shipped file** and write the new one elsewhere by setting
  `OPTIMAL_HYPERPARAMS_PATH=my_hyperparams.json`. Training only reads
  `optimal_hyperparams.json`, so copy your file over it when ready to use it:
  `cp my_hyperparams.json optimal_hyperparams.json`.
- **Hand-edit** `optimal_hyperparams.json` directly — any subset of keys is
  accepted, and unspecified values fall back to the built-in defaults.
- **Run a larger or parallel search** with more `OPTUNA_TRIALS`, or point several
  workers at one shared study via `OPTUNA_STORAGE` (an Optuna storage URL) or
  `OPTUNA_JOURNAL_PATH` with a common `OPTUNA_STUDY_NAME`.

### 6. Train the policy (10-seed sweep)

The reported results pool ten independently seeded policies. Each is trained with
a distinct `TRAINING_SEED_OFFSET` into its own checkpoint directory.

```bash
SEED_OFFSETS=(0 100000 200000 300000 400000 500000 600000 700000 800000 900000)
for i in "${!SEED_OFFSETS[@]}"; do
  TRAINING_SEED_OFFSET="${SEED_OFFSETS[$i]}" \
  CHECKPOINT_DIR="checkpoints/seed${i}" \
  NUM_TRAINING_EPISODES=40001 \
  BENCHMARK_QASM_DIR=benchmark_corpora/mqt_bench \
  BENCHMARK_CORPUS_NAME=mqt_bench BENCHMARK_CORPUS_PROB=1 \
  .venv/bin/python scalable_quantum.py
done
```

Training is compute-intensive; run the seeds in parallel across machines/jobs if
available. For a single quick sanity run, use `bash smoke_test.sh` instead.

### 7. Run the held-out benchmark

Evaluates every seed checkpoint against the SABRE baselines across all three
calibration snapshots (3 × 10 = 30 cells), writing per-cell shards and a pooled
summary.

```bash
REVIEW_RUN_DIRS="checkpoints/seed0,checkpoints/seed1,checkpoints/seed2,checkpoints/seed3,checkpoints/seed4,checkpoints/seed5,checkpoints/seed6,checkpoints/seed7,checkpoints/seed8,checkpoints/seed9" \
REVIEW_RUN_LABELS="seed0,seed1,seed2,seed3,seed4,seed5,seed6,seed7,seed8,seed9" \
REVIEW_CALIBRATION_FILES="downloaded_calibrations/ibm_fez_calibration.json,downloaded_calibrations/ibm_kingston_calibration.json,downloaded_calibrations/ibm_marrakesh_calibration.json" \
REVIEW_HOLDOUT_EPISODES=50 \
REVIEW_BASELINES="sabre,qiskit_noise_aware_vf2" \
BENCHMARK_QASM_DIR=benchmark_corpora/mqt_bench \
BENCHMARK_CORPUS_NAME=mqt_bench BENCHMARK_CORPUS_PROB=1 \
REVIEW_SHARD_DIR=results/reviewer_benchmark_shards \
REVIEW_OUTPUT_DIR=results/reviewer_benchmark \
.venv/bin/python reviewer_benchmark.py
```

### 8. Generate figures and tables

Turns the benchmark summary and shards into the publication assets under
`results/paper_assets/`. This step needs only the files from step 7 (or the ones
already shipped in the repo).

```bash
.venv/bin/python scripts/build_qce_workshop_assets.py \
  --summary results/reviewer_benchmark/reviewer_benchmark_summary.json \
  --shard-dir results/reviewer_benchmark_shards \
  --out-dir results/paper_assets
```

The regenerated `results/paper_assets/data/table01_overall_summary.csv` should
match the numbers in the [Results](#results) table above.

## Testing

```bash
.venv/bin/python -m unittest discover -s tests
```

Tests covering the benchmark schema and table generation run without optional
heavy dependencies; environment-invariant tests are skipped when Qiskit is not
installed. The same suite runs in CI (`.github/workflows/ci.yml`) on Python
3.10–3.12, installing `requirements.txt` so a broken install fails the build.


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
