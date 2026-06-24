#!/bin/bash
# Fast end-to-end check: tiny tuning -> training -> benchmark run on a small qubit count.
# Intended to validate the pipeline locally in a few minutes, not to reproduce paper results.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: ${PYTHON_BIN} is missing. Run: bash setup_env.sh"
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(
        f"Python >= 3.10 is required; got {sys.version.split()[0]}."
    )
print(f"Using Python {sys.version.split()[0]}")
PY

export TRAIN_NUM_QUBITS=5
export EXACT_DM_MAX_QUBITS=5
export USE_PROXY_REWARD=1
export NUM_TRAINING_EPISODES=50
export CHECKPOINT_FREQUENCY=50
export TRAIN_FREQUENCY=10
export CHECKPOINT_DIR="${PROJECT_ROOT}/smoke_checkpoints"
export TENSORBOARD_LOG_DIR="${PROJECT_ROOT}/smoke_tensorboard"
export OPTUNA_TRIALS=2
export TUNE_EPISODES=20
export TUNE_EXACT_EVAL_EPISODES=2
export TUNE_NUM_QUBITS=5
export TUNE_MAX_STEPS=80
export REVIEW_RUN_DIRS="${CHECKPOINT_DIR}"
export REVIEW_HOLDOUT_EPISODES=20
export REVIEW_HOLDOUT_START_SEED=90000
export REVIEW_OUTPUT_DIR="${PROJECT_ROOT}/smoke_reviewer_benchmark"
export REVIEW_REQUIRE_FRESH_CALIBRATIONS="${REVIEW_REQUIRE_FRESH_CALIBRATIONS:-0}"
export EVAL_HOLDOUT_EPISODES=2
export EVAL_BASELINES_ON_CHECKPOINT=0
export DISABLE_PLOT_SHOW=1
export REQUIRE_PIPELINE_ARTIFACTS=1

rm -rf "${CHECKPOINT_DIR}" "${TENSORBOARD_LOG_DIR}" "${REVIEW_OUTPUT_DIR}"

echo "== Smoke: hyperparameter tuning =="
"${PYTHON_BIN}" tune_hyperparameters.py

echo "== Smoke: training =="
"${PYTHON_BIN}" scalable_quantum.py

echo "== Smoke: benchmark =="
"${PYTHON_BIN}" reviewer_benchmark.py

test -s "${REVIEW_OUTPUT_DIR}/reviewer_benchmark_summary.json"
echo "Smoke test passed: ${REVIEW_OUTPUT_DIR}/reviewer_benchmark_summary.json"
