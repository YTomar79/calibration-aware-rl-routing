#!/bin/bash
# Create a local virtual environment and install pinned dependencies.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip freeze > requirements.lock.txt

echo "Environment ready at ${SCRIPT_DIR}/.venv"
echo "Run a quick end-to-end check with: bash smoke_test.sh"
