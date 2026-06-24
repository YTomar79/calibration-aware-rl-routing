#!/bin/bash
# Create a local virtual environment and install the declared dependencies.
#
# Default install uses requirements.txt (cross-platform, resolves currently
# compatible versions). It does NOT overwrite the committed requirements.lock.txt,
# which is a platform-specific record of the exact environment behind the
# reported results (see that file's header). To reproduce that environment
# bit-for-bit on matching hardware (Linux x86-64 + CUDA 13.2), run instead:
#   .venv/bin/pip install -r requirements.lock.txt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# Record the resolved environment for your own reference (gitignored, so it does
# not clobber the committed lockfile).
.venv/bin/pip freeze > requirements.resolved.txt

echo "Environment ready at ${SCRIPT_DIR}/.venv"
echo "Resolved versions written to requirements.resolved.txt"
echo "Run a quick end-to-end check with: bash smoke_test.sh"
