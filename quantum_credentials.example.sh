#!/bin/bash
# Template for IBM Quantum credentials.
# Copy to quantum_credentials.sh, fill in real values, and keep it out of version control:
#   cp quantum_credentials.example.sh quantum_credentials.sh
#   chmod 600 quantum_credentials.sh
# Then `source quantum_credentials.sh` before downloading calibrations or running fresh benchmarks.

export IBM_QUANTUM_CHANNEL="ibm_cloud"
export IBM_QUANTUM_API_TOKEN="REPLACE_WITH_IBM_QUANTUM_API_TOKEN"
export IBM_QUANTUM_CRN="REPLACE_WITH_IBM_QUANTUM_CRN"

# Clear any stale variables that could override the intended credential source.
unset IBM_QUANTUM_INSTANCE
unset QISKIT_IBM_TOKEN
