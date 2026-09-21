#!/bin/bash
# Install runtime dependencies for the Dormant Radar automation.
#
# The no-LLM automation pattern implies a stdlib-only script, but this project
# needs `requests` (chain HTTP) and `numpy` (the anomaly model), so a setup
# step is required. Without it the entrypoint fails on import.
set -e

python3 -m pip install --quiet --disable-pip-version-check \
  "requests>=2.31" \
  "numpy>=1.24" \
  "fastapi>=0.110" \
  "uvicorn[standard]>=0.29"

echo "dependencies installed"