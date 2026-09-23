#!/bin/bash
# Install runtime dependencies for the Dormant Radar automation.
#
# Only the two packages the `cloud-run` path actually imports:
#   requests  - chain HTTP (chain.py, price.py)
#   numpy     - the anomaly model (nn.py, anomaly.py, features.py)
#
# Deliberately NOT installing fastapi or uvicorn. The API and web UI are served
# by `_cmd_serve` and `_cmd_worker`, which this automation never invokes, and
# uvicorn[standard] drags in uvloop/httptools/watchfiles/websockets/PyYAML and
# can compile from source. An earlier version installed them anyway, and the
# setup step consumed the entire 11-minute run budget without ever reaching the
# entrypoint.
set -e

python3 -m pip install --quiet --disable-pip-version-check --no-cache-dir \
  "requests>=2.31,<3" \
  "numpy>=1.24"

# Prove the imports resolve here rather than discovering it at run time.
python3 -c "import requests, numpy; print('deps ok:', requests.__version__, numpy.__version__)"

echo "dependencies installed"