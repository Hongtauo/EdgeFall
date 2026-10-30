#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[1/2] Verifying EdgeFall-ST installation..."
python tools/verify_install.py --device auto

echo "[2/2] Running the bundled demo videos..."
python -u infer.py --input demo/source --device auto

echo "Demo completed. Results are under the results directory."
