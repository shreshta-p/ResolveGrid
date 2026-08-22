#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --package resolvegrid-api python -m resolvegrid_api.seed "$@"
