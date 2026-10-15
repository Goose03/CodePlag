#!/usr/bin/env bash
# Fit the combiner on validation, evaluate on held-out splits, benchmark efficiency,
# and write metrics + plots to docs/results/.
set -euo pipefail
cd "$(dirname "$0")/.."
codeplag evaluate "$@"
codeplag plots
