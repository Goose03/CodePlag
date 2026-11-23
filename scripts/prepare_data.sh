#!/usr/bin/env bash
# Download Project CodeNet Python800/Java250, clean, split by problem and build labeled pairs.
# Output: data/processed/{train,val,test,hard_*,unseen_java}.jsonl and docs/results/dataset_stats.json
set -euo pipefail
cd "$(dirname "$0")/.."
codeplag prepare-data "$@"
