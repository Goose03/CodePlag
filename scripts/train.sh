#!/usr/bin/env bash
# Fine-tune the siamese model. Uses CODEPLAG_DEVICE (cpu|cuda); on CPU consider
# `bash scripts/train.sh --max-pairs 4000 --epochs 2` to keep it under an hour.
set -euo pipefail
cd "$(dirname "$0")/.."
codeplag train "$@"
