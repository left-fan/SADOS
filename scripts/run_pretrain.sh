#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/amazon.yaml}"
SPLIT="${2:-cluster}"

python -m experiments.pretrain --config "$CONFIG" --split "$SPLIT"
