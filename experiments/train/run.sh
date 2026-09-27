#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export REPO_ROOT="$PWD"
export ACA_DATA_ROOT="${ACA_DATA_ROOT:-$PWD/data/generated}"
cfg=${1:?usage: run.sh CONFIG_NAME}
shift || true
exec python train.py --config-path "$PWD/experiments/train/generated" --config-name "${cfg%.yaml}" "$@"
