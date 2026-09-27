#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
task=${1:?usage: run_strict.sh cube|tworoom RUN_DIR DATASET_PATH OUTPUT [seed]}
run_dir=${2:?missing RUN_DIR}; dataset=${3:?missing DATASET_PATH}; output=${4:?missing OUTPUT}; seed=${5:-42}
case "$task" in
  cube) cfg=config/eval/env/cube-strict.yaml; manifest=config/strict_manifests/cube-strict-seed42-n100.json ;;
  tworoom) cfg=config/eval/env/tworoom-strict.yaml; manifest=config/strict_manifests/tworoom-strict-seed42-n100.json ;;
  *) echo "task must be cube or tworoom" >&2; exit 2 ;;
esac
exec python scripts/evaluate_strict.py --run-dir "$run_dir" --eval-config "$cfg" --manifest "$manifest" --dataset-path "$dataset" --output "$output" --planner-seed "$seed" --clear-root vendor
