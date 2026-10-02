#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
root=$PWD
source /root/autodl-tmp/sensorimotor-world-model/.venv/bin/activate



dry_run=${TWOROOM_STRICT_DRY_RUN:-0}




baseline_run=${TWOROOM_BASELINE_RUN:-/root/autodl-tmp/ACA1/results/seeded_tworoom_seed0/training/tworoom_full_inv}
best_inv_run=${TWOROOM_BEST_INV_RUN:-/root/autodl-tmp/ACA1/results/seeded_tworoom_seed0/training/tworoom_full_inv_aca_rho01}
dataset=${TWOROOM_HELDOUT_DATASET:-/root/autodl-tmp/sensorimotor-world-model/planning/data/external/tworoom_eval.h5}
output_root=${TWOROOM_STRICT_OUTPUT_ROOT:-$root/results/reeval_tworoom_strict_heldout}
planner_seeds=(52025 52026 52027 52028 52029)
run_dirs=("$baseline_run" "$best_inv_run")
labels=(inv_baseline inv_aca_w1_rho0p1)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled

[[ -s "$dataset" ]] || { echo "Missing held-out dataset: $dataset" >&2; exit 1; }
for run_dir in "${run_dirs[@]}"; do
  [[ -s "$run_dir/config.yaml" && -s "$run_dir/checkpoints/last.ckpt" ]] || {
    echo "Missing trained run: $run_dir" >&2
    exit 1
  }
done

python - "$root/config/strict_manifests/tworoom-strict-heldout-seed42-n100.json" "$dataset" "$root/vendor" <<'PY'
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from clear_lewm.datasets import metadata_fingerprint

manifest = json.loads(Path(sys.argv[1]).read_text())
dataset = Path(sys.argv[2]).resolve()
assert manifest["split"] == "heldout"
assert float(manifest["protocol"]["heldout_fraction"]) == 1.0
assert manifest["dataset"]["name"] == dataset.name == "tworoom_eval.h5"
assert manifest["dataset"]["fingerprint"]["value"] == metadata_fingerprint(dataset)
assert len(manifest["pairs"]) == 100
print("held-out manifest and dataset fingerprint OK")
PY

if [[ "$dry_run" == 1 ]]; then
  echo "===== DRY RUN OK: TwoRoom strict held-out best baseline + best INV ====="
  exit 0
fi

for i in "${!run_dirs[@]}"; do
  run_dir=${run_dirs[$i]}
  label=${labels[$i]}
  for planner_seed in "${planner_seeds[@]}"; do
    output="$output_root/$label/planner_seed_$planner_seed.json"
    if [[ -s "$output" ]]; then
      echo "===== SKIP $label planner_seed=$planner_seed ====="
    else
      echo "===== TWOROOM STRICT HELD-OUT $label planner_seed=$planner_seed ====="
      mkdir -p "$(dirname "$output")"
      scripts/run_strict.sh tworoom "$run_dir" "$dataset" "$output" "$planner_seed"
    fi
  done
done

echo "===== COMPLETE: TwoRoom strict held-out best baseline + best INV ====="
