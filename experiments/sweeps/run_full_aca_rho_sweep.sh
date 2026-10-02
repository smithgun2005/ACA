#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 cube|reacher|pusht|tworoom inv|sig" >&2
  exit 2
}
[[ $# -eq 2 ]] || usage
database=$1
objective=$2
case "$database" in cube|reacher|pusht|tworoom) ;; *) usage ;; esac
case "$objective" in inv|sig) ;; *) usage ;; esac

cd "$(dirname "$0")/../.."
root=$PWD
source /root/autodl-tmp/sensorimotor-world-model/.venv/bin/activate

train_seed=0
dry_run=${SWEEP_DRY_RUN:-0}
sweep_name="${database}_full_${objective}_aca_w1_rho_sweep_seed0"
sweep_root="$root/results/sweeps/$sweep_name"
train_root="$sweep_root/training"
eval_cfg="$root/config/eval/full/$database.yaml"
rhos=(0p05 0p1 0p25 0p5)
train_cfgs=("${database}_full_${objective}_baseline_seed0")
labels=(baseline)
for rho in "${rhos[@]}"; do
  train_cfgs+=("${database}_full_${objective}_aca_w1_rho${rho}_seed0")
  labels+=("aca_w1_rho${rho}")
done

case "$database" in
  cube)
    planner_seeds=(55025 55026 55027 55028 55029)
    task_seed=45025
    strict_task=cube
    strict_dataset=/root/autodl-tmp/sensorimotor-world-model/planning/data/external/cube_single_expert_eval.h5
    ;;
  reacher)
    planner_seeds=(53025 53026 53027 53028 53029)
    task_seed=43025
    strict_task=
    strict_dataset=
    ;;
  pusht)
    planner_seeds=(54025 54026 54027 54028 54029)
    task_seed=44025
    strict_task=
    strict_dataset=
    ;;
  tworoom)
    planner_seeds=(52025 52026 52027 52028 52029)
    task_seed=42025
    strict_task=tworoom
    strict_dataset=/root/autodl-tmp/sensorimotor-world-model/planning/data/external/tworoom_eval.h5
    ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED="$train_seed"
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export REPO_ROOT="$root"

[[ -s "$eval_cfg" ]] || { echo "Missing eval config: $eval_cfg" >&2; exit 1; }

for i in "${!train_cfgs[@]}"; do
  cfg=${train_cfgs[$i]}
  label=${labels[$i]}
  cfg_file="$root/experiments/train/generated/$cfg.yaml"
  run_dir="$train_root/$cfg"
  [[ -s "$cfg_file" ]] || { echo "Missing training config: $cfg_file" >&2; exit 1; }

  python - "$cfg_file" "$database" "$objective" "$label" <<'PY'
import sys
from omegaconf import OmegaConf

path, database, objective, label = sys.argv[1:]
c = OmegaConf.load(path)
assert int(c.seed) == 0
assert int(c.trainer.devices) == 1
assert int(c.trainer.max_epochs) == 10
assert float(c.optimizer.lr) == 1e-4
assert bool(c.scheduler.enabled) is True
assert float(c.loss.aca.get("noise_scale", 0.0)) == 0.0
if label == "baseline":
    assert float(c.loss.aca.weight) == 0.0
    assert float(c.loss.aca.rho) == 0.0
else:
    expected = {"aca_w1_rho0p05": 0.05, "aca_w1_rho0p1": 0.1,
                "aca_w1_rho0p25": 0.25, "aca_w1_rho0p5": 0.5}[label]
    assert float(c.loss.aca.weight) == 1.0
    assert float(c.loss.aca.rho) == expected
if objective == "inv":
    assert float(c.loss.inverse.weight) > 0 and float(c.loss.sigreg.weight) == 0
else:
    assert float(c.loss.sigreg.weight) == 0.09 and float(c.loss.inverse.weight) == 0
print(f"protocol OK: {database}/{objective}/{label}")
PY

  if [[ "$dry_run" == 1 ]]; then
    continue
  fi

  if [[ -s "$run_dir/checkpoints/last.ckpt" && -s "$run_dir/config.yaml" ]]; then
    echo "===== SKIP TRAIN $database/$objective/$label (checkpoint exists) ====="
  else
    echo "===== TRAIN $database/$objective/$label seed=$train_seed ====="
    mkdir -p "$run_dir"


    script -q -e -f -c \
      "RUNS_ROOT='$train_root' experiments/train/run.sh '$cfg' seed='$train_seed' +trainer.deterministic=true +trainer.benchmark=false" \
      "$run_dir/train.log"
  fi
done

if [[ "$dry_run" == 1 ]]; then
  echo "===== DRY RUN OK: $sweep_name ====="
  exit 0
fi


for i in "${!train_cfgs[@]}"; do
  cfg=${train_cfgs[$i]}
  label=${labels[$i]}
  run_dir="$train_root/$cfg"
  [[ -s "$run_dir/checkpoints/last.ckpt" && -s "$run_dir/config.yaml" ]] || {
    echo "Incomplete training run: $run_dir" >&2
    exit 1
  }

  for planner_seed in "${planner_seeds[@]}"; do
    out="$sweep_root/eval/standard/$label/planner_seed_$planner_seed"
    if [[ -s "$out/summary.json" ]]; then
      echo "===== SKIP EVAL $database/$objective/$label planner_seed=$planner_seed ====="
    else
      echo "===== EVAL $database/$objective/$label planner_seed=$planner_seed task_seed=$task_seed ====="
      mkdir -p "$out"
      MODEL_RUN_DIR="$run_dir" RUNS_ROOT="$out" PYTHONHASHSEED="$planner_seed" \
        python -u eval.py --config "$eval_cfg" seed="$planner_seed" eval.task_seed="$task_seed"
    fi
  done

  if [[ -n "$strict_task" ]]; then
    for planner_seed in "${planner_seeds[@]}"; do
      out="$sweep_root/eval/strict/$label/planner_seed_$planner_seed.json"
      if [[ -s "$out" ]]; then
        echo "===== SKIP STRICT $database/$objective/$label planner_seed=$planner_seed ====="
      else
        echo "===== STRICT $database/$objective/$label planner_seed=$planner_seed ====="
        mkdir -p "$(dirname "$out")"
        scripts/run_strict.sh "$strict_task" "$run_dir" "$strict_dataset" "$out" "$planner_seed"
      fi
    done
  fi
done

echo "===== COMPLETE: $sweep_name ====="
