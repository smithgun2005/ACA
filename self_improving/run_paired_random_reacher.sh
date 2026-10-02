#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 CFG INITIAL_RUN SELF_ROUND1_H5 SELF_ROUND2_H5 SOURCE_H5 EPISODES.npy OUTPUT_ROOT [rho] [seed] [resume_optimizer]" >&2
  exit 2
}
[[ $# -ge 7 ]] || usage

cfg=$1
initial_run=$(readlink -f "$2")
self_round1_h5=$(readlink -f "$3")
self_round2_h5=$(readlink -f "$4")
source_h5=$(readlink -f "$5")
episodes=$(readlink -f "$6")
output_root=$7
rho=${8:-0.2}
seed=${9:-0}
resume_optimizer=${10:-false}

cd "$(dirname "$0")/.."
root=$PWD
export RUNS_ROOT=$(readlink -m "$output_root")
mkdir -p "$RUNS_ROOT"

round1_run="$RUNS_ROOT/reacher_5pct_inv_paired_random_round1"
round2_run="$RUNS_ROOT/reacher_5pct_inv_paired_random_round2"
replay_root="$RUNS_ROOT/paired_random_replay"
mkdir -p "$replay_root"

extract_indices() {
  local source_replay=$1
  local output_npy=$2
  python - "$source_replay" "$output_npy" <<'PY'
import sys
import h5py
import numpy as np

with h5py.File(sys.argv[1], "r") as src:
    indices = np.asarray(src["source_index"][:], dtype=np.int64)
if indices.ndim != 1 or len(indices) == 0 or len(np.unique(indices)) != len(indices):
    raise ValueError(f"invalid paired source indices in {sys.argv[1]}")
np.save(sys.argv[2], indices)
print(f"paired indices: {len(indices):,} -> {sys.argv[2]}")
PY
}

check_indices() {
  local self_replay=$1
  local random_replay=$2
  python - "$self_replay" "$random_replay" <<'PY'
import sys
import h5py
import numpy as np

with h5py.File(sys.argv[1], "r") as a, h5py.File(sys.argv[2], "r") as b:
    ai = np.asarray(a["source_index"][:], dtype=np.int64)
    bi = np.asarray(b["source_index"][:], dtype=np.int64)
if not np.array_equal(ai, bi):
    raise AssertionError("paired-random source_index differs from self-improving source_index")
print(f"exact source_index match: {len(ai):,} transitions")
PY
}

train_stage() {
  local init_ckpt=$1
  local replay=$2
  local subdir=$3
  local epochs=$4
  if [[ "$resume_optimizer" == true ]]; then
    experiments/train/run.sh "$cfg" \
      subdir="$subdir" \
      init_from_checkpoint=null \
      resume_from_checkpoint="$init_ckpt" \
      data.counterfactual.enabled=true \
      data.counterfactual.path="$replay" \
      data.counterfactual.only=false \
      trainer.max_epochs="$epochs" \
      trainer.val_check_interval=1.0 \
      optimizer.type=AdamW optimizer.lr=1e-4 \
      scheduler.enabled=false loss.aca.weight=0.0 wandb.enabled=false
  else
    experiments/train/run.sh "$cfg" \
      subdir="$subdir" \
      init_from_checkpoint="$init_ckpt" \
      resume_from_checkpoint=null \
      data.counterfactual.enabled=true \
      data.counterfactual.path="$replay" \
      data.counterfactual.only=false \
      trainer.max_epochs="$epochs" \
      trainer.val_check_interval=1.0 \
      optimizer.type=AdamW optimizer.lr=1e-4 \
      scheduler.enabled=false loss.aca.weight=0.0 wandb.enabled=false
  fi
}

round1_indices="$replay_root/self_round1_source_indices.npy"
round1_replay="$replay_root/random_round1.h5"
extract_indices "$self_round1_h5" "$round1_indices"
if [[ ! -s "$round1_replay" ]]; then
  python scripts/collect_reacher_aca_counterfactuals.py \
    --run-dir "$initial_run" --source "$source_h5" --output "$round1_replay" \
    --rho "$rho" --top-fraction 1 --episode-indices "$episodes" \
    --source-indices "$round1_indices" --action-mode random \
    --seed "$seed" --action-seed "$seed"
fi
check_indices "$self_round1_h5" "$round1_replay"

if [[ ! -s "$round1_run/checkpoints/last.ckpt" ]]; then
  train_stage "$initial_run/checkpoints/last.ckpt" "$round1_replay" \
    "$(basename "$round1_run")" "$([[ "$resume_optimizer" == true ]] && echo 10 || echo 5)"
fi

round2_indices="$replay_root/self_round2_source_indices.npy"
round2_replay="$replay_root/random_round2.h5"
extract_indices "$self_round2_h5" "$round2_indices"
if [[ ! -s "$round2_replay" ]]; then
  python scripts/collect_reacher_aca_counterfactuals.py \
    --run-dir "$round1_run" --source "$source_h5" --output "$round2_replay" \
    --rho "$rho" --top-fraction 1 --episode-indices "$episodes" \
    --source-indices "$round2_indices" --action-mode random \
    --seed "$seed" --action-seed "$((seed + 1))"
fi
check_indices "$self_round2_h5" "$round2_replay"

random_union="$replay_root/random_round1_round2_union.h5"
if [[ ! -s "$random_union" ]]; then
  python scripts/merge_counterfactuals.py --output "$random_union" \
    "$round1_replay" "$round2_replay"
fi
if [[ ! -s "$round2_run/checkpoints/last.ckpt" ]]; then
  train_stage "$round1_run/checkpoints/last.ckpt" "$random_union" \
    "$(basename "$round2_run")" "$([[ "$resume_optimizer" == true ]] && echo 25 || echo 15)"
fi

echo "[paired random] complete: $round2_run"
