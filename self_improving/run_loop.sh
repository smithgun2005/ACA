#!/usr/bin/env bash
set -euo pipefail

# Complete self-improving ACA loop aligned with the old staged protocols:
#   Reacher: baseline 0->5, replay 5->10, cumulative replay 10->25
#   Cube:    baseline 0->5, replay 5->10, cumulative replay 10->30
# The initial RUN_DIR must already contain the completed epoch-5 5% run.

usage() {
  echo "usage: $0 ENV CFG INITIAL_RUN_DIR SOURCE_H5 ROUNDS rho top_fraction EPISODES.npy [eval_config.yaml] [second_stage_epochs] [lr]" >&2
  echo "  ENV: cube | cube-strict | reacher" >&2
  echo "  protocol: exactly 2 rounds; Reacher=5+5+15 -> epoch25, Cube=5+5+20 -> epoch30" >&2
  exit 2
}

[[ $# -ge 8 ]] || usage
env_name=$1
cfg=$2
initial_run=$3
source_h5=$4
rounds=$5
rho=${6:-0.2}
top_fraction=${7:-0.1}
episode_indices=${8:-}
eval_config=${9:-}
second_stage_epochs=${10:-}
lr=${11:-1e-4}

case "$env_name" in cube|cube-strict|reacher) ;; *) usage ;; esac
[[ "$rounds" =~ ^[1-9][0-9]*$ ]] || { echo "ROUNDS must be positive" >&2; exit 2; }
[[ "$rounds" == 2 ]] || { echo "This aligned protocol requires exactly 2 rounds." >&2; exit 2; }
if [[ "$env_name" == "reacher" ]]; then
  default_second_stage_epochs=15
else
  default_second_stage_epochs=20
fi
stage2_epochs=${second_stage_epochs:-$default_second_stage_epochs}
[[ "$stage2_epochs" =~ ^[1-9][0-9]*$ ]] || { echo "second_stage_epochs must be positive" >&2; exit 2; }
[[ "$stage2_epochs" == "$default_second_stage_epochs" ]] || {
  echo "Refusing noncanonical second stage: $env_name requires $default_second_stage_epochs epochs." >&2
  exit 2
}
[[ -d "$initial_run" ]] || { echo "initial run not found: $initial_run" >&2; exit 2; }
[[ -f "$source_h5" ]] || { echo "source HDF5 not found: $source_h5" >&2; exit 2; }
[[ -n "$episode_indices" && -f "$episode_indices" ]] || {
  echo "EPISODES.npy is required for both Cube and Reacher so mining stays inside the 5% pool." >&2
  exit 2
}

cd "$(dirname "$0")/.."
root=$PWD
export REPO_ROOT="$root"
export ACA_DATA_ROOT="${ACA_DATA_ROOT:-$root/data/generated}"
run_dir=$(readlink -f "$initial_run")
source_h5=$(readlink -f "$source_h5")
episode_indices=$(readlink -f "$episode_indices")
export RUNS_ROOT="$(dirname "$run_dir")"
if [[ "$env_name" == "reacher" ]]; then
  export REACHER_SUBSET_INDICES="$episode_indices"
else
  export CUBE_SUBSET_INDICES="$episode_indices"
fi

if [[ -f "$run_dir/config.yaml" ]]; then
  if ! grep -A8 -Eq '^[[:space:]]*aca:' "$run_dir/config.yaml" || \
     ! grep -A8 -Eq '^[[:space:]]+weight:[[:space:]]*[1-9]' "$run_dir/config.yaml"; then
    echo "initial run config does not appear to enable ACA; refusing self-improving loop" >&2
    exit 2
  fi
fi

replay_files=()
for ((round=1; round<=rounds; round++)); do
  checkpoint="$run_dir/checkpoints/last.ckpt"
  [[ -s "$checkpoint" ]] || { echo "missing checkpoint for round $round: $checkpoint" >&2; exit 2; }
  out_dir="$run_dir/self_improving_round${round}"
  cf="$out_dir/aca_counterfactual.h5"
  mkdir -p "$out_dir"

  if [[ "$env_name" == "reacher" ]]; then
    collector=(python scripts/collect_reacher_aca_counterfactuals.py
      --run-dir "$run_dir" --source "$source_h5" --output "$cf"
      --rho "$rho" --top-fraction "$top_fraction"
      --episode-indices "$episode_indices")
  else
    collector=(python scripts/collect_cube_aca_counterfactuals.py
      --run-dir "$run_dir" --source "$source_h5" --output "$cf"
      --rho "$rho" --top-fraction "$top_fraction")
    collector+=(--episode-indices "$episode_indices")
  fi
  echo "[ACA self-improving] round $round/$rounds: mining from $run_dir"
  "${collector[@]}"

  replay_files+=("$cf")
  replay="$cf"
  scheduler_args=(scheduler.enabled=false)
  if [[ "$round" == 2 ]]; then
    # Both legacy protocols train on the cumulative two-round replay.  The
    # Both environments keep the legacy fixed learning rate in the final
    # stage as well: fresh AdamW at 1e-4 with the scheduler disabled.
    union="$run_dir/self_improving_replay_union.h5"
    python scripts/merge_counterfactuals.py --output "$union" "${replay_files[@]}"
    replay="$union"
    stage_epochs="$stage2_epochs"
  else
    # Initial replay adaptation is always the legacy five fresh epochs.
    stage_epochs=5
  fi

  next_run="$RUNS_ROOT/$(basename "$run_dir")_self_round${round}"
  echo "[ACA self-improving] round $round/$rounds: epoch stage ${stage_epochs} into $next_run (replay=$(basename "$replay"))"
  # Old repository protocol: initialize weights from the previous stage,
  # create a fresh optimizer, use the stage-specific scheduler above, and
  # disable ACA during supervised replay adaptation.
  experiments/train/run.sh "$cfg" \
    subdir="$(basename "$next_run")" \
    init_from_checkpoint="$checkpoint" \
    resume_from_checkpoint=null \
    data.counterfactual.enabled=true \
    data.counterfactual.path="$replay" \
    data.counterfactual.only=false \
    trainer.max_epochs="$stage_epochs" \
    trainer.val_check_interval=1.0 \
    optimizer.type=AdamW \
    optimizer.lr="$lr" \
    "${scheduler_args[@]}" \
    loader.num_workers=0 \
    loader.persistent_workers=false \
    loader.prefetch_factor=null \
    loss.aca.weight=0.0 \
    wandb.enabled=false
  run_dir=$(readlink -f "$next_run")

  if [[ -n "$eval_config" ]]; then
    echo "[ACA self-improving] round $round/$rounds: standard CEM evaluation"
    experiments/eval/run.sh "$eval_config" "$run_dir"
  fi
done

echo "[ACA self-improving] complete: final run $run_dir"
