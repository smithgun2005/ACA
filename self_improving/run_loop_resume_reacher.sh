#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 CFG INITIAL_RUN SOURCE_H5 EPISODES.npy OUTPUT_ROOT [rho] [top_fraction] [seed]" >&2
  exit 2
}
[[ $# -ge 5 ]] || usage

cfg=$1
initial_run=$(readlink -f "$2")
source_h5=$(readlink -f "$3")
episodes=$(readlink -f "$4")
output_root=$(readlink -m "$5")
rho=${6:-0.2}
top_fraction=${7:-0.1}
seed=${8:-0}

cd "$(dirname "$0")/.."
export RUNS_ROOT="$output_root"
export REACHER_SUBSET_INDICES="$episodes"
mkdir -p "$RUNS_ROOT"

round1_run="$RUNS_ROOT/reacher_5pct_inv_self_resume_round1_epoch10"
round2_run="$RUNS_ROOT/reacher_5pct_inv_self_resume_round2_epoch25"
round1_replay="$RUNS_ROOT/self_replay_round1.h5"
round2_replay="$RUNS_ROOT/self_replay_round2.h5"
union_replay="$RUNS_ROOT/self_replay_round1_round2_union.h5"

[[ -s "$initial_run/checkpoints/last.ckpt" ]] || { echo "missing initial checkpoint" >&2; exit 1; }
python - "$initial_run/checkpoints/last.ckpt" 4 <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert int(c["epoch"]) == int(sys.argv[2]), (sys.argv[1], c.get("epoch"))
assert c.get("optimizer_states"), "checkpoint has no optimizer state"
PY

if [[ ! -s "$round1_replay" ]]; then
  python scripts/collect_reacher_aca_counterfactuals.py \
    --run-dir "$initial_run" --source "$source_h5" --output "$round1_replay" \
    --rho "$rho" --top-fraction "$top_fraction" \
    --episode-indices "$episodes" --seed "$seed"
fi

if [[ ! -s "$round1_run/checkpoints/last.ckpt" ]]; then


  experiments/train/run.sh "$cfg" \
    subdir="$(basename "$round1_run")" \
    init_from_checkpoint=null \
    resume_from_checkpoint="$initial_run/checkpoints/last.ckpt" \
    data.counterfactual.enabled=true \
    data.counterfactual.path="$round1_replay" \
    data.counterfactual.only=false \
    trainer.max_epochs=10 \
    optimizer.type=AdamW optimizer.lr=1e-4 \
    scheduler.enabled=false loss.aca.weight=0.0 wandb.enabled=false
fi
python - "$round1_run/checkpoints/last.ckpt" 9 <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert int(c["epoch"]) == int(sys.argv[2]), (sys.argv[1], c.get("epoch"))
assert c.get("optimizer_states"), "checkpoint has no optimizer state"
PY

if [[ ! -s "$round2_replay" ]]; then
  python scripts/collect_reacher_aca_counterfactuals.py \
    --run-dir "$round1_run" --source "$source_h5" --output "$round2_replay" \
    --rho "$rho" --top-fraction "$top_fraction" \
    --episode-indices "$episodes" --seed "$seed"
fi

if [[ ! -s "$union_replay" ]]; then
  python scripts/merge_counterfactuals.py --output "$union_replay" \
    "$round1_replay" "$round2_replay"
fi

if [[ ! -s "$round2_run/checkpoints/last.ckpt" ]]; then


  experiments/train/run.sh "$cfg" \
    subdir="$(basename "$round2_run")" \
    init_from_checkpoint=null \
    resume_from_checkpoint="$round1_run/checkpoints/last.ckpt" \
    data.counterfactual.enabled=true \
    data.counterfactual.path="$union_replay" \
    data.counterfactual.only=false \
    trainer.max_epochs=25 \
    optimizer.type=AdamW optimizer.lr=1e-4 \
    scheduler.enabled=false loss.aca.weight=0.0 wandb.enabled=false
fi
python - "$round2_run/checkpoints/last.ckpt" 24 <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert int(c["epoch"]) == int(sys.argv[2]), (sys.argv[1], c.get("epoch"))
assert c.get("optimizer_states"), "checkpoint has no optimizer state"
PY

echo "[self full-resume] complete: $round2_run"
