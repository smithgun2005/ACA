#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export REPO_ROOT="$PWD"
export ACA_DATA_ROOT="${ACA_DATA_ROOT:-$PWD/data/generated}"
cfg=${1:?usage: run.sh EVAL_CONFIG.yaml MODEL_RUN_DIR}
shift
export MODEL_RUN_DIR=${1:?missing MODEL_RUN_DIR}
shift
if [[ "$cfg" != /* ]]; then cfg="$PWD/$cfg"; fi
if [[ -n "${MODEL_RUN_DIR:-}" && "$MODEL_RUN_DIR" != /* ]]; then MODEL_RUN_DIR="$PWD/$MODEL_RUN_DIR"; fi
exec python eval.py --config "$cfg" solver.type=cem solver.num_samples=300 solver.n_steps=30 solver.topk=30 plan_config.horizon=5 plan_config.receding_horizon=5 plan_config.action_block=5 eval.goal_offset_steps=25 eval.eval_budget=50 "$@"
