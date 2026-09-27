#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
env_name=${1:?usage: run.sh cube|reacher|cube-strict RUN_CONFIG COUNTERFACTUAL_H5 [overrides...]}
cfg=${2:?missing training config}
cf_path=${3:?missing counterfactual .h5 path}
shift 3
case "$env_name" in
  cube|reacher|cube-strict) ;;
  *) echo "self-improving is supported only for 5% cube, reacher and cube-strict" >&2; exit 2 ;;
esac
echo "One self-improving retraining round: use run_loop.sh for automatic mine/execute/retrain iterations."
exec experiments/train/run.sh "$cfg" data.counterfactual.enabled=true data.counterfactual.path="$cf_path" "$@"
