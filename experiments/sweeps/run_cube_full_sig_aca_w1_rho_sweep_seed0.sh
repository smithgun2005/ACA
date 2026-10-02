#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/run_full_aca_rho_sweep.sh" cube sig
