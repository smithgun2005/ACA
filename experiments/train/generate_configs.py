#!/usr/bin/env python3
"""Generate the canonical full-dataset INV/SIG ACA rho sweeps.

Every database gets two matched comparisons:

* objective baseline (ACA weight 0), and
* ACA weight 1 at rho 0.05, 0.1, 0.25, and 0.5.

All variants use seed 0, zero ACA direction noise, and otherwise identical
training settings within a database/objective sweep.
"""
from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATABASES = {
    "cube": {"data": "ogbcube", "action_dim": 5, "inverse_weight": 1.0},
    "reacher": {"data": "reacher", "action_dim": 2, "inverse_weight": 5.0},
    "pusht": {"data": "pusht", "action_dim": 2, "inverse_weight": 30.0},
    "tworoom": {"data": "tworoom", "action_dim": 2, "inverse_weight": 0.1},
}
OBJECTIVES = ("inv", "sig")
RHO_GRID = ("0.05", "0.1", "0.25", "0.5")


def rho_tag(rho: str) -> str:
    return "rho" + rho.replace(".", "p")


def render(database: str, objective: str, rho: str | None) -> tuple[str, str]:
    profile = DATABASES[database]
    is_aca = rho is not None
    variant = f"aca_w1_{rho_tag(rho)}" if is_aca else "baseline"
    name = f"{database}_full_{objective}_{variant}_seed0"
    aca_weight = "1.0" if is_aca else "0.0"
    aca_rho = rho if is_aca else "0.0"
    sig_weight = "0.09" if objective == "sig" else "0.0"
    inverse_weight = str(profile["inverse_weight"]) if objective == "inv" else "0.0"
    text = f"""defaults:
  - /train/base
  - /train/data/{profile['data']}
  - _self_

hydra:
  searchpath:
    - file://${{oc.env:REPO_ROOT}}/config

seed: 0
subdir: {name}
artifacts:
  embedding_subset_size: 4096
trainer:
  max_epochs: 10
  devices: 1
  accelerator: gpu
  precision: bf16
  val_check_interval: 1.0
loader:
  batch_size: 256
optimizer:
  lr: 1e-4
  weight_decay: 1e-3
scheduler:
  enabled: true
  warmup_steps_override: 1
wandb:
  enabled: false
validation_monitoring:
  enabled: false
data:
  dataset:
    num_steps: 2
    cache_dir: ${{oc.env:EXTERNAL_DATA_ROOT,/root/autodl-tmp/sensorimotor-world-model/planning/data/external}}
wm:
  action_dim: {profile['action_dim']}
loss:
  sigreg:
    weight: {sig_weight}
  inverse:
    weight: {inverse_weight}
  aca:
    weight: {aca_weight}
    rho: {aca_rho}
    margin: 0.0
    mode: gradient
    adversary: gradient
    noise_scale: 0.0
    encoder_scale: none
    online:
      enabled: false
    epoch_active:
      enabled: false
"""
    return name, text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        nargs="+",
        choices=(*DATABASES, "all"),
        default=["all"],
    )
    parser.add_argument(
        "--objective",
        nargs="+",
        choices=(*OBJECTIVES, "all"),
        default=["all"],
    )
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/train/generated")
    args = parser.parse_args()

    databases = tuple(DATABASES) if "all" in args.database else tuple(args.database)
    objectives = OBJECTIVES if "all" in args.objective else tuple(args.objective)
    args.output.mkdir(parents=True, exist_ok=True)
    for database in databases:
        for objective in objectives:
            for rho in (None, *RHO_GRID):
                name, text = render(database, objective, rho)
                path = args.output / f"{name}.yaml"
                path.write_text(text)
                print(path)


if __name__ == "__main__":
    main()
