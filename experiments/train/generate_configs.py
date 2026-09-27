#!/usr/bin/env python3
"""Generate only the supported SIG/INV/ACA training configs."""
from __future__ import annotations
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = {"cube": "ogbcube", "cube-strict": "cube-strict", "reacher": "reacher", "pusht": "pusht", "tworoom": "tworoom", "tworoom-strict": "tworoom-strict"}

# Canonical paper runs.  ACA settings are method-specific; using one rho and
# one weight for every database was the bug in the previous generator.
# ACA is deterministic in this standalone package: no random direction noise.
PROFILES = {
    "cube": {
        "epochs": 10, "inv": 1.0,
        "sig_aca": (0.05, 1.5), "inv_aca": (0.10, 1.5),
    },
    "reacher": {
        "epochs": 10, "inv": 5.0,
        "sig_aca": (0.50, 1.0), "inv_aca": (0.05, 1.5),
    },
    "pusht": {
        "epochs": 10, "inv": 30.0,
        "sig_aca": (0.25, 1.5), "inv_aca": (0.25, 1.5),
    },
    "tworoom": {
        "epochs": 10, "inv": 0.1,
        "sig_aca": (0.05, 1.0), "inv_aca": (0.25, 1.0),
    },
}
SUBSET_PROFILES = {
    "cube": {
        "epochs": 10, "inv": 1.0,
        "sig_aca": (0.05, 1.5), "inv_aca": (0.50, 1.5),
    },
    "reacher": {
        "epochs": 20, "inv": 5.0,
        "sig_aca": (0.25, 1.0), "inv_aca": (0.05, 1.0),
    },
    "pusht": {
        "epochs": 40, "inv": 30.0,
        "sig_aca": (0.25, 1.5), "inv_aca": (0.25, 1.0),
    },
    "tworoom": {
        "epochs": 20, "inv": 0.1,
        "sig_aca": (0.05, 1.0), "inv_aca": (0.10, 1.0),
    },
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=sorted(DATA), required=True)
    p.add_argument("--subset", choices=["full", "5pct"], default="full")
    p.add_argument("--methods", nargs="+", default=["sig", "inv", "sig_aca", "inv_aca"], choices=["sig", "inv", "sig_aca", "inv_aca"])
    p.add_argument("--epochs", type=int, default=None, help="override protocol epoch count")
    p.add_argument("--aca-weight", type=float, default=None, help="override profile weight")
    p.add_argument("--aca-rho", type=float, default=None, help="override profile rho")
    p.add_argument("--inverse-weight", type=float, default=None)
    p.add_argument("--output", type=Path, default=ROOT / "experiments/train/generated")
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    profile_env = a.env.replace("-strict", "")
    profile = dict((SUBSET_PROFILES if a.subset == "5pct" else PROFILES).get(profile_env, PROFILES["cube"]))
    epochs = a.epochs if a.epochs is not None else profile.get("epochs", 10)
    inverse_weight = a.inverse_weight if a.inverse_weight is not None else profile["inv"]
    # Strict changes the evaluator/manifest, not the training corpus.  A
    # strict 5% run therefore still trains on the corresponding 5% subset.
    data_name = (DATA[profile_env] + "5pct") if a.subset == "5pct" else DATA[a.env]
    for method in a.methods:
        sig = method in ("sig", "sig_aca"); inv = method in ("inv", "inv_aca"); aca = method in ("sig_aca", "inv_aca")
        default_rho, default_aca_weight = profile.get(method, (0.0, 0.0)) if aca else (0.0, 0.0)
        aca_rho = a.aca_rho if a.aca_rho is not None else default_rho
        aca_weight = a.aca_weight if a.aca_weight is not None else default_aca_weight
        name = f"{a.env}_{a.subset}_{method}"
        action_dim = 5 if profile_env == "cube" else 2
        text = f'''defaults:\n  - /train/base\n  - /train/data/{data_name}\n  - _self_\n\nhydra:\n  searchpath:\n    - file://{ROOT}/config\n\nseed: 0\nsubdir: {name}\nartifacts:\n  embedding_subset_size: 4096\ntrainer:\n  max_epochs: {epochs}\n  devices: 1\n  accelerator: gpu\n  precision: bf16\nloader:\n  batch_size: 256\noptimizer:\n  lr: 1e-4\n  weight_decay: 1e-3\nscheduler:\n  enabled: true\n  warmup_steps_override: 0\nwandb:\n  enabled: false\nvalidation_monitoring:\n  enabled: false\ndata:\n  dataset:\n    num_steps: 2\nwm:\n  action_dim: {action_dim}\nloss:\n  sigreg:\n    weight: {0.09 if sig else 0.0}\n  inverse:\n    weight: {inverse_weight if inv else 0.0}\n  aca:\n    weight: {aca_weight if aca else 0.0}\n    rho: {aca_rho}\n    margin: 0.0\n    mode: gradient\n    adversary: gradient\n    noise_scale: 0.0\n    encoder_scale: none\n    online:\n      enabled: false\n    epoch_active:\n      enabled: false\n'''
        text = text.replace("  precision: bf16\n", "  precision: bf16\n  val_check_interval: 1.0\n")
        text = text.replace("scheduler:\n  enabled: true\n  warmup_steps_override: 0", "scheduler:\n  enabled: false")
        (a.output / f"{name}.yaml").write_text(text)
        print(a.output / f"{name}.yaml")

if __name__ == "__main__":
    main()
