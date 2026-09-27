# ACA experiments

This directory is a self-contained runner for the supported experiments. It
contains SIG baseline, INV baseline, SIG+ACA and INV+ACA training only. All
evaluation uses standard CEM (300 samples, 30 iterations, top-k 30, horizon 5,
receding horizon 5, action block 5, goal offset 25 and budget 50).

Set `EXTERNAL_DATA_ROOT` to the directory containing the training/evaluation
HDF5 files. The package includes default 5% episode-index files under
`data/generated`; set `ACA_DATA_ROOT` to relocate them. The subset index defaults can be overridden with
`CUBE_SUBSET_INDICES`, `REACHER_SUBSET_INDICES`, `PUSHT_SUBSET_INDICES` and
`TWOROOM_SUBSET_INDICES`.

Generate a training config, then run it:

```bash
python experiments/train/generate_configs.py --env cube --subset 5pct
experiments/train/run.sh cube_5pct_inv_aca
```

The generator uses the established database-specific protocol values rather
than one shared default: Cube (full SIG+ACA `.05/1.5`, Inv+ACA `.1/1.5`),
Reacher (`.5/1` and `.05/1.5`), PushT (`.25/1.5`), and TwoRoom (`.05/1` and
`.25/1`). The 5% schedules use 10/20/40/20 epochs for Cube/Reacher/PushT/
TwoRoom respectively. `--aca-rho`, `--aca-weight`, `--epochs`, and
`--inverse-weight` are available for an explicitly documented override.

Run evaluation with an environment config:

```bash
experiments/eval/run.sh config/eval/env/ogbcube.yaml /path/to/training/run
```

Strict evaluation uses the manifests under `config/strict_manifests/` and
`scripts/evaluate_strict.py`. AMP, custom planners, and unrelated auxiliary
losses are intentionally rejected.

The old `self_improving/run.sh` performs one retraining round from an already
collected HDF5 file. For the complete iterative workflow use
`self_improving/run_loop.sh`:

```bash
self_improving/run_loop.sh cube CFG_NAME INITIAL_RUN SOURCE_H5 2 0.2 0.1 EPISODES.npy config/eval/ogbcube.yaml
```

Each round mines ACA actions from the current checkpoint, executes them in
MuJoCo, writes a real-transition HDF5 file, and continues from the previous
checkpoint via `init_from_checkpoint`. The default stage follows the old 5+5
protocol: five fresh AdamW epochs at fixed `lr=1e-4`, scheduler off,
validation every epoch, and ACA disabled during replay adaptation. The round's
new replay is added to the original 5% dataset; the next round remine starts
from the updated checkpoint. The episode-index file is mandatory for both
Reacher and Cube/Cube-strict, so a full `reacher_train.h5` never expands the
5% mining pool.
An optional final argument evaluates every new checkpoint with the same
standard CEM command used by `experiments/eval/run.sh`.
