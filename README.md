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

Generate the canonical full-dataset sweep configs:

```bash
python experiments/train/generate_configs.py
```

The launchers are listed in `experiments/SWEEP_SCRIPTS.md`. Every Cube,
Reacher, PushT, and TwoRoom objective uses a baseline plus ACA weight `1` at
rho `0.05`, `0.1`, `0.25`, and `0.5`, with training seed `0` and zero ACA
direction noise. Both INV and SIG objectives are covered, and each launcher
automatically evaluates after training.

Run evaluation with an environment config:

```bash
experiments/eval/run.sh config/eval/env/ogbcube.yaml /path/to/training/run
```

Strict evaluation uses the manifests under `config/strict_manifests/` and
`scripts/evaluate_strict.py`. AMP, custom planners, and unrelated auxiliary
losses are intentionally rejected.

The old `self_improving/run.sh` performs one retraining round from an already
collected HDF5 file. For the complete two-stage workflow use
`self_improving/run_loop.sh`:

```bash
self_improving/run_loop.sh cube CFG_NAME INITIAL_RUN SOURCE_H5 2 0.2 0.1 EPISODES.npy config/eval/ogbcube.yaml
```

The loop is fixed to the legacy schedules (the `ROUNDS` argument must be `2`):

- Reacher: `5 + 5 + 15 = 25` epochs. The final 15-epoch stage resets AdamW,
  uses a constant `lr=1e-4` with scheduler disabled, and trains on the union
  of both replay rounds.
- Cube/Cube-strict: `5 + 5 + 20 = 30` epochs. The final 20-epoch stage resets
  AdamW, uses fixed `lr=1e-4` with scheduler disabled, and trains on the union
  of both replay rounds.

Every stage initializes model weights from the previous checkpoint via
`init_from_checkpoint` (optimizer state is intentionally reset). In the
controlled self-improving experiment, all stages use fixed `lr=1e-4`, the
same loader worker settings, and no ACA training loss; ACA is used only to
mine actions for real-environment replay. Each round mines ACA actions from the
current checkpoint, executes them in MuJoCo, and writes a real-transition HDF5
file. The episode-index file is mandatory for both Reacher and Cube/Cube-strict,
so a full `reacher_train.h5` never expands the 5% mining pool.
An optional final argument evaluates every new checkpoint with the same
standard CEM command used by `experiments/eval/run.sh`.
