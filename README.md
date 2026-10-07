# Action-Consequence Alignment (ACA)

The PyTorch implementation of Action-Consequence Alignment (ACA) 
https://arxiv.org/abs/2610.04539
The supported environments are **OGBench-Cube**, **Reacher**, **Push-T**, and
**TwoRoom**. Training uses a single CUDA GPU. Evaluation uses the standard CEM
planner, or the fixed-manifest strict protocol described below.


## Repository layout

```text
train.py, eval.py, jepa.py, module.py, utils.py  # model, training, planning
config/train/                                     # base and dataset configs
config/eval/                                      # standard and environment configs
experiments/train/run.sh                          # train one generated config
experiments/train/generate_configs.py             # generate canonical configs
experiments/eval/run.sh                           # standard CEM evaluation
experiments/sweeps/                                # complete rho sweeps
scripts/run_strict.sh                             # fixed-manifest evaluation
self_improving/                                   # mine -> replay -> retrain loops
data/generated/                                   # checked-in 5% episode indices
vendor/clear_lewm/                                # strict-evaluation adapter
```

Runs are written to `results/<subdir>` by default. Set `RUNS_ROOT` to relocate
them.

## Setup and data

Set `EXTERNAL_DATA_ROOT` to a directory containing these training files:

```text
tworoom_train.h5
reacher_train.h5
pusht_expert_train.h5
cube_single_expert_train.h5
```

Standard evaluation additionally needs `tworoom_eval.h5`, `reacher_eval.h5`,
`pusht_expert_eval.h5`, and `cube_single_expert_eval.h5`. These LeWorldModel
datasets are not checked into this repository. Dataset names in the configs must
match the HDF5 filename stem.

The checked-in 5% episode-index files are under `data/generated/`:

```text
cube_5pct/cube5pct_episodes.npy
reacher_5pct/reacher5pct_episodes.npy
pusht_5pct/pusht5pct_episodes.npy
tworoom_5pct/tworoom5pct_train_episodes.npy
```

Set `ACA_DATA_ROOT` to relocate these files, or override one with
`CUBE_SUBSET_INDICES`, `REACHER_SUBSET_INDICES`, `PUSHT_SUBSET_INDICES`, or
`TWOROOM_SUBSET_INDICES`.

## Quick Start: Run the complete rho sweep (recommend)
We strongly recommend training ACA with the full rho sweep: we have observed
substantial variation across devices even with fixed random seeds, and, in our
tested fixed-seed settings, the sweep consistently identifies ACA configurations
that outperform the corresponding baseline.
There is one launcher for each environment/objective pair:

```bash
bash experiments/sweeps/run_tworoom_full_inv_aca_w1_rho_sweep_seed0.sh
bash experiments/sweeps/run_tworoom_full_sig_aca_w1_rho_sweep_seed0.sh
bash experiments/sweeps/run_cube_full_inv_aca_w1_rho_sweep_seed0.sh
bash experiments/sweeps/run_cube_full_sig_aca_w1_rho_sweep_seed0.sh
bash experiments/sweeps/run_reacher_full_inv_aca_w1_rho_sweep_seed0.sh
bash experiments/sweeps/run_reacher_full_sig_aca_w1_rho_sweep_seed0.sh
bash experiments/sweeps/run_pusht_full_inv_aca_w1_rho_sweep_seed0.sh
bash experiments/sweeps/run_pusht_full_sig_aca_w1_rho_sweep_seed0.sh
```

Each launcher trains five matched variants (baseline plus four rho values),
then evaluates each variant with five fixed planner seeds. The standard
protocol's seed sweep is over planner seeds; the training seed stays `0`.
Details and planner seeds are listed in `experiments/SWEEP_SCRIPTS.md`.

Validate all referenced configs without starting training:

```bash
SWEEP_DRY_RUN=1 bash experiments/sweeps/run_cube_full_inv_aca_w1_rho_sweep_seed0.sh
```

Results are stored under:

```text
results/sweeps/<environment>_full_<objective>_aca_w1_rho_sweep_seed0/
    training/<config-name>/
    eval/standard/<label>/planner_seed_<seed>/summary.json
```

Set `CUDA_VISIBLE_DEVICES` to choose the GPU. The sweep scripts disable
Weights & Biases logging and set deterministic runtime variables.

## Evaluate an existing checkpoint

The portable wrapper uses standard CEM with 300 samples, 30 optimization steps,
top-k 30, horizon 5, receding horizon 5, action block 5, goal offset 25, and
evaluation budget 50:

```bash
experiments/eval/run.sh config/eval/env/reacher.yaml \
  /absolute/path/to/results/reacher_full_inv_baseline_seed0
```

Use `config/eval/full/{cube,reacher,pusht,tworoom}.yaml` for the full 100-task
protocol. Append Hydra overrides such as `seed=53026` or `eval.num_eval=10`.

## Strict fixed-manifest evaluation

Strict evaluation checks the dataset fingerprint and evaluates exact fixed pairs.
It is available for Cube and TwoRoom:

```bash
scripts/run_strict.sh cube \
  /absolute/path/to/run /absolute/path/to/cube_single_expert_eval.h5 \
  /absolute/path/to/output.json 42

scripts/run_strict.sh tworoom \
  /absolute/path/to/run /absolute/path/to/tworoom_eval.h5 \
  /absolute/path/to/output.json 42
```

The wrapper selects manifests from `config/strict_manifests/`. TwoRoom strict
evaluation must use the held-out `tworoom_eval.h5`; a full training HDF5 is
rejected. The complete Cube and TwoRoom sweep launchers run strict evaluation
after standard evaluation.

## Self-improving ACA replay

`self_improving/run_loop.sh` performs two rounds of ACA mining, MuJoCo
execution, replay merging, and retraining. It requires a 5% episode-index file
so that the mining pool remains fixed:

```bash
self_improving/run_loop.sh \
  reacher CFG_NAME INITIAL_RUN_DIR SOURCE_TRAIN_H5 2 \
  0.2 0.1 data/generated/reacher_5pct/reacher5pct_episodes.npy \
  config/eval/full/reacher.yaml
```

Use `cube` or `cube-strict` for Cube. The aligned schedules are Reacher
`5 + 5 + 15 = 25` epochs and Cube `5 + 5 + 20 = 30` epochs. The final stage
trains on both replay rounds, resets AdamW, and uses constant `lr=1e-4`.
`self_improving/run.sh` is the lower-level one-round command when a
counterfactual HDF5 already exists.

## Citation and provenance
This codebase builds on [SMWM](https://github.com/petr-ivashkov/sensorimotor-world-model).## Citation

If you found this work interesting, you can cite:

```bibtex
@misc{wang12026actionconsequencealignmentreliableplanning,
      title={Action-Consequence Alignment for Reliable Planning and Self-Improving in Latent World Models}, 
      author={Jinping Wang1 and Zhiqiang Gao and Xiantong Zhen and Ling Shao},
      year={2026},
      eprint={2610.04539},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2610.04539}, 
}
```
