# Full-dataset ACA rho sweeps

All launchers use the same naming and comparison protocol:

- training seed: `0`;
- one objective baseline with ACA disabled;
- ACA weight: `1`;
- ACA rho: `0.05`, `0.1`, `0.25`, `0.5`;
- ACA direction noise: `0`;
- five fixed planner seeds per database;
- evaluation starts only after all five training runs finish;
- a PTY preserves Lightning's live progress bar while saving `train.log`.

```bash
bash experiments/sweeps/run_cube_full_inv_aca_w1_rho_sweep_seed0.sh
bash experiments/sweeps/run_cube_full_sig_aca_w1_rho_sweep_seed0.sh
bash experiments/sweeps/run_reacher_full_inv_aca_w1_rho_sweep_seed0.sh
bash experiments/sweeps/run_reacher_full_sig_aca_w1_rho_sweep_seed0.sh
bash experiments/sweeps/run_pusht_full_inv_aca_w1_rho_sweep_seed0.sh
bash experiments/sweeps/run_pusht_full_sig_aca_w1_rho_sweep_seed0.sh
bash experiments/sweeps/run_tworoom_full_inv_aca_w1_rho_sweep_seed0.sh
bash experiments/sweeps/run_tworoom_full_sig_aca_w1_rho_sweep_seed0.sh
```

Cube and TwoRoom also run fixed-manifest strict evaluation after normal
evaluation. TwoRoom strict is bound to the pre-split `tworoom_eval.h5` and a
held-out-only manifest; the runner rejects the full `tworoom.h5`. Reacher and
PushT run normal CEM evaluation.

The shared implementation is `experiments/sweeps/run_full_aca_rho_sweep.sh`.
Set `SWEEP_DRY_RUN=1` to validate every referenced config and protocol without
starting training or evaluation.
