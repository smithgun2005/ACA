#!/usr/bin/env python3
"""Evaluate a local Lightning LeWM run on a fixed CLEAR-LeWM manifest.

This is an adapter, not a checkpoint conversion: it uses this repository's
``eval.py`` loader for a run directory (saved Hydra config + Lightning
``last.ckpt``), while importing CLEAR-LeWM solely for its fixed-pair manifest
contract and task-semantic success patch.  Results are explicitly marked
``clear-lightning-adapter-v1`` and are not CLEAR official-reference results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

# MuJoCo's EGL finalizer may run after EGL has already released its display
# (especially when several evaluator processes share one GPU).  In that case
# ``GLContext.__del__`` emits noisy ``EGL_NOT_INITIALIZED`` tracebacks even
# though the rollout and result file completed successfully.  Make context
# destruction idempotent and quiet at interpreter shutdown; runtime rendering
# errors are still raised by the normal code paths.
try:
    import mujoco.egl as _mujoco_egl

    if not getattr(_mujoco_egl.GLContext.free, "_clear_lewm_safe", False):
        _egl_free_original = _mujoco_egl.GLContext.free

        def _clear_lewm_safe_free(self):
            try:
                _egl_free_original(self)
            except Exception:
                # EGL may already be torn down during Python finalization.
                # Dropping the handle prevents repeated destructor warnings.
                pass
            finally:
                self._context = None

        _clear_lewm_safe_free._clear_lewm_safe = True
        _mujoco_egl.GLContext.free = _clear_lewm_safe_free
except Exception:
    # Headless/runtime setup will report any genuine import failure later.
    pass

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEAR_ROOT = ROOT / "vendor"
sys.path.insert(0, str(ROOT))

from eval import (  # noqa: E402
    build_solver,
    build_world,
    fit_processors,
    img_transform,
    load_jepa_from_run,
)
from utils import load_composed_config  # noqa: E402
import stable_worldmodel as swm  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--eval-config", type=Path, required=True,
                   help="Existing planning eval YAML defining CEM and environment.")
    p.add_argument("--manifest", type=Path, required=True,
                   help="CLEAR manifest generated from exactly --dataset-path.")
    p.add_argument("--dataset-path", type=Path, required=True)
    p.add_argument("--clear-root", type=Path, default=DEFAULT_CLEAR_ROOT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--planner-seed", type=int, required=True)
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--n-steps", type=int, default=30)
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--eval-budget", type=int, default=50)
    return p.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    args = parse_args()
    args.run_dir = args.run_dir.resolve(); args.eval_config = args.eval_config.resolve()
    args.manifest = args.manifest.resolve(); args.dataset_path = args.dataset_path.resolve()
    args.clear_root = args.clear_root.resolve(); args.output = args.output.resolve()
    for path in (args.run_dir / "config.yaml", args.run_dir / "checkpoints/last.ckpt", args.eval_config, args.manifest, args.dataset_path):
        if not path.is_file(): raise FileNotFoundError(path)
    if not args.clear_root.is_dir(): raise FileNotFoundError(args.clear_root)
    sys.path.insert(0, str(args.clear_root))
    from clear_lewm.datasets import metadata_fingerprint
    from clear_lewm.manifests import load_manifest
    from clear_lewm.protocols import normalize_task, protocol_from_dict
    from clear_lewm.runner import _install_task_success

    manifest = load_manifest(args.manifest)
    task = normalize_task(manifest["task"])
    expected = manifest["dataset"]["fingerprint"]
    if expected["kind"] != "metadata-sha256":
        raise ValueError("only CLEAR metadata-sha256 manifests are supported")
    actual = metadata_fingerprint(args.dataset_path)
    if actual != expected["value"]:
        raise ValueError(f"dataset fingerprint mismatch: {actual} != {expected['value']}")
    protocol = protocol_from_dict(manifest["protocol"])
    if int(protocol.goal_offset) <= 0:
        raise ValueError("manifest goal offset must be positive")

    # The repository eval YAML uses environment interpolations for its seed/run
    # fields; provide them locally before composing it.
    os.environ["PLANNER_SEED"] = str(args.planner_seed)
    os.environ["PLANNER_NAME"] = "clear_lightning"
    # The planning YAML uses these interpolations for its single run entry.
    # Keep MODEL_RUN_DIR as the canonical name (EVAL_RUN_DIR was an older
    # adapter-only variable and leaves OmegaConf unable to resolve runs[0]).
    os.environ["MODEL_RUN_DIR"] = str(args.run_dir)
    os.environ["EVAL_RUN_DIR"] = str(args.run_dir)
    os.environ.setdefault("REPO_ROOT", str(ROOT))

    # Reuse the normal planning evaluator's architecture, transforms, CEM and
    # data standardization.  Only its random task sampler is replaced.
    # Resolve the planning YAML's Hydra defaults exactly as the repository's
    # normal evaluator does; a raw OmegaConf.load would omit /eval/base and
    # /eval/env/ogbcube fields.
    cfg_node = load_composed_config(args.eval_config)
    OmegaConf.set_struct(cfg_node, False)
    cfg_node.seed = args.planner_seed
    cfg_node.eval.num_eval = len(manifest["pairs"])
    cfg_node.eval.eval_budget = args.eval_budget
    cfg_node.eval.goal_offset_steps = int(protocol.goal_offset)
    cfg_node.world.num_envs = len(manifest["pairs"])
    cfg_node.world.max_episode_steps = 2 * args.eval_budget
    cfg_node.solver.num_samples = args.num_samples
    cfg_node.solver.n_steps = args.n_steps
    cfg_node.solver.topk = args.topk
    cfg = OmegaConf.to_container(cfg_node, resolve=True)
    np.random.seed(args.planner_seed); torch.manual_seed(args.planner_seed)

    # stable_worldmodel's HDF5Dataset accepts a dataset *name* plus cache
    # directory (not an arbitrary path keyword).  Passing the local filename
    # this way keeps the CLEAR fingerprint check tied to the exact file.
    dataset = swm.data.HDF5Dataset(
        args.dataset_path.stem,
        frameskip=int(cfg["dataset"].get("frameskip", 1)),
        num_steps=int(cfg["dataset"].get("num_steps", 1)),
        keys_to_load=cfg["dataset"].get("keys_to_load"),
        keys_to_cache=cfg["dataset"]["keys_to_cache"],
        cache_dir=args.dataset_path.parent,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, train_cfg = load_jepa_from_run(args.run_dir, device=device)
    world = build_world(cfg, train_cfg=train_cfg)
    process = fit_processors(dataset, cfg["dataset"]["keys_to_cache"])
    transform = {"pixels": img_transform(cfg["eval"]["img_size"]), "goal": img_transform(cfg["eval"]["img_size"])}
    solver = build_solver(cfg, model)
    if hasattr(solver, "set_action_normalizer") and "action" in process:
        solver.set_action_normalizer(process["action"].mean_, process["action"].scale_)
    policy = swm.policy.WorldModelPolicy(solver=solver, config=swm.PlanConfig(**cfg["plan_config"]), process=process, transform=transform)
    episodes = [int(pair["episode_id"]) for pair in manifest["pairs"]]
    starts = [int(pair["start_step"]) for pair in manifest["pairs"]]
    world.set_policy(policy)
    _install_task_success(world, task, protocol)
    started = time.perf_counter()
    try:
        raw = world.evaluate_from_dataset(
            dataset, start_steps=starts, goal_offset_steps=int(protocol.goal_offset),
            eval_budget=args.eval_budget, episodes_idx=episodes,
            callables=cfg["eval"].get("callables"), save_video=False,
        )
    finally:
        world.close()
    successes = np.asarray(raw["episode_successes"], dtype=bool)
    result = {
        "schema_version": "clear-lightning-adapter-v1",
        "reference_status": "non-reference custom Lightning checkpoint",
        "task": task,
        "protocol": protocol.to_dict(),
        "manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest),
        "dataset": str(args.dataset_path), "dataset_fingerprint": actual,
        "run_dir": str(args.run_dir), "checkpoint_sha256": sha256(args.run_dir / "checkpoints/last.ckpt"),
        "planner_seed": args.planner_seed,
        "solver": {"num_samples": args.num_samples, "n_steps": args.n_steps, "topk": args.topk, "batch_size": 1},
        "eval_budget": args.eval_budget,
        "metrics": {"success_rate": float(successes.mean() * 100.0), "successes": int(successes.sum()), "episodes": int(len(successes))},
        "episode_successes": successes.tolist(), "raw_world_metrics": raw,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
