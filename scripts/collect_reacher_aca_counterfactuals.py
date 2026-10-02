#!/usr/bin/env python3
"""Collect real Reacher transitions for ACA-mined counterfactual actions.

The checkpoint is never updated.  For every usable offline transition this
script computes ACA's one native projected gradient action

    a_cf = clamp(a - rho * grad_a E(a) / ||grad_a E(a)||),

retains positive-hinge examples, globally keeps the hardest fraction, executes
those macro actions in MuJoCo, and writes an independent factual dataset.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import hdf5plugin
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import build_jepa


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Completed training run containing config.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Checkpoint to mine from (default: RUN_DIR/checkpoints/last.ckpt)")
    parser.add_argument("--source", type=Path,
                        default=ROOT / "data/external/reacher_train.h5")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--top-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-transitions", type=int, default=None,
                        help="Optional prefix for a smoke test; omit to scan all usable source transitions.")
    parser.add_argument("--max-executions", type=int, default=None,
                        help="Optional cap after global top-fraction selection; omitted means execute all selected.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-seed", type=int, default=None,
                        help="Random-direction seed; defaults to --seed. Environment reset always uses --seed.")
    parser.add_argument("--episode-indices", type=Path, default=None)
    parser.add_argument("--source-indices", type=Path, default=None,
                        help="Optional exact source transition rows; execute all without hinge filtering.")
    parser.add_argument("--action-mode", choices=("aca", "random"), default="aca",
                        help="ACA gradient perturbation or a seeded random unit perturbation. "
                             "Random mode requires --source-indices for a paired control.")
    return parser.parse_args()


def load_model(run_dir: Path, checkpoint: Path, device: torch.device):
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing saved config: {cfg_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
    cfg = OmegaConf.load(cfg_path)
    model = build_jepa(cfg)
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {
        key[len("model."):]: value
        for key, value in checkpoint_data["state_dict"].items()
        if key.startswith("model.")
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint/model mismatch: missing={missing}, unexpected={unexpected}")
    model.to(device).eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model, cfg


def valid_starts(h5, frameskip: int, max_transitions: int | None, episode_indices=None):
    ep_idx = h5["ep_idx"][:]
    valid = np.arange(len(ep_idx) - frameskip, dtype=np.int64)
    valid = valid[ep_idx[:-frameskip] == ep_idx[frameskip:]]
    if episode_indices is not None:
        keep = set(np.load(episode_indices).astype(np.int64).tolist())
        valid = valid[np.array([int(ep_idx[i]) in keep for i in valid], dtype=bool)]
    if max_transitions is not None:
        valid = valid[:max_transitions]
    return valid


def action_statistics(h5):
    actions = h5["action"][:]
    actions = actions[~np.isnan(actions).any(axis=1)]
    return actions.mean(axis=0), actions.std(axis=0), actions.min(axis=0), actions.max(axis=0)


def mine_positive_hinges(model, h5, starts, mean, std, low, high, frameskip, rho, margin, batch_size, device, keep_all=False):
    positive_indices, positive_hinges = [], []
    tiled_mean = np.tile(mean, frameskip).astype(np.float32)
    tiled_std = np.tile(std, frameskip).astype(np.float32)
    low_normalized = torch.as_tensor((np.tile(low, frameskip) - tiled_mean) / tiled_std, device=device).view(1, 1, -1)
    high_normalized = torch.as_tensor((np.tile(high, frameskip) - tiled_mean) / tiled_std, device=device).view(1, 1, -1)

    for begin in range(0, len(starts), batch_size):
        rows = starts[begin:begin + batch_size]

        pixels_t = np.asarray(h5["pixels"][rows])
        pixels_next = np.asarray(h5["pixels"][rows + frameskip])
        raw_actions = np.asarray([h5["action"][row:row + frameskip] for row in rows])
        macro_actions = raw_actions.reshape(len(rows), -1).astype(np.float32)
        actions = torch.as_tensor((macro_actions - tiled_mean) / tiled_std, device=device).unsqueeze(1)
        pixels = np.stack([pixels_t, pixels_next], axis=1)
        pixels = torch.from_numpy(pixels).permute(0, 1, 4, 2, 3).to(device)

        with torch.no_grad():
            embeddings = model.encode({"pixels": pixels})["emb"]
        z_t, z_next = embeddings[:, :1].detach(), embeddings[:, 1:2].detach()
        action_for_grad = actions.detach().clone().requires_grad_(True)
        action_input = model.action_encoder(action_for_grad) if model.action_encoder is not None else action_for_grad
        prediction = model.predict(z_t, action_input)
        factual_energy = (prediction - z_next).float().pow(2).mean(dim=(-2, -1))
        (gradient,) = torch.autograd.grad(factual_energy.sum(), action_for_grad)
        direction = gradient / gradient.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        mined = (actions - float(rho) * direction).clamp(low_normalized, high_normalized)
        if keep_all:


            hinge = torch.zeros_like(factual_energy)
            mask = np.ones(len(rows), dtype=bool)
        else:
            with torch.no_grad():
                mined_input = model.action_encoder(mined) if model.action_encoder is not None else mined
                mined_energy = (model.predict(z_t, mined_input) - z_next).float().pow(2).mean(dim=(-2, -1))
                hinge = float(margin) + factual_energy - mined_energy
            mask = hinge.detach().cpu().numpy() > 0
        if np.any(mask):
            positive_indices.append(rows[mask])
            positive_hinges.append(hinge.detach().cpu().numpy()[mask])
        if (begin // batch_size) % 100 == 0 or begin + len(rows) == len(starts):
            label = "selected" if keep_all else "positives"
            print(f"processed {begin + len(rows):,}/{len(starts):,}; {label}={sum(map(len, positive_indices)):,}", flush=True)

    if not positive_indices:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), tiled_mean, tiled_std
    return (np.concatenate(positive_indices), np.concatenate(positive_hinges), tiled_mean, tiled_std)


def make_environment(seed):
    import gymnasium as gym
    import stable_worldmodel
    return gym.make("swm/ReacherDMControl-v0", task="qpos_match").unwrapped


def restore_state(env, qpos, qvel, target_pos, seed):
    env.reset(seed=seed)
    env.set_state(np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64))
    env.env.physics.named.model.geom_pos["target", :2] = np.asarray(target_pos, dtype=np.float64)
    env.env.physics.forward()


def create_output(path, image_shape, observation_dim, action_dim, atomic_action_dim, args, run_dir, checkpoint):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():



        try:
            with h5py.File(path, "r") as existing:
                complete_rows = int(existing["action"].shape[0]) if "action" in existing else 0
        except (OSError, KeyError):
            complete_rows = 0
        if complete_rows:
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
        path.unlink()
    h5 = h5py.File(path, "w")
    h5.attrs["format"] = "reacher_aca_counterfactual_v1"
    h5.attrs["atomic_action_dim"] = atomic_action_dim
    h5.attrs["frameskip"] = action_dim // atomic_action_dim
    h5.attrs["source_run_dir"] = str(run_dir.resolve())
    h5.attrs["source_checkpoint"] = str(checkpoint.resolve())
    for name, value in vars(args).items():
        if name not in {"run_dir", "checkpoint", "source", "output"}:


            if isinstance(value, (Path, os.PathLike)):
                value = str(value)
            h5.attrs[name] = str(value) if value is None else value
    h5.create_dataset("pixels", shape=(0, 2, *image_shape), maxshape=(None, 2, *image_shape),
                      dtype=np.uint8, chunks=(1, 2, *image_shape), compression="lzf")
    h5.create_dataset("action", shape=(0, action_dim), maxshape=(None, action_dim), dtype=np.float32)
    h5.create_dataset("observation", shape=(0, 2, observation_dim), maxshape=(None, 2, observation_dim), dtype=np.float32)
    h5.create_dataset("source_index", shape=(0,), maxshape=(None,), dtype=np.int64)
    h5.create_dataset("hinge", shape=(0,), maxshape=(None,), dtype=np.float32)
    return h5


def append_transition(output, pixels, action, observation, source_index, hinge):
    count = output["action"].shape[0]
    for dataset in output.values():
        dataset.resize(count + 1, axis=0)
    output["pixels"][count] = pixels
    output["action"][count] = action
    output["observation"][count] = observation
    output["source_index"][count] = source_index
    output["hinge"][count] = hinge


def main():
    args = parse_args()
    if args.rho <= 0 or not 0 < args.top_fraction <= 1 or args.batch_size <= 0:
        raise ValueError("rho/batch-size must be positive and top-fraction must be in (0, 1]")
    args.run_dir = args.run_dir.expanduser().resolve()
    args.source = args.source.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.action_mode == "random" and args.source_indices is None:
        raise ValueError("--action-mode=random requires --source-indices for paired sampling")
    checkpoint = (args.checkpoint or args.run_dir / "checkpoints/last.ckpt").expanduser().resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda was requested but CUDA is not available")
    model, cfg = load_model(args.run_dir, checkpoint, device)
    frameskip = int(cfg.data.dataset.frameskip)
    if int(cfg.wm.history_size) != 1:
        raise ValueError("collector currently supports wm.history_size=1")

    with h5py.File(args.source, "r") as source:
        required = {"pixels", "action", "qpos", "qvel", "target_pos", "observation", "ep_idx"}
        missing = required.difference(source.keys())
        if missing:
            raise KeyError(f"source is missing required columns: {sorted(missing)}")
        starts = valid_starts(source, frameskip, args.max_transitions, args.episode_indices)
        if args.source_indices is not None:
            rows = np.asarray(np.load(args.source_indices), dtype=np.int64)
            valid_set = set(int(x) for x in starts.tolist())
            if rows.ndim != 1 or len(rows) == 0 or len(np.unique(rows)) != len(rows):
                raise ValueError("--source-indices must be a non-empty 1-D array without duplicates")
            if any(int(x) not in valid_set for x in rows):
                raise ValueError("--source-indices contains rows outside the requested source pool")


            starts = rows if args.action_mode == "random" else np.sort(rows)
        mean, std, low, high = action_statistics(source)
        std = np.maximum(std, 1e-6)
        print(f"Scanning {len(starts):,} valid source transitions with frameskip={frameskip}.")
        tiled_mean = np.tile(mean, frameskip).astype(np.float32)
        tiled_std = np.tile(std, frameskip).astype(np.float32)
        if args.action_mode == "random":


            positive_rows = starts
            hinges = np.zeros(len(starts), dtype=np.float32)
        else:
            positive_rows, hinges, tiled_mean, tiled_std = mine_positive_hinges(
                model, source, starts, mean, std, low, high, frameskip, args.rho,
                args.margin, args.batch_size, device, keep_all=args.source_indices is not None,
            )
        if len(positive_rows) == 0:
            raise RuntimeError("no positive-hinge actions found; no file was written")
        if args.source_indices is not None:

            rows = positive_rows
            hinges = np.zeros_like(hinges)
        else:
            keep = max(1, int(np.ceil(args.top_fraction * len(positive_rows))))
            selected = np.argsort(hinges)[-keep:][::-1]
            if args.max_executions is not None:
                selected = selected[:args.max_executions]
            rows, hinges = positive_rows[selected], hinges[selected]
        print(f"Positive hinge: {len(positive_rows):,}; executing selected: {len(rows):,}.")

        image_shape = tuple(source["pixels"].shape[1:])
        observation_dim = int(source["observation"].shape[1])
        atomic_dim = int(source["action"].shape[1])
        output = create_output(args.output, image_shape, observation_dim, frameskip * atomic_dim, atomic_dim, args, args.run_dir, checkpoint)
        env = make_environment(args.seed)
        env_rng = np.random.default_rng(args.seed)
        action_rng = np.random.default_rng(
            args.seed if args.action_seed is None else args.action_seed
        )
        try:
            for count, (row, hinge) in enumerate(zip(rows, hinges), start=1):
                restore_state(env, source["qpos"][row], source["qvel"][row], source["target_pos"][row], int(env_rng.integers(2**31 - 1)))



                start_pixels = np.asarray(source["pixels"][row])
                start_obs = np.asarray(source["observation"][row], dtype=np.float32)



                pixels = np.stack([source["pixels"][row], source["pixels"][row + frameskip]], axis=0)
                tensor_pixels = torch.from_numpy(pixels[None]).permute(0, 1, 4, 2, 3).to(device)
                raw = np.asarray(source["action"][row:row + frameskip]).reshape(1, -1).astype(np.float32)
                normalized = torch.as_tensor((raw - tiled_mean) / tiled_std, device=device).unsqueeze(1)
                if args.action_mode == "random":
                    direction = action_rng.normal(size=normalized.shape).astype(np.float32)
                    direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-8)
                    action_cf = normalized + args.rho * torch.as_tensor(direction, device=device)
                else:
                    with torch.no_grad():
                        emb = model.encode({"pixels": tensor_pixels})["emb"]
                    action_grad = normalized.detach().clone().requires_grad_(True)
                    action_input = model.action_encoder(action_grad) if model.action_encoder is not None else action_grad
                    energy = (model.predict(emb[:, :1].detach(), action_input) - emb[:, 1:2].detach()).float().pow(2).mean()
                    (grad,) = torch.autograd.grad(energy, action_grad)
                    action_cf = normalized - args.rho * grad / grad.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                action_cf = action_cf.clamp(
                    torch.as_tensor((np.tile(low, frameskip) - tiled_mean) / tiled_std, device=device).view(1, 1, -1),
                    torch.as_tensor((np.tile(high, frameskip) - tiled_mean) / tiled_std, device=device).view(1, 1, -1),
                )
                raw_cf = (action_cf.detach().cpu().numpy().reshape(-1) * tiled_std + tiled_mean).reshape(frameskip, atomic_dim)
                final_obs = start_obs
                for control in raw_cf:
                    final_obs, _, terminated, _, _ = env.step(control)
                    if terminated:
                        break
                final_pixels = env.render(*image_shape[:2])
                append_transition(
                    output,
                    np.stack([start_pixels, final_pixels]), raw_cf.reshape(-1).astype(np.float32),
                    np.stack([start_obs, np.asarray(final_obs, dtype=np.float32)]), row, hinge,
                )
                if count % 100 == 0 or count == len(rows):
                    output.flush()
                    print(f"executed {count:,}/{len(rows):,}", flush=True)
        finally:
            output.close()
            env.close()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
