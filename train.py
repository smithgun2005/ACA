import os
import shutil
import fcntl
import math
from collections import deque
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import lightning.fabric.utilities.registry as lightning_registry
import lightning.pytorch.trainer.connectors.callback_connector as callback_connector
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
import numpy as np
import h5py
import hdf5plugin  # registers the compression filter used by Reacher HDF5
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import (
    ASPDCHead,
    ActionConsistencyIDM,
    BlindPredictor,
    CAIBlindPredictor,
    CAITransitionPredictor,
    DeltaLiftHead,
    Embedder,
    GeoPDCHead,
    IIPDCHead,
    InverseModel,
    MLP,
    PDCHead,
    PredictiveMaskPolicy,
    SIGReg,
    build_predictor,
)
from utils import (
    ModelObjectCallBack,
    PreserveColumns,
    ResizeCompat,
    ValidationEmbeddingStatsCallback,
    get_column_normalizer,
    get_macro_action_normalizer,
    save_embeddings,
)
from counterfactual_data import CounterfactualTransitionDataset

REPO_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("REPO_ROOT", str(REPO_ROOT))


def get_runs_root() -> Path:
    default_root = Path.cwd() / "results"
    return Path(os.environ.get("RUNS_ROOT", str(default_root))).expanduser().resolve()


def initialize_model_from_checkpoint(module, checkpoint_path) -> None:
    """Initialize model weights while intentionally resetting optimizer/epochs."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"init_from_checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state = {
        key[len("model."):]: value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    if not state:
        raise KeyError(f"checkpoint has no model.* keys: {path}")
    missing, unexpected = module.model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint/model mismatch for {path}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    print(f"Initialized model weights from {path}; optimizer and epoch count reset.")


def prepare_full_resume_checkpoint(
    run_dir: Path,
    checkpoint_path,
    scheduler_max_steps: int | None = None,
    scheduler_warmup_steps: int | None = None,
) -> Path:
    """Copy a full Lightning state into this run's private resume location.

    ``spt.Manager`` intentionally resumes only from ``run_dir/checkpoints/
    last.ckpt``.  Copying, instead of pointing it at the source run, prevents
    the resumed final epoch from overwriting the saved epoch-nine checkpoint.
    """
    source = Path(checkpoint_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"resume_from_checkpoint does not exist: {source}")
    destination = run_dir / "checkpoints" / "last.ckpt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    # In DDP every rank executes Hydra's run function. Serialize the initial
    # copy: without this lock multiple ranks can concurrently write a 300+ MB
    # checkpoint and Lightning may read a partial file.
    lock_path = destination.with_suffix(".resume.lock")
    with open(lock_path, "w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if not destination.exists():
                temporary = destination.with_suffix(".copying")
                shutil.copy2(source, temporary)
                os.replace(temporary, destination)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    checkpoint = torch.load(destination, map_location="cpu", weights_only=False)
    if checkpoint.get("global_step") is None:
        raise RuntimeError(f"invalid full-resume checkpoint: {destination}")

    if scheduler_max_steps is not None and checkpoint.get("lr_schedulers"):
        for state in checkpoint["lr_schedulers"]:
            state["max_steps"] = int(scheduler_max_steps)
            if scheduler_warmup_steps is not None:
                state["warmup_steps"] = int(scheduler_warmup_steps)
            step = int(checkpoint["global_step"])
            warmup = int(state["warmup_steps"])
            base_lrs = state.get("base_lrs", [])
            if step < warmup:
                lrs = [
                    float(state.get("warmup_start_lr", 0.0))
                    + (float(base_lr) - float(state.get("warmup_start_lr", 0.0)))
                    * step / warmup
                    for base_lr in base_lrs
                ]
            else:
                lrs = [
                    float(state.get("eta_min", 0.0))
                    + (float(base_lr) - float(state.get("eta_min", 0.0)))
                    * (1 + math.cos(math.pi * (step - warmup) / (int(scheduler_max_steps) - warmup))) / 2
                    for base_lr in base_lrs
                ]
            state["_last_lr"] = lrs
            for optimizer in checkpoint.get("optimizer_states", []):
                for group, lr in zip(optimizer.get("param_groups", []), lrs):
                    group["lr"] = lr
        torch.save(checkpoint, destination)
        print(
            "Retargeted resumed LR scheduler: "
            f"warmup_steps={scheduler_warmup_steps}, max_steps={scheduler_max_steps}."
        )
    print(
        "Full resume prepared from "
        f"{source} (epoch={checkpoint.get('epoch')}, "
        f"global_step={checkpoint.get('global_step')})."
    )
    return destination


def scheduler_step_budget(cfg, train_set) -> tuple[int, int]:
    """Return explicit per-rank scheduler steps before DDP setup begins.

    The project scheduler is stepped manually each optimization step. During
    DDP's optimizer setup Lightning has not yet populated
    ``estimated_stepping_batches``, so relying on its implicit defaults makes
    a dynamically supplied counterfactual dataset fail at construction time.
    """
    devices = cfg.trainer.get("devices", 1)
    if devices == "auto":
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    elif isinstance(devices, (list, tuple)):
        world_size = len(devices)
    else:
        world_size = int(devices)
    world_size = max(1, world_size)
    # With epoch-active ACA the union grows only after an epoch completes.
    # Budget for the conservative maximum (every original row has positive
    # hinge) so cosine decay never reaches zero before the final new rows.
    active = cfg.loss.get("aca", {}).get("epoch_active", {})
    if bool(active.get("enabled", False)):
        frac = float(active.get("top_fraction", 0.0))
        epochs = int(cfg.trainer.max_epochs)
        expected_examples = len(train_set.base) * (
            epochs + frac * epochs * max(0, epochs - 1) / 2
        )
        per_rank_examples = int(np.ceil(expected_examples / (epochs * world_size)))
    else:
        per_rank_examples = int(np.ceil(len(train_set) / world_size))
    steps_per_epoch = max(1, per_rank_examples // int(cfg.loader.batch_size))
    max_steps = max(2, steps_per_epoch * int(cfg.trainer.max_epochs))
    # Controlled multi-stage runs can reserve a scheduler budget before the
    # data union changes (e.g. mine after epoch 4, then train epoch 5).
    override = cfg.get("scheduler", {}).get("max_steps_override")
    if override is not None:
        max_steps = max(2, int(override))
    warmup_override = cfg.get("scheduler", {}).get("warmup_steps_override")
    warmup_steps = (
        max(0, int(warmup_override))
        if warmup_override is not None
        else min(max_steps - 1, max(1, int(0.01 * max_steps)))
    )
    warmup_steps = min(max_steps - 1, warmup_steps)
    return warmup_steps, max_steps


def configure_external_callbacks(enabled: bool) -> None:
    if enabled:
        return

    def _no_external_callbacks(_group: str):
        return []

    lightning_registry._load_external_callbacks = _no_external_callbacks
    callback_connector._load_external_callbacks = _no_external_callbacks


def _rsi_encode(model, batch):
    """Encode a shallow batch copy; JEPA.encode adds ``emb``/``act_emb``."""
    return model.encode(dict(batch))


def _rsi_action_energy(idm, z_t, z_next, actions):
    """MSE of factual action reconstructed from an explicit latent transition."""
    pred_action = idm(z_t, z_next - z_t)
    return (pred_action - actions).pow(2).mean()


def _set_requires_grad(module, enabled):
    if module is not None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)


class OnlineACAReplay:
    """Execute hard ACA actions and replay their real transitions.

    This is deliberately a synchronous, single-process collector: the current
    batch contains raw qpos/qvel/target metadata, so every selected fake action
    can be executed from exactly its originating MuJoCo state. The offline HDF5
    file is never modified; collected samples live in RAM and are exported at
    the end of the run for reproducibility/plotting.
    """

    def __init__(self, cfg, action_mean, action_std, img_size, seed):
        online = cfg.loss.aca.online
        self.cfg = online
        self.frameskip = int(cfg.data.dataset.frameskip)
        self.capacity = int(online.replay_capacity)
        self.samples = deque(maxlen=self.capacity)
        self.rng = np.random.default_rng(seed)
        # HDF5 stores atomic 2-D controls; HDF5Dataset concatenates the
        # ``frameskip`` controls into each model action vector.
        self.action_mean = np.tile(
            action_mean.detach().cpu().float().numpy().reshape(-1), self.frameskip
        )
        self.action_std = np.tile(
            action_std.detach().cpu().float().numpy().reshape(-1), self.frameskip
        )
        self.img_size = int(img_size)
        self.env = None

    def _environment(self):
        if self.env is None:
            # EGL is needed on headless workers and must be selected before
            # dm_control initializes an OpenGL context.
            os.environ.setdefault("MUJOCO_GL", "egl")
            import gymnasium as gym
            import stable_worldmodel  # registers swm/ReacherDMControl-v0
            self.env = gym.make("swm/ReacherDMControl-v0", task="qpos_match")
        return self.env.unwrapped

    def _set_transition_state(self, env, qpos, qvel, target_pos):
        env.reset(seed=int(self.rng.integers(2**31 - 1)))
        env.set_state(np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64))
        # Reacher's target is a physical geom; move it to the factual state's
        # recorded location before rendering/executing the counterfactual.
        env.env.physics.named.model.geom_pos["target", :2] = np.asarray(
            target_pos, dtype=np.float64
        )
        env.env.physics.forward()

    def _raw_action(self, normalized_action):
        action = normalized_action.detach().cpu().float().numpy()
        return action * self.action_std + self.action_mean

    def add(self, batch, selected_actions):
        """Execute selected normalized action chunks from their batch states."""
        env = self._environment()
        qpos = batch["replay_qpos"].detach().cpu().numpy()
        qvel = batch["replay_qvel"].detach().cpu().numpy()
        target_pos = batch["replay_target_pos"].detach().cpu().numpy()
        pixels = batch["replay_pixels"].detach().cpu().numpy()
        for row, action in selected_actions:
            self._set_transition_state(env, qpos[row, 0], qvel[row, 0], target_pos[row, 0])
            raw_action = self._raw_action(action).reshape(self.frameskip, -1)
            for control in raw_action:
                obs_next, _, terminated, _, info = env.step(control)
                if terminated:
                    break
            # Dataset samples are CHW tensors; DM-Control renders HWC.
            next_pixels = env.render(self.img_size, self.img_size).transpose(2, 0, 1)
            next_obs = np.asarray(obs_next, dtype=np.float32)
            if pixels[row, 0].shape != next_pixels.shape:
                raise RuntimeError(
                    "online ACA pixel shape mismatch: "
                    f"offline={pixels[row, 0].shape}, rendered={next_pixels.shape}"
                )
            self.samples.append({
                "pixels": np.stack([pixels[row, 0], next_pixels], axis=0),
                "action": action.detach().cpu().float(),
                "observation": np.stack([
                    batch["replay_observation"][row, 0].detach().cpu().numpy(),
                    next_obs,
                ]),
            })

    def mix(self, batch):
        if not self.samples or self.cfg.replay_ratio <= 0:
            self.last_mix_count = 0
            return batch
        batch_size = batch["action"].size(0)
        count = min(len(self.samples), int(round(batch_size * float(self.cfg.replay_ratio))))
        if count <= 0:
            self.last_mix_count = 0
            return batch
        positions = self.rng.choice(len(self.samples), size=count, replace=False)
        chosen = [self.samples[int(i)] for i in positions]
        # Preserve batch shape: replacement does not alter the dataloader or
        # distributed batch accounting. Online actions are already normalized.
        replace = torch.randperm(batch_size, device=batch["action"].device)[:count]
        for dst, item in zip(replace.tolist(), chosen):
            batch["pixels"][dst] = torch.from_numpy(item["pixels"]).to(batch["pixels"])
            batch["action"][dst] = item["action"].to(batch["action"])
            batch["observation"][dst] = torch.from_numpy(item["observation"]).to(batch["observation"])
        self.last_mix_count = count
        return batch

    def export(self, path):
        if not self.samples:
            return
        import h5py
        path.parent.mkdir(parents=True, exist_ok=True)
        records = list(self.samples)
        with h5py.File(path, "w") as f:
            f.attrs["description"] = "Online ACA hard-action factual replay"
            f.create_dataset("pixels", data=np.stack([x["pixels"] for x in records]))
            f.create_dataset("action", data=torch.stack([x["action"] for x in records]).numpy())
            f.create_dataset("observation", data=np.stack([x["observation"] for x in records]))

    def close(self):
        if self.env is not None:
            self.env.close()


class EpochActiveACADataset(Dataset):
    """A mutable union of the offline data and delayed ACA transitions.

    The length is deliberately evaluated at every epoch by PyTorch's sampler.
    At the end of epoch ``e`` the collector appends real transitions to the
    HDF5 file and updates ``replay_len``; consequently they are visible to the
    *next* epoch only.  Keeping the added images on disk (rather than a Python
    list) matters here: a few percent of Reacher at 224x224 is already many GB.
    """

    def __init__(self, base, path, transform, image_shape, action_dim):
        self.base = base
        self.path = Path(path)
        self.transform = transform
        self.image_shape = tuple(image_shape)
        self.action_dim = int(action_dim)
        self.h5_file = None
        self.replay_len = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.path, "w") as f:
            f.attrs["description"] = "Epoch-delayed active ACA real transitions"
            f.attrs["action_dim"] = self.action_dim
            f.create_dataset(
                "pixels", shape=(0, 2, *self.image_shape), maxshape=(None, 2, *self.image_shape),
                dtype=np.uint8, chunks=(1, 2, *self.image_shape), compression="lzf",
            )
            f.create_dataset(
                "action", shape=(0, self.action_dim), maxshape=(None, self.action_dim),
                dtype=np.float32, chunks=True,
            )
            for name, width, dtype in (
                ("observation", 6, np.float32), ("qpos", 2, np.float64),
                ("qvel", 2, np.float64), ("target_pos", 2, np.float64),
            ):
                f.create_dataset(
                    name, shape=(0, 2, width), maxshape=(None, 2, width),
                    dtype=dtype, chunks=True,
                )

    def __len__(self):
        return len(self.base) + self.replay_len

    def _open(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.path, "r", swmr=True)

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["h5_file"] = None
        return state

    def __getitem__(self, index):
        base_len = len(self.base)
        if index < base_len:
            return self.base[index]
        self._open()
        row = index - base_len
        f = self.h5_file
        # HDF5Dataset returns two action rows.  Only the first one is used by
        # the forward loss; duplicating it makes the inverse loss's [:, :-1]
        # target have the same shape as ordinary sequence samples.
        sample = {
            "pixels": torch.from_numpy(f["pixels"][row]).permute(0, 3, 1, 2),
            "action": torch.from_numpy(f["action"][row]).repeat(2, 1),
            "observation": torch.from_numpy(f["observation"][row]),
            "qpos": torch.from_numpy(f["qpos"][row]),
            "qvel": torch.from_numpy(f["qvel"][row]),
            "target_pos": torch.from_numpy(f["target_pos"][row]),
            # ``-1`` cannot identify an HDF5 original row and is the explicit
            # marker that prevents generated samples being mined again.
            "id": torch.full((2,), -1, dtype=torch.long),
            "replay_pixels": torch.from_numpy(f["pixels"][row]).permute(0, 3, 1, 2),
            "replay_observation": torch.from_numpy(f["observation"][row]),
            "replay_qpos": torch.from_numpy(f["qpos"][row]),
            "replay_qvel": torch.from_numpy(f["qvel"][row]),
            "replay_target_pos": torch.from_numpy(f["target_pos"][row]),
            "replay_id": torch.full((2,), -1, dtype=torch.long),
        }
        return self.transform(sample) if self.transform else sample

    def append(self, records):
        """Append one already-executed chunk and expose it next epoch."""
        if not records:
            return
        self.close()  # never hold an HDF5 reader while opening its writer
        n = len(records["action"])
        with h5py.File(self.path, "a") as f:
            start = int(f["action"].shape[0])
            end = start + n
            for key, data in records.items():
                ds = f[key]
                ds.resize((end, *ds.shape[1:]))
                ds[start:end] = data
            f.flush()
        self.replay_len += n


class EpochActiveACACollector:
    """Globally rank one native ACA action per original transition per epoch."""

    def __init__(self, cfg, dataset, action_mean, action_std, action_low, action_high, img_size, seed):
        self.cfg = cfg.loss.aca.epoch_active
        self.dataset = dataset
        self.frameskip = int(cfg.data.dataset.frameskip)
        self.action_mean = np.tile(action_mean.detach().cpu().numpy().reshape(-1), self.frameskip)
        self.action_std = np.tile(action_std.detach().cpu().numpy().reshape(-1), self.frameskip)
        self.action_low = action_low
        self.action_high = action_high
        self.img_size = int(img_size)
        self.rng = np.random.default_rng(seed)
        self.env = None
        self.candidates = []

    def start_epoch(self):
        self.candidates.clear()

    def _environment(self):
        if self.env is None:
            os.environ.setdefault("MUJOCO_GL", "egl")
            import gymnasium as gym
            import stable_worldmodel  # registers swm/ReacherDMControl-v0
            self.env = gym.make("swm/ReacherDMControl-v0", task="qpos_match")
        return self.env.unwrapped

    def _set_transition_state(self, env, qpos, qvel, target_pos):
        env.reset(seed=int(self.rng.integers(2**31 - 1)))
        env.set_state(np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64))
        env.env.physics.named.model.geom_pos["target", :2] = np.asarray(target_pos, dtype=np.float64)
        env.env.physics.forward()

    def mine(self, model, batch, z_t, actions, tgt_emb, global_step):
        """Keep compact metadata for *all* positive hinges, then rank globally."""
        every = int(self.cfg.get("collect_every_n_batches", 1))
        if every <= 0:
            raise ValueError("epoch_active.collect_every_n_batches must be positive")
        # ``global_step`` refers to the current optimization step here. This
        # calls ACA on batch 0, 10, 20, ... and leaves all other batches as
        # ordinary forward/inverse training only.
        if int(global_step) % every != 0:
            return 0
        original = batch["replay_id"][:, 0] != -1
        if not bool(original.any()):
            return 0
        indices = torch.nonzero(original, as_tuple=False).squeeze(-1)
        mined_actions, hinge = model.mine_aca_actions(
            z_t[indices], actions[indices], tgt_emb[indices],
            rho=float(self.cfg.rho), margin=float(self.cfg.margin),
            action_low=self.action_low, action_high=self.action_high,
        )
        # JEPA's action energy averages over all non-batch dimensions, hence
        # hinge is normally (batch,).  Accept a trailing singleton too so the
        # collector stays valid for predictors returning an explicit horizon.
        hinge_rows = hinge if hinge.ndim == 1 else hinge[:, 0]
        positive = torch.nonzero(hinge_rows > 0, as_tuple=False).squeeze(-1)
        if positive.numel() == 0:
            return 0
        source = indices[positive]
        self.candidates.append({
            "hinge": hinge_rows[positive].detach().cpu(),
            "action": mined_actions[positive, 0].detach().cpu(),
            "qpos": batch["replay_qpos"][source, 0].detach().cpu(),
            "qvel": batch["replay_qvel"][source, 0].detach().cpu(),
            "target_pos": batch["replay_target_pos"][source, 0].detach().cpu(),
            "observation": batch["replay_observation"][source, 0].detach().cpu(),
        })
        return int(positive.numel())

    def finish_epoch(self):
        if not self.candidates:
            return 0, 0, float("nan")
        merged = {key: torch.cat([x[key] for x in self.candidates]) for key in self.candidates[0]}
        total = int(merged["hinge"].numel())
        keep = max(1, int(np.ceil(float(self.cfg.top_fraction) * total)))
        optional_cap = self.cfg.get("max_samples_per_epoch", None)
        if optional_cap is not None:
            keep = min(keep, int(optional_cap))
        top = merged["hinge"].topk(keep).indices
        selected = {key: value[top] for key, value in merged.items()}
        env = self._environment()
        chunk = {"pixels": [], "action": [], "observation": [], "qpos": [], "qvel": [], "target_pos": []}
        # Write in small chunks: top 1% can still be thousands of 224px images.
        for i in range(keep):
            self._set_transition_state(env, selected["qpos"][i].numpy(), selected["qvel"][i].numpy(), selected["target_pos"][i].numpy())
            start_pixels = env.render(self.img_size, self.img_size)
            raw_action = selected["action"][i].numpy() * self.action_std + self.action_mean
            obs_next = selected["observation"][i].numpy()
            for control in raw_action.reshape(self.frameskip, -1):
                obs_next, _, terminated, _, _ = env.step(control)
                if terminated:
                    break
            chunk["pixels"].append(np.stack([start_pixels, env.render(self.img_size, self.img_size)], axis=0))
            chunk["action"].append(raw_action.astype(np.float32))
            chunk["observation"].append(np.stack([selected["observation"][i].numpy(), np.asarray(obs_next, dtype=np.float32)]))
            chunk["qpos"].append(np.stack([selected["qpos"][i].numpy(), selected["qpos"][i].numpy()]))
            chunk["qvel"].append(np.stack([selected["qvel"][i].numpy(), selected["qvel"][i].numpy()]))
            chunk["target_pos"].append(np.stack([selected["target_pos"][i].numpy(), selected["target_pos"][i].numpy()]))
            if len(chunk["action"]) == 32 or i + 1 == keep:
                self.dataset.append({key: np.asarray(value) for key, value in chunk.items()})
                chunk = {key: [] for key in chunk}
        mean_hinge = float(selected["hinge"].float().mean())
        self.candidates.clear()
        return total, keep, mean_hinge

    def close(self):
        self.dataset.close()
        if self.env is not None:
            self.env.close()


class EpochActiveACACallback(Callback):
    """Finalize epoch e's mining after optimization; replay starts at e+1."""

    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.epoch_active_aca.start_epoch()

    def on_train_epoch_end(self, trainer, pl_module):
        total, added, mean_hinge = pl_module.epoch_active_aca.finish_epoch()
        pl_module.log("fit/epoch_aca_positive_candidates", float(total), on_step=False, on_epoch=True)
        pl_module.log("fit/epoch_aca_added_next_epoch", float(added), on_step=False, on_epoch=True)
        if total:
            pl_module.log("fit/epoch_aca_selected_hinge", mean_hinge, on_step=False, on_epoch=True)
        print(
            f"Epoch-active ACA: {total} positive candidates; appended {added} "
            f"real transitions for epoch {trainer.current_epoch + 2} "
            f"(union size={len(pl_module.epoch_active_aca.dataset)})."
        )


@contextmanager
def _freeze_batchnorm_running_stats(module):
    """Keep train-mode BatchNorm's batch-normalized forward, but do not mutate
    its running statistics during a frozen game branch.

    ``requires_grad_(False)`` only freezes parameters; it does *not* prevent
    BatchNorm buffers from updating in ``train()`` mode.  The game makes extra
    discriminator and frozen-encoder forwards every batch, whereas the
    baseline makes one.  Holding ``momentum=0`` preserves the train-time batch
    normalization used by the live branch while preventing those extra passes
    from silently changing validation-time running mean/variance.
    """
    norms = [m for m in module.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in norms]
    try:
        for norm in norms:
            norm.momentum = 0.0
        yield
    finally:
        for norm, momentum in zip(norms, momenta):
            norm.momentum = momentum


def forward_inverse_game_training_step(self, batch, batch_idx, cfg):
    """Three-step Forward--Inverse Adversarial Game update.

    1. IDM sees detached real/fake transitions and learns their factual-action
       consistency gap.  2. The shared encoder receives an ordinary forward
       prediction loss plus real-transition inverse/separation losses.  3.
       The predictor receives the same forward prediction loss plus the
       catch-up loss through a frozen IDM.  The opposing separation/catch
       gradients therefore never cancel in one backward pass.
    """
    if batch_idx is None:
        batch_idx = 0
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    history_size = int(cfg.wm.get("history_size", 1))
    rsi = cfg.loss.rsi
    real_weight = float(rsi.get("real_weight", 1.0))
    separation_weight = float(rsi.get("separation_weight", 1.0))
    catch_weight = float(rsi.get("catch_weight", 1.0))
    margin = float(rsi.get("margin", 0.01))

    encoder_modules = [self.model.encoder, self.model.projector]
    predictor_modules = [
        self.model.predictor, self.model.action_encoder, self.model.pred_proj,
    ]
    idm = self.rsi_idm
    optimizers = self.optimizers()
    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]
    if len(optimizers) != 3:
        raise RuntimeError(
            "Forward--Inverse Game requires encoder_opt, predictor_opt, and idm_opt"
        )
    encoder_opt, predictor_opt, idm_opt = optimizers
    schedulers = self.lr_schedulers()
    if schedulers is None:
        schedulers = [None, None, None]
    elif not isinstance(schedulers, (list, tuple)):
        schedulers = [schedulers]
    if len(schedulers) != 3:
        schedulers = list(schedulers) + [None] * (3 - len(schedulers))

    def step(optimizer, scheduler, loss):
        optimizer.zero_grad(set_to_none=True)
        self.manual_backward(loss)
        self.clip_gradients(
            optimizer,
            gradient_clip_val=self.trainer.gradient_clip_val,
            gradient_clip_algorithm=self.trainer.gradient_clip_algorithm,
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    # 1) Discriminator / IDM: all transition endpoints are stopped.
    _set_requires_grad(self.model, False)
    _set_requires_grad(idm, True)
    with torch.no_grad(), _freeze_batchnorm_running_stats(self.model):
        disc_out = _rsi_encode(self.model, batch)
        disc_emb = disc_out["emb"]
        disc_actions = batch["action"][:, :history_size]
        disc_z_t = disc_emb[:, :history_size]
        disc_z_tp1 = disc_emb[:, 1 : history_size + 1]
        disc_fake = self.model.predict(
            disc_z_t, disc_out["act_emb"][:, :history_size]
        )
    disc_real_energy = _rsi_action_energy(idm, disc_z_t, disc_z_tp1, disc_actions)
    disc_fake_energy = _rsi_action_energy(idm, disc_z_t, disc_fake, disc_actions)
    idm_loss = disc_real_energy + F.relu(margin + disc_real_energy - disc_fake_energy)
    step(idm_opt, schedulers[2], idm_loss)

    # 2) Encoder: baseline prediction loss remains live into the shared
    # encoder. Predictor/IDM weights are frozen, but autograd still traverses
    # their operations to obtain d L_pred / d z_t and d L_real / d z.
    _set_requires_grad(idm, False)
    _set_requires_grad(self.model, False)
    for module in encoder_modules:
        _set_requires_grad(module, True)
    # pred_proj belongs to the predictor update, but must stay in train-mode
    # batch normalization so its frozen function is baseline-equivalent.  Its
    # buffers are held fixed during the encoder-only update.
    with _freeze_batchnorm_running_stats(self.model.pred_proj):
        enc_out = _rsi_encode(self.model, batch)
        enc_emb = enc_out["emb"]
        enc_actions = batch["action"][:, :history_size]
        enc_z_t = enc_emb[:, :history_size]
        enc_z_tp1 = enc_emb[:, 1 : history_size + 1]
        enc_fake = self.model.predict(enc_z_t, enc_out["act_emb"][:, :history_size])
    encoder_pred_loss = (enc_fake - enc_z_tp1).pow(2).mean()
    # Hard audit of the requested baseline property: with P frozen, the
    # ordinary prediction loss must still carry gradient into both shared
    # encoder endpoints.  Compute it only on the first batch to avoid making
    # the full training run pay for an extra backward traversal.
    if batch_idx == 0:
        pred_grad_z_t, pred_grad_z_tp1 = torch.autograd.grad(
            encoder_pred_loss,
            (enc_z_t, enc_z_tp1),
            retain_graph=True,
            allow_unused=True,
        )
        pred_grad_z_t_norm = (
            pred_grad_z_t.detach().norm()
            if pred_grad_z_t is not None else encoder_pred_loss.new_zeros(())
        )
        pred_grad_z_tp1_norm = (
            pred_grad_z_tp1.detach().norm()
            if pred_grad_z_tp1 is not None else encoder_pred_loss.new_zeros(())
        )
        if pred_grad_z_t is None or pred_grad_z_tp1 is None:
            raise RuntimeError(
                "RSI invariant failed: encoder prediction loss was detached "
                "from a transition endpoint"
            )
    else:
        pred_grad_z_t_norm = encoder_pred_loss.new_zeros(())
        pred_grad_z_tp1_norm = encoder_pred_loss.new_zeros(())
    encoder_real_energy = _rsi_action_energy(idm, enc_z_t, enc_z_tp1, enc_actions)
    with torch.no_grad():
        encoder_fake_energy = _rsi_action_energy(idm, enc_z_t, enc_fake, enc_actions)
    encoder_sep_loss = F.relu(margin + encoder_real_energy - encoder_fake_energy)
    encoder_loss = (
        encoder_pred_loss
        + real_weight * encoder_real_energy
        + separation_weight * encoder_sep_loss
    )
    step(encoder_opt, schedulers[0], encoder_loss)

    # 3) Predictor: E is frozen and its endpoints are detached. IDM weights
    # are frozen but its input path remains differentiable, so catch-up repairs
    # only the generated transition rather than the discriminator.
    _set_requires_grad(self.model, False)
    for module in predictor_modules:
        _set_requires_grad(module, True)
    _set_requires_grad(idm, False)
    with torch.no_grad(), _freeze_batchnorm_running_stats(
        torch.nn.ModuleList([self.model.encoder, self.model.projector])
    ):
        pred_out = _rsi_encode(self.model, batch)
        pred_emb = pred_out["emb"]
        pred_z_t = pred_emb[:, :history_size].detach()
        pred_z_tp1 = pred_emb[:, 1 : history_size + 1].detach()
        pred_actions = batch["action"][:, :history_size]
        pred_real_energy = _rsi_action_energy(idm, pred_z_t, pred_z_tp1, pred_actions)
    pred_act_emb = (
        self.model.action_encoder(batch["action"])
        if self.model.action_encoder is not None
        else batch["action"]
    )
    pred_fake = self.model.predict(pred_z_t, pred_act_emb[:, :history_size])
    predictor_pred_loss = (pred_fake - pred_z_tp1).pow(2).mean()
    predictor_fake_energy = _rsi_action_energy(idm, pred_z_t, pred_fake, pred_actions)
    catch_loss = F.relu(predictor_fake_energy - pred_real_energy)
    predictor_loss = predictor_pred_loss + catch_weight * catch_loss
    step(predictor_opt, schedulers[1], predictor_loss)

    # Restore flags for validation/checkpointing and log losses separately.
    _set_requires_grad(self.model, True)
    _set_requires_grad(idm, True)
    state = {
        "loss": encoder_loss.detach() + predictor_loss.detach() + idm_loss.detach(),
        "pred_loss": encoder_pred_loss.detach(),
        "encoder_pred_loss": encoder_pred_loss.detach(),
        "predictor_pred_loss": predictor_pred_loss.detach(),
        "rsi_idm_loss": idm_loss.detach(),
        "rsi_real_energy": encoder_real_energy.detach(),
        "rsi_fake_energy": predictor_fake_energy.detach(),
        "rsi_sep_loss": encoder_sep_loss.detach(),
        "rsi_catch_loss": catch_loss.detach(),
        "rsi_pred_grad_to_z_t": pred_grad_z_t_norm,
        "rsi_pred_grad_to_z_tp1": pred_grad_z_tp1_norm,
    }
    self.log_dict(
        {f"fit/{key}": value for key, value in state.items()},
        on_step=True,
        on_epoch=True,
        sync_dist=True,
    )
    return state


def action_innovation_gradient_training_step(self, batch, batch_idx, cfg):
    """Separate nuisance-mean and AIG world-model updates.

    The world-model step uses the original AR/AdaLN forward function exactly:
    each conditioner numerically receives the factual action.  AIG only
    changes the conditioner backward path to factual minus behaviour-mean
    action.  The behavior head itself sees detached history and is frozen for
    this world-model step.
    """
    if batch_idx is None:
        batch_idx = 0
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    history_size = int(cfg.wm.get("history_size", 1))
    predictor = self.model.predictor
    if not getattr(predictor, "aig_enabled", False):
        raise TypeError("AIG training requires predictor.type=ar and aig_enabled=true")

    optimizers = self.optimizers()
    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]
    if len(optimizers) != 2:
        raise RuntimeError("AIG requires behavior_opt and wm_opt")
    # Optimizer ordering follows the config declaration.  ``behavior_opt`` is
    # intentionally declared first so its explicit regex wins over the broad
    # ``model`` world-model group for the nested behavior_head.
    behavior_opt, wm_opt = optimizers
    schedulers = self.lr_schedulers()
    if schedulers is None:
        schedulers = [None, None]
    elif not isinstance(schedulers, (list, tuple)):
        schedulers = [schedulers]
    if len(schedulers) != 2:
        schedulers = list(schedulers) + [None] * (2 - len(schedulers))

    def step(optimizer, scheduler, loss):
        optimizer.zero_grad(set_to_none=True)
        self.manual_backward(loss)
        self.clip_gradients(
            optimizer,
            gradient_clip_val=self.trainer.gradient_clip_val,
            gradient_clip_algorithm=self.trainer.gradient_clip_algorithm,
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    # 1) Nuisance behaviour fit.  Do not make an extra train-mode BatchNorm
    # update merely to train this tiny head: that would change the baseline
    # encoder statistics once per batch.
    _set_requires_grad(self.model, False)
    _set_requires_grad(predictor.behavior_head, True)
    # This extra no-grad encoder pass must not consume the random sequence of
    # the subsequent LeWM world-model pass.  With this fork, a continued
    # baseline and AIG see the same dropout masks for their live prediction
    # forward (given the same initial RNG state), in addition to AIG's exact
    # numerical forward equivalence at every AdaLN conditioner.
    rng_devices = []
    if batch["pixels"].is_cuda:
        rng_devices = [batch["pixels"].device.index or torch.cuda.current_device()]
    with torch.random.fork_rng(devices=rng_devices, enabled=True):
        with torch.no_grad(), _freeze_batchnorm_running_stats(self.model):
            behavior_out = self.model.encode(dict(batch))
            behavior_h = predictor.history_features(
                behavior_out["emb"][:, :history_size]
            )
    behavior_actions = batch["action"][:, :history_size]
    behavior_loss = predictor.behavior_loss_from_features(
        behavior_h.detach(), behavior_actions
    )
    step(behavior_opt, schedulers[0], behavior_loss)
    # Clear the behaviour-step gradients before the WM pass.  Besides saving
    # memory, this makes the isolation invariant observable: any gradient on
    # this head after the prediction backward would be an implementation bug.
    behavior_opt.zero_grad(set_to_none=True)

    # 2) Baseline-forward-equivalent world-model update.  Behavior parameters
    # stay frozen; prediction loss remains live into both encoder endpoints.
    _set_requires_grad(self.model, True)
    _set_requires_grad(predictor.behavior_head, False)
    output = forward_step(self, batch, "fit", cfg, aig_reference_action_mean=
        predictor.behavior_mean_from_features(behavior_h.detach(), detach_input=True).detach())
    wm_loss = output["loss"]

    # Audit the principal implementation invariant once per run: AIG must not
    # accidentally detach the usual prediction objective from the encoder's
    # input or target endpoint. ``forward_step`` exposes these only for AIG.
    if batch_idx == 0:
        pred_z_t = output.pop("_aig_z_t")
        pred_z_tp1 = output.pop("_aig_z_tp1")
        grad_z_t, grad_z_tp1 = torch.autograd.grad(
            output["pred_loss"], (pred_z_t, pred_z_tp1), retain_graph=True,
            allow_unused=True,
        )
        if grad_z_t is None or grad_z_tp1 is None:
            raise RuntimeError(
                "AIG invariant failed: prediction loss was detached from an "
                "encoder transition endpoint"
            )
        pred_grad_z_t = grad_z_t.detach().norm()
        pred_grad_z_tp1 = grad_z_tp1.detach().norm()
    else:
        output.pop("_aig_z_t", None)
        output.pop("_aig_z_tp1", None)
        pred_grad_z_t = wm_loss.new_zeros(())
        pred_grad_z_tp1 = wm_loss.new_zeros(())
    step(wm_opt, schedulers[1], wm_loss)
    if any(p.grad is not None for p in predictor.behavior_head.parameters()):
        raise RuntimeError(
            "AIG invariant failed: prediction loss reached behavior_head"
        )

    # Restore normal flags for validation/checkpointing.  The head never sees
    # a prediction gradient: its reference action was detached before AdaLN.
    _set_requires_grad(self.model, True)
    state = {
        "loss": wm_loss.detach() + behavior_loss.detach(),
        "pred_loss": output["pred_loss"].detach(),
        "aig_behavior_loss": behavior_loss.detach(),
        "aig_innovation_rms": (behavior_actions - predictor.behavior_mean_from_features(
            behavior_h.detach(), detach_input=True
        ).detach()).pow(2).mean().sqrt().detach(),
        "aig_pred_grad_to_z_t": pred_grad_z_t,
        "aig_pred_grad_to_z_tp1": pred_grad_z_tp1,
    }
    # ``forward_step`` already logged fit/loss and fit/pred_loss for the live
    # baseline-forward-equivalent world-model pass. Log AIG-specific quantities
    # here so there is no duplicate key with a different meaning.
    self.log_dict(
        {
            f"fit/{key}": state[key]
            for key in (
                "aig_behavior_loss", "aig_innovation_rms",
                "aig_pred_grad_to_z_t", "aig_pred_grad_to_z_tp1",
            )
        },
        on_step=True,
        on_epoch=True,
        sync_dist=True,
    )
    return state


def predictive_mask_training_step(self, batch, batch_idx, cfg):
    """Alternating fixed-budget prediction-driven visual information selection.

    First optimize only the state-only patch policy against the frozen world
    model, using the ST TopK mask. Then re-select a hard mask and optimize the
    ordinary LeWM objective plus the masked prediction branch.  The latter's
    next-frame latent is always a stop-gradient target; the full-observation
    branch remains exactly the usual ``forward_step`` objective.
    """
    if batch_idx is None:
        batch_idx = 0
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    mask_cfg = cfg.loss.predictive_mask
    history_size = int(cfg.wm.get("history_size", 1))
    beta = float(mask_cfg.beta)
    num_masked = int(mask_cfg.num_masked_patches)
    policy = self.mask_policy

    optimizers = self.optimizers()
    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]
    if len(optimizers) != 2:
        raise RuntimeError("Predictive-mask training requires policy_opt and wm_opt")
    policy_opt, wm_opt = optimizers
    schedulers = self.lr_schedulers()
    if schedulers is None:
        schedulers = [None, None]
    elif not isinstance(schedulers, (list, tuple)):
        schedulers = [schedulers]
    if len(schedulers) != 2:
        schedulers = list(schedulers) + [None] * (2 - len(schedulers))

    def step(optimizer, scheduler, loss):
        optimizer.zero_grad(set_to_none=True)
        self.manual_backward(loss)
        self.clip_gradients(
            optimizer,
            gradient_clip_val=self.trainer.gradient_clip_val,
            gradient_clip_algorithm=self.trainer.gradient_clip_algorithm,
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    # The mask policy is intentionally state-only. Its feature extraction and
    # target encoding do not update (or alter BatchNorm buffers of) the world
    # model, but gradients from masked pixels still pass through the frozen
    # encoder/predictor into the policy's ST score path.
    _set_requires_grad(self.model, False)
    _set_requires_grad(policy, True)
    current_pixels = batch["pixels"][:, :history_size]
    current_actions = batch["action"][:, :history_size]
    rng_devices = []
    if current_pixels.is_cuda:
        rng_devices = [current_pixels.device.index or torch.cuda.current_device()]
    with torch.random.fork_rng(devices=rng_devices, enabled=True):
        with torch.no_grad(), _freeze_batchnorm_running_stats(self.model):
            target_out = self.model.encode(dict(batch))
            policy_features = target_out["cls_features"][:, :history_size]
            target_emb = target_out["emb"][:, 1 : history_size + 1].detach()
        _, st_mask, soft_mask, _ = policy(
            policy_features.detach(), num_masked
        )
        with _freeze_batchnorm_running_stats(self.model):
            masked_out = self.model.encode_masked(
                {"pixels": current_pixels, "action": current_actions}, st_mask
            )
            policy_pred = self.model.predict(
                masked_out["emb"], masked_out["act_emb"]
            )
            policy_loss = (policy_pred - target_emb).pow(2).mean()
    step(policy_opt, schedulers[0], policy_loss)

    # Re-select after the policy update, freeze its discrete decision, and
    # train the complete (full-observation) LeWM + Inv + ACA objective with
    # the additive stopped-target masked branch.
    _set_requires_grad(policy, False)
    _set_requires_grad(self.model, True)
    output = forward_step(self, batch, "fit", cfg)
    with torch.no_grad():
        hard_mask, _, _, _ = policy(
            output["cls_features"][:, :history_size], num_masked
        )
    fixed_mask = hard_mask.detach()
    with _freeze_batchnorm_running_stats(self.model):
        masked_out = self.model.encode_masked(
            {"pixels": current_pixels, "action": current_actions}, fixed_mask
        )
    masked_pred = self.model.predict(masked_out["emb"], masked_out["act_emb"])
    # ``forward_step(..., stage="fit")`` deliberately keeps embeddings live
    # for its base loss but does not expose ``output["emb"]``. Re-encode only
    # the target endpoint under no-grad for the masked branch's stop-gradient
    # target, exactly matching the stated objective.
    with torch.no_grad(), _freeze_batchnorm_running_stats(self.model):
        world_target_out = self.model.encode(dict(batch))
        world_target_emb = world_target_out["emb"][:, 1 : history_size + 1]
    masked_loss = (masked_pred - world_target_emb).pow(2).mean()
    world_loss = output["loss"] + beta * masked_loss
    step(wm_opt, schedulers[1], world_loss)

    # Do not leave checkpointing/validation with a frozen policy. The policy
    # is not used by planning; it is a training-only information selector.
    _set_requires_grad(self.model, True)
    _set_requires_grad(policy, True)
    masked_ratio = fixed_mask.float().mean()
    state = {
        "loss": world_loss.detach(),
        "pred_loss": output["pred_loss"].detach(),
        "mask_policy_loss": policy_loss.detach(),
        "mask_loss": masked_loss.detach(),
        "mask_ratio": masked_ratio.detach(),
        "mask_soft_budget": soft_mask.detach().sum(dim=-1).mean(),
    }
    # ``forward_step`` has already logged the base LeWM loss. Avoid emitting
    # the same key twice with a different value; the additive total is kept
    # separately for experiment tracking.
    self.log_dict(
        {
            "fit/mask_world_loss": state["loss"],
            "fit/mask_policy_loss": state["mask_policy_loss"],
            "fit/mask_loss": state["mask_loss"],
            "fit/mask_ratio": state["mask_ratio"],
            "fit/mask_soft_budget": state["mask_soft_budget"],
        },
        on_step=True,
        on_epoch=True,
        sync_dist=True,
    )
    return state


def forward_step(self, batch, stage, cfg, aig_reference_action_mean=None):
    lambd_sigreg = cfg.loss.sigreg.weight
    lambd_innov = cfg.loss.get("innov", {}).get("weight", 0.0)
    lambd_inv = cfg.loss.inverse.weight
    lambd_pdc = cfg.loss.pdc.weight
    lambd_geo_pdc = cfg.loss.geo_pdc.weight
    lambd_as_pdc = cfg.loss.get("as_pdc", {}).get("weight", 0.0)
    lambd_lift = cfg.loss.get("delta_lift", {}).get("weight", 0.0)
    lambd_aca = cfg.loss.get("aca", {}).get("weight", 0.0)
    lambd_an = cfg.loss.get("action_normal", {}).get("weight", 0.0)
    lambd_random_aca = cfg.loss.get("random_aca", {}).get("weight", 0.0)
    lambd_gradient_penalty = cfg.loss.get("gradient_penalty", {}).get("weight", 0.0)
    lambd_cai = cfg.loss.get("cai", {}).get("weight", 0.0)
    lambd_si = cfg.loss.get("si", {}).get("weight", 0.0)
    online_aca = cfg.loss.get("aca", {}).get("online", {}).get("enabled", False)
    epoch_active_aca = cfg.loss.get("aca", {}).get("epoch_active", {}).get("enabled", False)
    online_collect = (
        stage == "fit"
        and online_aca
        and self.global_step % int(cfg.loss.aca.online.every_n_steps) == 0
    )
    inverse_type = cfg.inverse.get("type", "mlp")
    history_size = int(cfg.wm.get("history_size", 1))
    required_steps = history_size + 1

    # A collection step must use only its originating offline state metadata.
    # All other steps may replace a fraction of transitions with real replay.
    if stage == "fit" and online_aca and not online_collect:
        batch = self.online_aca_replay.mix(batch)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)

    emb = output["emb"]
    act_emb = output["act_emb"]
    if emb.size(1) < required_steps:
        raise ValueError(
            f"Batch sequence length {emb.size(1)} is too short for "
            f"wm.history_size={history_size}; need at least {required_steps} "
            "steps for one-step supervision."
        )

    z_t = emb[:, :history_size]
    actions = batch["action"][:, :history_size]
    tgt_emb = emb[:, 1:required_steps]
    endpoint_completion = str(cfg.predictor.get("type", "ar")) in (
        "endpoint_completion", "edge_completion"
    )
    masked_transition = str(cfg.predictor.get("type", "ar")) in (
        "masked_transition", "mtm", "transition_jepa"
    )

    # An action-labelled transition is trained as masked endpoint completion:
    # choose exactly one endpoint per sample, then predict it from the other
    # endpoint and the *same* logged action.  This is one prediction loss and
    # one predictor evaluation per item, not forward loss plus a reverse head.
    if endpoint_completion:
        direction = torch.rand(z_t.shape[:-1], device=z_t.device) < 0.5
        known_emb = torch.where(direction.unsqueeze(-1), z_t, tgt_emb)
        endpoint_target = torch.where(direction.unsqueeze(-1), tgt_emb, z_t)
        pred_emb = self.model.complete_endpoint(
            known_emb, act_emb[:, :history_size], forward=direction
        )
        output["pred_loss"] = (pred_emb - endpoint_target).pow(2).mean()
        output["loss"] = output["pred_loss"]
        output["edge_forward_ratio"] = direction.float().mean().detach()

    # Masked Transition World Model: every item performs both complementary
    # queries through the same Transformer.  The action query sees detached
    # endpoints, so inverse ambiguity cannot distort the encoder; it still
    # updates all shared predictor parameters and the thin action projection.
    if masked_transition:
        pred_emb = self.model.predict(z_t, actions)
        output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
        pred_actions = self.model.complete_transition_action(
            z_t.detach(), tgt_emb.detach()
        )
        output["mtm_action_loss"] = (pred_actions - actions).pow(2).mean()
        mtm_cfg = cfg.loss.get("masked_transition", {})
        pred_weight = float(mtm_cfg.get("pred_weight", 0.5))
        action_weight = float(mtm_cfg.get("action_weight", 0.5))
        if pred_weight < 0.0 or action_weight < 0.0:
            raise ValueError("masked-transition loss weights must be non-negative")
        if pred_weight + action_weight <= 0.0:
            raise ValueError("at least one masked-transition loss weight must be positive")
        output["loss"] = (
            pred_weight * output["pred_loss"]
            + action_weight * output["mtm_action_loss"]
        )

    # Adversarial Action Mining (ACA): mines, at a fixed radius from the real
    # action, the nearby impostor action that best fools the *current*
    # predictor into reproducing the real transition, then requires the real
    # action to explain that transition strictly better (see
    # JEPA.adversarial_action_energy).
    if online_aca:
        # Online ACA does not optimize a hinge. Instead, its positive-hinge
        # hard negatives are executed in MuJoCo and return as factual replay.
        pred_emb = self.model.predict(z_t, act_emb[:, :history_size])
        output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
        output["loss"] = output["pred_loss"]
        online_cfg = cfg.loss.aca.online
        if online_collect:
            selected, mean_hinge = self.model.sample_positive_aca_actions(
                z_t,
                actions,
                tgt_emb,
                rho=float(online_cfg.rho),
                margin=float(online_cfg.margin),
                top_fraction=float(online_cfg.top_fraction),
                max_samples=int(online_cfg.max_samples_per_round),
                action_low=self.online_aca_action_low,
                action_high=self.online_aca_action_high,
            )
            self.online_aca_replay.add(batch, selected)
            output["online_aca_positive_hinge"] = mean_hinge
            output["online_aca_collected"] = output["pred_loss"].new_tensor(len(selected))
        else:
            output["online_aca_positive_hinge"] = output["pred_loss"].new_zeros(())
            output["online_aca_collected"] = output["pred_loss"].new_zeros(())
        output["online_aca_replayed"] = output["pred_loss"].new_tensor(
            float(getattr(self.online_aca_replay, "last_mix_count", 0))
        )
    elif lambd_aca:
        aca_cfg = cfg.loss.aca
        adversary = str(aca_cfg.get("adversary", "gradient"))
        aca_mode = str(aca_cfg.get("mode", "gradient"))
        if aca_mode == "counterfactual_only":
            if endpoint_completion or masked_transition:
                raise ValueError(
                    "counterfactual_only ACA requires the ordinary forward predictor"
                )
            if adversary != "gradient":
                raise ValueError(
                    "counterfactual_only ACA supports loss.aca.adversary=gradient only"
                )
            aca_out = self.model.counterfactual_only_action_energy(
                z_t,
                actions,
                tgt_emb,
                rho=float(aca_cfg.get("rho", 1.0)),
                noise_scale=0.0,
            )
        elif endpoint_completion:
            if adversary != "gradient":
                raise ValueError(
                    "endpoint_completion currently supports BiACA with "
                    "loss.aca.adversary=gradient only"
                )
            aca_out = self.model.bidirectional_adversarial_action_energy(
                known_emb,
                actions,
                endpoint_target,
                forward=direction,
                rho=float(aca_cfg.get("rho", 1.0)),
                noise_scale=0.0,
                margin=float(aca_cfg.get("margin", 0.0)),
            )
        elif adversary == "closed_form":
            aca_out = self.model.closed_form_aca(
                z_t, actions, tgt_emb, rho=float(aca_cfg.get("rho", 1.0))
            )
        elif adversary in ("convergent", "aca_infinity"):
            aca_out = self.model.convergent_aca(
                z_t,
                actions,
                tgt_emb,
                rho=float(aca_cfg.get("rho", 1.0)),
                max_steps=int(aca_cfg.get("max_steps", 20)),
                step_size=aca_cfg.get("step_size", None),
                grad_tol=float(aca_cfg.get("grad_tol", 1e-5)),
                num_restarts=int(aca_cfg.get("num_restarts", 4)),
                margin=float(aca_cfg.get("margin", 0.0)),
            )
        elif adversary in ("random_shooting", "cem"):
            aca_out = self.model.planner_consistent_aca(
                z_t,
                actions,
                tgt_emb,
                rho=float(aca_cfg.get("rho", 1.0)),
                adversary=adversary,
                n_samples=int(aca_cfg.get("n_samples", 32)),
                n_elites=int(aca_cfg.get("n_elites", 8)),
                n_iters=int(aca_cfg.get("n_iters", 3)),
                margin=float(aca_cfg.get("margin", 0.0)),
            )
        else:
            aca_out = self.model.adversarial_action_energy(
                z_t,
                actions,
                tgt_emb,
                rho=float(aca_cfg.get("rho", 1.0)),
                noise_scale=0.0,
                margin=float(aca_cfg.get("margin", 0.0)),
                encoder_scale=aca_cfg.get("encoder_scale", "none"),
            )
        pred_emb = aca_out["pred_emb"]
        output["pred_loss"] = aca_out["pred_loss"]
        output["aca_hat_loss"] = aca_out["pred_loss_hat"]
        output["aca_loss"] = aca_out["aca_loss"]
        # Keep predictor- and encoder-recipient ACA terms separate until the
        # inverse loss is available below, where the requested loss-ratio
        # scaling can be computed.
        output["aca_pred_component"] = aca_out.get("aca_loss_pred", aca_out["aca_loss"])
        output["aca_encoder_component"] = aca_out.get("aca_loss_encoder", aca_out["aca_loss"])
        for key in ("aca_margin", "causal_capacity", "causal_min_eig", "aca_minimum_distance", "aca_active_ratio", "aca_factual_energy"):
            if key in aca_out:
                output[key] = aca_out[key]
        if aca_mode == "counterfactual_only":
            # This ablation has exactly one forward-energy term: E(a_cf).
            # ``pred_loss`` is retained purely as a named diagnostic; adding
            # it again here would accidentally optimize (1 + lambda) E(a_cf).
            output["loss"] = lambd_aca * output["aca_loss"]
        else:
            output["loss"] = output["pred_loss"]
    elif not endpoint_completion and not masked_transition:
        use_aig = aig_reference_action_mean is not None
        if use_aig:
            if self.model.action_encoder is None:
                raise RuntimeError("AIG requires LeWM's learned action encoder")
            reference_act_emb = self.model.action_encoder(aig_reference_action_mean)
        else:
            reference_act_emb = None
        if lambd_innov:
            if use_aig or history_size != 1:
                raise ValueError(
                    "NIC innovation loss currently requires history_size=1 "
                    "and the ordinary forward predictor"
                )
            # NIC needs two adjacent one-step residuals.  The AR predictor is
            # instantiated for one history token, so flatten transition
            # positions into the batch and evaluate the same predictor once.
            n_transitions = emb.size(1) - 1
            if n_transitions < 2:
                raise ValueError(
                    "NIC innovation loss requires data.dataset.num_steps >= 3"
                )
            z_seq = emb[:, :-1]
            a_seq = act_emb[:, :-1]
            bsz, n_trans, dim = z_seq.shape
            pred_flat = self.model.predict(
                z_seq.reshape(bsz * n_trans, 1, dim),
                a_seq.reshape(bsz * n_trans, 1, a_seq.size(-1)),
            )
            pred_seq = pred_flat.reshape(bsz, n_trans, -1)
            pred_emb = pred_seq[:, :history_size]
            residual = emb[:, 1:] - pred_seq
            scale = residual.float().square().mean((0, 1), keepdim=True).add(
                float(cfg.loss.innov.get("eps", 1e-6))
            ).sqrt()
            residual_norm = residual.float() / scale
            e0, e1 = residual_norm[:, :-1], residual_norm[:, 1:]
            c1 = torch.einsum("btd,bte->de", e0, e1) / max(
                1, e0.shape[0] * e0.shape[1]
            )
            output["innov_loss"] = c1.square().mean()
            output["innov_lag1_abs"] = c1.abs().mean().detach()
            output["innov_residual_rms"] = residual.float().square().mean().sqrt().detach()
            # Every transition used to form the innovation sequence remains
            # prediction-supervised; the second transition must not become a
            # regularizer-only example.
            output["pred_loss"] = (pred_seq - emb[:, 1:]).pow(2).mean()
            pred_emb = pred_seq[:, :history_size]
        else:
            pred_emb = self.model.predict(
                z_t, act_emb[:, :history_size],
                reference_act_emb=reference_act_emb,
                use_aig=use_aig,
            )
        if not lambd_innov:
            output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
        output["loss"] = output["pred_loss"]

    # Epoch-active ACA is a data-generation mechanism, not a loss term.  The
    # current batch is still trained with the ordinary forward/inverse loss;
    # its original rows merely contribute one (action, hinge) record to the
    # global ranker.  Added rows are marked replay_id=-1 and never re-mined.
    if stage == "fit" and epoch_active_aca:
        mined = self.epoch_active_aca.mine(
            self.model, batch, z_t, actions, tgt_emb, self.global_step
        )
        output["epoch_aca_positive_batch"] = output["pred_loss"].new_tensor(float(mined))

    # The AIG manual update validates once that the ordinary prediction loss
    # remains connected to both encoder endpoints.  These private values are
    # removed before the custom training step returns and are never logged.
    if cfg.loss.get("aig", {}).get("weight", 0.0) and stage == "fit":
        output["_aig_z_t"] = z_t
        output["_aig_z_tp1"] = tgt_emb

    if lambd_inv:
        if masked_transition:
            raise ValueError(
                "masked_transition has an intrinsic action-mask objective; "
                "set loss.inverse.weight=0"
            )
        if inverse_type == "iipdc":
            # Blind step: trains only the blind predictor, on detached z_t/z_tp1
            # so no gradient reaches the encoder through this term.
            blind_out = self.blind(z_t.detach())
            output["blind_loss"] = (blind_out - tgt_emb.detach()).pow(2).mean()
            output["loss"] = output["loss"] + output["blind_loss"]

            # World step: blind_ref is computed from the live z_t (so the blind
            # predictor keeps tracking the current encoder), but IIPDCHead
            # stop-gradients it internally, isolating blind from the encoder/
            # action-predictor/Q update carried by this term.
            blind_ref = self.blind(z_t)
            output["inv_loss"] = self.iipdc(pred_emb, blind_ref, actions)
        else:
            # Preserve the factual InvReg objective exactly: it sees the two
            # real encoded endpoints and therefore retains its original
            # encoder/IDM gradients.
            pred_actions = self.model.predict_action(emb[:, :-1], emb[:, 1:])
            output["inv_loss"] = (pred_actions - batch["action"][:, :-1]).pow(2).mean()
        output["loss"] = output["loss"] + lambd_inv * output["inv_loss"]

    if lambd_an:
        an_out = self.model.action_normal_energy(
            z_t,
            actions,
            tgt_emb,
            create_graph=(stage == "fit"),
        )
        output.update(an_out)
        output["loss"] = output["loss"] + float(lambd_an) * output["action_normal_loss"]

    if lambd_random_aca:
        random_cfg = cfg.loss.random_aca
        random_out = self.model.random_aca_energy(
            z_t,
            actions,
            tgt_emb,
            rho=float(random_cfg.rho),
            margin=float(random_cfg.get("margin", 0.0)),
        )
        output["random_aca_hat_loss"] = random_out["random_aca_hat_loss"]
        output["random_aca_loss"] = random_out["random_aca_loss"]
        output["loss"] = output["loss"] + float(lambd_random_aca) * random_out["random_aca_loss"]

    if lambd_gradient_penalty:
        grad_cfg = cfg.loss.gradient_penalty
        grad_out = self.model.action_energy_gradient_penalty(
            z_t,
            actions,
            tgt_emb,
            rho=float(grad_cfg.rho),
            create_graph=(stage == "fit"),
        )
        output["gradient_penalty_loss"] = grad_out["gradient_penalty_loss"]
        output["gradient_penalty_grad_norm"] = grad_out["gradient_penalty_grad_norm"]
        output["loss"] = output["loss"] + float(lambd_gradient_penalty) * grad_out["gradient_penalty_loss"]

    # ACA predictor gradients stay unscaled.  Only its encoder-side copy is
    # multiplied so its scale matches the loss magnitude received by the
    # encoder versus the predictor (prediction + inverse, divided by the
    # predictor's prediction loss).  All ratio statistics are detached.
    if lambd_aca and "aca_pred_component" in output and aca_mode != "counterfactual_only":
        aca_cfg = cfg.loss.aca
        mode = str(aca_cfg.get("encoder_scale", "none"))
        if mode == "loss_ratio":
            base_pred = output["pred_loss"].detach()
            base_encoder = output["pred_loss"].detach() + float(lambd_inv) * output.get(
                "inv_loss", base_pred.new_zeros(())
            ).detach()
            ratio = base_encoder / base_pred.clamp_min(1e-8)
            output["aca_encoder_scale"] = ratio
            output["loss"] = output["loss"] + lambd_aca * (
                output["aca_pred_component"]
                + ratio * output["aca_encoder_component"]
            )
        else:
            # Historical/default behavior: one ordinary ACA hinge only.
            output["aca_encoder_scale"] = output["aca_pred_component"].new_tensor(0.0)
            output["loss"] = output["loss"] + lambd_aca * output["aca_loss"]

    if lambd_cai:
        # CAI estimator best-response losses are evaluated on detached
        # representations, so only the explicit CAI repair term updates E.
        cai_actions = batch["action"][:, :history_size]
        cai_z_t = z_t.detach()
        cai_z_tp1 = tgt_emb
        blind_pred = self.cai_blind(cai_z_t)
        trans_pred = self.cai_transition(cai_z_t, cai_z_tp1.detach())
        blind_loss = (blind_pred - cai_actions).pow(2).mean()
        trans_loss = (trans_pred - cai_actions).pow(2).mean()
        output["cai_blind_loss"] = blind_loss
        output["cai_trans_loss"] = trans_loss
        # Freeze estimator parameters for the repair pass while preserving
        # d(loss)/d(z_{t+1}); z_t is detached and cannot be used as a shortcut.
        req_blind = [p.requires_grad for p in self.cai_blind.parameters()]
        req_trans = [p.requires_grad for p in self.cai_transition.parameters()]
        for p in self.cai_blind.parameters(): p.requires_grad_(False)
        for p in self.cai_transition.parameters(): p.requires_grad_(False)
        try:
            blind_ref = self.cai_blind(cai_z_t)
            trans_ref = self.cai_transition(cai_z_t, cai_z_tp1)
            l_blind_ref = (blind_ref - cai_actions).pow(2).mean(dim=-1)
            l_trans_ref = (trans_ref - cai_actions).pow(2).mean(dim=-1)
            output["cai_gain"] = (l_blind_ref - l_trans_ref).mean().detach()
            output["cai_failure"] = (l_trans_ref >= l_blind_ref).float().mean().detach()
            output["cai_loss"] = F.softplus(l_trans_ref - l_blind_ref.detach()).mean()
        finally:
            for p, r in zip(self.cai_blind.parameters(), req_blind): p.requires_grad_(r)
            for p, r in zip(self.cai_transition.parameters(), req_trans): p.requires_grad_(r)
        output["loss"] = output["loss"] + lambd_cai * (output["cai_loss"] + blind_loss + trans_loss)

    if lambd_sigreg:
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
        output["loss"] = output["loss"] + lambd_sigreg * output["sigreg_loss"]

    if lambd_innov:
        output["loss"] = output["loss"] + lambd_innov * output["innov_loss"]

    if lambd_si:
        if endpoint_completion:
            raise ValueError(
                "SI-WM currently requires a forward predictor; "
                "endpoint_completion is bidirectional rather than forward-only"
            )
        si_cfg = cfg.loss.si
        si_out = self.model.self_inverting_action_denoising(
            z_t,
            actions,
            tgt_emb,
            sigma=float(si_cfg.get("sigma", 0.5)),
            step_size=si_cfg.get("step_size", None),
            create_graph=(stage == "fit"),
        )
        output.update(si_out)
        output["loss"] = output["loss"] + float(lambd_si) * output["si_loss"]

    if lambd_pdc:
        resized_pixels = self.model.resized_pixels(batch["pixels"])
        output["pdc_loss"] = self.pdc(emb, resized_pixels)
        output["loss"] = output["loss"] + lambd_pdc * output["pdc_loss"]

        if cfg.loss.pdc.get("jensen", False):
            # Jensen-PDC: counterfactual action a_cf via batch permutation
            # (a valid, in-distribution action), midpoint action a_m =
            # (a+a_cf)/2. z_next is kept differentiable (real embedding, not
            # detached) so this shapes the encoder, not only the predictor.
            perm = torch.randperm(emb.size(0), device=emb.device)
            actions_cf = actions[perm]
            actions_mid = 0.5 * (actions + actions_cf)
            if self.model.action_encoder is not None:
                act_emb_cf = self.model.action_encoder(actions_cf)
                act_emb_mid = self.model.action_encoder(actions_mid)
            else:
                act_emb_cf = actions_cf
                act_emb_mid = actions_mid
            z_other_pred = self.model.predict(z_t, act_emb_cf)
            z_mid_pred = self.model.predict(z_t, act_emb_mid)
            output["jpdc_loss"] = self.pdc.forward_jpdc(z_mid_pred, tgt_emb, z_other_pred)
            output["loss"] = output["loss"] + lambd_pdc * output["jpdc_loss"]

    if lambd_geo_pdc:
        resized_pixels = self.model.resized_pixels(batch["pixels"])
        output["geo_pdc_loss"] = self.geo_pdc(emb, resized_pixels)
        output["loss"] = output["loss"] + lambd_geo_pdc * output["geo_pdc_loss"]

    if lambd_as_pdc:
        # Counterfactual action a_cf: permute the logged (normalized) actions
        # across the batch -> a valid, in-distribution action, not synthetic.
        perm = torch.randperm(emb.size(0), device=emb.device)
        actions_cf = actions[perm]
        act_emb_cf = (
            self.model.action_encoder(actions_cf)
            if self.model.action_encoder is not None
            else actions_cf
        )
        # Counterfactual future under the SAME state z_t: F(z_t, a_cf).
        cf_pred = self.model.predict(z_t, act_emb_cf)
        # One endpoint is the real next embedding tgt_emb (kept differentiable
        # so this constrains the encoder, not only the predictor).
        output["as_pdc_loss"] = self.as_pdc(
            tgt_emb, cf_pred, actions, actions_cf
        )
        output["loss"] = output["loss"] + lambd_as_pdc * output["as_pdc_loss"]

    if lambd_lift:
        resized_pixels = self.model.resized_pixels(batch["pixels"])
        output["delta_lift_loss"] = self.delta_lift(emb, resized_pixels)
        output["loss"] = output["loss"] + lambd_lift * output["delta_lift_loss"]

    if stage != "fit":
        output["emb"] = emb.detach()

    logs = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k
        or k in (
            "cai_gain", "cai_failure", "si_energy", "si_score_norm",
            "si_recovery_error", "online_aca_positive_hinge",
            "online_aca_collected", "online_aca_replayed",
            "aca_encoder_scale",
            "epoch_aca_positive_batch",
            "innov_lag1_abs", "innov_residual_rms",
        )
    }
    if stage == "fit":
        self.log_dict(logs, on_step=True, on_epoch=True, sync_dist=True)
    else:
        self.log_dict(logs, on_step=False, on_epoch=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path=None, config_name=None)
def run(cfg):
    # ACA package deliberately exposes only the four supported objectives.
    # Keep legacy fields internally for checkpoint compatibility, but reject
    # accidental activation of unrelated research methods.
    forbidden = ("action_normal", "random_aca", "gradient_penalty", "si", "innov",
                 "rsi", "aig", "cai", "pdc", "geo_pdc", "as_pdc", "delta_lift")
    active = [name for name in forbidden if float(cfg.loss.get(name, {}).get("weight", 0.0) or 0.0) != 0.0]
    if active:
        raise ValueError("ACA package allows only SIGReg, inverse, and ACA; disabled: " + ", ".join(active))
    if cfg.predictor.get("type", "ar") != "ar":
        raise ValueError("ACA package training requires predictor.type=ar")
    if bool(cfg.loss.get("predictive_mask", {}).get("enabled", False)):
        raise ValueError("predictive_mask is not part of the ACA package")
    # Action-Normal and SI-WM differentiate through an action gradient
    # (``autograd.grad(create_graph=True)``), which requires a second backward
    # through every operation in the predictor.  CUDA FlashAttention,
    # memory-efficient SDPA, and cuDNN SDPA do not implement that higher-order
    # derivative in all supported PyTorch/CUDA combinations.  Force the
    # reference math SDPA backend for these objectives; ordinary training keeps
    # the faster fused kernels.
    higher_order_loss = (
        bool(cfg.loss.get("action_normal", {}).get("weight", 0.0))
        or bool(cfg.loss.get("gradient_penalty", {}).get("weight", 0.0))
        or bool(cfg.loss.get("si", {}).get("weight", 0.0))
    )
    if higher_order_loss and torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        print(
            "Higher-order action loss enabled; using math SDPA attention "
            "(Flash/memory-efficient/cuDNN kernels disabled)."
        )
    history_size = int(cfg.wm.get("history_size", 1))
    required_steps = history_size + 1
    with open_dict(cfg):
        cfg.wm.history_size = history_size
        cfg.wm.num_preds = int(cfg.wm.get("num_preds", 1))
        if cfg.loss.geo_pdc.weight:
            max_scale = max(int(m) for m in cfg.loss.geo_pdc.kwargs.scales)
            required_steps = max(required_steps, max_scale + 1)
        current_steps = int(cfg.data.dataset.get("num_steps", required_steps))
        if current_steps < required_steps:
            cfg.data.dataset.num_steps = required_steps
        online_aca = bool(cfg.loss.get("aca", {}).get("online", {}).get("enabled", False))
        epoch_active_aca = bool(
            cfg.loss.get("aca", {}).get("epoch_active", {}).get("enabled", False)
        )
        counterfactual_cfg = cfg.data.get("counterfactual", {})
        counterfactual_enabled = bool(counterfactual_cfg.get("enabled", False))
        if cfg.get("init_from_checkpoint") and cfg.get("resume_from_checkpoint"):
            raise ValueError(
                "init_from_checkpoint and resume_from_checkpoint are mutually exclusive"
            )
        if sum((online_aca, epoch_active_aca, counterfactual_enabled)) > 1:
            raise ValueError(
                "choose exactly one data-collection mode: offline counterfactual, "
                "online ACA, or epoch-active ACA"
            )
        if counterfactual_enabled and not counterfactual_cfg.get("path"):
            raise ValueError("data.counterfactual.path is required when enabled=true")
        if counterfactual_enabled and bool(counterfactual_cfg.get("only", False)):
            # A one-epoch adaptation on collected transitions has no held-out
            # counterfactual validation split.  Its short epoch can also be
            # shorter than the base config's 500-step validation interval.
            # Disable validation rather than constructing an unrelated
            # original-dataset validation pass.
            cfg.trainer.limit_val_batches = 0.0
        if online_aca or epoch_active_aca:
            required_online_columns = ("qpos", "qvel", "target_pos", "id")
            cfg.data.dataset.keys_to_load = list(cfg.data.dataset.keys_to_load)
            for key in required_online_columns:
                if key not in cfg.data.dataset.keys_to_load:
                    cfg.data.dataset.keys_to_load.append(key)

    # Keep this base dataset even when training only counterfactual samples:
    # its atomic action mean/std are the coordinate system of the checkpoint.
    base_train_set = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    subset_cfg = cfg.data.get("subset", {})
    subset_indices_path = subset_cfg.get("episode_indices", None)
    if subset_indices_path:
        subset_path = Path(str(subset_indices_path)).expanduser().resolve()
        if not subset_path.is_file():
            raise FileNotFoundError(f"subset episode index file not found: {subset_path}")
        selected_episodes = np.load(subset_path).astype(np.int64)
        if selected_episodes.ndim != 1 or len(selected_episodes) == 0:
            raise ValueError(f"subset indices must be a non-empty 1-D array: {subset_path}")
        if selected_episodes.min() < 0 or selected_episodes.max() >= len(base_train_set.lengths):
            raise ValueError(f"subset episode index out of range for {base_train_set.h5_path}")
        selected = set(int(x) for x in selected_episodes.tolist())
        base_train_set.clip_indices = [(ep, start) for ep, start in base_train_set.clip_indices if ep in selected]
        print(f"Fixed dataset subset: {len(selected):,} episodes, {len(base_train_set):,} clips; source={base_train_set.h5_path}")
    train_set = base_train_set
    if online_aca or epoch_active_aca:
        if cfg.data.dataset.name != "reacher_train":
            raise ValueError("active ACA collection currently supports reacher_train only")
        required_online_columns = ("pixels", "action", "observation", "qpos", "qvel", "target_pos", "id")
        base_train_set._open()
        missing = [key for key in required_online_columns if key not in base_train_set.h5_file]
        if missing:
            raise KeyError(f"online ACA requires HDF5 columns {missing}")
    # Read only the physical pixel column; merged cached keys may not exist in HDF5.
    base_train_set._open()
    pixel_shape = base_train_set.h5_file["pixels"][0].shape
    pixel_hw = tuple(pixel_shape[:2]) if len(pixel_shape) >= 2 else ()
    img_size = None if pixel_hw == (cfg.img_size, cfg.img_size) else cfg.img_size
    transforms = []
    if img_size is not None:
        transforms.append(ResizeCompat(img_size, source="pixels", target="pixels"))

    if online_aca or epoch_active_aca:
        transforms.append(
            PreserveColumns("pixels", "observation", "qpos", "qvel", "target_pos", "id")
        )

    with open_dict(cfg):
        if online_aca or epoch_active_aca:
            # The collector/replay lives in the Lightning process; forked
            # workers would retain stale dataset handles and cannot observe it.
            cfg.loader.num_workers = 0
            cfg.loader.persistent_workers = False
            cfg.loader.pop("prefetch_factor", None)
            # The environment lives in the training process and appends a
            # mutable dataset at epoch boundaries.  This is intentionally one
            # GPU; distributed ranks would otherwise each generate a separate
            # and nondeterministic replay file.
            cfg.trainer.devices = 1
        macro_transforms = list(transforms)
        metadata_columns = {"id", "qpos", "qvel", "target_pos"}
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            if col not in metadata_columns:
                transforms.append(get_column_normalizer(base_train_set, col, col))
            if col == "action":
                macro_transforms.append(get_macro_action_normalizer(
                    base_train_set, col, col, cfg.data.dataset.frameskip
                ))
            elif col not in metadata_columns:
                macro_transforms.append(get_column_normalizer(base_train_set, col, col))
            if col not in metadata_columns:
                setattr(cfg.wm, f"{col}_dim", base_train_set.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    base_train_set.transform = transform
    epoch_active_dataset = None
    if epoch_active_aca:
        run_id = cfg.get("subdir") or "epoch_active_aca"
        replay_path = get_runs_root() / run_id / "epoch_active_aca_replay.h5"
        epoch_active_dataset = EpochActiveACADataset(
            base_train_set,
            replay_path,
            transform=transform,
            image_shape=pixel_shape,
            action_dim=int(cfg.data.dataset.frameskip) * int(cfg.wm.action_dim),
        )
        train_set = epoch_active_dataset
        print(
            "Epoch-active ACA enabled: rho="
            f"{cfg.loss.aca.epoch_active.rho}, global top "
            f"{100 * float(cfg.loss.aca.epoch_active.top_fraction):g}% positive hinges; "
            "new data begins next epoch."
        )
    if counterfactual_enabled:
        counterfactual_set = CounterfactualTransitionDataset(
            cfg.data.counterfactual.path,
            transform=spt.data.transforms.Compose(*macro_transforms),
            include_proprio="proprio" in cfg.data.dataset.keys_to_load,
        )
        expected_action_dim = int(cfg.data.dataset.frameskip) * int(cfg.wm.action_dim)
        if counterfactual_set.action_dim != expected_action_dim:
            raise ValueError(
                f"counterfactual action_dim={counterfactual_set.action_dim}; "
                f"expected {expected_action_dim}"
            )
        train_set = (
            counterfactual_set if bool(cfg.data.counterfactual.get("only", False))
            else swm.data.ConcatDataset([base_train_set, counterfactual_set])
        )
        print(
            f"Counterfactual dataset enabled: {len(counterfactual_set)} transitions, "
            f"only={bool(cfg.data.counterfactual.get('only', False))}."
        )

    eval_kwargs = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    if not eval_kwargs["name"].endswith("_train"):
        raise ValueError(
            f"cfg.data.dataset.name={eval_kwargs['name']!r} must end with "
            "'_train' so the held-out '_eval' split name can be derived"
        )
    eval_kwargs["name"] = eval_kwargs["name"][: -len("_train")] + "_eval"
    val_set = swm.data.HDF5Dataset(**eval_kwargs, transform=transform)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        shuffle=True,
        drop_last=True,
        generator=rnd_gen,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        **cfg.loader,
        shuffle=False,
        drop_last=False,
    )
    data_module = spt.data.DataModule(train=train_loader, val=val_loader)
    scheduler_warmup_steps, scheduler_max_steps = scheduler_step_budget(cfg, train_set)
    print(
        f"Explicit LR schedule: warmup_steps={scheduler_warmup_steps}, "
        f"max_steps={scheduler_max_steps} (per rank)."
    )

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim
    predictor_type = cfg.predictor.get("type", "ar")

    predictor_kwargs = {
        k: v for k, v in cfg.predictor.items() if k != "type"
    }
    # These predictors operate in raw normalized action coordinates.  AIG
    # deliberately does *not* appear here: it preserves the normal LeWM
    # action embedder and AdaLN conditioner exactly.
    use_raw_actions = predictor_type in ("aft", "cafe", "laf", "masked_transition", "mtm", "transition_jepa")

    # AFT's action-faithfulness identity (Q(z)^T[T(z,a)-T(z,0)] = a) only
    # holds for the predictor's own output. Any nonlinear pred_proj applied
    # afterwards could re-collapse the action-transport term it guarantees,
    # so AFT must predict directly into the target embed_dim space and skip
    # pred_proj entirely (unlike ARPredictor, which predicts into hidden_dim
    # and relies on pred_proj to project down to embed_dim).
    predictor_output_dim = embed_dim if use_raw_actions else hidden_dim

    inverse_type = cfg.inverse.get("type", "mlp")
    use_inverse_model = bool(cfg.loss.inverse.weight) and inverse_type != "iipdc"
    use_rsi_game = bool(cfg.loss.get("rsi", {}).get("weight", 0.0))
    use_aig = bool(cfg.loss.get("aig", {}).get("weight", 0.0))
    use_predictive_mask = bool(
        cfg.loss.get("predictive_mask", {}).get("enabled", False)
    )
    use_masked_transition = predictor_type in (
        "masked_transition", "mtm", "transition_jepa"
    )
    if use_rsi_game and (cfg.loss.inverse.weight or cfg.loss.aca.weight):
        raise ValueError(
            "Forward--Inverse Game owns its IDM and predictor updates; set "
            "loss.inverse.weight=loss.aca.weight=0 for this configuration"
        )
    if use_aig:
        if predictor_type != "ar" or not bool(cfg.predictor.get("aig_enabled", False)):
            raise ValueError(
                "loss.aig.weight > 0 requires the original predictor.type=ar "
                "with predictor.aig_enabled=true"
            )
        incompatible = {
            "inverse": cfg.loss.inverse.weight,
            "rsi": cfg.loss.get("rsi", {}).get("weight", 0.0),
            "aca": cfg.loss.aca.weight,
            "cai": cfg.loss.get("cai", {}).get("weight", 0.0),
            "pdc": cfg.loss.pdc.weight,
            "geo_pdc": cfg.loss.geo_pdc.weight,
            "as_pdc": cfg.loss.get("as_pdc", {}).get("weight", 0.0),
            "delta_lift": cfg.loss.get("delta_lift", {}).get("weight", 0.0),
            "si": cfg.loss.get("si", {}).get("weight", 0.0),
        }
        enabled = [name for name, weight in incompatible.items() if weight]
        if enabled:
            raise ValueError(
                "AIG uses only prediction (+ optional SIGReg); disable "
                + ", ".join(enabled)
            )
    if use_predictive_mask and (use_rsi_game or use_aig):
        raise ValueError(
            "predictive_mask has its own alternating optimizer; it cannot be "
            "combined with RSI or AIG"
        )
    if use_masked_transition:
        incompatible = {
            "inverse": cfg.loss.inverse.weight,
            "rsi": cfg.loss.get("rsi", {}).get("weight", 0.0),
            "aca": cfg.loss.aca.weight,
            "cai": cfg.loss.get("cai", {}).get("weight", 0.0),
        }
        enabled = [name for name, weight in incompatible.items() if weight]
        if enabled:
            raise ValueError(
                "masked_transition uses its intrinsic action mask; disable "
                + ", ".join(enabled)
            )
    if cfg.loss.get("si", {}).get("weight", 0.0) and cfg.loss.aca.weight:
        raise ValueError(
            "SI-WM and ACA are alternative action-energy objectives; "
            "set loss.aca.weight=0 when loss.si.weight > 0"
        )
    world_model = JEPA(
        encoder=encoder,
        predictor=build_predictor(
            predictor_type,
            num_frames=cfg.wm.history_size,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=predictor_output_dim,
            action_dim=effective_act_dim,
            **predictor_kwargs,
        ),
        action_encoder=None if use_raw_actions else Embedder(input_dim=effective_act_dim, emb_dim=embed_dim),
        projector=MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        ),
        pred_proj=None if use_raw_actions else MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        ),
        # II-PDC replaces the free-form InverseModel with BlindPredictor +
        # IIPDCHead (constructed below, attached to the Lightning Module
        # instead of JEPA -- same pattern as PDCHead/GeoPDCHead/SIGReg).
        inverse_model=InverseModel(
            embed_dim=embed_dim,
            action_dim=effective_act_dim,
            hidden_dim=cfg.inverse.get("hidden_dim", 256),
        ) if use_inverse_model else None,
        grounded_coordinate=bool(cfg.wm.get("grounded_coordinate", False)),
        grounded_coordinate_scale=float(cfg.wm.get("grounded_coordinate_scale", 1.0)),
    )

    model_opt = {
        # II-PDC's BlindPredictor/IIPDCHead and AS-PDC's ASPDCHead all
        # carry learnable parameters (blind MLP, learnable orthogonal
        # Q) attached as top-level `self.blind`/`self.iipdc`/
        # `self.as_pdc` attributes (same pattern as sigreg/pdc/geo_pdc
        # below), so they must be included here too or their
        # parameters are silently never optimized (regex match
        # determines which params get an optimizer at all -- no match
        # means no gradient step).
        "modules": "model|blind|iipdc|as_pdc|cai_blind|cai_transition",
        "optimizer": dict(cfg.optimizer),
        "interval": "epoch",
    }
    if bool(cfg.get("scheduler", {}).get("enabled", True)):
        model_opt["scheduler"] = {
            "type": "LinearWarmupCosineAnnealingLR",
            "warmup_steps": scheduler_warmup_steps,
            "max_steps": scheduler_max_steps,
            "warmup_start_lr": 0.0,
            "eta_min": 0.0,
        }
    else:
        # ``stable_pretraining.Module`` uses a cosine scheduler by default
        # for every entry in the multi-optimizer form when the key is absent.
        # Supply an identity ConstantLR explicitly so fixed-LR experiments
        # remain exactly at optimizer.lr instead of silently decaying to zero.
        model_opt["scheduler"] = {
            "type": "ConstantLR",
            "factor": 1.0,
            "total_iters": 1,
        }

    module_kwargs = {
        "model": world_model,
        "forward": partial(forward_step, cfg=cfg),
        "optim": {
            "model_opt": model_opt
        },
    }
    if use_predictive_mask:
        image_patches = (int(cfg.img_size) // int(cfg.patch_size)) ** 2
        if int(cfg.loss.predictive_mask.num_masked_patches) >= image_patches:
            raise ValueError(
                "loss.predictive_mask.num_masked_patches must be smaller than "
                f"the {image_patches} ViT image patches"
            )
        policy_lr = cfg.loss.predictive_mask.get("policy_lr", None)
        policy_optimizer = dict(cfg.optimizer)
        if policy_lr is not None:
            policy_optimizer["lr"] = float(policy_lr)
        module_kwargs["mask_policy"] = PredictiveMaskPolicy(
            input_dim=hidden_dim,
            num_patches=image_patches,
            hidden_dim=int(cfg.loss.predictive_mask.hidden_dim),
            alpha=float(cfg.loss.predictive_mask.alpha),
            temperature=float(cfg.loss.predictive_mask.temperature),
            base_seed=int(cfg.loss.predictive_mask.base_seed),
        )
        # Put the explicit policy group first: stable_pretraining assigns the
        # first matching module group, so it cannot leak into the broad world
        # optimizer. Both are stepped once per batch and share the matched
        # fixed-horizon cosine schedule.
        module_kwargs["optim"] = {
            "policy_opt": {
                "modules": "mask_policy",
                "optimizer": policy_optimizer,
                "scheduler": dict(model_opt["scheduler"]),
                "interval": "epoch",
            },
            "wm_opt": {
                "modules": "model|blind|iipdc|as_pdc|cai_blind|cai_transition",
                "optimizer": dict(cfg.optimizer),
                "scheduler": dict(model_opt["scheduler"]),
                "interval": "epoch",
            },
        }
    if use_rsi_game:
        # These are deliberately disjoint: the game has three opposed updates
        # rather than one shared loss backward pass.
        module_kwargs["optim"] = {
            "encoder_opt": {
                "modules": "model.encoder|model.projector",
                "optimizer": dict(cfg.optimizer),
                "scheduler": "LinearWarmupCosineAnnealingLR",
                "interval": "epoch",
            },
            "predictor_opt": {
                "modules": "model.predictor|model.action_encoder|model.pred_proj",
                "optimizer": dict(cfg.optimizer),
                "scheduler": "LinearWarmupCosineAnnealingLR",
                "interval": "epoch",
            },
            "idm_opt": {
                "modules": "rsi_idm",
                "optimizer": dict(cfg.optimizer),
                "scheduler": "LinearWarmupCosineAnnealingLR",
                "interval": "epoch",
            },
        }
    if use_aig:
        # The behavior head is listed first: the grouping utility assigns the
        # first explicit regex match, so this excludes it from the broad
        # ``model`` world-model optimizer below.
        module_kwargs["optim"] = {
            "behavior_opt": {
                "modules": "model.predictor.behavior_head",
                "optimizer": dict(cfg.optimizer),
                "scheduler": "LinearWarmupCosineAnnealingLR",
                "interval": "epoch",
            },
            "wm_opt": {
                "modules": "model",
                "optimizer": dict(cfg.optimizer),
                "scheduler": "LinearWarmupCosineAnnealingLR",
                "interval": "epoch",
            },
        }
    if cfg.loss.sigreg.weight:
        module_kwargs["sigreg"] = SIGReg(**cfg.loss.sigreg.kwargs)
    if cfg.loss.pdc.weight:
        module_kwargs["pdc"] = PDCHead(
            embed_dim=embed_dim,
            image_shape=(3, cfg.img_size, cfg.img_size),
            **cfg.loss.pdc.kwargs,
        )
    if cfg.loss.geo_pdc.weight:
        module_kwargs["geo_pdc"] = GeoPDCHead(
            embed_dim=embed_dim,
            image_shape=(3, cfg.img_size, cfg.img_size),
            **cfg.loss.geo_pdc.kwargs,
        )
    if cfg.loss.inverse.weight and inverse_type == "iipdc":
        module_kwargs["blind"] = BlindPredictor(
            embed_dim=embed_dim,
            hidden_dim=cfg.inverse.get("blind_hidden_dim", 256),
        )
        module_kwargs["iipdc"] = IIPDCHead(
            embed_dim=embed_dim,
            action_dim=effective_act_dim,
        )
    if cfg.loss.get("cai", {}).get("weight", 0.0):
        module_kwargs["cai_blind"] = CAIBlindPredictor(
            embed_dim=embed_dim, action_dim=effective_act_dim,
            hidden_dim=cfg.loss.cai.get("blind_hidden_dim", cfg.inverse.get("blind_hidden_dim", 256)),
        )
        module_kwargs["cai_transition"] = CAITransitionPredictor(
            embed_dim=embed_dim, action_dim=effective_act_dim,
            hidden_dim=cfg.loss.cai.get("transition_hidden_dim", cfg.inverse.get("hidden_dim", 256)),
        )
    if cfg.loss.get("as_pdc", {}).get("weight", 0.0):
        module_kwargs["as_pdc"] = ASPDCHead(
            embed_dim=embed_dim,
            action_dim=effective_act_dim,
        )
    if cfg.loss.get("delta_lift", {}).get("weight", 0.0):
        module_kwargs["delta_lift"] = DeltaLiftHead(
            embed_dim=embed_dim,
            image_shape=(3, cfg.img_size, cfg.img_size),
            **cfg.loss.delta_lift.kwargs,
        )
    if use_rsi_game:
        module_kwargs["rsi_idm"] = ActionConsistencyIDM(
            embed_dim=embed_dim,
            action_dim=effective_act_dim,
            hidden_dim=cfg.inverse.get("hidden_dim", 256),
        )

    module = spt.Module(**module_kwargs)
    if cfg.get("init_from_checkpoint"):
        initialize_model_from_checkpoint(module, cfg.init_from_checkpoint)
    if online_aca or epoch_active_aca:
        action_data = torch.from_numpy(np.array(base_train_set.get_col_data("action")))
        action_valid = action_data[~torch.isnan(action_data).any(dim=1)]
        action_mean = action_valid.mean(0, keepdim=True)
        action_std = action_valid.std(0, keepdim=True).clamp_min(1e-6)
        normalized_low = ((action_valid.amin(0, keepdim=True) - action_mean) / action_std).repeat(
            1, int(cfg.data.dataset.frameskip)
        ).view(1, 1, -1)
        normalized_high = ((action_valid.amax(0, keepdim=True) - action_mean) / action_std).repeat(
            1, int(cfg.data.dataset.frameskip)
        ).view(1, 1, -1)
    if online_aca:
        module.online_aca_replay = OnlineACAReplay(
            cfg, action_mean, action_std, cfg.img_size, cfg.seed
        )
        module.online_aca_action_low = normalized_low
        module.online_aca_action_high = normalized_high
    if epoch_active_aca:
        module.epoch_active_aca = EpochActiveACACollector(
            cfg, epoch_active_dataset, action_mean, action_std,
            normalized_low, normalized_high, cfg.img_size, cfg.seed,
        )
    if use_rsi_game:
        # stable_pretraining.Module's default manual loop assumes one joint
        # loss.  Replace only the training hook; validation still calls the
        # ordinary forward prediction path for baseline-compatible monitoring.
        module.training_step = lambda batch, batch_idx: forward_inverse_game_training_step(
            module, batch, batch_idx, cfg
        )
    if use_aig:
        # AIG has a separate nuisance-mean update; validation/planning keep
        # the ordinary numerical LeWM forward path.
        module.training_step = lambda batch, batch_idx: action_innovation_gradient_training_step(
            module, batch, batch_idx, cfg
        )
    if use_predictive_mask:
        module.training_step = lambda batch, batch_idx: predictive_mask_training_step(
            module, batch, batch_idx, cfg
        )
    module._log_hyperparams = cfg.get("artifacts", {}).get(
        "log_lightning_hparams",
        False,
    )

    run_id = cfg.get("subdir") or ""
    run_root = get_runs_root()
    run_dir = run_root / run_id if run_id else run_root
    run_dir.mkdir(parents=True, exist_ok=True)

    # Do this before creating Manager. Manager resumes exclusively from the
    # current run's checkpoints/last.ckpt, while the copy preserves source
    # weights/optimizer/LR state as well as protecting the source run.
    if cfg.get("resume_from_checkpoint"):
        retarget_scheduler = bool(
            cfg.get("resume", {}).get("retarget_scheduler", False)
        )
        prepare_full_resume_checkpoint(
            run_dir,
            cfg.resume_from_checkpoint,
            scheduler_max_steps=scheduler_max_steps if retarget_scheduler else None,
            scheduler_warmup_steps=scheduler_warmup_steps if retarget_scheduler else None,
        )

    if cfg.get("artifacts", {}).get("save_resolved_config", True):
        with open(run_dir / "config.yaml", "w") as f:
            OmegaConf.save(cfg, f)

    configure_external_callbacks(cfg.get("artifacts", {}).get("use_external_callbacks", False))

    lightning_dir = run_dir / "lightning" / "local"
    if lightning_dir.exists():
        shutil.rmtree(lightning_dir)

    callbacks = []
    if epoch_active_aca:
        callbacks.append(EpochActiveACACallback())
    if cfg.get("artifacts", {}).get("save_model_object", False):
        callbacks.append(
            ModelObjectCallBack(
                dirpath=run_dir,
                filename=cfg.output_model_name,
                epoch_interval=1,
            )
        )
    if cfg.get("artifacts", {}).get("save_epoch_checkpoints", False):
        callbacks.append(
            ModelCheckpoint(
                dirpath=run_dir / "checkpoints",
                filename="epoch={epoch:02d}",
                every_n_epochs=1,
                save_on_train_epoch_end=True,
                save_top_k=-1,
                save_last=False,
            )
        )
    if cfg.get("validation_monitoring", {}).get("enabled", False):
        callbacks.append(
            ValidationEmbeddingStatsCallback(
                alpha=float(
                    cfg.validation_monitoring.get("probe_ridge_alpha", 1.0)
                )
            )
        )

    csv_logger = CSVLogger(save_dir=str(run_dir), name="lightning", version="local")
    logger = csv_logger
    if cfg.wandb.enabled:
        wandb_logger = WandbLogger(save_dir=str(run_dir), **cfg.wandb.config)
        if cfg.wandb.get("log_config", False):
            wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        logger = [csv_logger, wandb_logger]

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        default_root_dir=str(run_dir),
        logger=logger,
        enable_checkpointing=bool(
            cfg.get("artifacts", {}).get("save_epoch_checkpoints", False)
        ),
    )

    ckpt_path = run_dir / "checkpoints" / "last.ckpt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
        seed=cfg.seed,
        ckpt_path=ckpt_path,
        compile=bool(cfg.get("compile", {}).get("enabled", False)),
    )
    manager()

    barrier = getattr(manager._trainer.strategy, "barrier", None)
    if barrier is not None:
        barrier("save_embeddings_start")
    if manager._trainer.is_global_zero:
        embedding_subset_size = cfg.get("artifacts", {}).get("embedding_subset_size")
        if embedding_subset_size != 0:
            save_embeddings(
                manager.instantiated_module.model,
                val_set,
                run_dir / "final_embeddings.pt",
                batch_size=int(cfg.loader.batch_size),
                max_items=embedding_subset_size,
            )
        if online_aca and cfg.loss.aca.online.get("save_replay", True):
            module.online_aca_replay.export(run_dir / "online_aca_replay.h5")
    if online_aca:
        module.online_aca_replay.close()
    if epoch_active_aca:
        module.epoch_active_aca.close()
    if barrier is not None:
        barrier("save_embeddings_end")


if __name__ == "__main__":
    run()
