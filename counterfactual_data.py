"""Datasets written by ``collect_reacher_aca_counterfactuals.py``.

The normal Reacher HDF5 stores one atomic 2-D control per row and lets
``HDF5Dataset`` concatenate five controls.  A collected transition is already
one five-control macro action, so storing it as a two-frame record avoids
duplicating ten images just to satisfy that storage convention.
"""

from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import torch
from torch.utils.data import Dataset


class CounterfactualTransitionDataset(Dataset):
    """Read two-frame, pre-concatenated action counterfactual transitions.

    Required datasets are ``pixels`` with shape ``(N, 2, H, W, C)`` and
    ``action`` with shape ``(N, A)``.  It is returned as ``(2, A)`` to match
    normal Reacher ``HDF5Dataset(frameskip=5, num_steps=2)`` output. Only row
    zero is supervised by the one-step objective; row one is duplicated since
    the collected file contains one macro transition per item.
    """

    def __init__(self, path, transform=None, include_proprio: bool = False):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"counterfactual dataset not found: {self.path}")
        self.transform = transform
        self.include_proprio = include_proprio
        self.h5_file = None
        with h5py.File(self.path, "r") as f:
            self.index_mode = str(f.attrs.get("format", "")) == "reacher_factual_index_v1"
            self.source_path = Path(str(f.attrs.get("source", ""))).expanduser().resolve() if self.index_mode else None
            if self.index_mode:
                self.source_indices = np.asarray(f["source_index"][:], dtype=np.int64)
                self.source_h5 = None
                self.length = len(self.source_indices)
                shape = tuple(int(x) for x in f.attrs.get("pixel_shape", (224, 224, 3)))
                self.pixel_shape = (2, *shape)
                self.atomic_action_dim = int(f.attrs.get("atomic_action_dim", 2))
                self.frameskip = int(f.attrs.get("frameskip", 5))
                self.action_dim = self.frameskip * self.atomic_action_dim
                if self.source_path is None or not self.source_path.is_file():
                    raise FileNotFoundError(f"source HDF5 missing for index dataset: {self.source_path}")
                self.lengths = np.ones(self.length, dtype=np.int64)
                self.offsets = np.arange(self.length, dtype=np.int64)
                return
            required = {"pixels", "action", "observation"}
            missing = required.difference(f.keys())
            if missing:
                raise KeyError(f"{self.path} is missing columns: {sorted(missing)}")
            self.length = int(f["action"].shape[0])
            if f["pixels"].shape[0] != self.length or f["pixels"].shape[1] != 2:
                raise ValueError("counterfactual pixels must have shape (N, 2, H, W, C)")
            self.pixel_shape = tuple(f["pixels"].shape[2:])
            self.action_dim = int(f["action"].shape[1])
            self.atomic_action_dim = int(f.attrs.get("atomic_action_dim", 0))
        if self.atomic_action_dim <= 0 or self.action_dim % self.atomic_action_dim:
            raise ValueError("invalid atomic_action_dim metadata in counterfactual dataset")


        self.lengths = np.ones(self.length, dtype=np.int64)
        self.offsets = np.arange(self.length, dtype=np.int64)

    @property
    def column_names(self):
        return ["pixels", "action", "observation"]

    def _open(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.path, "r", swmr=True)
        if getattr(self, "index_mode", False) and getattr(self, "source_h5", None) is None:
            self.source_h5 = h5py.File(self.source_path, "r", swmr=True)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["h5_file"] = None
        state["source_h5"] = None
        return state

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self._open()
        if getattr(self, "index_mode", False):
            row = int(self.source_indices[index]); fs = self.frameskip
            src = self.source_h5
            pixels = np.stack([src["pixels"][row], src["pixels"][row+fs]])
            action = np.asarray(src["action"][row:row+fs], dtype=np.float32).reshape(-1)
            obs = np.asarray([src["observation"][row], src["observation"][row+fs]], dtype=np.float32)
            sample = {"pixels": torch.from_numpy(pixels).permute(0,3,1,2), "action": torch.from_numpy(action).repeat(2,1), "observation": torch.from_numpy(obs)}
            return self.transform(sample) if self.transform else sample
        pixels = self.h5_file["pixels"][index]

        sample = {
            "pixels": torch.from_numpy(pixels).permute(0, 3, 1, 2),
            "action": torch.from_numpy(self.h5_file["action"][index]).repeat(2, 1),
            "observation": torch.from_numpy(self.h5_file["observation"][index]),
        }













        if self.include_proprio:
            sample["proprio"] = torch.zeros(
                sample["observation"].shape[0], 19,
                dtype=sample["observation"].dtype,
            )
        return self.transform(sample) if self.transform else sample

    def get_col_data(self, column):
        if getattr(self, "index_mode", False):



            self._open()
            src = self.source_h5
            if column == "action":
                return np.asarray([src["action"][r:r+self.frameskip].reshape(-1) for r in self.source_indices])
            if column == "pixels":
                return np.asarray([np.stack([src["pixels"][r], src["pixels"][r+self.frameskip]]) for r in self.source_indices])
            if column == "observation":
                return np.asarray([np.stack([src["observation"][r], src["observation"][r+self.frameskip]]) for r in self.source_indices])
            raise KeyError(column)
        self._open()
        return self.h5_file[column][:]

    def get_dim(self, column):
        if column == "action":
            return self.action_dim
        if getattr(self, "index_mode", False):
            self._open()
            if column == "pixels":
                return int(np.prod(self.pixel_shape[1:]))
            if column == "observation":
                return int(self.source_h5["observation"].shape[1])
            raise KeyError(column)
        self._open()
        value = self.h5_file[column]
        return int(np.prod(value.shape[2:] if column in ("pixels", "observation") else value.shape[1:]))
