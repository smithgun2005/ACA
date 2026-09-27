#!/usr/bin/env python3
"""Concatenate ACA counterfactual HDF5 transition files."""
import argparse
from pathlib import Path
import h5py
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("inputs", nargs="+", type=Path)
    a = p.parse_args()
    if a.output in a.inputs:
        raise ValueError("output must not also be an input")
    for f in a.inputs:
        if not f.is_file():
            raise FileNotFoundError(f)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.inputs[0], "r") as first:
        keys = [k for k, v in first.items() if isinstance(v, h5py.Dataset) and v.shape and v.shape[0] >= 0]
        attrs = dict(first.attrs)
        shapes = {k: first[k].shape[1:] for k in keys}
        dtypes = {k: first[k].dtype for k in keys}
    counts = []
    with h5py.File(a.output, "w") as out:
        for k, v in attrs.items():
            out.attrs[k] = v
        out.attrs["merged_inputs"] = ",".join(str(x) for x in a.inputs)
        total = 0
        for src_path in a.inputs:
            with h5py.File(src_path, "r") as src:
                if set(keys) - set(src.keys()):
                    raise ValueError(f"{src_path} is missing datasets")
                n = int(src[keys[0]].shape[0])
                if any(src[k].shape[1:] != shapes[k] for k in keys):
                    raise ValueError(f"dataset shapes do not match: {src_path}")
                counts.append(n)
                total += n
        for k in keys:
            out.create_dataset(k, shape=(total, *shapes[k]), dtype=dtypes[k])
        offset = 0
        for src_path, n in zip(a.inputs, counts):
            with h5py.File(src_path, "r") as src:
                for k in keys:
                    out[k][offset:offset+n] = src[k][:]
            offset += n
    print(f"merged {sum(counts):,} transitions from {len(a.inputs)} files -> {a.output}")

if __name__ == "__main__":
    main()
