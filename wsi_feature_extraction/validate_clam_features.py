#!/usr/bin/env python3
"""Fast sanity checks for CLAM feature outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_dir", required=True, type=Path)
    parser.add_argument("--max_files", type=int, default=3)
    args = parser.parse_args()

    h5_dir = args.feat_dir / "h5_files"
    pt_dir = args.feat_dir / "pt_files"
    h5_files = sorted(h5_dir.glob("*.h5"))[: args.max_files]
    pt_files = sorted(pt_dir.glob("*.pt"))

    print(f"feat_dir: {args.feat_dir}")
    print(f"h5 files: {len(list(h5_dir.glob('*.h5')))}")
    print(f"pt files: {len(pt_files)}")

    failures = 0
    for h5_file in h5_files:
        pt_file = pt_dir / f"{h5_file.stem}.pt"
        with h5py.File(h5_file, "r") as f:
            features = f["features"][:]
            coords = f["coords"][:]

        if not pt_file.exists():
            print(f"[BAD] missing pt file: {pt_file.name}")
            failures += 1
            continue

        pt = torch.load(pt_file, map_location="cpu")
        bad = (
            features.ndim != 2
            or coords.ndim != 2
            or features.shape[0] != coords.shape[0]
            or tuple(pt.shape) != tuple(features.shape)
            or not np.isfinite(features).all()
            or float(features.std()) <= 1e-6
        )
        status = "BAD" if bad else "OK"
        if bad:
            failures += 1
        print(
            f"[{status}] {h5_file.stem}: features={features.shape}, "
            f"coords={coords.shape}, mean={features.mean():.4f}, std={features.std():.4f}"
        )

    if failures:
        raise SystemExit(f"{failures} validation checks failed")
    print("validation passed")


if __name__ == "__main__":
    main()
