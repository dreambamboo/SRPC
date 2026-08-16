#!/usr/bin/env python3
"""Validate that CLAM feature outputs are complete for every patch h5 file."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def sanitize_thread_env(default: str = "1") -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        value = os.environ.get(name, "").strip()
        if not value or not value.isdigit() or int(value) < 1:
            os.environ[name] = default


sanitize_thread_env()

import h5py
import numpy as np
import torch


def h5_shape(path: Path, key: str) -> tuple[int, ...]:
    with h5py.File(path, "r") as handle:
        if key not in handle:
            raise KeyError(f"missing dataset: {key}")
        return tuple(handle[key].shape)


def validate_one(coord_h5: Path, feat_h5: Path, feat_pt: Path, check_finite: bool) -> dict[str, object]:
    row: dict[str, object] = {
        "slide_id": coord_h5.stem,
        "status": "ok",
        "coord_h5": str(coord_h5),
        "feat_h5": str(feat_h5),
        "feat_pt": str(feat_pt),
        "coord_rows": "",
        "feature_shape": "",
        "pt_shape": "",
        "message": "",
    }
    try:
        coord_shape = h5_shape(coord_h5, "coords")
        row["coord_rows"] = coord_shape[0]
        if len(coord_shape) != 2 or coord_shape[1] != 2:
            raise RuntimeError(f"coord shape should be Nx2, got {coord_shape}")
        if not feat_h5.exists():
            raise FileNotFoundError(f"missing feature h5: {feat_h5}")
        if not feat_pt.exists():
            raise FileNotFoundError(f"missing feature pt: {feat_pt}")

        with h5py.File(feat_h5, "r") as handle:
            if "features" not in handle or "coords" not in handle:
                raise RuntimeError("feature h5 missing features or coords")
            feature_shape = tuple(handle["features"].shape)
            feat_coord_shape = tuple(handle["coords"].shape)
            if check_finite:
                features = handle["features"][:]
                if not np.isfinite(features).all():
                    raise RuntimeError("feature h5 contains NaN or Inf")
                if float(features.std()) <= 1e-6:
                    raise RuntimeError("feature h5 has near-zero std")

        row["feature_shape"] = "x".join(str(x) for x in feature_shape)
        if len(feature_shape) != 2:
            raise RuntimeError(f"features should be 2D, got {feature_shape}")
        if feat_coord_shape != coord_shape:
            raise RuntimeError(f"feature coords {feat_coord_shape} != patch coords {coord_shape}")
        if feature_shape[0] != coord_shape[0]:
            raise RuntimeError(f"feature rows {feature_shape[0]} != patch rows {coord_shape[0]}")

        pt = torch.load(feat_pt, map_location="cpu")
        pt_shape = tuple(pt.shape if hasattr(pt, "shape") else np.asarray(pt).shape)
        row["pt_shape"] = "x".join(str(x) for x in pt_shape)
        if pt_shape != feature_shape:
            raise RuntimeError(f"pt shape {pt_shape} != h5 feature shape {feature_shape}")
    except Exception as exc:
        row["status"] = "bad"
        row["message"] = f"{exc.__class__.__name__}: {exc}"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_h5_dir", required=True, type=Path)
    parser.add_argument("--feat_dir", required=True, type=Path)
    parser.add_argument("--out_csv", type=Path, default=None)
    parser.add_argument("--check_finite", action="store_true")
    args = parser.parse_args()

    coord_files = sorted(args.patch_h5_dir.glob("*.h5"))
    h5_dir = args.feat_dir / "h5_files"
    pt_dir = args.feat_dir / "pt_files"
    out_csv = args.out_csv or (args.feat_dir / "feature_completion_report.csv")

    rows = []
    for coord_h5 in coord_files:
        rows.append(
            validate_one(
                coord_h5=coord_h5,
                feat_h5=h5_dir / coord_h5.name,
                feat_pt=pt_dir / f"{coord_h5.stem}.pt",
                check_finite=args.check_finite,
            )
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "slide_id",
        "status",
        "coord_rows",
        "feature_shape",
        "pt_shape",
        "message",
        "coord_h5",
        "feat_h5",
        "feat_pt",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_count = sum(1 for row in rows if row["status"] == "ok")
    bad_count = len(rows) - ok_count
    print("\nFEATURE COMPLETION VALIDATION")
    print(f"feat_dir: {args.feat_dir}")
    print(f"expected from patch h5: {len(rows)}")
    print(f"complete: {ok_count}")
    print(f"bad_or_missing: {bad_count}")
    print(f"report: {out_csv}")

    if bad_count:
        bad_examples = [row for row in rows if row["status"] != "ok"][:10]
        print("\nFirst bad examples:")
        for row in bad_examples:
            print(f"- {row['slide_id']}: {row['message']}")
        raise SystemExit(f"Feature validation failed: {bad_count}/{len(rows)} incomplete outputs")


if __name__ == "__main__":
    main()

