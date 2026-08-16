#!/usr/bin/env python3
"""Extract WSI patch features using CLAM modules with GPU-first inference.

This script does not perform extra foreground filtering. Background rejection is
handled by CLAM during create_patches_fp.py via tissue segmentation, contour
checking, and the generated patch coordinate h5 files.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path


def sanitize_thread_env(default: str = "1") -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        value = os.environ.get(name, "").strip()
        if not value or not value.isdigit() or int(value) < 1:
            os.environ[name] = default


sanitize_thread_env()

import h5py
import numpy as np
import openslide
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def add_clam_to_path(clam_dir: Path) -> None:
    sys.path.insert(0, str(clam_dir.resolve()))


def choose_device(device_arg: str, allow_cpu: bool) -> torch.device:
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if device.type != "cuda" and not allow_cpu:
        raise RuntimeError("GPU is required by default. Use --allow_cpu only for debugging.")
    return device




def get_encoder_for_model(model_name, target_img_size, clam_get_encoder):
    """Use local checkpoints when configured; otherwise fall back to CLAM."""
    import os
    import timm
    from torchvision import transforms

    if model_name == "resnet50_trunc":
        ckpt = os.environ.get("RESNET50_CKPT_PATH", "")
        if ckpt:
            class PooledTimmEncoder(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.model = timm.create_model(
                        "resnet50.tv_in1k",
                        features_only=True,
                        out_indices=(3,),
                        pretrained=False,
                        num_classes=0,
                    )
                    self.pool = torch.nn.AdaptiveAvgPool2d(1)

                def forward(self, x):
                    out = self.model(x)
                    if isinstance(out, list):
                        out = out[0]
                    return self.pool(out).squeeze(-1).squeeze(-1)

            model = PooledTimmEncoder()
            state = torch.load(ckpt, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            missing, unexpected = model.model.load_state_dict(state, strict=False)
            print(f"loaded local ResNet50 checkpoint: {ckpt}")
            print(f"resnet50 missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
            img_transforms = transforms.Compose(
                [
                    transforms.Resize(target_img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
            return model, img_transforms
        return clam_get_encoder(model_name, target_img_size=target_img_size)

    if model_name == "conch_v1_5":
        from transformers import AutoModel

        titan_dir = os.environ.get("TITAN_CKPT_DIR", "")
        if not titan_dir:
            raise RuntimeError("TITAN_CKPT_DIR is required for local conch_v1_5 loading")
        titan_path = Path(titan_dir)
        required = ["config.json", "modeling_titan.py", "configuration_titan.py"]
        missing = [x for x in required if not (titan_path / x).exists()]
        if missing:
            raise RuntimeError(f"TITAN local directory is missing required files: {missing}")
        print(f"loading local TITAN/CONCH1.5 directory: {titan_path}")

        # transformers copies trust_remote_code files into a dynamic-module cache.
        # Some TITAN releases import helper files such as conch_tokenizer.py that
        # are not always auto-copied, so mirror all local .py files there first.
        import shutil
        cache_root = Path(os.environ.get("HF_MODULES_CACHE", Path.home() / ".cache" / "huggingface" / "modules"))
        titan_cache = cache_root / "transformers_modules" / titan_path.name
        titan_cache.mkdir(parents=True, exist_ok=True)
        (titan_cache / "__init__.py").touch()
        for py_file in titan_path.glob("*.py"):
            cache_file = titan_cache / py_file.name
            shutil.copy2(py_file, cache_file)
            if py_file.name == "conch_tokenizer.py":
                text = cache_file.read_text(encoding="utf-8")
                text = text.replace(
                    'PreTrainedTokenizerFast.from_pretrained("MahmoodLab/TITAN")',
                    f'PreTrainedTokenizerFast.from_pretrained(r"{titan_path}", local_files_only=True)',
                )
                cache_file.write_text(text, encoding="utf-8")
            if py_file.name == "conch_v1_5.py":
                conch_weight = titan_path / "conch_v1_5_pytorch_model.bin"
                if not conch_weight.exists():
                    raise RuntimeError(f"Missing local CONCH1.5 weight file: {conch_weight}")
                text = cache_file.read_text(encoding="utf-8")
                text = text.replace(
                    '    from huggingface_hub import hf_hub_download\n    checkpoint_path = hf_hub_download(\n        "MahmoodLab/TITAN", \n        filename="conch_v1_5_pytorch_model.bin",\n    )',
                    f'    checkpoint_path = r"{conch_weight}"',
                )
                cache_file.write_text(text, encoding="utf-8")

        titan = AutoModel.from_pretrained(str(titan_path), trust_remote_code=True, local_files_only=True)

        conch_weight = titan_path / "conch_v1_5_pytorch_model.bin"
        if not conch_weight.exists():
            raise RuntimeError(f"Missing local CONCH1.5 weight file: {conch_weight}")

        import huggingface_hub
        original_hf_hub_download = huggingface_hub.hf_hub_download

        def local_hf_hub_download(repo_id, filename=None, *args, **kwargs):
            if repo_id == "MahmoodLab/TITAN" and filename == "conch_v1_5_pytorch_model.bin":
                print(f"using local CONCH1.5 weights: {conch_weight}")
                return str(conch_weight)
            return original_hf_hub_download(repo_id, filename=filename, *args, **kwargs)

        huggingface_hub.hf_hub_download = local_hf_hub_download
        try:
            model, _ = titan.return_conch()
        finally:
            huggingface_hub.hf_hub_download = original_hf_hub_download
        if target_img_size != 448:
            raise RuntimeError("CONCH1.5/TITAN should use target_patch_size=448")
        img_transforms = transforms.Compose(
            [
                transforms.Resize(target_img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        return model, img_transforms

    if model_name != "uni_v2":
        return clam_get_encoder(model_name, target_img_size=target_img_size)

    if "UNI_CKPT_PATH" not in os.environ:
        raise RuntimeError("UNI_CKPT_PATH is required for uni_v2")

    timm_kwargs = {
        "model_name": "vit_giant_patch14_224",
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    model = timm.create_model(pretrained=False, **timm_kwargs)
    state = torch.load(os.environ["UNI_CKPT_PATH"], map_location="cpu")
    model.load_state_dict(state, strict=True)
    img_transforms = transforms.Compose(
        [
            transforms.Resize(target_img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return model, img_transforms


def coord_count_from_h5(coord_h5_path: Path) -> int:
    with h5py.File(coord_h5_path, "r") as f:
        return int(f["coords"].shape[0])


def inspect_existing_feature(pt_path: Path, out_h5_path: Path, coord_h5_path: Path) -> tuple[str, str]:
    """Return (status, reason) for an existing feature pair."""
    if not out_h5_path.exists():
        return "missing", "missing feature h5"
    try:
        expected_rows = coord_count_from_h5(coord_h5_path)
        with h5py.File(out_h5_path, "r") as f:
            if "features" not in f or "coords" not in f:
                return "invalid", "feature h5 missing features or coords dataset"
            features_shape = tuple(f["features"].shape)
            coords_shape = tuple(f["coords"].shape)
        if len(features_shape) != 2:
            return "invalid", f"features should be 2D, got {features_shape}"
        if len(coords_shape) != 2 or coords_shape[1] != 2:
            return "invalid", f"coords should be Nx2, got {coords_shape}"
        if features_shape[0] != expected_rows or coords_shape[0] != expected_rows:
            return "invalid", f"row mismatch feature={features_shape} coords={coords_shape} expected={expected_rows}"
        if not pt_path.exists():
            return "h5_only", "feature h5 complete, pt missing"
        pt = torch.load(pt_path, map_location="cpu")
        pt_shape = tuple(pt.shape if hasattr(pt, "shape") else np.asarray(pt).shape)
        if pt_shape != features_shape:
            return "invalid", f"pt shape {pt_shape} != h5 features {features_shape}"
        return "complete", f"features={features_shape}"
    except Exception as exc:
        return "invalid", f"{exc.__class__.__name__}: {exc}"


def write_pt_from_h5(out_h5_path: Path, pt_path: Path) -> tuple[int, int]:
    with h5py.File(out_h5_path, "r") as f:
        features = f["features"][:]
    torch.save(torch.from_numpy(features), pt_path)
    return int(features.shape[0]), int(features.shape[1])

def compute_features(output_path, loader, model, device, save_hdf5, use_amp: bool):
    mode = "w"
    total = 0
    for data in tqdm(loader, desc=Path(output_path).stem):
        with torch.inference_mode():
            batch = data["img"].to(device, non_blocking=True)
            coords = data["coord"].numpy().astype(np.int32)
            if use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = model(batch)
            else:
                features = model(batch)
            features = features.float().cpu().numpy().astype(np.float32)
            save_hdf5(output_path, {"features": features, "coords": coords}, mode=mode)
            mode = "a"
            total += len(coords)
    return output_path, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clam_dir", required=True, type=Path)
    parser.add_argument("--patch_h5_dir", required=True, type=Path)
    parser.add_argument("--slide_dir", required=True, type=Path)
    parser.add_argument("--feat_dir", required=True, type=Path)
    parser.add_argument("--model_name", default="resnet50_trunc")
    parser.add_argument("--slide_ext", default=".svs")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--target_patch_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use CUDA fp16 autocast for faster inference.")
    parser.add_argument(
        "--persistent_workers",
        action="store_true",
        help="Keep DataLoader workers alive. Disabled by default because one DataLoader is created per slide.",
    )
    parser.add_argument("--no_auto_skip", action="store_true")
    args = parser.parse_args()

    add_clam_to_path(args.clam_dir)
    from dataset_modules.dataset_h5 import Whole_Slide_Bag_FP
    from models import get_encoder as clam_get_encoder
    from utils.file_utils import save_hdf5

    device = choose_device(args.device, args.allow_cpu)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    print(f"device: {device}")
    print(f"amp: {args.amp and device.type == 'cuda'}")

    args.feat_dir.mkdir(parents=True, exist_ok=True)
    h5_out_dir = args.feat_dir / "h5_files"
    pt_out_dir = args.feat_dir / "pt_files"
    h5_out_dir.mkdir(exist_ok=True)
    pt_out_dir.mkdir(exist_ok=True)

    model, img_transforms = get_encoder_for_model(args.model_name, args.target_patch_size, clam_get_encoder)
    model.eval()
    model = model.to(device)

    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs.update({
            "num_workers": args.num_workers,
            "pin_memory": device.type == "cuda",
            "persistent_workers": args.persistent_workers,
            "prefetch_factor": 2,
        })

    h5_files = sorted(args.patch_h5_dir.glob("*.h5"))
    print(f"model_name: {args.model_name}")
    print(f"batch_size: {args.batch_size}")
    print(f"num_workers: {args.num_workers}")
    print(f"patch h5 files: {len(h5_files)}")

    stats_rows = []
    for h5_file in h5_files:
        slide_id = h5_file.stem
        pt_path = pt_out_dir / f"{slide_id}.pt"
        out_h5_path = h5_out_dir / h5_file.name
        slide_path = args.slide_dir / f"{slide_id}{args.slide_ext}"

        if not args.no_auto_skip:
            existing_status, existing_reason = inspect_existing_feature(pt_path, out_h5_path, h5_file)
            if existing_status == "complete":
                print(f"[skip complete] {slide_id}: {existing_reason}")
                stats_rows.append((slide_id, "skipped_complete", "", 0.0, existing_reason))
                continue
            if existing_status == "h5_only":
                try:
                    rows, dim = write_pt_from_h5(out_h5_path, pt_path)
                    reason = f"repaired missing pt from h5 features=({rows}, {dim})"
                    print(f"[repair] {slide_id}: {reason}")
                    stats_rows.append((slide_id, "repaired_pt", rows, 0.0, reason))
                    continue
                except Exception as exc:
                    existing_reason = f"could not repair pt: {exc.__class__.__name__}: {exc}"
            if existing_status != "missing":
                print(f"[recompute] {slide_id}: existing output {existing_status}: {existing_reason}")

        if not slide_path.exists():
            reason = f"missing slide: {slide_path}"
            print(f"[skip missing] {slide_id}: {reason}")
            stats_rows.append((slide_id, "missing_slide", "", 0.0, reason))
            continue

        start = time.time()
        wsi = None
        try:
            wsi = openslide.open_slide(str(slide_path))
            dataset = Whole_Slide_Bag_FP(file_path=str(h5_file), wsi=wsi, img_transforms=img_transforms)
            loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
            _, total = compute_features(str(out_h5_path), loader, model, device, save_hdf5, args.amp)

            with h5py.File(out_h5_path, "r") as f:
                features = f["features"][:]
                coords_shape = f["coords"].shape
            torch.save(torch.from_numpy(features), pt_path)
            elapsed = time.time() - start
            stats_rows.append((slide_id, "done", total, elapsed, f"features={features.shape}; coords={coords_shape}"))
            print(f"[done] {slide_id}: features={features.shape}, coords={coords_shape}, time={elapsed:.1f}s")
        except Exception as exc:
            elapsed = time.time() - start
            reason = f"{exc.__class__.__name__}: {exc}"
            stats_rows.append((slide_id, "failed", "", elapsed, reason))
            print(f"[failed] {slide_id}: {reason}")
        finally:
            if wsi is not None:
                wsi.close()
            try:
                del loader
            except UnboundLocalError:
                pass
            try:
                del dataset
            except UnboundLocalError:
                pass
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    stats_path = args.feat_dir / "feature_extraction_stats.csv"
    with stats_path.open("w", encoding="utf-8") as f:
        f.write("slide_id,status,patches,seconds,message\n")
        for slide_id, status, patches, seconds, message in stats_rows:
            safe_message = str(message).replace('"', '""')
            f.write(f'{slide_id},{status},{patches},{seconds:.3f},"{safe_message}"\n')
    print(f"stats: {stats_path}")

    status_counts = {}
    for _, status, _, _, _ in stats_rows:
        status_counts[status] = status_counts.get(status, 0) + 1
    print("\nFEATURE EXTRACTION SUMMARY")
    for status in sorted(status_counts):
        print(f"{status}: {status_counts[status]}")

    bad_statuses = {"failed", "missing_slide"}
    bad_count = sum(count for status, count in status_counts.items() if status in bad_statuses)
    if bad_count:
        raise SystemExit(
            f"Feature extraction incomplete: {bad_count} slides have failed/missing status. "
            f"See {stats_path}"
        )


if __name__ == "__main__":
    main()








