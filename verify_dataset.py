"""Verify the exact supplied FF++ dataset pipeline against train/val/test pickles."""

import argparse
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from dataset import FaceForensics


def load_cfg(config_path, root, split, compression, method):
    with open(config_path, "r", encoding="utf-8") as f:
        all_cfg = yaml.safe_load(f)
    cfg = deepcopy(all_cfg["train_cfg" if split == "train" else "test_cfg"])
    cfg["root"] = str(Path(root).expanduser().resolve())
    cfg["split"] = split
    cfg["compression"] = compression
    cfg["method"] = method
    return cfg


def label_counts(ds):
    real = sum(int(t == 0) for t in ds.targets)
    fake = sum(int(t == 1) for t in ds.targets)
    return real, fake


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--compression", default="c23", choices=["c0", "c23", "c40"])
    p.add_argument("--method", default="all", choices=["all", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"])
    p.add_argument("--config", default="config/dataset/faceforensics.yml")
    args = p.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    print("dataset root:", root)
    required = [
        root / "original_sequences",
        root / "manipulated_sequences",
        root / f"train_{args.compression}.pickle",
        root / f"val_{args.compression}.pickle",
        root / f"test_{args.compression}.pickle",
    ]
    for pth in required:
        print(f"  {'OK' if pth.exists() else 'MISSING'}  {pth}")
    if not all(pth.exists() for pth in required):
        raise SystemExit("Dataset structure check failed.")

    for split in ("train", "val", "test"):
        cfg = load_cfg(args.config, root, split, args.compression, args.method)
        ds = FaceForensics(cfg, seed=2022)
        real, fake = label_counts(ds)
        paths, targets = [], []
        # __getitem__ is intentionally path/target based in the supplied loader.
        for i in range(min(2, len(ds))):
            path, target = ds[i]
            paths.append(path)
            targets.append(target)
        images = ds.load_item(paths) if paths else torch.empty(0)

        print(f"\n{split}: {len(ds):,} samples")
        print(f"  real: {real:,}")
        print(f"  fake: {fake:,}")
        print(f"  loaded batch shape: {tuple(images.shape)}")
        print(f"  labels: {targets}")
        if paths:
            print(f"  first path: {paths[0]}")

    print("\nDataset verification passed.")


if __name__ == "__main__":
    main()
