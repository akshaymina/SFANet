"""
Preprocess WildDeepfake archives into the layout expected by dataset/wild_deepfake.py.

Kaggle / official layout (before preprocessing):
  deepfake_in_the_wild/
    real train/   or real_train/   -> *.tar.gz
    real test/
    fake train/
    fake test/

Each tar.gz contains sequence folders with 224x224 face PNGs.

Expected output:
  <out_root>/
    train/
      real/<seq>/000.png
      fake/<seq>/000.png
      real.pickle    # list of relative paths, e.g. "real/131/1057.png"
      fake.pickle
    test/
      ...
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

import torch

SPLIT_MAP = {
    "train": [("real", ("real train", "real_train")), ("fake", ("fake train", "fake_train"))],
    "test": [("real", ("real test", "real_test")), ("fake", ("fake test", "fake_test"))],
}


def _find_split_dir(raw_root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        p = raw_root / name
        if p.exists():
            return p
    return None


def _extract_tarballs(split_dir: Path, out_split_dir: Path, label_name: str):
    """Extract all tar.gz into out_split_dir/<label_name>/<seq_id>/*.png"""
    archives = sorted(split_dir.glob("*.tar.gz")) + sorted(split_dir.glob("*.tar"))
    if not archives:
        # Maybe already extracted flat folders
        for sub in sorted(split_dir.iterdir()):
            if sub.is_dir():
                dest = out_split_dir / label_name / sub.name
                if not dest.exists():
                    shutil.copytree(sub, dest)
        return

    for archive in archives:
        print(f"  extracting {archive.name} ...")
        with tarfile.open(archive, "r:*") as tar:
            tar.extractall(path=out_split_dir / "_tmp")
        tmp = out_split_dir / "_tmp"
        # Move sequence folders under label_name
        for seq_dir in tmp.iterdir():
            if seq_dir.is_dir():
                dest = out_split_dir / label_name / seq_dir.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(seq_dir), str(dest))
        if tmp.exists():
            shutil.rmtree(tmp)


def _build_pickle(out_split_dir: Path, label_name: str) -> list[str]:
    label_root = out_split_dir / label_name
    paths = []
    if not label_root.exists():
        return paths
    for seq_dir in sorted(label_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        for img in sorted(seq_dir.glob("*.png")) + sorted(seq_dir.glob("*.jpg")):
            rel = f"{label_name}/{seq_dir.name}/{img.name}"
            paths.append(rel)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Preprocess WildDeepfake for SFANet.")
    parser.add_argument(
        "--raw_root",
        required=True,
        help="Folder containing deepfake_in_the_wild (or that folder itself)",
    )
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--max_images", type=int, default=None, help="Subsample paths in pickle (demo)")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    if (raw_root / "deepfake_in_the_wild").exists():
        raw_root = raw_root / "deepfake_in_the_wild"

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for split, label_dirs in SPLIT_MAP.items():
        out_split = out_root / split
        out_split.mkdir(parents=True, exist_ok=True)

        for label_name, dir_names in label_dirs:
            split_dir = _find_split_dir(raw_root, dir_names)
            if split_dir is None:
                print(f"[warn] missing {split}/{label_name}: tried {dir_names}")
                continue
            print(f"\n=== {split}/{label_name} from {split_dir} ===")
            _extract_tarballs(split_dir, out_split, label_name)

            paths = _build_pickle(out_split, label_name)
            if args.max_images is not None:
                paths = paths[: args.max_images]
            pickle_path = out_split / f"{label_name}.pickle"
            torch.save(paths, pickle_path)
            print(f"Wrote {pickle_path} ({len(paths)} images)")

    print(f"\nDone. WildDeepfake root for SFANet: {out_root}")


if __name__ == "__main__":
    main()
