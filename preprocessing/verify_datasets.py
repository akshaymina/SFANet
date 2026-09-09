"""Verify preprocessed datasets match SFANet loader expectations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def verify_ffpp(root: Path, compression: str = "c23") -> bool:
    ok = True
    for split in ("train", "val", "test"):
        pkl = root / f"{split}_{compression}.pickle"
        if not pkl.exists():
            print(f"[FAIL] FF++ missing {pkl.name}")
            ok = False
            continue
        import torch

        index = torch.load(pkl, weights_only=False)
        print(f"[OK] {pkl.name}: {len(index)} entries")
        if index:
            sample_path = root / index[0][0]
            if not sample_path.exists():
                print(f"  [FAIL] sample missing: {sample_path}")
                ok = False
            else:
                print(f"  sample: {index[0]}")
    return ok


def verify_celebdf(root: Path) -> bool:
    ok = True
    for subset in ("YouTube-real", "Celeb-real", "Celeb-synthesis"):
        img_root = root / subset / "images"
        if not img_root.exists():
            print(f"[FAIL] Celeb-DF missing {img_root}")
            ok = False
            continue
        n_vids = len(list(img_root.iterdir()))
        print(f"[OK] {subset}/images: {n_vids} video folders")

    list_file = root / "List_of_testing_videos.txt"
    if not list_file.exists():
        print("[FAIL] Celeb-DF missing List_of_testing_videos.txt")
        ok = False
    else:
        print(f"[OK] {list_file.name} present")
    return ok


def verify_wilddeepfake(root: Path) -> bool:
    ok = True
    for split in ("train", "test"):
        split_root = root / split
        for label in ("real", "fake"):
            pkl = split_root / f"{label}.pickle"
            if not pkl.exists():
                print(f"[FAIL] WildDeepfake missing {pkl}")
                ok = False
                continue
            import torch

            paths = torch.load(pkl, weights_only=False)
            print(f"[OK] {split}/{label}.pickle: {len(paths)} entries")
            if paths:
                sample = split_root / paths[0]
                if not sample.exists():
                    print(f"  [FAIL] sample missing: {sample}")
                    ok = False
                else:
                    print(f"  sample: {paths[0]}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ffpp", "celebdf", "wilddeepfake"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--compression", default="c23")
    args = parser.parse_args()

    root = Path(args.root)
    checks = {
        "ffpp": lambda: verify_ffpp(root, args.compression),
        "celebdf": verify_celebdf,
        "wilddeepfake": verify_wilddeepfake,
    }
    ok = checks[args.dataset]()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
