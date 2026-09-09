"""
Preprocess FaceForensics++ videos into the layout expected by dataset/faceforensics.py.

Output layout (example for c23 compression):
  <out_root>/
    original_sequences/youtube/c23/images/<video_stem>/0000.png
    manipulated_sequences/Deepfakes/c23/images/<video_stem>/0000.png
    ...
    train_c23.pickle
    val_c23.pickle
    test_c23.pickle

Each pickle entry: (relative_path_str, label)  where label 0=real, 1=fake.
Split is at VIDEO level: 72% train / 14% val / 14% test (paper default).
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import torch

from face_extract import extract_faces_from_video, get_face_detector, iter_videos

REAL_SUBSETS = [
    ("original_sequences/youtube", 0),
    ("original_sequences/actors", 0),
]
FAKE_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def _find_comp_dir(base: Path, compression: str) -> Path:
    """Return directory containing videos for a subset (handles nested c23/c40/raw)."""
    candidates = [
        base / compression / "videos",
        base / compression,
        base / "videos",
        base,
    ]
    for c in candidates:
        if c.exists() and any(c.glob("*.mp4")):
            return c
    return base


def _video_rel_prefix(subset: str, compression: str, video_stem: str) -> str:
    return f"{subset}/{compression}/images/{video_stem}"


def _collect_video_jobs(raw_root: Path, compression: str, methods: list[str], max_videos: int | None):
    jobs = []
    for subset, label in REAL_SUBSETS:
        subset_path = raw_root / subset.replace("/", os.sep)
        if not subset_path.exists():
            continue
        comp_dir = _find_comp_dir(subset_path, compression)
        for vp in iter_videos(comp_dir):
            if max_videos is not None and sum(1 for j in jobs if j["label"] == label) >= max_videos:
                break
            jobs.append(
                {
                    "video": vp,
                    "label": label,
                    "rel_prefix": _video_rel_prefix(subset, compression, vp.stem),
                }
            )

    for method in methods:
        subset = f"manipulated_sequences/{method}"
        subset_path = raw_root / subset.replace("/", os.sep)
        if not subset_path.exists():
            print(f"[warn] missing subset: {subset_path}")
            continue
        comp_dir = _find_comp_dir(subset_path, compression)
        fake_count = 0
        for vp in iter_videos(comp_dir):
            if max_videos is not None and fake_count >= max_videos:
                break
            jobs.append(
                {
                    "video": vp,
                    "label": 1,
                    "rel_prefix": _video_rel_prefix(subset, compression, vp.stem),
                }
            )
            fake_count += 1
    return jobs


def _split_videos(jobs: list[dict], seed: int = 2022):
    """Split by unique video rel_prefix (video-level split)."""
    random.seed(seed)
    groups: dict[str, list[dict]] = {}
    for j in jobs:
        groups.setdefault(j["rel_prefix"], []).append(j)

    keys = list(groups.keys())
    random.shuffle(keys)
    n = len(keys)
    n_train = int(0.72 * n)
    n_val = int(0.14 * n)
    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train : n_train + n_val])
    test_keys = set(keys[n_train + n_val :])

    splits = {"train": [], "val": [], "test": []}
    for key, items in groups.items():
        if key in train_keys:
            splits["train"].extend(items)
        elif key in val_keys:
            splits["val"].extend(items)
        else:
            splits["test"].extend(items)
    return splits


def _build_index(out_root: Path, split_jobs: list[dict]) -> list[tuple[str, int]]:
    index = []
    for job in split_jobs:
        img_dir = out_root / job["rel_prefix"]
        if not img_dir.exists():
            continue
        for png in sorted(img_dir.glob("*.png")):
            rel = str((Path(job["rel_prefix"]) / png.name).as_posix())
            index.append((rel, job["label"]))
    return index


def main():
    parser = argparse.ArgumentParser(description="Preprocess FaceForensics++ for SFANet.")
    parser.add_argument("--raw_root", required=True, help="Unzipped FF++ root folder")
    parser.add_argument("--out_root", required=True, help="Output folder for faces + pickles")
    parser.add_argument("--compression", default="c23", choices=["raw", "c23", "c40"])
    parser.add_argument("--frame_stride", type=int, default=5)
    parser.add_argument("--max_frames", type=int, default=32, help="Max face frames per video")
    parser.add_argument("--max_videos", type=int, default=None, help="Limit videos per class (demo mode)")
    parser.add_argument("--out_size", type=int, default=299)
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    detect_fn = get_face_detector()
    jobs = _collect_video_jobs(raw_root, args.compression, FAKE_METHODS, args.max_videos)
    print(f"Found {len(jobs)} videos to process.")

    for i, job in enumerate(jobs):
        out_dir = out_root / job["rel_prefix"]
        if out_dir.exists() and any(out_dir.glob("*.png")):
            continue
        print(f"[{i+1}/{len(jobs)}] {job['video'].name}")
        extract_faces_from_video(
            job["video"],
            out_dir,
            detect_fn,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            out_size=args.out_size,
        )

    splits = _split_videos(jobs, seed=args.seed)
    for split_name, split_jobs in splits.items():
        index = _build_index(out_root, split_jobs)
        pickle_path = out_root / f"{split_name}_{args.compression}.pickle"
        torch.save(index, pickle_path)
        print(f"Wrote {pickle_path}  ({len(index)} images, {len(split_jobs)} videos)")


if __name__ == "__main__":
    main()
