"""
Preprocess the extracted FaceForensics++ dataset for SFANet.

RAW DATASET STRUCTURE USED BY THIS PROJECT
------------------------------------------

<raw_root>/
    original/
        ... videos ...
    Deepfakes/
        ... videos ...
    Face2Face/
        ... videos ...
    FaceShifter/
        ... videos ...
    FaceSwap/
        ... videos ...
    NeuralTextures/
        ... videos ...
    DeepFakeDetection/
        ... videos ...
    csv/

The preprocessing stage converts this into the directory structure
expected by the original SFANet FF++ dataset loader:

<out_root>/
    original_sequences/youtube/c23/images/<video_stem>/0000.png
    manipulated_sequences/Deepfakes/c23/images/<video_stem>/0000.png
    manipulated_sequences/Face2Face/c23/images/<video_stem>/0000.png
    ...
    train_c23.pickle
    val_c23.pickle
    test_c23.pickle

Pickle entry:
    (relative_image_path, label)

label:
    0 = real
    1 = fake

Split:
    72% train
    14% validation
    14% test
"""

from __future__ import annotations

# ---------------------------------------------------------------------
# Make sure Python can find face_extract.py even when this script is
# launched from an arbitrary Colab working directory.
# ---------------------------------------------------------------------

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# ---------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------

import argparse
import os
import random

import torch

from face_extract import (
    extract_faces_from_video,
    get_face_detector,
    iter_videos,
)


# ---------------------------------------------------------------------
# YOUR RAW DATASET STRUCTURE
# ---------------------------------------------------------------------

REAL_FOLDER = "original"

FAKE_FOLDERS = [
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
    "DeepFakeDetection",
]


# ---------------------------------------------------------------------
# Find where videos are actually stored inside each dataset folder.
#
# We support several possibilities:
#
#   folder/c23/videos/
#   folder/videos/c23/
#   folder/c23/
#   folder/videos/
#   folder/
#
# iter_videos() searches recursively, so this is deliberately flexible.
# ---------------------------------------------------------------------

def _find_video_root(base: Path, compression: str) -> Path:

    candidates = [
        base / compression / "videos",
        base / "videos" / compression,
        base / compression,
        base / "videos",
        base,
    ]

    for candidate in candidates:

        if not candidate.exists():
            continue

        videos = iter_videos(candidate)

        if videos:
            return candidate

    return base


# ---------------------------------------------------------------------
# Build the OUTPUT path that SFANet expects.
#
# Raw "original" is mapped to:
#
#   original_sequences/youtube
#
# Fake methods are mapped to:
#
#   manipulated_sequences/<method>
# ---------------------------------------------------------------------

def _output_prefix(
    raw_folder: str,
    compression: str,
    video_stem: str,
) -> str:

    if raw_folder == REAL_FOLDER:

        return (
            f"original_sequences/youtube/"
            f"{compression}/images/{video_stem}"
        )

    return (
        f"manipulated_sequences/{raw_folder}/"
        f"{compression}/images/{video_stem}"
    )


# ---------------------------------------------------------------------
# Collect videos
# ---------------------------------------------------------------------

def _collect_video_jobs(
    raw_root: Path,
    compression: str,
    max_videos: int | None,
):
    jobs = []

    # ================================================================
    # REAL VIDEOS
    # ================================================================

    real_root = raw_root / REAL_FOLDER

    if not real_root.exists():

        print(f"[ERROR] Real folder not found:")
        print(f"        {real_root}")

    else:

        video_root = _find_video_root(
            real_root,
            compression,
        )

        real_videos = iter_videos(video_root)

        if not real_videos:

            print(f"[WARN] No videos found in:")
            print(f"       {real_root}")

        else:

            if max_videos is not None:
                real_videos = real_videos[:max_videos]

            for video_path in real_videos:

                jobs.append(
                    {
                        "video": video_path,
                        "label": 0,
                        "raw_folder": REAL_FOLDER,
                        "rel_prefix": _output_prefix(
                            REAL_FOLDER,
                            compression,
                            video_path.stem,
                        ),
                    }
                )

    # ================================================================
    # FAKE VIDEOS
    # ================================================================

    for method in FAKE_FOLDERS:

        method_root = raw_root / method

        if not method_root.exists():

            print(f"[WARN] Fake folder not found:")
            print(f"       {method_root}")

            continue

        video_root = _find_video_root(
            method_root,
            compression,
        )

        fake_videos = iter_videos(video_root)

        if not fake_videos:

            print(f"[WARN] No videos found in:")
            print(f"       {method_root}")

            continue

        if max_videos is not None:
            fake_videos = fake_videos[:max_videos]

        for video_path in fake_videos:

            jobs.append(
                {
                    "video": video_path,
                    "label": 1,
                    "raw_folder": method,
                    "rel_prefix": _output_prefix(
                        method,
                        compression,
                        video_path.stem,
                    ),
                }
            )

    return jobs


# ---------------------------------------------------------------------
# VIDEO-LEVEL TRAIN / VAL / TEST SPLIT
# ---------------------------------------------------------------------

def _split_videos(
    jobs: list[dict],
    seed: int = 2022,
):

    random.seed(seed)

    groups: dict[str, list[dict]] = {}

    for job in jobs:

        key = job["rel_prefix"]

        groups.setdefault(key, []).append(job)

    keys = list(groups.keys())

    random.shuffle(keys)

    n = len(keys)

    n_train = int(0.72 * n)

    n_val = int(0.14 * n)

    train_keys = set(
        keys[:n_train]
    )

    val_keys = set(
        keys[n_train:n_train + n_val]
    )

    test_keys = set(
        keys[n_train + n_val:]
    )

    splits = {
        "train": [],
        "val": [],
        "test": [],
    }

    for key, items in groups.items():

        if key in train_keys:

            splits["train"].extend(items)

        elif key in val_keys:

            splits["val"].extend(items)

        else:

            splits["test"].extend(items)

    return splits


# ---------------------------------------------------------------------
# BUILD PICKLE INDEX
# ---------------------------------------------------------------------

def _build_index(
    out_root: Path,
    split_jobs: list[dict],
) -> list[tuple[str, int]]:

    index = []

    for job in split_jobs:

        img_dir = (
            out_root /
            job["rel_prefix"]
        )

        if not img_dir.exists():
            continue

        png_files = sorted(
            img_dir.glob("*.png")
        )

        for png in png_files:

            relative_path = str(
                (
                    Path(job["rel_prefix"])
                    / png.name
                ).as_posix()
            )

            index.append(
                (
                    relative_path,
                    job["label"],
                )
            )

    return index


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Preprocess extracted FaceForensics++ "
            "dataset for SFANet."
        )
    )

    parser.add_argument(
        "--raw_root",
        required=True,
        help="Root containing original/ and fake-method folders",
    )

    parser.add_argument(
        "--out_root",
        required=True,
        help="Output directory for extracted faces and pickles",
    )

    parser.add_argument(
        "--compression",
        default="c23",
        choices=["raw", "c23", "c40"],
    )

    parser.add_argument(
        "--frame_stride",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=32,
        help="Maximum number of extracted face frames per video",
    )

    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Maximum videos per class/method",
    )

    parser.add_argument(
        "--out_size",
        type=int,
        default=299,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2022,
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------

    raw_root = Path(args.raw_root).resolve()

    out_root = Path(args.out_root).resolve()

    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print("SFANet FaceForensics++ Preprocessing")
    print("=" * 70)

    print(f"Raw dataset : {raw_root}")
    print(f"Output      : {out_root}")
    print(f"Compression : {args.compression}")
    print(f"Frame stride: {args.frame_stride}")
    print(f"Max frames  : {args.max_frames}")
    print(f"Max videos  : {args.max_videos}")
    print("=" * 70)
    print()

    # -----------------------------------------------------------------
    # Check raw root
    # -----------------------------------------------------------------

    if not raw_root.exists():

        raise FileNotFoundError(
            f"Raw dataset does not exist:\n{raw_root}"
        )

    # -----------------------------------------------------------------
    # Face detector
    # -----------------------------------------------------------------

    detect_fn = get_face_detector()

    # -----------------------------------------------------------------
    # Collect videos
    # -----------------------------------------------------------------

    jobs = _collect_video_jobs(
        raw_root=raw_root,
        compression=args.compression,
        max_videos=args.max_videos,
    )

    print()
    print(f"Found {len(jobs)} videos to process.")
    print()

    if not jobs:

        raise RuntimeError(
            "No videos were found. "
            "Check the raw dataset structure."
        )

    # -----------------------------------------------------------------
    # Extract faces
    # -----------------------------------------------------------------

    for i, job in enumerate(jobs):

        out_dir = (
            out_root /
            job["rel_prefix"]
        )

        # Skip if already processed
        if (
            out_dir.exists()
            and any(out_dir.glob("*.png"))
        ):

            print(
                f"[{i + 1}/{len(jobs)}] "
                f"SKIP {job['video'].name} "
                f"(already processed)"
            )

            continue

        print(
            f"[{i + 1}/{len(jobs)}] "
            f"{job['video'].name}"
        )

        saved = extract_faces_from_video(
            job["video"],
            out_dir,
            detect_fn,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            out_size=args.out_size,
        )

        print(
            f"    Saved {saved} face frames"
        )

    # -----------------------------------------------------------------
    # Build train / validation / test splits
    # -----------------------------------------------------------------

    splits = _split_videos(
        jobs,
        seed=args.seed,
    )

    print()
    print("=" * 70)
    print("Building dataset indexes")
    print("=" * 70)

    for split_name, split_jobs in splits.items():

        index = _build_index(
            out_root,
            split_jobs,
        )

        pickle_path = (
            out_root /
            f"{split_name}_{args.compression}.pickle"
        )

        torch.save(
            index,
            pickle_path,
        )

        real_count = sum(
            1
            for _, label in index
            if label == 0
        )

        fake_count = sum(
            1
            for _, label in index
            if label == 1
        )

        print(
            f"{split_name:5s} -> "
            f"{len(index)} images | "
            f"real={real_count} | "
            f"fake={fake_count} | "
            f"videos={len(split_jobs)}"
        )

        print(
            f"    {pickle_path}"
        )

    print()
    print("=" * 70)
    print("Preprocessing complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()