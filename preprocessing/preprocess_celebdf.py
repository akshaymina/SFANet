"""
Preprocess Celeb-DF v2 videos into the layout expected by dataset/celeb_df.py.

Expected output:
  <out_root>/
    YouTube-real/images/<video_id>/0000.png
    Celeb-real/images/<video_id>/0000.png
    Celeb-synthesis/images/<video_id>/0000.png
    List_of_testing_videos.txt   (copied from raw root)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from face_extract import extract_faces_from_video, get_face_detector, iter_videos

SUBSETS = {
    "YouTube-real": 0,
    "Celeb-real": 0,
    "Celeb-synthesis": 1,
}


def _video_id_from_path(video_path: Path, subset: str) -> str:
    # e.g. YouTube-real/00170.mp4 -> 00170 ; Celeb-real/id1_0007.mp4 -> id1_0007
    return video_path.stem


def main():
    parser = argparse.ArgumentParser(description="Preprocess Celeb-DF v2 for SFANet.")
    parser.add_argument("--raw_root", required=True, help="Unzipped Celeb-DF root (contains mp4 folders)")
    parser.add_argument("--out_root", required=True, help="Output folder")
    parser.add_argument("--frame_stride", type=int, default=5)
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--max_videos", type=int, default=None, help="Limit videos per subset (demo mode)")
    parser.add_argument("--out_size", type=int, default=299)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    list_file = raw_root / "List_of_testing_videos.txt"
    if list_file.exists():
        shutil.copy2(list_file, out_root / "List_of_testing_videos.txt")
        print(f"Copied {list_file.name}")
    else:
        print("[warn] List_of_testing_videos.txt not found — test split will fail until you add it.")

    detect_fn = get_face_detector()

    for subset in SUBSETS:
        subset_raw = raw_root / subset
        if not subset_raw.exists():
            print(f"[warn] missing subset folder: {subset_raw}")
            continue

        videos = iter_videos(subset_raw)
        if args.max_videos is not None:
            videos = videos[: args.max_videos]

        print(f"\n=== {subset}: {len(videos)} videos ===")
        for i, vp in enumerate(videos):
            vid = _video_id_from_path(vp, subset)
            out_dir = out_root / subset / "images" / vid
            if out_dir.exists() and any(out_dir.glob("*.png")):
                continue
            print(f"[{i+1}/{len(videos)}] {vp.name}")
            extract_faces_from_video(
                vp,
                out_dir,
                detect_fn,
                frame_stride=args.frame_stride,
                max_frames=args.max_frames,
                out_size=args.out_size,
            )

    print(f"\nDone. Celeb-DF root for SFANet: {out_root}")


if __name__ == "__main__":
    main()
