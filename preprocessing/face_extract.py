"""Shared face extraction utilities for SFANet dataset preprocessing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np


def _lazy_retinaface_detector():
    """Return a RetinaFace-detect callable, or None if unavailable."""
    try:
        from retinaface import RetinaFace  # pip install retina-face
    except ImportError:
        return None

    def detect(image_bgr: np.ndarray):
        # RetinaFace expects RGB; returns dict keyed by face id.
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        faces = RetinaFace.detect_faces(rgb)
        if not isinstance(faces, dict) or len(faces) == 0:
            return None
        # Pick the face with highest detection score.
        best = max(faces.values(), key=lambda f: f.get("score", 0.0))
        x1, y1, x2, y2 = best["facial_area"]
        return int(x1), int(y1), int(x2), int(y2)

    return detect


def _lazy_mtcnn_detector(device: str = "cuda"):
    """Fallback detector when RetinaFace is not installed."""
    try:
        from facenet_pytorch import MTCNN
        import torch
    except ImportError as exc:
        raise ImportError(
            "Install a face detector: pip install retina-face  OR  pip install facenet-pytorch"
        ) from exc

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(keep_all=False, device=dev, margin=0)

    def detect(image_bgr: np.ndarray):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb)
        if boxes is None:
            return None
        x1, y1, x2, y2 = boxes[0]
        return int(x1), int(y1), int(x2), int(y2)

    return detect


def get_face_detector(prefer_retinaface: bool = True, device: str = "cuda") -> Callable:
    if prefer_retinaface:
        detector = _lazy_retinaface_detector()
        if detector is not None:
            print("Using RetinaFace for face detection.")
            return detector
        print("RetinaFace not found; falling back to MTCNN.")
    return _lazy_mtcnn_detector(device=device)


def crop_face(
    image_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    scale: float = 1.3,
    out_size: int = 299,
) -> Optional[np.ndarray]:
    """Conservative crop (RECCE / SFANet paper uses scale=1.3) and resize."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale
    nx1 = max(0, int(cx - bw / 2))
    ny1 = max(0, int(cy - bh / 2))
    nx2 = min(w, int(cx + bw / 2))
    ny2 = min(h, int(cy + bh / 2))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    crop = image_bgr[ny1:ny2, nx1:nx2]
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def extract_faces_from_video(
    video_path: Path,
    output_dir: Path,
    detect_fn: Callable,
    frame_stride: int = 5,
    max_frames: Optional[int] = 32,
    scale: float = 1.3,
    out_size: int = 299,
) -> int:
    """Extract face PNGs from one video. Returns number of saved frames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [skip] cannot open video: {video_path}")
        return 0

    saved = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        box = detect_fn(frame)
        if box is not None:
            face = crop_face(frame, box, scale=scale, out_size=out_size)
            if face is not None:
                out_path = output_dir / f"{saved:04d}.png"
                cv2.imwrite(str(out_path), face)
                saved += 1
                if max_frames is not None and saved >= max_frames:
                    break
        frame_idx += 1

    cap.release()
    return saved


def iter_videos(root: Path, extensions: Iterable[str] = (".mp4", ".avi", ".mov", ".mkv")) -> list[Path]:
    exts = {e.lower() for e in extensions}
    videos = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if Path(name).suffix.lower() in exts:
                videos.append(Path(dirpath) / name)
    return sorted(videos)
