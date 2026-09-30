"""Inspect an FF++ split file using the same serialization loader as the supplied pipeline."""

import argparse
import pickle
import torch


def load_serialized(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False), "torch.load"
    except Exception as torch_error:
        with open(path, "rb") as f:
            try:
                return pickle.load(f), "pickle.load"
            except Exception:
                raise RuntimeError(
                    f"Could not read {path!r} with torch.load or pickle.load. "
                    f"torch.load error: {torch_error}"
                ) from torch_error


def preview(obj, limit):
    print("top-level type:", type(obj).__name__)
    if isinstance(obj, dict):
        print("keys:", list(obj.keys())[:20])
    try:
        n = len(obj)
        print("length:", n)
    except TypeError:
        n = None

    if isinstance(obj, (list, tuple)):
        print("first entries:")
        for item in obj[:limit]:
            print(" ", repr(item))
    elif isinstance(obj, dict):
        print("first entries:")
        for i, (k, v) in enumerate(obj.items()):
            if i >= limit:
                break
            print(" ", repr(k), "->", repr(v))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pickle_path")
    p.add_argument("--show", type=int, default=10)
    args = p.parse_args()

    obj, loader = load_serialized(args.pickle_path)
    print("file:", args.pickle_path)
    print("loaded with:", loader)
    preview(obj, args.show)


if __name__ == "__main__":
    main()
