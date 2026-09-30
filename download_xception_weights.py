"""Download the Xception ImageNet checkpoint used by classic PyTorch Xception code."""

from pathlib import Path
import urllib.request

URL = "https://huggingface.co/spaces/asdasdasdasd/Face-forgery-detection/resolve/main/xception-b5690688.pth"
OUT = Path("pretrained/xception-b5690688.pth")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        print(f"Already exists: {OUT}")
        return
    print(f"Downloading to {OUT} ...")
    urllib.request.urlretrieve(URL, OUT)
    print(f"Done: {OUT} ({OUT.stat().st_size / (1024**2):.1f} MB)")


if __name__ == "__main__":
    main()
