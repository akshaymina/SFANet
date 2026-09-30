import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import SFANet


EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", default=None)
    p.add_argument("--image-folder", default=None)
    p.add_argument("--xception-weights", default=None)
    args = p.parse_args()

    if not args.image and not args.image_folder:
        raise SystemExit("Provide --image or --image-folder")

    paths = [Path(args.image)] if args.image else sorted(
        p for p in Path(args.image_folder).rglob("*") if p.suffix.lower() in EXTS
    )
    if not paths:
        raise SystemExit("No images found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SFANet(num_classes=2, pretrained=args.xception_weights is not None,
                   xception_weights=args.xception_weights).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    with torch.no_grad():
        for path in paths:
            with Image.open(path) as im:
                x = tfm(im.convert("RGB")).unsqueeze(0).to(device)
            prob_fake = model(x).softmax(1)[0, 1].item()
            label = "FAKE" if prob_fake >= 0.5 else "REAL"
            print(f"{path}\tfake_probability={prob_fake:.6f}\tprediction={label}")


if __name__ == "__main__":
    main()
