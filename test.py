import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from dataset import FaceForensics
from losses import SFANetLoss
from metrics import compute_metrics
from model import SFANet


def load_cfg(config_path, root, split, compression, method):
    with open(config_path, "r", encoding="utf-8") as f:
        all_cfg = yaml.safe_load(f)
    cfg = deepcopy(all_cfg["train_cfg" if split == "train" else "test_cfg"])
    cfg["root"] = str(Path(root).expanduser().resolve())
    cfg["split"] = split
    cfg["compression"] = compression
    cfg["method"] = method
    return cfg


def main():
    p = argparse.ArgumentParser(description="Evaluate SFANet on the supplied FF++ loader.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--compression", default="c23", choices=["c0", "c23", "c40"])
    p.add_argument("--method", default="all", choices=["all", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"])
    p.add_argument("--config", default="config/dataset/faceforensics.yml")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(args.config, args.data_root, args.split, args.compression, args.method)
    dataset = FaceForensics(cfg, seed=2022)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda")

    model = SFANet(num_classes=2, pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    criterion = SFANetLoss()

    ys, probs = [], []
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for paths, y in loader:
            x = dataset.load_item(paths).to(device)
            y = y.to(device).long()
            logits = model(x)
            reconstruction = model.reconstruction
            loss, _, _ = criterion(logits, y, x, reconstruction)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            ys.extend(y.cpu().numpy().tolist())
            probs.extend(logits.softmax(1)[:, 1].cpu().numpy().tolist())

    metrics = compute_metrics(ys, probs)
    metrics["loss"] = total_loss / max(1, n)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
