import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FaceForensics
from losses import SFANetLoss
from metrics import compute_metrics
from model import SFANet


def parse_args():
    p = argparse.ArgumentParser(description="Train SFANet on FF++ using the supplied dataset pipeline.")
    p.add_argument("--data-root", required=True,
                   help="Directory containing original_sequences/, manipulated_sequences/, and *_c23.pickle")
    p.add_argument("--output-dir", default="results/ffpp_c23")
    p.add_argument("--xception-weights", default=None)
    p.add_argument("--compression", default="c23", choices=["c0", "c23", "c40"])
    p.add_argument("--method", default="all", choices=["all", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"])
    p.add_argument("--config", default="config/dataset/faceforensics.yml")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=0,
                   help="Keep 0 first on Windows; the supplied dataset class performs image loading in load_item().")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--step-size", type=int, default=10,
                   help="Paper specifies StepLR but does not publish step_size.")
    p.add_argument("--gamma", type=float, default=0.1,
                   help="Paper does not publish scheduler gamma.")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default=None)
    return p.parse_args()


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cfg(config_path, root, split, compression, method):
    with open(config_path, "r", encoding="utf-8") as f:
        all_cfg = yaml.safe_load(f)

    key = "train_cfg" if split == "train" else "test_cfg"
    cfg = deepcopy(all_cfg[key])
    cfg["root"] = str(Path(root).expanduser().resolve())
    cfg["split"] = split
    cfg["compression"] = compression
    cfg["method"] = method
    return cfg


def make_dataset(args, split):
    cfg = load_cfg(args.config, args.data_root, split, args.compression, args.method)
    return FaceForensics(cfg, seed=args.seed)


def make_loader(dataset, batch_size, workers, device, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def evaluate(model, loader, dataset, device, criterion):
    model.eval()
    total_loss = total_rec = total_cls = 0.0
    n = 0
    ys, probs = [], []

    with torch.no_grad():
        for paths, y in loader:
            # This is the key integration with the supplied dataset code:
            # FaceForensics.__getitem__ returns paths/targets and load_item()
            # loads + transforms the whole mini-batch.
            x = dataset.load_item(paths)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            logits = model(x)
            reconstruction = model.module.reconstruction if isinstance(model, torch.nn.DataParallel) else model.reconstruction
            loss, rec, cls = criterion(logits, y, x, reconstruction)

            bs = x.size(0)
            n += bs
            total_loss += loss.item() * bs
            total_rec += rec.item() * bs
            total_cls += cls.item() * bs
            ys.extend(y.cpu().numpy().tolist())
            probs.extend(logits.softmax(1)[:, 1].cpu().numpy().tolist())

    result = compute_metrics(ys, probs)
    result.update(
        loss=total_loss / max(1, n),
        rec_loss=total_rec / max(1, n),
        cls_loss=total_cls / max(1, n),
    )
    return result


def main():
    args = parse_args()
    seed_everything(args.seed)

    out = Path(args.output_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_ds = make_dataset(args, "train")
    val_ds = make_dataset(args, "val")
    test_ds = make_dataset(args, "test")

    print(f"Train samples: {len(train_ds):,}")
    print(f"Val samples:   {len(val_ds):,}")
    print(f"Test samples:  {len(test_ds):,}")

    train_loader = make_loader(train_ds, args.batch_size, args.workers, device, shuffle=True)
    val_loader = make_loader(val_ds, args.batch_size, args.workers, device, shuffle=False)
    test_loader = make_loader(test_ds, args.batch_size, args.workers, device, shuffle=False)

    model = SFANet(
        num_classes=args.num_classes,
        pretrained=args.xception_weights is not None,
        xception_weights=args.xception_weights,
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion = SFANetLoss(lambda_rec=0.1, lambda_cls=1.0)

    start_epoch = 0
    best_auc = -float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_auc = ckpt.get("best_auc", best_auc)
        print(f"Resumed from epoch {start_epoch}")

    history = []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = running_rec = running_cls = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for paths, y in pbar:
            x = train_ds.load_item(paths)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            reconstruction = model.module.reconstruction if isinstance(model, torch.nn.DataParallel) else model.reconstruction
            loss, rec, cls = criterion(logits, y, x, reconstruction)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_rec += rec.item()
            running_cls += cls.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                rec=f"{rec.item():.4f}",
                cls=f"{cls.item():.4f}",
            )

        val = evaluate(model, val_loader, val_ds, device, criterion)
        scheduler.step()

        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": running_loss / max(1, len(train_loader)),
            "train_rec_loss": running_rec / max(1, len(train_loader)),
            "train_cls_loss": running_cls / max(1, len(train_loader)),
            **{f"val_{k}": v for k, v in val.items()},
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_auc": best_auc,
            "args": vars(args),
        }
        torch.save(state, out / "checkpoints" / "last.pt")

        if np.isfinite(val.get("auc", float("nan"))) and val["auc"] > best_auc:
            best_auc = val["auc"]
            state["best_auc"] = best_auc
            torch.save(state, out / "checkpoints" / "best.pt")

        with open(out / "history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

    print("Best validation AUC:", best_auc)
    best = out / "checkpoints" / "best.pt"
    if best.exists():
        model.load_state_dict(torch.load(best, map_location=device, weights_only=False)["model"])

    test = evaluate(model, test_loader, test_ds, device, criterion)
    print("Test:", json.dumps(test, indent=2))
    with open(out / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test, f, indent=2)


if __name__ == "__main__":
    main()
