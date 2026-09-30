import argparse
import torch

from model import SFANet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--xception-weights", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SFANet(num_classes=2, pretrained=args.xception_weights is not None,
                   xception_weights=args.xception_weights).to(device)
    model.eval()

    x = torch.randn(args.batch_size, 3, 299, 299, device=device).clamp(-1, 1)
    with torch.no_grad():
        logits = model(x)
    print("input:", tuple(x.shape))
    print("logits:", tuple(logits.shape))
    print("reconstruction:", tuple(model.reconstruction.shape))
    print("softmax:", logits.softmax(1))
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
