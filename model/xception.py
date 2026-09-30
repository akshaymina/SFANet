"""PyTorch Xception backbone compatible with the SFANet implementation.

The module layout intentionally matches the classic Cadene/TStandley Xception
implementation expected by SFANet: conv1/conv2, block1..block12, conv3/conv4,
and bn/act layers.
"""

from pathlib import Path
import torch
import torch.nn as nn
from .common import SeparableConv2d, Block


class Xception(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU(inplace=True)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.act3 = nn.ReLU(inplace=True)

        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.act4 = nn.ReLU(inplace=True)

        self.fc = nn.Linear(2048, num_classes)

    def features(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        for i in range(1, 13):
            x = getattr(self, f"block{i}")(x)
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        return x

    def forward(self, x):
        x = self.features(x)
        x = torch.relu(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def _fix_pointwise_weights(state_dict):
    """Convert old 2-D pointwise kernels to 4-D Conv2d kernels if needed."""
    fixed = {}
    for k, v in state_dict.items():
        if "pointwise.weight" in k and v.ndim == 2:
            v = v.unsqueeze(-1).unsqueeze(-1)
        fixed[k] = v
    return fixed


def load_imagenet_weights(model, path):
    path = Path(path)
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = _fix_pointwise_weights(state)
    # The original classifier is 1000-way. SFANet replaces it with its own head,
    # so classifier weights are deliberately excluded.
    state = {k: v for k, v in state.items() if not k.startswith("fc.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected


def xception(pretrained=False, weights_path=None):
    model = Xception(num_classes=1000)
    if pretrained:
        if not weights_path:
            raise ValueError(
                "pretrained=True requires weights_path. Download the Xception "
                "checkpoint first and pass --xception-weights."
            )
        missing, unexpected = load_imagenet_weights(model, weights_path)
        if missing:
            print(f"[Xception] missing keys while loading weights: {len(missing)}")
        if unexpected:
            print(f"[Xception] unexpected keys while loading weights: {len(unexpected)}")
    return model
