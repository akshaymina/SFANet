from .abstract_dataset import AbstractDataset
from .faceforensics import FaceForensics

# Only FaceForensics is included at the moment because that is the dataset
# being reproduced. WildDeepfake/CelebDF can be added later without changing
# the SFANet model.
LOADERS = {
    "FaceForensics": FaceForensics,
}


def load_dataset(name="FaceForensics"):
    print(f"Loading dataset: '{name}'...")
    return LOADERS[name]


__all__ = ["AbstractDataset", "FaceForensics", "LOADERS", "load_dataset"]
