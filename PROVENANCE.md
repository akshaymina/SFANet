# Provenance / exactness map

## User-supplied core model

- `model/network/WTFD.py`: cleaned/copy of the user-supplied WTFD implementation.
- `model/network/CMF.py`: cleaned/copy of the user-supplied CMF implementation.
- `model/network/Dualfusion.py`: cleaned/copy of the user-supplied Dualfusion implementation.
- `model/SFANet.py`: cleaned active path of the user-supplied SFANet.py; dead/undefined leftovers were removed so the model can instantiate.

## User-supplied preprocessing / dataset pipeline

- `dataset/abstract_dataset.py`: the user-supplied `abstract_dataset(1).py`, copied into the `dataset` package without changing its dataset behavior.
- `dataset/faceforensics.py`: the user-supplied `faceforensics(2).py`; the import was changed from `from dataset import AbstractDataset` to the package-relative `from .abstract_dataset import AbstractDataset` so it works reliably inside this project.
- `dataset/__init__.py`: adapted from the user-supplied registry. It currently registers only `FaceForensics` because the supplied `wild_deepfake.py` and `celeb_df.py` files were not provided. This prevents imports from failing while preserving the `load_dataset("FaceForensics")` interface.

## Integration changes

- `config/dataset/faceforensics.yml`: added configuration for the supplied loader, using `method=all`, `compression=c23`, 299x299 resize, random horizontal flip for training, and normalization.
- `train.py`: changed to use the supplied loader exactly as written: DataLoader yields `(paths, targets)`, and `FaceForensics.load_item(paths)` performs image loading and transforms for each mini-batch.
- `test.py`: same integration pattern as training.
- `verify_dataset.py`: verifies the exact supplied `FaceForensics` class against all three split pickle files and actually loads sample images through `load_item()`.
- `inspect_pickle.py`: inspects split files using `torch.load(..., weights_only=False)` first, matching the supplied `FaceForensics` implementation, with a pickle fallback.

## Reconstructed model/training support

- `model/common.py`: reconstructed Xception-style `SeparableConv2d`/`Block` support used by this SFANet family.
- `model/xception.py`: local Xception implementation with the layer names required by SFANet and a loader for the classic Xception checkpoint.
- `losses.py`, `metrics.py`, `train.py`, `test.py`, `inference.py`: runnable scaffolding built from the paper's published architecture and training description.

## Reproducibility note

The supplied SFANet source is incomplete relative to the published architecture/repository structure. The package therefore distinguishes user-supplied source from integration/reconstruction code. Undocumented implementation choices (for example the exact StepLR `step_size` and `gamma`) are exposed as options and are not represented as author-verified values.
