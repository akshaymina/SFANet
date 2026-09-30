# SFANet + supplied FaceForensics++ preprocessing/data loader

This directory integrates the dataset pipeline supplied with the project into the runnable SFANet reproduction.

## Important: keep the FF++ layout unchanged

Use exactly:

```text
processed/ffpp/
├── original_sequences/youtube/c23/images/<video_id>/0000.png
├── manipulated_sequences/Deepfakes/c23/images/<pair_id>/0000.png
├── manipulated_sequences/Face2Face/c23/images/<pair_id>/0000.png
├── manipulated_sequences/FaceSwap/c23/images/<pair_id>/0000.png
├── manipulated_sequences/NeuralTextures/c23/images/<pair_id>/0000.png
├── train_c23.pickle
├── val_c23.pickle
└── test_c23.pickle
```

Do not reorganize the images into `train/real` and `train/fake`.

## Dataset code used

The `dataset/` package is based directly on the three supplied preprocessing/data-loading files:

```text
__init__.py
abstract_dataset.py
faceforensics.py
```

The only required code-level changes are integration changes:

1. They are placed under the top-level `dataset/` package.
2. `faceforensics.py` uses a package-relative import:
   `from .abstract_dataset import AbstractDataset`.
3. `dataset/__init__.py` registers only `FaceForensics` for now, because the supplied `wild_deepfake.py` and `celeb_df.py` files are not present.
4. Training/evaluation use the supplied loader's intended API: DataLoader returns `(paths, targets)`, then `dataset.load_item(paths)` loads/transforms the image batch.

The original `AbstractDataset.__getitem__` returns path/target, rather than an image tensor. This is why the training and testing loops below call `load_item()` explicitly.

## Configuration

`config/dataset/faceforensics.yml` supplies the model-side image transforms:

- Resize to 299x299
- Random horizontal flip for training
- Normalize with mean/std 0.5/0.5 so pixel values reach approximately [-1, 1]
- `ToTensorV2()` is added by the supplied `AbstractDataset`

The paper specifies 299x299 input and random horizontal flipping; the normalization is required to match the [-1,1] input convention expected by the supplied model implementation.

## Install

```powershell
pip install -r requirements.txt
```

## Verify your actual pickle serialization first

```powershell
python inspect_pickle.py "D:\datasets\processed\ffpp\train_c23.pickle"
```

The supplied `FaceForensics` code loads the split files with:

```python
torch.load(path, weights_only=False)
```

so the inspection script tries the same method first.

## Verify dataset + transforms

```powershell
python verify_dataset.py `
  --data-root "D:\datasets\processed\ffpp" `
  --compression c23 `
  --method all
```

This checks the directory structure, constructs the supplied `FaceForensics` class for all three splits, and actually loads a couple of images through `load_item()`.

Expected tensor shape:

```text
(2, 3, 299, 299)
```

for the small verification batch.

## Model smoke test

The existing model smoke test remains:

```powershell
python smoke_test.py --xception-weights pretrained/xception-b5690688.pth
```

## One-epoch end-to-end test

On Windows, begin with `--workers 0`:

```powershell
python train.py `
  --data-root "D:\datasets\processed\ffpp" `
  --compression c23 `
  --method all `
  --xception-weights pretrained/xception-b5690688.pth `
  --batch-size 4 `
  --workers 0 `
  --epochs 1 `
  --output-dir results/ffpp_smoke
```

The important part is that the training loop receives paths from your supplied loader and then calls:

```python
x = train_ds.load_item(paths)
```

before sending the tensor to SFANet.

## Full training

The paper reports Xception, batch size 32, 40 epochs, learning rate 2e-4, Adam, weight decay 1e-5, and random horizontal flipping. It also says a StepLR scheduler is used, but the published table does not give the step size or gamma. This implementation exposes them as command-line arguments and defaults to `step_size=10`, `gamma=0.1`.

```powershell
python train.py `
  --data-root "D:\datasets\processed\ffpp" `
  --compression c23 `
  --method all `
  --xception-weights pretrained/xception-b5690688.pth `
  --batch-size 32 `
  --workers 0 `
  --epochs 40 `
  --lr 2e-4 `
  --weight-decay 1e-5 `
  --step-size 10 `
  --gamma 0.1 `
  --output-dir results/ffpp_c23
```

Once everything is stable, `--workers 4` can be tested, but the supplied loader performs the actual image decoding in `load_item`, so increasing DataLoader workers does not give the same benefit as a conventional image-returning Dataset.

## Test

```powershell
python test.py `
  --data-root "D:\datasets\processed\ffpp" `
  --compression c23 `
  --method all `
  --checkpoint results/ffpp_c23/checkpoints/best.pt `
  --batch-size 32 `
  --workers 0
```

## Binary labels

The supplied FaceForensics loader uses:

```text
original_sequences       -> target 0 (real)
manipulated sequences    -> target 1 (fake)
```

with `method=all`, the pickle labels are used directly. The four manipulated techniques therefore remain four sources of fake examples, while the classifier remains binary.
