# Kaggle ASL Alphabet: first static-letter baseline

Source: [grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
Downloaded on 2026-11-01 via the public Kaggle dataset download endpoint:
`https://www.kaggle.com/api/v1/datasets/download/grassknoted/asl-alphabet`.
The downloaded archive contains 87,000 training images in 29 classes and 28 test images.
Only `asl_alphabet_train/asl_alphabet_train` is used in this experiment.
The small supplied test folder is not used for model selection or reported validation.

Local archive: `data/external/asl-alphabet.zip`.
Extracted dataset: `data/external/asl-alphabet/`.
These paths, landmark caches, local webcam recordings and model weights are git-ignored.
Retain the original source terms when redistributing images; this repository does not include them.
The download endpoint did not pin a Kaggle version number; the archive SHA-256 in the
experiment log identifies the downloaded bytes.

## Scope and cache

Static labels in checkpoint order: `A B C D E F G H I K L M N O P Q R S T U V W X Y`.
Exclude J and Z (motion-dependent), space, del and nothing.
First select 50 images per label without replacement with NumPy seed 42.
Then extract all 3,000 images per selected label (72,000 attempted images).
MediaPipe Hand Landmarker runs in IMAGE mode, one hand, detection and presence
thresholds 0.5. Failed detections are retained as failed cache rows, never imputed.
`results/kaggle_pilot_tracking.json` and `results/kaggle_full_tracking.json` contain per-label coverage.

`data/cache/kaggle_landmarks.sqlite` stores source path, label, decoded-RGB SHA-256,
raw 21x3 float32 landmarks, predicted physical handedness and its confidence.
Cache settings record model checksum, MediaPipe version and thresholds; mismatches are rejected.
The source directory is treated as immutable for cache reuse. Delete/rebuild the cache if images change.
`data/cache/static_split.json` records paths and pixel hashes for the actual split.

## Features and evaluation protocol

Subtract the wrist, reflect x only for a left physical hand, divide by maximum
3D distance from the wrist, and flatten to 63 values. No rotation to a canonical
finger direction. This is shared by Kaggle extraction consumers, clip evaluation
and live inference. The physical hand in the Kaggle images is undocumented/unknown. Image handedness comes from MediaPipe and can be wrong; webcam
handedness is supplied explicitly from the known physical signing hand. Mirroring
the displayed preview does not mirror the classifier input. Both sources use
MediaPipe normalized image coordinates; image/camera aspect ratios may differ.

Collapse exact decoded-pixel duplicates before splitting; discard any hash with
conflicting labels. Stratify unique detected images 80/20 by letter with seed 42.
Near duplicates and signer overlap may remain: this is not signer-independent
validation. Split before augmentation. Only training batches receive +/-10 degree
xy jitter and Gaussian landmark noise (standard deviation 0.005).

Train a fresh PyTorch 63-256-128-24 MLP, ReLU, dropout 0.2, AdamW
(lr 0.001, weight decay 0.0001), batch 256, at most 100 epochs.
Select minimum validation cross-entropy, early stopping patience 15.
All random seeds are 42. Training and validation accuracy are evaluated without
augmentation or dropout using that same best checkpoint.

Keep every `data/clips/` recording outside training and model selection.
For `p01/s01`, use the corrected physical `hand=left`, filter by saved valid mask
and finite landmarks, average softmax probabilities over valid frames and take
one argmax per clip. No-valid-frame clips count as incorrect. Evaluate once after
model selection. This small session covers eight letters and is not a 24-letter benchmark.

## Reproduce

```bash
uv sync --locked --group train --group dev
uv run --locked --group train python scripts/extract_kaggle.py \
  --root data/external/asl-alphabet/asl_alphabet_train/asl_alphabet_train --pilot
uv run --locked --group train python scripts/extract_kaggle.py \
  --root data/external/asl-alphabet/asl_alphabet_train/asl_alphabet_train
uv run --locked --group train python scripts/train_static.py
uv run --locked --group train python scripts/evaluate.py --person p01 --session s01
uv run --locked --group train python -m fingerspelling.app \
  --model models/mlp_static.pt --hand left
```

Live mode shows current letter predictions and uses the timestamp-based word builder to assemble editable text. See docs/experiments.md for the frozen-baseline audit and temporal tests. Only the
24 included letters are recognized; there is no trained unknown/background class.
