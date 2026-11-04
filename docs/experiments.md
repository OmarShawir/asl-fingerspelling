# Experiment log

Record each comparison as you make it. Cite the dataset version or split manifest and the machine used for live timing. Keep example targets and measured results separate.

| Date | Git revision | Training data / split | Change | Validation metric | Live timing | Decision |
| --- | --- | --- | --- | --- | --- | --- |

## 2026-11-01 — First Kaggle static MLP baseline

Base Git revision: `647f2ba` plus uncommitted implementation. Code and checkpoint hashes: [provenance](../results/static_provenance.json).

Source: [Kaggle ASL Alphabet — grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
Archive SHA-256: `7c572f14fbaff94f98835cfe71c7582dd379a5176e7c4f83dbf3a30e4b3f68c4`. Downloaded archive: 1,100,887,034 bytes.

Protocol and commands: [dataset guide](../data/KAGGLE_ASL_ALPHABET.md). MediaPipe IMAGE mode, thresholds 0.5, seed 42; exclude J, Z, space, del, nothing.

Full extraction: **55,368/72,000 (76.90%)** detected hands. Failures are excluded from classification accuracy, not counted as successful recognition.

| Letter | Pilot detection (50 each) | Full detection (3,000 each) | Validation recall |
| --- | ---: | ---: | ---: |
| A | 37/50 (74.0%) | 2182/3000 (72.7%) | 98.62% |
| B | 35/50 (70.0%) | 2202/3000 (73.4%) | 99.55% |
| C | 26/50 (52.0%) | 1962/3000 (65.4%) | 99.74% |
| D | 38/50 (76.0%) | 2457/3000 (81.9%) | 99.59% |
| E | 40/50 (80.0%) | 2307/3000 (76.9%) | 99.13% |
| F | 49/50 (98.0%) | 2864/3000 (95.5%) | 99.48% |
| G | 43/50 (86.0%) | 2428/3000 (80.9%) | 100.00% |
| H | 38/50 (76.0%) | 2373/3000 (79.1%) | 99.16% |
| I | 38/50 (76.0%) | 2379/3000 (79.3%) | 98.95% |
| K | 45/50 (90.0%) | 2699/3000 (90.0%) | 99.07% |
| L | 47/50 (94.0%) | 2521/3000 (84.0%) | 99.60% |
| M | 31/50 (62.0%) | 1603/3000 (53.4%) | 95.95% |
| N | 20/50 (40.0%) | 1275/3000 (42.5%) | 96.08% |
| O | 39/50 (78.0%) | 2258/3000 (75.3%) | 99.34% |
| P | 29/50 (58.0%) | 2040/3000 (68.0%) | 98.53% |
| Q | 35/50 (70.0%) | 2118/3000 (70.6%) | 99.76% |
| R | 42/50 (84.0%) | 2545/3000 (84.8%) | 98.82% |
| S | 41/50 (82.0%) | 2555/3000 (85.2%) | 99.41% |
| T | 41/50 (82.0%) | 2354/3000 (78.5%) | 98.94% |
| U | 46/50 (92.0%) | 2509/3000 (83.6%) | 98.01% |
| V | 35/50 (70.0%) | 2548/3000 (84.9%) | 99.02% |
| W | 41/50 (82.0%) | 2453/3000 (81.8%) | 98.37% |
| X | 35/50 (70.0%) | 2156/3000 (71.9%) | 98.14% |
| Y | 44/50 (88.0%) | 2580/3000 (86.0%) | 99.61% |

Deduplication: `{"detected_rows": 55368, "unique_hashes": 55368, "conflicting_hashes_dropped": 0, "duplicate_rows": 0}`. Decoded-pixel hashes are disjoint across the 80/20 stratified split. Augmentation occurs only after splitting and only in training batches.

Shared features: wrist subtraction, physical-left x reflection to a canonical right hand, maximum-radius scaling, 63 values. No rotation normalization. Image hand labels are predicted by MediaPipe; camera hand labels come from the recording metadata.

MLP: 63 → 256 → 128 → 24, ReLU, dropout 0.2; 52,376 parameters trained from scratch. AdamW lr 0.001, weight decay 0.0001, batch 256, maximum 100 epochs, patience 15. Training augmentation: xy rotation ±10 degrees and Gaussian noise σ=0.005.
Best checkpoint: epoch **81**, selected by minimum validation loss; 96 epochs run. CPU training/preparation elapsed: 43.2 seconds. PyTorch 2.6.0+cpu; Python 3.12.16; Linux-7.0.0-30-generic-x86_64-with-glibc2.43.

| Evaluation | Samples | Accuracy | Cross-entropy |
| --- | ---: | ---: | ---: |
| train (best checkpoint, no augmentation/dropout) | 44,293 | 99.60% | 0.0130 |
| validation (best checkpoint, no augmentation/dropout) | 11,075 | 98.97% | 0.0425 |

### Held-out webcam: p01/s01

**20 correct clips out of 30** (66.67%). 1,445 total frames; 1,445 valid finite frames. All selected clips have physical hand `left`.
Exactly one prediction per clip: mean softmax probabilities over valid finite frames, then argmax. No training, early stopping, hyperparameter selection or post-evaluation tuning used these clips. Existing s02 recordings are also excluded.

| Letter | Correct clips | Total clips |
| --- | ---: | ---: |
| A | 3 | 3 |
| E | 6 | 6 |
| H | 6 | 6 |
| L | 3 | 3 |
| M | 0 | 3 |
| N | 0 | 3 |
| O | 2 | 3 |
| T | 0 | 3 |

The Kaggle validation split is not signer-independent: near-duplicate images and signer overlap may remain. Webcam recordings cover only eight letters and one signer/session. Differences in image/camera aspect ratio, backgrounds, poses and handedness-estimation errors may affect transfer. Neither validation accuracy nor classifier-only inference demonstrates complete live word recognition.

Artifacts: `models/mlp_static.pt` (weights, label order, feature version, seed, epoch, split hash), `models/mlp_static.labels.json`, `data/cache/kaggle_landmarks.sqlite`, `data/cache/static_split.json`. Full metrics/confusion matrices and per-clip decisions are in `results/static_training.json`, `results/static_per_letter.csv`, and `results/webcam_evaluation.json`.

Validation: six automated tests passed; Ruff passed. Checkpoint inference and clip aggregation ran successfully. The webcam GUI was not opened by the agent.

Live command (physical left hand):

```bash
.venv/bin/python -m fingerspelling.app --model models/mlp_static.pt --hand left
```

Decision: retain this first frozen baseline and report the webcam result separately. Further data collection and domain adaptation require a new experiment; s02 remains deferred.

## 2026-11-01 — Frozen baseline audit and temporal text builder

No retraining or checkpoint modification. Baseline SHA-256: `f26200e9da636868ed8ae31ab958fcbe76f3edfe64b288b59c2792084efd06d6`.

Re-evaluation retained all 24 checkpoint classes and the original aggregation: mean frame softmax probabilities over valid finite frames, followed by one clip-level argmax. Scores are **not vote fractions**. Result remains **20/30**, with 1,445 valid frames. Every clip prediction and confidence matches the prior evaluation exactly.

Full 30-clip export: [CSV](../results/baseline_recheck.csv), [JSON with checkpoint hash and label order](../results/baseline_recheck.json). Only errors are listed below.

| Clip ID | True | Predicted | Top 3 mean probabilities | Valid frames |
| --- | --- | --- | --- | ---: |
| `67f8949c33d94785aca173294a36f41a` | T | A | A 97.01%, X 2.49%, S 0.40% | 53 |
| `6f71b731adc14c288aa4bcbeb47a05f0` | N | S | S 90.55%, E 7.36%, X 1.49% | 67 |
| `72b14bad9dfc4a118f3962e2aef1332e` | T | A | A 99.05%, X 0.58%, S 0.36% | 50 |
| `759a2a7a82fe452badbdb34a332e0ed0` | O | Y | Y 74.67%, Q 11.36%, E 8.56% | 67 |
| `9a54359c00524bd39638ecd76a344e14` | M | S | S 98.80%, E 1.16%, I 0.02% | 38 |
| `a3cbb264bcc544fba2c7c99788462dbe` | M | S | S 88.14%, E 11.81%, X 0.04% | 50 |
| `afa2c1d3f4a14113a3584297e0b6584d` | N | S | S 92.34%, E 7.20%, X 0.43% | 41 |
| `b9b250ab56584f09b54d67e0404fa5b8` | M | S | S 72.67%, E 27.27%, X 0.05% | 34 |
| `bb268d8f61294458baabb3134193c01b` | N | S | S 91.46%, E 6.26%, X 1.76% | 41 |
| `e2fb45d69b4d4ad99cefb3719f7f3f44` | T | A | A 82.48%, X 15.03%, S 1.98% | 40 |

Existing full extraction coverage for the error target classes (no extraction rerun):

| Letter | Detected / attempted | Coverage |
| --- | ---: | ---: |
| M | 1603/3000 | 53.43% |
| N | 1275/3000 | 42.50% |
| O | 2258/3000 | 75.27% |
| T | 2354/3000 | 78.47% |

These rates measure tracker coverage, not classifier accuracy, and do not establish the cause of a particular error. All 24 pilot/full rates remain in the preceding experiment and in `results/kaggle_full_tracking.json`.

### Handedness path audit

1. `app.capture` reads the original camera frame. `extract_points` converts BGR to RGB and runs MediaPipe VIDEO mode without a horizontal flip.
2. `--hand left` is passed as `physical_hand` to `StaticPredictor.probabilities`; it does not change camera acquisition or ask the tracker to select a left hand.
3. `prepare_features` subtracts the wrist, then applies `centered[..., 0] *= -1` for `left`, then divides by the maximum 3D radius. This reflects wrist-relative x to canonical right-hand coordinates. It does not rotate the hand to erase finger direction.
4. `cv2.flip(frame, 1)` affects only the displayed preview. The predictor consumes landmark arrays extracted before preview mirroring. The mocked live-loop test verifies unchanged predictor inputs while preview flipping is enabled.
5. Training calls the same `prepare_features` on cached image landmarks; evaluation and live inference both use `StaticPredictor.probabilities`. Evaluation uses corrected clip metadata `left`; live mode uses the explicit physical-hand argument.
6. **The physical signing hand in the Kaggle photographs is unknown from the available dataset documentation.** The cached hand field is MediaPipe-predicted handedness, not independently verified ground truth. Shared preprocessing is consistent in code, but it cannot guarantee those predictions are correct. MediaPipe IMAGE vs VIDEO extraction and image/camera geometry differences remain.

### Temporal acceptance and editing

Configuration: `hold_ms=700`, `grace_ms=100`, `release_ms=300`, `min_letter_confidence=0.80`; require `0 <= grace_ms < release_ms`. All 24 static labels remain eligible; `pilot_letters` does not filter model outputs.
Qualified evidence accumulates only between consecutive visible, above-threshold predictions of the same candidate. A different prediction or visible low confidence resets the candidate. A missing-hand interval of at most grace preserves the candidate but contributes no evidence, including intervals adjacent to missing samples. Longer gaps discard the candidate. Only continuous observed absence reaching release unlocks repetition of the last accepted letter. A different stable letter may be accepted without release. Duration is timestamp-based, not missing-frame-count-based.
After acceptance the same letter remains locked even during visible low confidence. Space appends a space, Backspace (key code 8 or 127) deletes one character, and C clears text in recognition mode. Edits cancel a pending candidate but preserve the physical repeat lock and historical acceptance count. Recording mode retains Space start/save and does not route keys to the text builder.

### Software timing results — not a camera trial

Synthetic samples are 50ms apart and use default.yaml. [Reproducible report](../results/word_builder_simulation.json):

| Scenario | Text | Accepted letters | Mean acceptance time (ms) |
| --- | --- | ---: | ---: |
| long_L_once | `L` | 1 | 700 |
| switching_and_low_confidence | `(empty)` | 0 | n/a |
| two_missing_frames_no_repeat | `L` | 1 | 850 |
| long_release_allows_LL | `LL` | 2 | 700 |
| HELLO | `HELLO` | 5 | 700 |

Across these synthetic scenarios: **9 accepted letters**, mean **716.7ms**, median **700.0ms**. The short-gap candidate has 700ms qualified evidence but 850ms acceptance time. HELLO accepts five letters with mean/median 700ms.

**Letter acceptance time** is elapsed time between the first qualifying prediction of the surviving candidate and the acceptance timestamp. It can include a preserved grace gap; stability evidence cannot. The count covers acceptance events, including subsequently deleted characters, and excludes typed spaces. No accepted letters yields null mean/median. This is not time from physical user-motion onset, which needs an independent timing reference.

Checks: **21 tests passed**, including mean probabilities versus majority vote, short/long gaps, weak confidence, interrupted absence, HELLO, editing, mock preview/input separation, and preservation of recording shortcuts. Ruff and diff whitespace checks passed. Baseline checkpoint and original webcam clips/CSV remained byte-identical.

The live GUI was **not** opened for a real camera trial. The camera-loop tests used fake camera frames and keyboard events; the 20/30 result is offline clip classification, not live text accuracy. No real-camera letter acceptance timing is claimed.

Run:

```bash
.venv/bin/python scripts/evaluate.py --output-prefix results/baseline_recheck
.venv/bin/python scripts/simulate_word_builder.py
.venv/bin/python -m fingerspelling.app --model models/mlp_static.pt --hand left
```

On exit, recognition prints acceptance count/mean/median and saves events, final text, settings and timing definitions to a unique `results/live_sessions/*.json`. Use `--session-log PATH` to select another log location. The full text is logged; the preview displays its last 40 characters.
