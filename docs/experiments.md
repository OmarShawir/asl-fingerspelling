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

## 2026-11-07 — M → N diagnosis with the frozen baseline

Checkpoint SHA-256: `f26200e9da636868ed8ae31ab958fcbe76f3edfe64b288b59c2792084efd06d6`. No weights, preprocessing, extraction settings or recordings were changed. Existing `results/s02_baseline.*` files were left untouched.

Reproduce: `MPLCONFIGDIR=/tmp/asl-mpl .venv/bin/python scripts/diagnose_mn.py`. Full numerical results, point coordinates, selection seed and counterfactuals: diagnostic JSON (local-only artifact: `../results/mn_diagnostic_2026-11-07/report.json`).

### Official test images through the actual training extractor

Found both files under `data/external/asl-alphabet/asl_alphabet_test/asl_alphabet_test/`. Both are **200 × 200** pixels. The diagnostic imports and calls `init_worker`/`detect` from `scripts/extract_kaggle.py` with the training IMAGE-mode settings: one hand, detection=0.5, presence=0.5. Predictions, where points exist, go through `StaticPredictor.probabilities` and the unchanged `prepare_features`, using all 24 classes.

| Image | True label | Baseline prediction | Top 3 | Handedness used |
| --- | --- | --- | --- | --- |
| M_test.jpg | M (dataset label) | No hand detected | Unavailable: MLP not run | Unavailable |
| N_test.jpg | N (dataset label) | No hand detected | Unavailable: MLP not run | Unavailable |

Saved M_test overlay (local-only artifact: `../results/mn_diagnostic_2026-11-07/M_test_landmarks.png`) and N_test overlay (local-only artifact: `../results/mn_diagnostic_2026-11-07/N_test_landmarks.png`). They explicitly mark detection failure: no landmarks or numbers were fabricated.

### Explicit diagnostic-only threshold/scale variants

Original images and 2×/4× resized images all fail at thresholds 0.5. Separately tested both thresholds at 0.25 and 0.1, with scales 1×/2×/4×. M becomes detectable; N remains undetected in all nine tested combinations. These variants are not baseline results and were not written to config or the training cache.

| Diagnostic example | Prediction | Top 3 mean probabilities | Predicted hand used | Hand score |
| --- | --- | --- | --- | ---: |
| M_test, thresholds 0.25, original size | M | M 99.1773%, N 0.8200%, H 0.0008% | right | 68.06% |

Numbered M_test diagnostic overlay (local-only artifact: `../results/mn_diagnostic_2026-11-07/M_test_relaxed_landmarks.png`). Its high classifier probability does not verify anatomical landmark correctness. At thresholds 0.25, the handedness prediction changes from right at 1×/2× to left at 4×, demonstrating extraction sensitivity. The actual physical hand in the dataset is not independently documented/verified here.

### Multiple training-folder examples

Seed 20261107 selects six images per letter from successful cached detections. This is a conditional diagnostic sample, not a coverage/accuracy benchmark. Fresh extraction matched cached coordinates exactly for all twelve examples (maximum absolute difference 0.0). All selected images are 200 × 200 and use predicted handedness right.

| File | Dataset label | Prediction | Top 3 probabilities | Hand used |
| --- | --- | --- | --- | --- |
| M304.jpg | M | M | M 99.79%, N 0.21%, I 0.00% | right |
| M1384.jpg | M | M | M 91.10%, N 8.90%, A 0.00% | right |
| M141.jpg | M | M | M 99.91%, N 0.09%, E 0.00% | right |
| M1525.jpg | M | M | M 87.85%, N 12.15%, I 0.00% | right |
| M2140.jpg | M | M | M 80.95%, N 19.05%, A 0.00% | right |
| M467.jpg | M | M | M 99.83%, N 0.14%, X 0.02% | right |
| N237.jpg | N | N | N 99.91%, M 0.09%, I 0.00% | right |
| N2986.jpg | N | N | N 100.00%, M 0.00%, A 0.00% | right |
| N608.jpg | N | N | N 98.95%, M 1.05%, I 0.00% | right |
| N578.jpg | N | N | N 99.91%, M 0.09%, S 0.00% | right |
| N1659.jpg | N | M | M 60.09%, N 39.91%, A 0.00% | right |
| N1106.jpg | N | N | N 99.09%, M 0.91%, O 0.00% | right |

Training images with numbered landmarks (local-only artifact: `../results/mn_diagnostic_2026-11-07/training_examples.png`). Individual full-size overlays are saved beside the montage. This sample predicts 6/6 M and 5/6 N correctly; N1659 is predicted M.

Existing full extraction statistics remain M **1,603/3,000 (53.43%)** and N **1,275/3,000 (42.50%)**. The coverage rates are not classifier accuracy.

### p01/s02, date-filtered camera comparison

Filter `created_at` into **Asia/Kuala_Lumpur**, date 2026-11-07, person p01, session s02, kind letter, source webcam. There are **36 clips**, 1,494 frames, 1,490 valid finite frames on this date. Five older s02 clips are excluded, explaining why the pre-existing session-wide report contains 41 clips. The M/N subset contains **8 M + 8 N**, all labeled physical left.
Clip predictions use the original average of valid-frame softmax probabilities, one argmax over all 24 letters. **M: 0/8 correct (all N); N: 8/8 correct.** [Date-filtered clip CSV](../results/mn_diagnostic_2026-11-07/clip_predictions.csv).

Camera landmark panels (local-only artifact: `../results/mn_diagnostic_2026-11-07/camera_landmarks.png`) show one actual frame per M/N clip, selected as nearest to that clip’s median feature vector. The JSON records exact frame indices and timestamps. These are landmark-only plots: RGB frames were not saved, so we cannot inspect the true camera wrist or finger locations against the image.

### Wrist, ring finger and thumb

Landmark IDs: wrist **0**, thumb **1–4** (tip 4), ring finger **13–16** (MCP 13, tip 16). Both extraction paths preserve the MediaPipe landmark order; no index swap was found. See the [official landmark map and coordinate definition](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker?tab=Python).

Geometry summary (local-only artifact: `../results/mn_diagnostic_2026-11-07/geometry_summary.png`) shows coordinate medians, explicitly labeled as summaries rather than real hands. Numeric geometry below uses all successful cached M/N images; camera values are medians of per-clip medians, so long clips do not dominate.

| Feature-space metric (median) | Kaggle M | Camera M | Kaggle N | Camera N |
| --- | ---: | ---: | ---: | ---: |
| Ring chord / joint-chain length (13→16) | 0.963 | 0.999 | 0.942 | 0.430 |
| Thumb tip 4 to ring MCP 13 | 0.691 | 0.356 | 0.809 | 0.399 |
| Thumb tip 4 to ring tip 16 | 0.426 | 0.391 | 0.484 | 0.222 |
| Ring tip 16 to wrist 0 | 0.773 | 0.958 | 0.585 | 0.478 |
| Middle MCP 9 y relative to wrist 0 | -0.381 | 0.280 | -0.427 | 0.227 |

Observed differences:
- Camera M landmarks depict an almost straight ring chain; camera N depicts a much more folded ring chain. Cached Kaggle N commonly has a far straighter estimated ring chain than camera N. These are estimated landmark geometry measurements, not verified anatomical joint angles.
- The middle MCP is above the estimated wrist in the median Kaggle examples (negative y offset), but below it in the camera examples (positive offset). X reflection cannot change this y-direction difference. It may involve pose/viewpoint, wrist localization, or both; the camera RGB needed to distinguish them is unavailable.
- Visual inspection of numbered Kaggle overlays (especially M1384 and N1106) makes estimated point 0 appear too far inside the hand/near the knuckles rather than at the forearm junction. This is a localization concern visible in individual examples, not a quantitative ground-truth landmark error measurement.
- Thumb-to-ring distances also shift substantially across sources. Missing/occluded thumb detail and inferred depth may contribute. Current data cannot establish a single anatomical cause of M→N.
- Wrist subtraction sets point 0 to zero by design. An inaccurate wrist estimate shifts all other relative coordinates and changes the maximum-radius scale; centering does not repair a misplaced wrist.

### Preprocessing consistency and confirmed aspect-ratio limitation

Training reads BGR, converts to RGB, extracts `(p.x,p.y,p.z)` in IMAGE mode and uses MediaPipe-predicted handedness. Live capture performs the same color conversion/order in VIDEO mode and uses explicit `--hand left`. Recorded evaluation uses metadata left. All three call the same feature function: subtract wrist → reflect x for left → maximum 3D radius normalization → flatten. `cv2.flip(frame,1)` only mirrors the preview; it does not reflect the landmark input a second time. Physical dataset handedness remains unknown, and image handedness is only a tracker prediction.

**Confirmed feature limitation:** normalized x uses image width and normalized y uses image height. The current Euclidean scale mixes these axes without aspect compensation. A single scalar normalization cannot undo their unequal scaling. In an isolated planar (z=0) test, identical pixel geometry encoded in a 200×200 versus 400×200 canvas produced a maximum feature difference of **0.259893**. Thus feature geometry is not invariant to canvas aspect ratio, even before considering changes in detection. Changing resolution at the same aspect ratio does not introduce this particular mathematical distortion, though it may change detector output.

**Confirmed metadata gap:** neither NPZ metadata nor metadata.csv stores original frame width/height. The historical camera resolution is unknown; opening the camera now would not prove the dimensions of these recordings.

Counterfactual sensitivity only: apply y *= H/W before the existing feature function, holding x/z fixed, under assumed square, 4:3 and 16:9 frames. These are hypothetical width-unit conversions, not measured resolutions or a validated fix.

| Assumed H/W | Correct M clips | Correct N clips |
| --- | ---: | ---: |
| 1.0 | 0/8 | 8/8 |
| 0.75 | 1/8 | 8/8 |
| 0.5625 | 3/8 | 8/8 |

Using the deliberately wrong hand=right instead of recorded left yields only 1/8 correct M (many become P), while N remains 8/8. That counterfactual does not justify changing physical-hand labels or removing x reflection.

### Conclusions and next evidence needed

Confirmed: baseline detection fails on the two official test images; M→N occurs in all eight dated M clips; code paths share point order and preprocessing; feature geometry depends on aspect ratio; historical image dimensions and camera RGB are missing. No checkpoint corruption or double-X-flip bug was found.
Not established: whether the camera’s M pose, wrist localization, hidden-thumb inference, handedness prediction, IMAGE/VIDEO behavior or aspect ratio is the dominant cause. The hypothetical aspect adjustments help only part of this small inspected set and are not held-out validation.
Next experiment: retain the baseline, capture matched RGB + landmarks + width/height for M/N with known physical hand, review wrist/thumb/ring annotations, compare IMAGE and VIDEO extraction on the same frames, and evaluate an explicitly versioned aspect-aware feature pipeline on a fresh held-out session. Any retraining or feature change should create a new checkpoint; no fix was applied during this diagnosis.

## 2026-11-07 — Aspect-aware features v2 with frozen weights

Implemented a controlled same-weight comparison. No retraining was performed. No historical frame dimensions were guessed or backfilled. No original checkpoint, recording or metadata CSV was altered during this experiment.

### Recording and diagnostic evidence

New camera takes record `frame_width` and `frame_height` from the actual BGR `frame.shape[:2]` passed to MediaPipe, with `dimensions_source=mediapipe_input_frame.shape`. Every frame in a take must have the same dimensions; a resolution change discards the take with a visible notice. Metadata is stored in NPZ `meta_json` and included in rebuilt CSV rows. Old clips remain readable and have blank dimension columns when reindexed.
`--save-diagnostic-frames` enables original PNG snapshots before landmark drawing and preview mirroring. `--diagnostic-every N` samples every Nth recorded frame (default 10, use 1 for every frame). The clip’s `diagnostic_manifest` points to `data/diagnostic_frames/<clip_id>/manifest.json`, which maps each PNG to its zero-based frame index, clip-relative `t_ms`, monotonic capture timestamp and dimensions. These images/manifests are git-ignored. Abandoned takes do not save diagnostic PNGs.

### Versioned preprocessing

v1: `wrist-maxradius-canonical-right-v1`, original calculations unchanged.
v2: `width-units-wrist-maxradius-canonical-right-v2`: convert `(x,y,z)` to `(x,y*frame_height/frame_width,z)` first, then subtract the wrist, reflect x for physical left, apply the same maximum-radius scale and flatten. Width/height must be documented positive integers. Missing dimensions fail explicitly, including in evaluation; the code never silently substitutes 640×480, skips undocumented clips, or falls back to v1.
`StaticPredictor` selects preprocessing from checkpoint `feature_version`. Live inference forwards actual frame dimensions; clip evaluation forwards recorded metadata; training v2 loads image dimensions verified against cached RGB hashes. v1 continues to evaluate historical clips without dimensions.

### Actual Kaggle audit and checkpoint derivation

`scripts/derive_aspect_v2.py` decoded **72,000 source images**, checked their RGB hashes against the landmark cache, and found **all 72,000 are 200×200**. It also matched the original split-manifest hash: **44,293 training + 11,075 validation** detected images. Comparing v1/v2 for **55,368 detected landmark sets**, with their cached handedness, gave **maximum absolute feature difference 0.0**.
Consequently the first comparison preserves every weight tensor bit-for-bit. Only preprocessing version and derivation metadata differ. This isolates camera aspect correction rather than retraining randomness. Tensor identity was checked after saving and reloading.

| Artifact | Value |
| --- | --- |
| Original checkpoint | `models/mlp_static.pt` |
| Original file SHA-256 | `f26200e9da636868ed8ae31ab958fcbe76f3edfe64b288b59c2792084efd06d6` |
| Derived checkpoint | `models/mlp_static_aspect_v2.pt` |
| Derived file SHA-256 | `8f514f6180a1c5ac65c29483201f91365d76fab67d410263a1efc2c9f35423e9` |
| Shared tensor-content SHA-256 | `db33387a03213bd9c462eb571d4c852539b7726e2553101ca79c5b0b39facc4e` |

[Derivation report](../results/aspect_v2_derivation.json) specifies the hash serialization and audited split/dimension manifests. The checkpoint includes the source path/hash, feature version and tensor hash. The full verified dimensions map is cached locally in `data/cache/kaggle_dimensions.json`. Existing checkpoints are never overwritten by the derivation command.

### Verification results

- **39 tests passed**, Ruff passed, and whitespace checks passed.
- The same synthetic pixel-space 3D hand encoded in 640×480, 1280×720 and 800×800 frames gives matching v2 features (absolute tolerance 2e-7) for both physical hands. v1 differs as expected.
- Square-frame v1/v2 features are bit-identical; inputs remain unmodified.
- Missing/zero/negative/non-integer dimensions are rejected by v2. v1 checkpoint loading without dimensions remains valid.
- Camera-loop tests verify exact dimension propagation, original unmirrored PNG pixel contents before drawing, clip/frame/time associations, and rejection of a mid-take resolution change.
- A subprocess integration test compares two identical-weight, differently versioned models on the exact same synthetic clips across all 24 classes. It also verifies that an undocumented legacy clip rejects the comparison. This is a software test, not a camera accuracy result.
- Real historical v1 re-evaluation: **p01/s01 = 20/30**, **p01/s02 = 27/41**. Every clip prediction and score matches its existing report exactly. See `results/aspect_v1_s01_recheck.*` and `results/aspect_v1_s02_recheck.*`.
- Running v2 on historical s01 correctly fails on missing `frame_width` and writes no evaluation report. Those recordings cannot support the measured aspect comparison.

**No new physical camera captures were made, so there is no measured v2 camera accuracy yet.** Do not substitute the earlier hypothetical aspect sensitivity numbers for this controlled experiment.

### Matched capture and evaluation commands

Use a fresh session (s03 below), keeping the physical hand label accurate. Record several separate takes per letter; Space starts/saves, Q advances to the next letter. `--diagnostic-every 1` saves every recorded frame for short diagnostic takes.

```bash
for letter in A B C D E F G H I K L M N O P Q R S T U V W X Y; do
  .venv/bin/python scripts/collect.py \
    --person p01 --session s03 --hand left --kind letter --text "$letter" \
    --save-diagnostic-frames --diagnostic-every 1 || break
done
```

Close the collector, then run one paired command. It checks identical weights, all 24 ordered classes, known dimensions, and identical clip IDs/content hashes before producing a comparison.

```bash
.venv/bin/python scripts/compare_aspect.py \
  --person p01 --session s03 \
  --v1 models/mlp_static.pt \
  --v2 models/mlp_static_aspect_v2.pt \
  --output-prefix results/s03_aspect
```

Outputs: `results/s03_aspect_v1.csv/json`, `results/s03_aspect_v2.csv/json`, and paired summary `results/s03_aspect.json`. Both models score all 24 classes and use the same mean-valid-frame-probability aggregation.

For live v2:

```bash
.venv/bin/python -m fingerspelling.app --model models/mlp_static_aspect_v2.pt --hand left
```

Future retraining (separate experiment, not performed here) explicitly selects a new checkpoint path and preprocessing version:

```bash
.venv/bin/python scripts/train_static.py --feature-version v2 \
  --output models/mlp_static_v2_retrained.pt --result-prefix results/static_v2_retrained
```

The training command now requires `--output` and refuses existing files to protect the fixed baseline.

## 2026-11-15 — Source/clip-balanced v2 MLP fine-tuning

Initialized all MLP layers from `models/mlp_static_aspect_v2.pt`. Preserved features v2 and the exact ordered 24 checkpoint labels. Saved a separate `models/mlp_static_adapted.pt`; the source checkpoint remains unchanged.

### Data and split audit

Kaggle: reused the original frozen split manifest, **44,293 train / 11,075 validation**. Verified its SHA-256 against the source checkpoint, each path/label/pixel hash against the landmark cache, and dimensions against the verified image manifest. Train/validation path and decoded-pixel hash intersections are empty.
Camera training: **p01/s04, 29 clips / 1,203 valid frames** (E=6, M=11, N=7, S=5 clips). Clip lengths range from 22 to 77 valid frames. Camera validation: **p01/s03, 40 clips / 1,603 valid frames**, covering A/E/I/L/M/N/S/T. Both sessions record physical left hand and **640×480** frames.
All s04/s03 clip IDs are disjoint. Valid-landmark-sequence hashes are also disjoint. Dimensions are required and no missing values were inferred. Data hashes were checked again after training. No s01/s02 data was used. [Manifest and checks](../results/adaptation/manifest.json).

### Sampling and optimization

- Each batch has **128 Kaggle + 128 camera frames**, sampled with replacement. Thus each source contributes exactly 50% of mean cross-entropy.
- Kaggle sampling is uniform over all 24 classes, then uniform over images within the selected class.
- Camera sampling is uniform over the four observed training classes, then uniform over clips within a class, then uniform over valid frames inside a clip. Frame weight is `1 / (number_of_classes × clips_in_class × valid_frames_in_clip)`. A 77-frame clip receives the same expected total mass as a 22-frame clip of the same class. M has more clips, but not more total class mass.
- Each camera class therefore receives 12.5% of the full batch in expectation; every Kaggle class receives about 2.083%. Classes absent from s04 continue to receive Kaggle samples.
- AdamW, learning rate **1e-4**, weight decay **1e-4**, batch **256**, seed **42**, 100 steps per epoch, maximum 40 epochs, patience 8. All layers train; existing dropout 0.2 is active during training. No additional augmentation was used. Deterministic PyTorch CPU execution with four threads.
- Each epoch draws 12,800 examples per source. Actual per-clip and Kaggle per-class draw counts are saved in the report. The 27-epoch run drew 345,600 samples per source; the selected epoch-19 weights had seen 243,200 samples per source.

### Model selection

Used **s03 macro clip accuracy over its eight represented letters** as the primary criterion; tied accuracy uses lowest macro clip negative log-likelihood. Clip probabilities are the mean of valid-frame softmax probabilities, with one argmax over all 24 classes. The unmodified epoch-0 baseline was eligible to avoid replacing it with a worse candidate. Kaggle validation was measured before/after for monitoring, not used to select an epoch.
Best epoch **19**; stopped at **27** after eight epochs without a better selection key. The saved checkpoint was reloaded through the ordinary `scripts/evaluate.py` path and confirmed the reported predictions. s04 training performance is **29/29**, reported only as training fit.

### s03: same 40 clips before and after

Baseline **35/40 (87.5%)**, adapted **40/40 (100%)**. **Five clips improved; zero regressed.** The other sixteen model classes have no s03 target examples, so their camera validation accuracy is unmeasured, not zero.

| Letter | Baseline correct / total | Adapted correct / total |
| --- | ---: | ---: |
| A | 6/6 | 6/6 |
| E | 5/5 | 5/5 |
| I | 4/4 | 4/4 |
| L | 5/5 | 5/5 |
| M | 0/5 | 5/5 |
| N | 5/5 | 5/5 |
| S | 5/5 | 5/5 |
| T | 5/5 | 5/5 |

Improved clips (all true M, previously predicted N):

| Clip ID | Before | After |
| --- | --- | --- |
| `0cebcc54656a48269b331d1d46271c47` | N | M |
| `127def6c23174b718a9c842a256be7a2` | N | M |
| `3faea694875448bdb009efe017f248c8` | N | M |
| `6bc2c2dfdd654a1aa45d073513624cc0` | N | M |
| `a2961067daf546b6841e2cdd2a271929` | N | M |

No regressed clips. [All paired clip decisions](../results/adaptation/s03_comparison.csv), [top-three scores and learning history](../results/adaptation/report.json).

### Kaggle validation: all 24 letters

Baseline **10,961/11,075 (98.9707%)**, adapted **10,955/11,075 (98.9165%)**: six fewer correct images, a decrease of about **0.0542 percentage points**. Cross-entropy increased from **0.04249 to 0.04588**. Macro accuracy changed from **98.8702% to 98.7826%**.

| Letter | Baseline correct / total | Adapted correct / total |
| --- | ---: | ---: |
| A | 430/436 | 431/436 |
| B | 438/440 | 438/440 |
| C | 391/392 | 391/392 |
| D | 489/491 | 487/491 |
| E | 457/461 | 457/461 |
| F | 570/573 | 569/573 |
| G | 486/486 | 486/486 |
| H | 471/475 | 472/475 |
| I | 471/476 | 471/476 |
| K | 535/540 | 534/540 |
| L | 502/504 | 502/504 |
| M | 308/321 | 311/321 |
| N | 245/255 | 238/255 |
| O | 449/452 | 449/452 |
| P | 402/408 | 404/408 |
| Q | 423/424 | 421/424 |
| R | 503/509 | 503/509 |
| S | 508/511 | 507/511 |
| T | 466/471 | 466/471 |
| U | 492/502 | 492/502 |
| V | 505/510 | 504/510 |
| W | 483/491 | 485/491 |
| X | 423/431 | 423/431 |
| Y | 514/516 | 514/516 |

Notable tradeoff: Kaggle M improved from 308/321 to 311/321, while N declined from 245/255 to 238/255. Retaining high overall Kaggle accuracy does not mean every class was preserved. [Combined per-letter CSV](../results/adaptation/per_letter.csv).

### Checks, provenance and interpretation

**42 tests passed**, Ruff passed, and whitespace checks passed. New tests cover class/clip balancing despite unequal clip lengths, exactly balanced source batch sizes, seeded reproducibility, overlap rejection and inconsistent clip labels. Source weights and data remained unchanged; adapted weights differ while feature version and class order match.
Output SHA-256: `d38c640d8b1bd92f98761a153e596212225600d3b6901810ca655760664cd2ef`. Source SHA-256: `8f514f6180a1c5ac65c29483201f91365d76fab67d410263a1efc2c9f35423e9`. Checkpoint metadata records initialization, configuration and the adaptation manifest hash.

**s03 was used for checkpoint selection, so 40/40 is a validation result, not an untouched test result.** It covers one signer and eight target letters. No new live camera test or independent-session generalization claim was made. A fresh session is needed to evaluate the adapted model without selection reuse.

Reproduce the experiment with a new output path (existing checkpoints are protected):

```bash
.venv/bin/python scripts/finetune_static.py --output models/mlp_static_adapted_repeat.pt --result-dir results/adaptation_repeat
```

Live command:

```bash
.venv/bin/python -m fingerspelling.app --model models/mlp_static_adapted.pt --hand left
```

For before/after re-evaluation, use `scripts/evaluate.py` separately with each checkpoint and the same `--person p01 --session s03`. Do not use `compare_aspect.py` for this adaptation: that tool intentionally requires identical weights, whereas fine-tuning changes them.

## 2026-11-15 — s05 paired local diagnostic (no training)

Matched all 122 unique clip IDs and targets in s05_v2.csv/json and
s05_adapted.csv/json, and verified original NPZ labels, dimensions, physical hand
and frame counts. Total: 4,149 valid frames. H and T each have six clips; every
other static letter has five. No s05 clip ID overlaps the recorded s03 selection
or s04 training IDs.

- Aspect v2: **105/122 (86.07%)**; adapted: **112/122 (91.80%)**.
- Both correct: **104**; improved: **8**; regressed: **1**; both wrong: **9**.
- Improvements: all five M clips and three S clips.
- Only regression: V clip b6e8dcbf36744f5bb56f109d7200db86, V → W.
- Adapted errors: K → V ×4, Q → P ×3, O → D ×1, T → N ×1,
  V → W ×1. K remains 1/5, Q 2/5, O 4/5, T 5/6; V changes 5/5 → 4/5.
- Scores are mean valid-frame probabilities across all 24 classes, not votes.

Reviewed 29 original error PNGs with adjacent numbered saved-landmark overlays,
nine correct camera controls and 18 local Kaggle references. Each diagnostic
image was linked through diagnostic_manifest to the exact NPZ frame_index and
t_ms; dimensions and capture-time origin also agree. No missing manifests or
referenced error PNGs. The first K error has two available images; the other
errors each display three. No tracker rerun was used. Kaggle examples were
sampled from successful cache entries with seed 20261115; cached handedness is
a prediction, not physical-hand ground truth.

Confirmed visual observations: K errors share a two-raised-finger silhouette
with V, with thumb/folded fingers overlapping; Q errors are side-on with
projected folded-finger overlap. The regressed V shows two extended fingers and
a folded ring finger in all three sampled images, unlike the W controls.
The O error shows a loop near the face; the T error and correct T control both
show a lateral thumb beside a compact fist. A local D reference (D2109.jpg)
visibly has a raised finger missed by its cached tip placement, demonstrating
that successful extraction need not imply anatomically accurate landmarks.

Possible explanations, not established causes: thumb occlusion, wrist
orientation, inferred depth and adaptation-induced decision-boundary changes.
Correct controls share some of the same broad poses, so those features alone
cannot explain the errors. These findings do not justify changing any label.
Sparse snapshots do not establish tracking quality for every frame.

Full report with all 24 per-letter counts, changed clip IDs, original/overlay
galleries and observations (local-only artifact: `../results/s05_diagnostic/report.md`).
[Paired CSV](../results/s05_diagnostic/comparison.csv).
Evidence, exact frames, top-three scores and input hashes (local-only artifact: `../results/s05_diagnostic/report.json`).

Reproduce generated evidence from the project root:
MPLCONFIGDIR=/tmp/asl-mpl .venv/bin/python scripts/diagnose_s05.py

This is offline diagnosis of existing reports and saved camera images, not a
new camera trial. Input hashes verify that checkpoints, reports, NPZ clips,
metadata and inspected source images remain unchanged.

## 2026-11-15 — final static adaptation, s04+s06 training / s03+s05 development

Started from models/mlp_static_adapted.pt, retained features v2 and all 24 labels.
s06 contains 45 clips / 2,658 valid frames: D K N O P Q T V W, five each,
all physical left hand at 640×480. Epoch-zero predictions reproduce the saved
45/45 result. s04+s06 training totals 74 clips / 3,861 valid frames across
12 classes. s03+s05 development totals 162 clips / 5,752 valid frames with all
24 classes. The original Kaggle split remains 44,293 train / 11,075 validation.

Used previous AdamW settings (lr and weight decay 0.0001, batch 256,
100 steps/epoch, max 40, patience 8, seed 42, dropout 0.2).
Every batch is 128 Kaggle and 128 camera samples. Camera sampling is uniform
class, then uniform clip, then uniform valid frame, preventing long clips from
dominating; Kaggle is uniform class then image. Train/development IDs, exact
valid landmark sequences and exact feature sequences are disjoint. Kaggle
train/validation paths and pixel hashes are disjoint. Exact duplicate checks
are not a claim of signer independence or absence of near duplicates.

**Adopted epoch 4 after 12 epochs run**, saved models/mlp_static_final.pt.
Epoch 0 participated in selection; new weights required strictly greater
combined 24-class macro clip accuracy. Lower macro clip NLL only broke ties
among improved candidates. Clip probabilities are averaged across valid frames.

| Dataset | Before | After |
|---|---:|---:|
| s03 development | 40/40 | 40/40 |
| s05 development | 112/122 | 116/122 |
| Combined development | 152/162 | 156/162 |
| Combined macro accuracy | 92.1212% | 95.4545% |
| Kaggle validation | 10955/11075 | 10952/11075 |
| Camera training s04+s06 | 74/74 | 74/74 |
| s06 (training) | 45/45 | 45/45 |
| Kaggle training | 44107/44293 | 44090/44293 |

Four s05 clips improved: Q ×2, V ×1, O ×1; no clip regressed.
Six errors remain: K → V ×4, Q → P ×1, T → N ×1.
The three-image Kaggle validation decline is an observed tradeoff; it did not
participate in selection. Full per-letter tables, exact improved clip IDs,
training-loss history, before/after JSON and checkpoint/tensor-state hashes
are in [the final report](../results/final_static/README.md) and
[manifest](../results/final_static/manifest.json).

**s03 and s05 were used for selection; s06 entered training. These are not
independent final-test results.** No new live-camera test was performed.

Validation: **48 tests passed**, Ruff and git diff --check passed.
New tests cover the strict epoch-zero gate, NLL tie-break and mean-probability
macro clip scoring. Saved-checkpoint reload and tensor-state/manifest/source
hash checks passed, including correspondence to the saved s06_before report.

## 2026-11-21 — first own-trained offline GRU + CTC pilot

Used the supplied ChicagoFSWild audit, not the blocked Kaggle route.
Pinned conversion c738cce88fcae8bcdb0811738c44a0b41701ad4d:
414,214,372 Parquet bytes plus 1 MiB of the university archive. Hashes matched.
Raw labels and fixed signer-disjoint pilot IDs remain: 5,433 train /
978 validation / 867 test. Test payload was hashed only; **no test inference**.

Experimental features sequence-unordered-hand-shape-wrist-scale-v1 preserve
missing frames, masks, raw point-0 position and scale alongside centered shape.
No static v2 aspect correction or anatomical mirror flip. 21-point blocks and
point-0 wrist semantics are assumptions requiring verification. Shared hand MLP
67→96→64, symmetric masked mean/max plus count, causal 2-layer GRU (128 hidden),
28-class CTC (a–z, space, blank). 214,876 randomly initialized parameters;
no pretrained recognition or static weights.

Four debug attempts on the original short training subset stalled at 14/16,
CER 4.65%, including a discarded blank-bias initialization experiment.
Failures had leading missing frames; causality is unproven. A fixed debug-only
restriction (hand present in frame 0) gave **16/16, CER 0 after 700 steps**.
All attempts and IDs are preserved. Full pilot membership did not change;
the pilot started independently from scratch.

Adam lr=0.001, batch 32, seed 42, four CPU threads, dropout 0.1,
gradient clipping 5, no augmentation or temporal downsampling.
Bound: 15 epochs, 600-second loop budget, patience 5.
Completed 15 epochs; **selected epoch 14 by validation micro CER**.
Epoch 0 participated. Epoch 15 had more exact transcripts but worse CER.

| Metric | Selected validation result |
|---|---:|
| CER | 2513/4967 = **50.5939%** |
| Exact transcripts | **110/978 = 11.2474%** |
| Empty predictions | **35/978 = 3.5787%** |
| Finite CTC loss | **1.85563429**, denominator **976** |
| J-containing clips | CER **62.50%**, exact **0/15** |
| Z-containing clips | CER **56.6176%**, exact **0/18** |

Short dev clips (emmett: 7<8; all: 3<4) remain in all 978 transcript metrics;
only unalignable CTC losses are excluded. zero_infinity=False.
J/Z figures are whole-transcript metrics, not isolated-letter accuracy.

Training-only CER: 49.3333%;
exact 803/5433.
Successful invocation elapsed 178.87 s; peak RSS
567.73 MiB. These exclude download and earlier
debug invocation setup; they do not measure live latency.

Saved models/sequence/gru_ctc_pilot_v1.pt, SHA-256
6dfa215d59bc7f614e9bf89d442ef274ae1aa1f092733c2ce073fe987c680031.
[Full report and artifacts](../results/sequence/gru_ctc_pilot_v1/README.md).
[All validation predictions](../results/sequence/gru_ctc_pilot_v1/validation_predictions.csv).
Raw unpadded logits are saved locally. Checkpoint reload/inference succeeded.
60 repository tests and 11 audit tests passed; Ruff passed with the immutable
audit snapshot excluded.

**Alphabet-plus-space offline feasibility baseline only.** Validation selected
the model; no test, streaming, live whole-word or delay claim follows. J/Z
reliability is not established. Converter orientation/aspect/tracker/FPS remain
unknown. Static models/mlp_static_final.pt and the current live path remain
intact. A future streaming experiment requires separate camera compatibility
and output-delay measurements.

Final integrity verification: all 318 protected static/recording files remained
byte-identical; all 978 raw-logit decodes match saved transcripts. Trained
hand-block swapping preserves the transcript, with maximum float32 logit
difference 1.43e-6 (atol 2e-6, rtol 1e-5), rather than bitwise equality.
[Learning curves](../results/sequence/gru_ctc_pilot_v1/learning_curves.png).

## 2026-11-21 — GRU + CTC decoder diagnosis and longer budget

Preflight independently reproduced 978 validation clips, 4,967 target characters, 3,703 predicted characters, 2,513 edits, 110 exact and 35 empty predictions. Edit breakdown: 1,395 deletions, 987 substitutions and 131 insertions.

No decoder or vocabulary bug found: blank=0, literal space=27. Synthetic logits correctly decode all 26 letters and space, jazz, all, a b, and leading/trailing and repeated spaces. J/Z/Q/space never occur even in pilot frame argmax; this is a learned prediction limitation, not a strip/collapse bug. Targets contain 16/19/4/193 occurrences respectively. Original decoder/features/model/trainer unchanged.

Edit alignment uses unit-cost Levenshtein DP and traceback from the end, preferring equal diagonal match, then substitution, then deletion, then insertion. The total distance is independent of tie choices, but error categories and per-symbol correct counts can vary. The same rule is used for both models.

One new run from seed-42 random initialization. Only max_epochs 15→60 and loop budget 600→900 seconds changed. Adam lr=0.001, batch=32, dropout=0.1, patience=5, architecture, features, split, no augmentation/balancing/scheduler, gradient clipping and greedy decoding all match the pilot. The same training-only debug protocol precedes a fresh pilot seed reset.

All 16 shared epoch records (0–15) reproduce the original training loss and validation metrics exactly: True. Run ended after 43 epochs by patience=5 early stopping. Selected long epoch 38 by minimum validation CER; pilot was retained as an eligible reference, with ties favoring the pilot.

| Split/model | CER | Exact | Empty | Deletions | Substitutions | Insertions | Output characters |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation/pilot | 50.5939% | 110/978 | 35/978 | 1395 | 987 | 131 | 3703 |
| Validation/selected | 42.7018% | 170/978 | 17/978 | 1072 | 855 | 194 | 4089 |
| Training/pilot | 49.3333% | 803/5433 | 299/5433 | 9047 | 4482 | 567 | 20093 |
| Training/selected | 28.9119% | 2210/5433 | 121/5433 | 5620 | 2217 | 424 | 23377 |

Validation CTC loss: 1.75283041 over 976 alignable clips. Both short clips remain in the 978 transcript-metric denominator; zero_infinity=False.

| Symbol | Validation target | Pilot emitted | Selected emitted | Selected aligned correct |
|---|---:|---:|---:|---:|
| J | 16 | 0 | 4 | 4 |
| Z | 19 | 0 | 3 | 0 |
| Q | 4 | 0 | 0 | 0 |
| space | 193 | 0 | 3 | 3 |

Emission counts alone are not correct recognition. Aligned-correct counts are text matches under the documented rule, not isolated gesture accuracy or temporal alignment ground truth. Training metrics are not held out.

Training invocation elapsed 473.69 s, peak process RSS 622.39 MiB. This elapsed time includes setup/debug/final scoring outside the loop budget; the subsequent diagnostic comparison is a separate process.

Checkpoint: models/sequence/gru_ctc_long_v1.pt; SHA-256 762d11636d38f35ae8590fbef6bfb9edd2687b5f0acdd5c027dba063146973fc.
[Full comparison and diagnostics](../results/sequence/gru_ctc_long_v1/README.md). [Numerical comparison](../results/sequence/gru_ctc_long_v1/comparison.csv). [All character counts](../results/sequence/gru_ctc_long_v1/character_counts.csv).

All 375 protected static/pilot/recording files and source hashes remain unchanged. **No test predictions or test-based selection.** Validation selected this offline model; these results do not establish live recognition or an independent final-test score.

Verification: **63 tests passed**, Ruff and git diff --check passed. Synthetic
logits cover all letters, jazz/all/a b and literal boundary/repeated spaces.
Saved-logit decodes reproduce all 978 predictions; checkpoint reload and protected
hash checks passed.
[Learning and edit-count plot](../results/sequence/gru_ctc_long_v1/learning_diagnostics.png).

Final verification completed on **2026-11-22 (Asia/Kuala_Lumpur)**; the entry
above retains the experiment start date.

## 2026-11-22 — fixed-width CTC decoder selection and experimental camera integration

No retraining. Reference models/sequence/gru_ctc_long_v1.pt and static
models/mlp_static_final.pt remain unchanged. Saved validation logits were checked
against checkpoint/vocabulary metadata, CSV/JSON, ordered clip IDs, fixed
validation membership, frame lengths and finite 28-class arrays. Greedy exactly
reproduces 2,121/4,967 edits, 170/978 exact and 17 empty transcripts.

| Decoder | CER | Exact | Empty | Mean decode ms/clip | p95 ms/clip |
|---|---:|---:|---:|---:|---:|
| Greedy | 42.7018% | 170/978 | 17 | 0.227 | 0.509 |
| Prefix beam 10 | 41.3932% | 177/978 | 13 | 3.541 | 9.786 |

Selected prefix beam 10 by lower validation CER (2,056 edits). All 28 symbols,
blank/nonblank prefix states in log-space; no LM/dictionary/token pruning or
modified probabilities. Greedy would win a CER tie. Exhaustive small-path tests
verify probabilities; repeat/space/blank tests pass. Comparison completed in
13.67 s under the 900-second cap. Decode timings exclude
model/camera/tracker time and are not live latency.

Added --mode sequence to the camera application, preserving default static
operation. Reuses camera/tracker, with up to two unordered hands. R starts and
resets, Enter finalizes/saves, C clears/resets, Q exits; user controls boundaries.
GRU hidden state and selected decoder persist across frames. Missing hands are
zero-mask input timesteps, never a forced blank; no static 700-ms rule.
Exact sequence features are reused without static aspect correction or hand flip.

Saved validation replay (66 and 35 frames) matches full forward text and logits
within atol=2e-5, rtol=1e-5; max difference 4.77e-6. Reset tests prevent state
leakage between words. Attempt logs preserve raw text, null target/accuracy,
model/decoder settings, frame logits/timestamps and start/end. Segment duration
is distinct from per-frame feature/model/decoder compute time.

**Real camera check blocked:** no /dev/video* device is visible; OpenCV cannot
open index 0. Synthetic camera controls and offline replay passed, but no live
accuracy trial occurred. Converter point ordering/coordinate convention,
mirroring/cropping/tracker/FPS remain unverified relative to the webcam.
Array-shape agreement alone is not compatibility evidence.

76 tests passed, Ruff passed. No test inference or new data audit. Weights and
recordings are unchanged. [Decoder results and operating instructions](../results/sequence/ctc_decode_v1/README.md).


## GitHub handoff — 2026-11-25

The runtime decoder configuration moved byte-for-byte to `configs/sequence_decoder.json`;
its checkpoint hash binding is unchanged. The two final runtime checkpoints are
included in Git. Private camera clips/images, raw logits, session logs, failed debug
runs and redundant exports remain local under targeted ignore rules. Public summary
reports and per-clip comparison tables preserve the reported metrics. Historical
implementation hashes identify the code at the time of each experiment, not the
current checkout. The hash-pinned `asl_sequence_audit` snapshot is unchanged.

Static development uses s03/s05 for selection and s06 for training. Sequence decoder
selection uses validation; 41.3932% is CER, not accuracy. No independent test result
or actual camera accuracy is established by this handoff. Exact private-camera
retraining requires the original private recordings and cached split manifest.

Handoff verification: 76 tests passed both in the working tree and in a temporary
projection containing only publishable files, using the existing installed dependencies.
Ruff, `git diff --check`, local Markdown links and the locked dependency manifest
passed. Both included checkpoints loaded and produced finite outputs in that
projection; decoder vocabulary/hash compatibility passed. Checkpoint bytes, the
audit snapshot and the pre-existing Git index were unchanged. A pattern-based
credential scan of publishable files found no candidates; this is not a full
repository-history secret audit. No training, dataset download or camera trial ran.


### Windows handoff follow-up — 2026-11-25

Root README commands now use `uv run --no-sync` after locked installation, with
single-line commands usable in PowerShell and Bash. Native Windows users need
64-bit Python 3.12 and desktop-app camera permission. Sequence training no longer
requires Unix `resource`: unsupported peak-RSS telemetry is null, while Linux
KiB and macOS byte units are handled explicitly. Historical reported metrics and
checkpoint bytes are unchanged. A simulated missing-resource import test checks
the fallback; it does not establish native Windows webcam compatibility.
