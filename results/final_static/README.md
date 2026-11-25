# Final static MLP development experiment — 2026-11-15

New weights **adopted**: selected epoch 4 of 12 executed epochs. Combined development macro accuracy strictly improves from 92.1212% to 95.4545%. Epoch 0 was eligible; a lower NLL at equal baseline accuracy could not authorize new weights. Among accuracy-improving candidates, lower macro clip NLL breaks accuracy ties. Patience is eight epochs since the last accepted checkpoint.

## Data audit and split

| Session | Role | Clips | Valid / total frames | Actual class counts |
|---|---|---:|---:|---|
| s03 | development selection | 40 | 1603/1603 | A:6, E:5, I:4, L:5, M:5, N:5, S:5, T:5 |
| s04 | training | 29 | 1203/1203 | E:6, M:11, N:7, S:5 |
| s05 | development selection | 122 | 4149/4149 | A:5, B:5, C:5, D:5, E:5, F:5, G:5, H:6, I:5, K:5, L:5, M:5, N:5, O:5, P:5, Q:5, R:5, S:5, T:6, U:5, V:5, W:5, X:5, Y:5 |
| s06 | training | 45 | 2658/2658 | D:5, K:5, N:5, O:5, P:5, Q:5, T:5, V:5, W:5 |

All camera clips use documented physical left hand and 640×480 dimensions. Features v2 and the source checkpoint order ABCDEFGHIKLMNOPQRSTUVWXY are preserved. s06 covers nine letters (D K N O P Q T V W); camera training together covers 12 letters (D E K M N O P Q S T V W), while Kaggle training and combined development cover all 24.

Kaggle source: grassknoted/asl-alphabet, local landmarks and dimension cache. The original data/cache/static_split.json is unchanged: 44,293 training images and 11,075 validation images. Split paths and decoded-pixel hashes are disjoint. Train/development camera IDs, complete valid landmark sequences and complete v2 feature sequences are disjoint. These exact-duplicate checks do not guarantee absence of similar performances or independence across recordings of one signer.

The saved s06_before report and epoch-zero inference agree on all 45 clip predictions. s06 is 45/45 before training and remains 45/45 after, using all 2,658 valid frames.

## Optimization and balancing

Same starting hyperparameters as the previous adaptation: seed 42, AdamW lr=0.0001, weight decay=0.0001, batch size 256, 100 steps/epoch, maximum 40 epochs, patience 8, dropout 0.2 and no added augmentation. CPU, four PyTorch threads, deterministic algorithms.

Each batch draws exactly 128 Kaggle and 128 camera frames with replacement. Kaggle: class uniformly then image uniformly. Camera: one of 12 observed classes uniformly, then clip uniformly within that class across s04+s06, then valid frame uniformly within that clip. Frame probability is 1/(number of camera classes × clips in that class × valid frames in that clip). This controls long-clip contribution; sessions are not separately balanced. Actual clip draw counts are saved in report.json.

Combined selection computes one mean-softmax prediction per clip across all 24 output classes, then averages the 24 per-letter accuracies. Macro clip NLL averages clip NLL within each letter, then averages the 24 letters. s03 alone covers only eight letters; its reported macro uses observed letters, but it is not the selection metric. Kaggle validation is monitored before/after, not used to select epochs.

## Before / after

| Dataset | Before correct / total | After correct / total | Before macro | After macro |
|---|---:|---:|---:|---:|
| s03 | 40/40 | 40/40 | 100.000000% | 100.000000% |
| s05 | 112/122 | 116/122 | 91.805556% | 95.138889% |
| s06 | 45/45 | 45/45 | 100.000000% | 100.000000% |
| combined | 152/162 | 156/162 | 92.121212% | 95.454545% |
| camera_training | 74/74 | 74/74 | 100.000000% | 100.000000% |
| kaggle_validation | 10955/11075 | 10952/11075 | 98.782593% | 98.774093% |
| kaggle_training | 44107/44293 | 44090/44293 | 99.481239% | 99.463286% |

Combined macro clip NLL: 0.49433753 → 0.36417562.

Kaggle validation loses three correct images overall. This tradeoff is reported rather than hidden; the specified selection criterion is combined camera development macro accuracy.

## All 24 letters: development correct / total

| Letter | s03 before → after / total | s05 before → after / total | Combined before → after / total | Kaggle validation before → after / total |
|---|---:|---:|---:|---:|
| A | 6 → 6 / 6 | 5 → 5 / 5 | 11 → 11 / 11 | 431 → 431 / 436 |
| B | 0 → 0 / 0 | 5 → 5 / 5 | 5 → 5 / 5 | 438 → 438 / 440 |
| C | 0 → 0 / 0 | 5 → 5 / 5 | 5 → 5 / 5 | 391 → 391 / 392 |
| D | 0 → 0 / 0 | 5 → 5 / 5 | 5 → 5 / 5 | 487 → 486 / 491 |
| E | 5 → 5 / 5 | 5 → 5 / 5 | 10 → 10 / 10 | 457 → 457 / 461 |
| F | 0 → 0 / 0 | 5 → 5 / 5 | 5 → 5 / 5 | 569 → 569 / 573 |
| G | 0 → 0 / 0 | 5 → 5 / 5 | 5 → 5 / 5 | 486 → 486 / 486 |
| H | 0 → 0 / 0 | 6 → 6 / 6 | 6 → 6 / 6 | 472 → 472 / 475 |
| I | 4 → 4 / 4 | 5 → 5 / 5 | 9 → 9 / 9 | 471 → 471 / 476 |
| K | 0 → 0 / 0 | 1 → 1 / 5 | 1 → 1 / 5 | 534 → 533 / 540 |
| L | 5 → 5 / 5 | 5 → 5 / 5 | 10 → 10 / 10 | 502 → 502 / 504 |
| M | 5 → 5 / 5 | 5 → 5 / 5 | 10 → 10 / 10 | 311 → 308 / 321 |
| N | 5 → 5 / 5 | 5 → 5 / 5 | 10 → 10 / 10 | 238 → 242 / 255 |
| O | 0 → 0 / 0 | 4 → 5 / 5 | 4 → 5 / 5 | 449 → 450 / 452 |
| P | 0 → 0 / 0 | 5 → 5 / 5 | 5 → 5 / 5 | 404 → 403 / 408 |
| Q | 0 → 0 / 0 | 2 → 4 / 5 | 2 → 4 / 5 | 421 → 420 / 424 |
| R | 0 → 0 / 0 | 5 → 5 / 5 | 5 → 5 / 5 | 503 → 503 / 509 |
| S | 5 → 5 / 5 | 5 → 5 / 5 | 10 → 10 / 10 | 507 → 507 / 511 |
| T | 5 → 5 / 5 | 5 → 5 / 6 | 10 → 10 / 11 | 466 → 468 / 471 |
| U | 0 → 0 / 0 | 5 → 5 / 5 | 5 → 5 / 5 | 492 → 491 / 502 |
| V | 0 → 0 / 0 | 4 → 5 / 5 | 4 → 5 / 5 | 504 → 505 / 510 |
| W | 0 → 0 / 0 | 5 → 5 / 5 | 5 → 5 / 5 | 485 → 482 / 491 |
| X | 0 → 0 / 0 | 5 → 5 / 5 | 5 → 5 / 5 | 423 → 423 / 431 |
| Y | 0 → 0 / 0 | 5 → 5 / 5 | 5 → 5 / 5 | 514 → 514 / 516 |

## Improved clips; no regressions

| clip_id | Session | True | Before | After |
|---|---|---|---|---|
| 6d2942fb61e04a45b48f1e99f66c4cfe | s05 | Q | P | Q |
| 8d9437260e93427ba086d98aa246cde5 | s05 | Q | P | Q |
| b6e8dcbf36744f5bb56f109d7200db86 | s05 | V | W | V |
| c5a0576b2c3b49dbbdd5bda9c8446290 | s05 | O | D | O |

Remaining six errors: four K → V, one Q → P and one T → N. There are 152 clips correct in both models, four improved, six wrong in both and no regressions.

## Training history (separate from validation)

Sampled training loss includes dropout on balanced sampled batches, so it is not directly comparable with deterministic full-dataset evaluation losses. Camera training remains 74/74. Kaggle training changes from 44,107/44,293 to 44,090/44,293; these are training metrics, not held-out results.

| Epoch | Sampled training CE | Development correct / 162 | Development macro accuracy | Development macro NLL | Accepted |
|---|---:|---:|---:|---:|---|
| 0 | — | 152 | 92.121212% | 0.494338 | baseline |
| 1 | 0.037492 | 156 | 95.454545% | 0.424004 | True |
| 2 | 0.027342 | 155 | 94.621212% | 0.399124 | False |
| 3 | 0.024035 | 154 | 93.787879% | 0.396568 | False |
| 4 | 0.021927 | 156 | 95.454545% | 0.364176 | True |
| 5 | 0.019136 | 155 | 94.621212% | 0.397405 | False |
| 6 | 0.018682 | 156 | 95.454545% | 0.401420 | False |
| 7 | 0.019596 | 156 | 95.454545% | 0.392331 | False |
| 8 | 0.018849 | 154 | 93.787879% | 0.417958 | False |
| 9 | 0.017793 | 156 | 95.454545% | 0.395577 | False |
| 10 | 0.022280 | 155 | 95.037879% | 0.419410 | False |
| 11 | 0.018680 | 156 | 95.454545% | 0.381084 | False |
| 12 | 0.017610 | 156 | 95.454545% | 0.413850 | False |

## Artifacts and provenance

- Separate before/after JSON: s03, s05, combined, s06, camera_training, kaggle_training and kaggle_validation.
- Each camera group has comparison, improved, regressed and per-letter CSV files. Empty regression CSVs retain headers.
- Kaggle validation/training have all-24-letter before/after CSV tables.
- manifest.json contains exact clip IDs, source hashes, class/frame counts, settings and overlap checks; report.json contains scores, history and actual sampling counts.
- Source: models/mlp_static_adapted.pt; output: models/mlp_static_final.pt.

Source checkpoint SHA-256: d38c640d8b1bd92f98761a153e596212225600d3b6901810ca655760664cd2ef

Output checkpoint SHA-256: 7821db6c17a04ba0be940fbca10a85fcb40db41b3aaa1dff1b33fdbaec08ce0a

Source tensor-state SHA-256: 8c29b31d6f290bf912d32995ab266b1d007ea02489eef3c4487ec930832738d5

Output tensor-state SHA-256: 0b68ad1a041bb53a8107629c43e65016bee0d42b5a8df20e44621f7700d2391f

Original Kaggle split SHA-256: 6980a06a014fa221b20ed1aaceb9eee0abd9af42efa9f53429815647a79c95cd

Run from project root with .venv/bin/python scripts/final_static.py. The script refuses to overwrite an existing final checkpoint or results directory. Source checkpoints and camera data are unchanged.

**Interpretation:** s03 and s05 were used for model selection, and s06 entered training. This is one-signer development performance, not an independent final test. No new manual recording or live-camera trial was performed.

## Verification

48 tests passed, including strict epoch-zero accuracy gating, NLL tie-breaking,
mean-frame-probability aggregation and macro-versus-micro scoring. Ruff and
git diff --check passed. Reloaded the saved checkpoint through StaticPredictor;
tensor-state, checkpoint, manifest and all protected source hashes matched.
The saved s06 pre-training report matched source checkpoint, feature version,
clip targets, predictions and 2,658/2,658 valid-frame count.

Live command (not executed as part of this offline experiment):

    .venv/bin/python -m fingerspelling.app --model models/mlp_static_final.pt --hand left
