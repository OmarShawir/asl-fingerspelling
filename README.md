# ASL fingerspelling

Webcam ASL letter recognition with MediaPipe landmarks and PyTorch. The **static
24-letter MLP is the primary mode**. The **GRU + CTC sequence mode is experimental**:
the user marks each word's start/end, and real-camera accuracy has not been established.

## Install

Use Python **3.12**, [uv](https://docs.astral.sh/uv/), and a desktop with a working
webcam/OpenCV display. Run commands from the repository root. The locked PyTorch
build uses CPU; a GPU is not required. On Windows, use **64-bit Python 3.12**
and native **PowerShell** (no WSL required). Install uv using the linked official
instructions, then open a new terminal. The commands below work in PowerShell
and Bash without activating the environment; run `uv sync` first.

```text
uv sync --locked --group train --group sequence --group dev
```

The two final checkpoints are included in this repository (about 210 KiB and
847 KiB). No dataset is needed for inference. On first camera use, the existing
tracker downloads Google's MediaPipe Hand Landmarker to `models/hand_landmarker.task`;
this requires internet access. See [model provenance](models/README.md) for hashes
and [configuration](configs/default.yaml) for tracker/camera settings.

## Run

Static recognition (use the physical signing hand, `left` or `right`):

```text
uv run --no-sync python -m fingerspelling.app --model models/mlp_static_final.pt --hand left
```

Hold a letter to accept it. Space inserts a space, Backspace deletes, C clears,
and Q exits. A sustained absence releases a letter for repetition. J/Z require
motion and are excluded; there is no trained unknown/background class. Preview
mirroring does not change the model input. In recording mode, Space retains its
start/stop-recording function.

Experimental sequence recognition:

```text
uv run --no-sync python -m fingerspelling.app --mode sequence
```

R starts a new word and resets state; Enter finalizes; C clears/resets; Q exits.
The displayed text is provisional while recording. Runtime settings are in
[configs/sequence_decoder.json](configs/sequence_decoder.json), bound to the
sequence checkpoint hash. This mode uses its own sequence preprocessing, not
static features v2. Add `--camera 1` if needed. Attempt logs stay local under
`results/sequence_live_sessions/`; elapsed word duration is not inference latency.

### Windows camera checklist

Allow camera access for desktop apps in Windows Settings, and close other apps
using the camera. If the default device fails, try `--camera 1`. Use a local desktop
terminal with a GUI, not a headless session. Historical experiment commands may
use `.venv/bin/python`; on Windows substitute `.venv\Scripts\python.exe`, or use
`uv run --no-sync python` as above. Do not paste Bash line-continuation backslashes
into PowerShell.

Windows wheels for the main dependencies are present in the lockfile. The automated
checks ran on Linux; native Windows installation and webcam operation still need
a real Windows smoke test. Sequence training records peak memory as `null` when
the operating system does not provide the Unix `resource` module.

## Measured results and limits

| Model / evaluation | Result | Interpretation |
| --- | --- | --- |
| Final static, s03 | 40/40 clips | Development validation |
| Final static, s05 | 116/122 clips | Development validation |
| Final static, combined s03+s05 | 156/162; macro letter accuracy 95.4545% | Used for model selection |
| Final static, Kaggle validation | 10,952/11,075 images | Image split, not signer-independent |
| Sequence greedy, validation | CER 42.7018%; exact 170/978 | 2,121 edits / 4,967 target characters |
| Sequence beam width 10, validation | **CER 41.3932%**; exact 177/978 | 2,056 edits; 13 empty outputs |

CER is character error rate, **not accuracy**. Static s06 entered training;
s03/s05 selected the final model. The sequence decoder was also selected on
validation. These are not independent final-test results or real-camera sequence
accuracy. The camera integration passed simulated-camera and saved-sequence replay
tests; actual camera testing was unavailable in the execution environment.

See [experiments](docs/experiments.md), [final static evidence](results/final_static/README.md),
and [decoder comparison and domain limitations](results/sequence/ctc_decode_v1/README.md).

## Data and reproduction

Static training uses [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
plus private webcam recordings for adaptation. Obtain/extract the images separately;
[the source/protocol guide](data/KAGGLE_ASL_ALPHABET.md) documents image hashes,
extraction and baseline training commands. Images, caches, recordings and diagnostic
photos are not distributed. The final model uses features v2 with documented frame
width/height. Exact final adaptation requires the original private recordings and
cached split manifest; it cannot be reproduced from this public repository alone.
The included checkpoint allows inference without those recordings. Training and
selection settings remain in [final provenance](results/final_static/manifest.json)
and [the adaptation manifest](results/adaptation/manifest.json).

Evaluate your own dimensioned recordings, keeping one probability-averaged prediction
per clip across all 24 letters (replace the person/session codes):

```text
uv run --no-sync python scripts/evaluate.py --checkpoint models/mlp_static_final.pt --person p02 --session s01 --output-prefix results/local_evaluation
```

[Recording format and collection](data/README.md) describes local data and diagnostic
image handling. Old recordings without documented dimensions cannot use v2.

The sequence experiment uses a pinned **ChicagoFSWild community landmark conversion**,
not the Kaggle competition initially considered. The immutable
[sequence audit snapshot](asl_sequence_audit/README.md) supplies source hashes and
fixed signer-disjoint membership. Its historical “next prompt” is not a current task.
Training uses 5,433 sequences, validation 978, and holds out 867 test sequences.
To fetch the approximately 415 MB source payload and reproduce the training protocol
into new paths (these commands perform downloads/training):

```text
uv run --no-sync python asl_sequence_audit/fetch_data.py --cache-dir data/external/chicagofswild
uv run --no-sync python scripts/train_ctc.py --epochs 60 --seconds 900 --output results/sequence/reproduced --checkpoint models/sequence/reproduced.pt
```

The training script evaluates validation and preserves test for future independent
assessment. Wall-clock budgets and hardware can change the stopping epoch; the
historical run selected epoch 38. A newly trained checkpoint needs its own validated
decoder configuration: the published configuration intentionally rejects a different
checkpoint hash. Source hand semantics, mirroring, tracker version and frame rate
are incompletely documented; matching tensor shapes does not prove camera compatibility.

## Repository and checks

`src/fingerspelling/` contains runtime/models/data processing, `scripts/` contains
training/evaluation/diagnosis, and `tests/` contains automated checks.
[Published results](results/README.md) retain numerical evidence and provenance.
Raw logits, personal camera data, session logs and redundant reports remain local.
Historical implementation hashes refer to their original experiment revision.

```text
uv run --no-sync python -m pytest -q
uv run --no-sync ruff check .
git diff --check
```
