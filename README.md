# ASL fingerspelling

A two-week project to turn live ASL fingerspelling into text. This starter implements webcam hand tracking and labeled landmark collection. The static letter classifier, word builder and sequence model are the next milestones. Their planned source files are included in the scaffold.

## Run the webcam on Ubuntu

Extract the ZIP and open a terminal in the **asl-fingerspelling** directory, next to this README. Install pipx once:

```bash
sudo apt update
sudo apt install -y pipx
```

From the project directory:

```bash
pipx run uv run --locked python -m fingerspelling.app
```

If uv is already installed, use `uv run` wherever this guide shows `pipx run uv run`. uv selects Python 3.12 and installs the project with the versions from uv.lock. The first run requires internet for the packages and the official hand tracker. Raise one hand to see **21 landmarks**. Press **Q** or **Esc** in the camera window to quit.

- To choose a second webcam, append `--camera 1`.
- If another program has the camera open, close it and retry.
- If you already downloaded the tracker for asl_camera.py, copy the file hand_landmarker.task into models/. An interrupted network download can also be completed manually with the [official v1 model](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task).
- The same application entry point is intended for Windows x64. With uv installed, open PowerShell in the extracted directory and run `uv run --locked python -m fingerspelling.app`. Ubuntu x86-64 is the development target.

## Record your first clip

Use a person code and a session code. Choose the actual signing hand. The following example records static letter L:

```bash
pipx run uv run --locked python scripts/collect.py --person p01 --session s01 --hand right --kind letter --text L
```

1. Form L with your hand fully visible. Press **Space** in the preview to start a take.
2. Hold the shape, then press **Space** again to save. The terminal prints the clip path.
3. Relax and form the letter again for a new take. Press **Q** between takes to exit. Quitting during a take discards that unfinished take.

For a word, start recording just before signing and save after the final letter:

```bash
pipx run uv run --locked python scripts/collect.py --person p01 --session s02 --hand right --kind word --text HELLO
```

Inspect `data/metadata.csv` after your first save. Each take has one row and one compressed file under `data/clips/`. The recorded frame times include pauses and missed detections. See [the data contract](data/README.md) and [the format example](data/sample/README.md).

## Stack

| Responsibility | Choice |
| --- | --- |
| Runtime and install | Python 3.12, uv, pyproject.toml, requirements.txt and uv.lock |
| Capture and tracking | OpenCV desktop through opencv-contrib-python, MediaPipe Tasks Hand Landmarker |
| Capture storage | NumPy NPZ per clip and a CSV index written using Python's csv module |
| Configuration | PyYAML and configs/default.yaml |
| Week 1 recognition | A from-scratch PyTorch MLP plus stable-letter and release logic |
| Week 2 experiment | A PyTorch GRU trained with CTC to emit ordered characters |
| Evaluation | scikit-learn for group-aware splits and confusion matrices, jiwer for character error rate, Matplotlib for plots |
| Local checks | pytest and Ruff in the optional dev group |

The current runtime dependencies are pinned in requirements.txt. The build reads them directly from that file. uv.lock also locks transitive dependencies. Add PyTorch and the evaluation libraries when their scripts are implemented and the CPU or GPU build is chosen. The tracker returns coordinates; your trained models will make the ASL predictions.

## Layout

```text
asl-fingerspelling/
  README.md, requirements.txt, pyproject.toml, uv.lock, .python-version, .gitignore
  configs/default.yaml
  data/                  clip schema, sample, local clips and split lists
  models/                tracker download and future trained weights
  src/fingerspelling/
    config.py            project configuration
    hand_tracker.py      landmark extraction on original camera frames
    app.py               live preview and capture loop
    dataset.py           NPZ writes, timestamps, visibility and CSV index
    features.py          planned shared feature preparation
    augment.py, synth.py planned training-data transformations
    models/              planned MLP and GRU definitions
    word_builder.py      planned static-model text output
    ctc_decode.py        planned sequence-model text output
    metrics.py           planned recognition evaluation
  scripts/               collect.py and planned training/evaluation entry points
  tests/                 clip format and index recovery checks
  notebooks/             inspect the public example or your local recordings
  results/               evaluation tables and plots
  docs/                  roadmap and experiment log
```

The public sample is a synthetic format fixture. Actual camera takes and metadata stay local by default. The week 1 target is a live word from your own trained model. See [the two-week plan](docs/roadmap.md) and [the experiment log](docs/experiments.md).

## Development checks

```bash
pipx run uv run --locked --group dev pytest -q
pipx run uv run --locked --group dev ruff check .
```

## Start the Git repository

For a freshly extracted folder:

```bash
git init -b main
git add .
git status
git commit -m "Add webcam tracker and labeled clip collector"
```

When you are ready to publish, create a public GitHub repository from this folder using an authenticated GitHub CLI:

```bash
gh repo create asl-fingerspelling --public --source=. --remote=origin --push
```

If you cloned an existing repository, copy the starter files into that clone and use its existing remote. Document model results, setup, retraining and a demo in the public README when those milestones are complete.

## References

- [Google Hand Landmarker for Python](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)
- [ASL University fingerspelling reference](https://www.lifeprint.com/asl101/pages-layout/fingerspelling.htm)
- [PyTorch CTC loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html)
- [scikit-learn grouped splits](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)
