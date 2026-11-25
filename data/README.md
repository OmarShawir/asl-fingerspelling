# One clip per take

Run the collector from the project root. Use an anonymous person code such as p01 and a session code such as s01. The hand option is the physical hand used by the signer. Only the preview is mirrored.

Each completed take adds a compressed file to clips/ and updates metadata.csv. Clips contain image-space MediaPipe coordinates as emitted by Hand Landmarker, before our own normalization. x and y use the camera image axes. z is MediaPipe's estimated relative depth. Retain x and y of the wrist across frames for motion features.

| NPZ key | Shape and type | Meaning |
| --- | --- | --- |
| xyz | (T, 21, 3), float32 | Frames in capture order, each with 21 points in MediaPipe order and x, y, z columns |
| t_ms | (T,), int64 | Milliseconds since the first captured frame. Values strictly increase. |
| valid | (T,), bool | False when the hand is missing. All 63 coordinates are NaN on that frame. |
| meta_json | scalar Unicode string | A copy of the index row. Read using json.loads and allow_pickle=False. |

The CSV fields are clip_id, text, kind, person, session, hand, fps, frames, created_at, source, frame_width, frame_height, dimensions_source and diagnostic_manifest. New camera clips record actual MediaPipe input dimensions. Legacy clips may omit the added fields; reindexing leaves their CSV cells blank. The filename is clips/<clip_id>.npz. FPS is measured from the capture times. created_at is UTC at save. The source field distinguishes a webcam recording from the artificial format example. Video images stay in memory unless diagnostic PNG saving is enabled.

Kinds used by the collector:

- letter captures one ASL letter. Hold only the target pose while collecting static examples. A J or Z take must include its motion and is excluded from static-classifier training.
- transition captures a specified pair in order, e.g. AB, including the change in handshape.
- word captures one complete fingerspelled word, e.g. HELLO. Keep the motion between letters.

Use separate takes, lighting and sessions. Reset your hand between takes. Confirm sign shapes against a reliable ASL reference before labeling recordings.

## Splits

Reserve separate groups before extracting individual frames or joining clips. For a signer-independent score, a person belongs to only one of train, validation or test. If only one signer is available, use (person, session) as the grouping key and report results as cross-session. Keep all frames from a clip together and check that enough groups and class coverage exist for the intended comparison.

Store one clip ID per line in splits/train.txt, splits/val.txt and splits/test.txt when you create the actual partitions. Source clips for synthetic words must all belong to train. Store their IDs with each synthetic example. Keep genuine recordings for validation and final testing. Use separate real clips for any fine-tuning. Reserve some new names and words for the end-to-end test.

## Local data and example

Open sample/ to inspect the format without a camera. The example lives outside clips/ and is excluded from training and evaluation. metadata.example.csv illustrates the header. Your actual metadata.csv is created on the first save. To rebuild the index from all complete local takes, run:

```bash
uv run --locked python -m fingerspelling.dataset
```

For a public release, select recordings whose participants agreed to share the landmarks, publish anonymous IDs, and distribute a matching split manifest. Keep your local metadata private.

## Aspect-aware recordings and diagnostic PNGs

Use `scripts/collect.py --save-diagnostic-frames --diagnostic-every 1` alongside
the required signer/session/hand/label arguments to save every original recorded
frame. The default sampling interval is 10. Snapshots are copied before drawing
or preview mirroring. `diagnostic_frames/<clip_id>/manifest.json` associates
filenames with zero-based recorded-frame indices, clip-relative `t_ms`, original
monotonic capture timestamps and frame dimensions. This directory is git-ignored.
Changing resolution during a take discards it; incomplete takes write no PNGs.

Feature v2 requires documented width/height. Historical clips and the public
format fixture remain valid for v1 but cannot be evaluated with v2 without original
capture evidence. Never infer old dimensions from today's camera settings.
See [the aspect experiment](../docs/experiments.md) for paired evaluation commands.
