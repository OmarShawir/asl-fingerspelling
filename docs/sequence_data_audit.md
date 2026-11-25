# Sequence data audit and pilot plan — 2026-11-15

## Status: index access blocked, no landmark download

The approved static model remains models/mlp_static_final.pt. No static weights,
labels or recordings were modified. GRU/CTC training has not started.
scripts/train_ctc.py, models/gru_ctc.py and ctc_decode.py are placeholders.
This audit document did not previously exist.

The Kaggle file-list endpoint and both individual index download endpoints
returned **HTTP 401 Unauthorized** when accessed outside the network sandbox.
Inside the sandbox, DNS resolution failed first. The successful external
connection establishes an authentication obstacle, not a dataset-not-found
result. No Kaggle CLI/module, KaggleHub, configured environment credentials,
~/.kaggle/kaggle.json or ~/.kaggle/access_token were found. Competition-rule
acceptance could not be checked without an authenticated account; it is not
established as a separate failure yet.

[Access evidence](../results/sequence_audit/access_attempts_network.json).
[Environment snapshot](../results/sequence_audit/environment.json).

| Requested result | Actual status |
|---|---|
| Complete filenames and exact download sizes | Unavailable: list endpoint 401 |
| train.csv | Not downloaded; endpoint 401 |
| character_to_prediction_index.json | Not downloaded; endpoint 401 |
| Sequence / participant counts | Unknown until index access |
| Text-length distribution and symbol vocabulary | Unknown until index access |
| J/j and Z/z occurrences / sequence coverage | Unknown; not assumed zero |
| Pilot filenames, bytes, participant IDs | Pending index and file sizes |
| Large landmark downloads | None |

The public [competition data page](https://www.kaggle.com/competitions/asl-fingerspelling/data)
reports **124 files, 189.09 GB**, and a **405 B** character mapping.
These are page-reported totals, not a verified API inventory or the train-only
download size. Approximately 18 GiB free disk means the complete dataset cannot
fit locally.

## Local resources

Initial checks: about 14 GiB total RAM, 5.2 GiB available, 4 GiB swap, and 18 GiB
free disk. Values fluctuate; exact byte snapshot is saved in environment.json.
nvidia-smi is unavailable. Installed PyTorch reports CUDA unavailable and zero
CUDA devices; this does not prove that no physical GPU exists.
pandas and pyarrow are absent. The index audit uses Python's standard library
and needs neither them nor the Kaggle CLI. Parquet inspection later will need
a reader such as PyArrow; do not load full wide shards into RAM.

## Reproducible index-only audit

Script: [scripts/audit_sequence_data.py](../scripts/audit_sequence_data.py).

After configuring Kaggle credentials locally (do not paste secrets into chat),
run from the repository root:

    .venv/bin/python scripts/audit_sequence_data.py --fetch

It supports KAGGLE_API_TOKEN or the legacy KAGGLE_USERNAME/KAGGLE_KEY pair,
and the standard local Kaggle credential files. Sign into Kaggle and verify
competition access/rules if authentication still does not grant access.
The script does not accept rules or download landmarks. Responses are capped
at 64 MiB, including extracted index files, and downloaded CSV/JSON are checked
before being saved. It follows file-list pagination when supplied by the API.

Alternatively place train.csv, character_to_prediction_index.json and an API
files.json listing (with exact byte sizes) under data/external/asl-fingerspelling,
then run without --fetch. This directory is already Git-ignored.

Outputs under results/sequence_audit:
- environment.json and access-attempt evidence;
- summary.json: actual sequence/person counts, complete text-length histogram,
  min/max/mean, exact character counts and separate lower/uppercase counts,
  case-insensitive ASCII letter occurrences and sequence counts, digits,
  whitespace, symbols, non-ASCII characters and mapping mismatches;
- participant_files.json: every Parquet path and its participant sequence counts;
- file_sizes.json: API-derived exact bytes;
- sample_plan.json and sample_sequences.csv: actual selected sequence IDs,
  participant IDs, verbatim phrases, split assignments, files and exact bytes.

The current summary and plan explicitly contain null counts because inputs are
missing. They are not a completed numerical audit. Running the script again
after access is available replaces these blocked outputs with measured results.
Index SHA-256 hashes are recorded after successful analysis.

## Pilot policy (not an instantiated sample)

Default download budget: **2 GiB**, not a claim about actual download size.
Select up to three smallest training shards fitting that budget, then up to ten
participants in deterministic hash order (seed 42), at most 200 whole sequences
per participant. Partition people roughly 70% train / 15% validation / remainder
test, always reserving at least one person for each held-out group. For ten
people this is 7/1/2 and at most 2,000 sequences. Real counts may be smaller.

Charge the union of needed Parquet files once even if a file contains several
participants assigned to different splits. A file boundary is not a person
boundary. The loader must filter by the saved participant/sequence assignments;
it must never train on an entire shard simply because one selected training
sequence is in it.

This simple pilot is for pipeline feasibility, not a representative benchmark.
It reports all missing letters, including J/Z, separately in each split.
If coverage is insufficient, revise the index-only plan explicitly before
downloading; do not silently move participants after examining test performance.
If fewer than three participants fit, the plan stays blocked.
Exact filenames, actual bytes and actual person split remain unknown today.

Preserve each original phrase, including case, spaces, digits and punctuation.
Case-insensitive counts are reporting only. Do not strip target symbols while
retaining their motion. Build the sequence vocabulary from the verified
character mapping and allocate a distinct CTC blank; the static model's 24
classes are not the CTC vocabulary.

## Landmark documentation and preprocessing boundary

The competition describes MediaPipe Holistic float32 landmarks in wide Parquet
columns named by coordinate, body-part type and landmark index: 543 points,
1,629 coordinate columns, with sequence_id as index and frame order. Frames may
lack detected hands. CSV links path/file_id, participant_id, sequence_id and
phrase. It warns that depth estimates are imperfect.
Source: [competition data documentation](https://www.kaggle.com/competitions/asl-fingerspelling/data).

[MediaPipe Holistic documentation](https://chuoling.github.io/mediapipe/solutions/holistic.html)
describes 21 landmarks per hand, 33 pose and 468 face points. Read columns by
explicit names, never inferred lexicographic positions:
x_left_hand_0 through x_left_hand_20, and corresponding y/z and right_hand
columns. Construct a declared tensor order, e.g. time × hand × point × xyz.
Verify actual schema and frame ordering when a pilot Parquet is available.

[MediaPipe Hands documentation](https://chuoling.github.io/mediapipe/solutions/hands.html)
defines wrist 0, thumb 1–4, index 5–8, middle 9–12, ring 13–16, little finger
17–20. x/y are normalized by image width/height respectively; hand z is
wrist-relative with a scale approximately like x, not calibrated metric depth.
The standalone Hands handedness convention assumes mirrored input; that does
not establish whether these competition videos were mirrored. Holistic's
semantic left/right channels and actual capture/display mirroring must be
distinguished.

Not established in the inspected competition documentation:
- capture orientation and whether frames were mirrored before extraction;
- original video width/height or per-sequence aspect ratios;
- exact MediaPipe package/model version and extraction settings;
- timestamps/FPS beyond frame indices.

No claim is made that source videos were square. Static features v2 need a
documented height/width ratio, wrist centering and physical-hand canonicalization.
Applying its y correction or left-hand X flip blindly could distort motion or
hand identity. It also normalizes each frame independently, potentially removing
motion relevant to J/Z and continuous fingerspelling.

Next inspection, after approving/downloading the concrete pilot: verify schema,
NaN masks and left/right availability; frame gaps; coordinate ranges; sequence
lengths; CTC minimum alignment length (target length plus adjacent repeated
characters); and orientation/aspect provenance. Retain raw coordinates and
missingness masks. Compare sequence-specific preprocessing experimentally;
do not reuse the static feature version identifier for an unverified pipeline.

## Verification

Synthetic tests cover case-sensitive J/Z counts, preserving punctuation and
spaces, shared-file participant-disjoint splits, minimum three-person splits,
missing-letter reporting, unknown-size rejection and budget enforcement.
They do not substitute for an audit of the unavailable real index.

Run result: **51 tests passed**, Ruff passed, and git diff --check passed.
No checkpoint was changed and no Parquet file was downloaded.
