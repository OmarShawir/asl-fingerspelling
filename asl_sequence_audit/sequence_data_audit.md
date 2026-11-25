# ASL sequence dataset audit — completed 21 November 2026

## Decision

**Proceed with a controlled offline GRU + CTC pilot on the scoped subset below.**
This is a recommendation based on a completed structural audit, not a claim that
a sequence recognizer has been trained or that this conversion is validated for
Omar's webcam. Keep `models/mlp_static_final.pt` as the current static reference.

بالسوداني: لقينا بيانات كلمات متحرّكة صغيرة نسبياً، وفحصنا الملفات نفسها.
ممكن نبدأ بيها أول تجربة للموديل الزمني. وجود J وZ في البيانات ما معناه إننا
اتعلّمناهم بالفعل؛ ده بيتثبت بعد التدريب والتقييم واللايف.

## Sources and exact scope of the work

- Original university data page: https://home.ttic.edu/~klivescu/ChicagoFSWild.htm
- Original archive: https://dl.ttic.edu/ChicagoFSWild.tgz
- Community landmark conversion: https://huggingface.co/datasets/Kibalama/ChicagoFSWild-Landmarks
- Pinned conversion revision: `c738cce88fcae8bcdb0811738c44a0b41701ad4d`.

Downloaded and SHA-256 verified all three Parquet files: **414,214,372 bytes**
(414.21 decimal MB; approximately 395.03 MiB). No login was required in this
audit. The university archive is 14,270,632,063 bytes; only bytes 0–1,048,575
were requested/read. Its initial tar members contained the full original CSV
and README, so no complete video archive was downloaded.

The official CSV is 904,482 bytes with SHA-256
`ed0698ae6a606742030bb6130daddef115a2032414efc34a92000029ddd17e61`.
The three Parquet hashes, sizes and URLs are saved in `provenance/download_manifest.json`.

Work was performed in this chat's workspace. The current VS Code repository,
its checkpoints, recordings, configuration and Git history were not modified.

## Full-dataset measured results

Counts in this table are for the unfiltered source files. J/Z columns count
**sequences containing the letter**, not individual occurrences or successes.

| Source split | Sequences | Original signer IDs | Contains J | Contains Z | Frames |
|---|---:|---:|---:|---:|---:|
| train | 5,455 | 87 | 133 | 72 | 150,937 |
| validation (official dev) | 981 | 37 | 15 | 20 | 28,124 |
| test | 868 | 36 | 30 | 9 | 26,078 |
| Total | **7,304** | **160** | **178** | **101** | **205,139** |

- All 7,304 sequence IDs, exact labels, split assignments and frame counts
  match the original university CSV. Every original sequence is represented.
- The original train/dev/test split has zero pairwise overlap in signer IDs.
  The conversion preserves that split. No signer identities were inferred from
  filenames: the original CSV's `signer` field was used.
- No sequence-ID overlap or duplicate nonempty complete hand sequences were
  found across splits. Exact checks do not prove absence of near duplicates.
- All 26 lowercase English letters occur in every split. Uppercase labels and
  digits do not occur. Spaces and a few punctuation marks also occur.
- Q is scarce: 15 character occurrences in raw training. J has 140 training
  occurrences and Z has 76. Presence alone does not ensure balanced learning.
- One validation symbol, `@`, is absent from raw training. This is explicitly
  recorded, not silently mapped to another character.

## Hand representation and missing data

The schema contains `id`, `label`, `hands`, `pose`, and `face`.

| Hand points in a frame | Train frames | Validation frames | Test frames |
|---|---:|---:|---:|
| 0 (empty list) | 2,869 | 415 | 317 |
| 21 | 51,047 | 12,851 | 11,227 |
| 42 | 97,021 | 14,858 | 14,534 |

All nonempty hand points are finite coordinate triples. No malformed hand-frame
shapes, NaN/Inf hand points, or all-zero hand points were found. There are
201,538 frames with hand input out of 205,139 total; this is **input presence**,
not landmark correctness or model accuracy. Empty frames must remain in the
sequence with explicit masks.

The pose field contains either 0 or 6 points per frame, and face contains 0 or
85. All three channels have the original frame count. Their exact selected
indices and the anatomical ordering of hand blocks are undocumented.

The 42-point representation is structurally compatible with two 21-point hand
blocks. However, the data does **not** say which block is left/right, whether
its ordering is stable, or which hand a lone 21-point block represents. Do not
pad a missing second block and call the first block the same anatomical hand.

For the pilot, a shared per-hand encoder with symmetric masked pooling can
avoid relying on block order. This is a proposed model design, not something
tested or trained in this audit. Verify block-swap invariance before training.
Retain global position/wrist trajectory with shape features so normalization
does not erase motion useful for J/Z.

## What remains unverified about extraction

The community dataset card contains schema/split metadata but no explanatory
extraction documentation. No extraction script was found in its repository.
The following remain unknown:

- MediaPipe implementation, version, settings and exact point selection/order.
- Mirroring, resizing/cropping and whether coordinates refer to original frames.
- Semantic accuracy of the detections, especially occluded fingers and hands.
- Original frame indices/timestamps/FPS and independently verified chronology.

The original CSV supplies video dimensions, including multiple aspect ratios.
Matching frame counts does not prove chronological correctness or establish
the converter's resizing policy. Do not automatically apply static v2's aspect
correction or physical-left-hand X flip to this conversion. Preserve its raw
coordinates, declare a separate experimental feature version, and perform a
later comparison with the actual webcam tracker before claiming live support.

Some x/y coordinates fall outside [0,1]; these are recorded as observed and not
clamped. They may reflect extrapolated/off-image detections. This audit does
not infer their semantic correctness.

The original site describes a public research release. The community card has
no explicit license field or attribution narrative. Cite both sources and keep
downloaded data out of public code commits; this report does not grant any
additional redistribution rights.

## Fixed pilot policy and exact membership

Use the project-relevant vocabulary **a–z plus literal space**, with a distinct
CTC blank (index 0) and symbol indices 1–27. No digits are supported by this
dataset; the additional-gesture bonus remains a separate task.

Policy: keep an entire sequence only when all target characters are in the
declared vocabulary. Do not remove punctuation from a target and keep its
motion. Also exclude the 18 training clips with no hand detections anywhere.
No validation/test clip has completely missing hand input.

| Split | Raw | Punctuation clips excluded | Empty-input clips excluded | Pilot | J clips | Z clips |
|---|---:|---:|---:|---:|---:|---:|
| train | 5,455 | 4 | 18 | **5,433** | 133 | 72 |
| validation | 981 | 3 | 0 | **978** | 15 | 18 |
| test | 868 | 1 | 0 | **867** | 30 | 8 |

All 160 original signer IDs remain represented with no cross-split movement.
Every exclusion is listed in `results/pilot_exclusions.csv`; fixed IDs are in
`results/pilot_*_ids.txt`. These scoped results must be described as the
alphabet-plus-space subset, not a full-dataset benchmark score.

No model predictions have been generated on the test split. Keep it unused
until model and preprocessing decisions are frozen.

## CTC alignment finding

CTC needs at least the target length plus one extra timestep for each equal
adjacent pair. For example, `all` needs four steps, because the two Ls must be
separated by a CTC blank. Two validation clips are too short at the supplied
frame rate:

| Sequence | Target | Frames | CTC minimum |
|---|---|---:|---:|
| youtube_4/sean_berdy_5756 | emmett | 7 | 8 |
| youtube_5/jeffrey_spinale_6043 | all | 3 | 4 |

All 5,433 scoped training clips and 867 scoped test clips satisfy the necessary
length bound. It is a necessary condition, not proof of sufficient hand motion.

For the first pilot, do not temporally downsample. Keep the two short dev clips
in character-error and exact-transcript metrics, whose denominator remains
978. Report finite CTC validation loss separately over the 976 alignable clips,
with the skipped count explicit. Do not quietly zero infinite losses or use
target length to stretch individual test inputs. A later uniform output-time
resolution change is a separate documented experiment.

## First experiment after this audit

1. Implement a masked sequence loader using the saved memberships and labels.
2. Define a small own-trained hand encoder + causal GRU + CTC head. Keep this
   separate from the existing static live mode.
3. Check decoding of repeated letters, blanks, padding and missing hands, then
   deliberately overfit 16 training-only clips to debug the training path.
4. Train an initial CPU pilot and select by validation character error rate
   over all 978 scoped dev clips. Also report exact-transcript accuracy and
   empty prediction rate; preserve the full raw outputs.
5. Evaluate J/Z-containing clips separately with their small sample counts.
6. Freeze a sequence candidate before one final test-set evaluation. Subsequent
   live testing must separately measure streaming behavior, word errors and
   delay. Pre-segmented offline clips do not prove continuous live recognition.

The detailed handoff is in `NEXT_CODEX_PROMPT.md`.

## Verification performed

- Download hashes and original CSV hash verified.
- Full audit completed twice with identical raw summary results.
- Seven logic tests cover CTC repeats, declared scope, split naming and hand
  block structure. Four download tests cover byte caps, hash failure and
  preserving a modified cache.
- This is an independent audit package. Its tests do not replace the existing
  project's 51-test report or imply new recognition accuracy.

Reproduction instructions are in `README.md`. Full machine-readable values,
including coordinate ranges and per-character counts, are in `results/summary.json`.
