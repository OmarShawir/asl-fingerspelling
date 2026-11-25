# Decoder selection and experimental camera sequence mode — 2026-11-22

Selected **CTC prefix beam search, width 10**. No retraining or checkpoint modification.

## Saved-logit provenance and baseline

Checked reference checkpoint SHA-256 against the long-run report, checkpoint feature/vocabulary/blank metadata, vocabulary.json, CSV versus JSON targets/predictions, fixed validation membership, audit target/frame counts, ordered logit IDs, each length and total finite 28-class logit shape. All saved logits decode to their stored greedy transcripts. Hashes are in comparison.json; no entire dataset re-audit or test inference was performed.

Greedy exactly reproduces 978 clips, 4,967 target characters, 2,121 edits, CER 42.7018%, 170 exact and 17 empty transcripts.

## Fixed comparison

Log-space prefix probabilities retain separate blank/nonblank ending states. All 28 symbols participate at every frame; no dictionary, LM, token pruning, probability temperature or length penalty. Blank=0, space=27. Only prefix hypotheses are restricted to the requested beam width 10. Ties in beam scores use lexicographic token IDs. Greedy wins if CER ties.

Exhaustive tests sum every path of small three-frame alphabets (including space), checking all prefix probabilities against the beam states where ten beams contain every possible prefix. Full-vocabulary synthetic tests cover jazz, all, a b, repeated spaces and all-blank input. No other production beam width was explored.

| Decoder | Edits / target chars | CER | Exact / 978 | Empty | Mean ms/clip | p95 ms/clip |
|---|---:|---:|---:|---:|---:|---:|
| Greedy | 2121/4967 | 42.7018% | 170/978 | 17 | 0.227 | 0.509 |
| Prefix beam 10 | 2056/4967 | 41.3932% | 177/978 | 13 | 3.541 | 9.786 |

Full comparison wall time: 13.67 seconds, below the 900-second limit. Per-clip decoding timing includes float64 log-softmax and incremental decoder steps; it excludes model forward, camera/tracker and data loading. It is not end-to-end camera latency.

The runtime selected configuration is [configs/sequence_decoder.json](../../../configs/sequence_decoder.json), tied to the reference checkpoint hash. Comparison CSVs preserve raw text including spaces. The comparison script refuses an existing output directory; no prior experiment was overwritten.

## Run the experimental camera mode

From the project root:

    .venv/bin/python -m fingerspelling.app --mode sequence

Defaults: models/sequence/gru_ctc_long_v1.pt and configs/sequence_decoder.json. An incompatible checkpoint hash or vocabulary is rejected. Optional --camera N selects another device; --sequence-log-dir changes the attempt-log directory.

- R: start a new user-segmented word and reset GRU and decoder states. An interrupted previous attempt is logged as restarted.
- Text in brackets is provisional and may be revised by beam search; preview shows its tail without modifying stored text.
- Enter: finish/save the raw final text and reset recognition states. The completed text stays visible.
- C: log an active attempt as cleared, clear display and reset states.
- Q/Esc: log an unfinished attempt as cancelled and close the camera.

The user currently determines word start/end. No automatic boundary detector, MLP 700-ms stability gate, or static word_builder is used. No CTC blank is imposed on missing-hand frames.

Static mode remains the default and keeps its existing command:

    .venv/bin/python -m fingerspelling.app --model models/mlp_static_final.pt --hand left

## Camera feature contract and unknowns

The existing OpenCV camera and MediaPipe VIDEO HandLandmarker are reused. Static defaults to one hand; sequence mode requests at most two. Camera BGR is converted to RGB, without mirroring before tracking. Each detected hand returns 21 image-space xyz points in MediaPipe order; point 0 is the camera tracker wrist. x/y are normalized image coordinates, and z is tracker depth rather than metric distance. Preview mirroring occurs after extraction and only affects display.

Sequence preprocessing calls the exact checkpoint hand_features function: unordered 21-point blocks; wrist-relative normalized shape plus raw point-0 xyz and log scale, with masks. No static v2 height/width correction or physical-left-hand X flip. Zero detected hands becomes an empty block list for a genuine timestep. GRU state advances using the zero masked frame embedding, and the decoder consumes the resulting model logits.

The community conversion did not document anatomical block order, precise within-hand semantics, normalization/cropping/resizing/mirroring, tracker implementation/version or FPS. Camera tracking uses the existing MediaPipe task/settings; matching (21,3) arrays does not establish domain compatibility. Source point-0 wrist semantics remain an assumption. Detected block order is handled symmetrically, but explicit hand identity over time is not tracked by the encoder. One timestep is processed per captured frame; no resampling is justified by the unknown source frame rate.

## Replay, state isolation and logs

Two saved validation sequences (66 and 35 frames) produced identical decoded text with full forward versus frame stepping. Maximum logit difference: 4.7683716e-06; checked with atol=2e-5, rtol=1e-5. The first contains 11 missing frames. Reset between different words and repeat replay verified state isolation. A separate artificial missing-frame sequence matched full forward, without forced blank. This checks implementation parity, not recognition correctness.

Each attempt writes a unique JSON under ignored results/sequence_live_sessions/. It contains raw text, completed/cancelled status, checkpoint/hash/architecture, decoder settings, feature version, tracker configuration, actual frame dimensions/timestamps, hand count, raw logits and start/end UTC and monotonic timestamps. Target and accuracy are null because no target word was supplied.

segment_duration_ms is user-controlled elapsed time. recognition_compute_ms measures feature/model/decoder work per frame, excluding camera/tracker/display; these quantities are deliberately distinct. No physical motion-onset latency is claimed.

## Verification and remaining blocker

76 repository tests passed; Ruff and git diff --check passed. Camera keys/logging/release were exercised with a synthetic camera, while existing static tests remain passing. Checkpoint, original recordings, pilot and long-run reference artifacts are unchanged.

An actual launch was attempted. This execution environment exposes no /dev/video* device, and OpenCV could not open camera 0. The original camera_open_attempt.log remains local. Therefore no real webcam recognition trial or live accuracy measurement was completed. The command is implemented and ready to run when the camera is available.

Validation selected the decoder; no test predictions or test-based settings. This remains experimental, manually segmented camera integration, not proof of live accuracy or bonus completion.
