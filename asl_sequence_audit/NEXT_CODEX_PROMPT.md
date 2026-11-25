# Next task for Codex in the current VS Code project

Continue the existing ASL project from `models/mlp_static_final.pt`. Keep the
static checkpoint, static preprocessing, original recordings and current live
mode intact. Do not resume the rejected Kaggle competition route. The attached
`asl_sequence_audit` package contains a completed alternative-data audit; read
its report, summary and pilot plan before implementing anything.

Implement and run a **first offline GRU + CTC experiment** on the audited,
pinned ChicagoFSWild landmark conversion. Build the recognition network and
train it ourselves; do not import pretrained letter/word-recognition weights.
This is a feasibility experiment, not a claim of bonus completion.

1. Inspect the current repository, instructions, Git status and installed
   dependencies. Integrate the audit utilities and cite both the original
   university dataset and the community converter. Download only the pinned
   ~415-MB payload using the bounded fetcher. Keep it ignored by Git. Preserve
   the source files and their hashes.
2. Use the supplied original signer-disjoint memberships. The alphabet-plus-
   space pilot has 5,433 train, 978 validation and 867 test clips. Preserve raw
   labels. Exclude whole punctuation-containing clips using the declared scope;
   do not erase symbols from a label while retaining that symbol's motion.
   The 18 training clips without any hand input are explicitly excluded.
   Do not evaluate test predictions during development.
3. Hands are variable lists of 0, 21 or 42 three-coordinate points per frame.
   No anatomical handedness is provided. Do not assume the first 21 are left,
   the next 21 are right, or that a lone block always represents the same hand.
   A reasonable experimental implementation is a shared per-hand MLP followed
   by a symmetric masked set aggregation and a small causal GRU. Test that
   swapping the two hand blocks leaves frame embeddings/logits unchanged in
   eval mode. Document the 21-point block interpretation as an extraction
   assumption that still requires semantic verification.
4. Preserve missing-frame positions and masks. Preserve global hand position
   or wrist trajectory alongside hand shape so per-frame centering cannot
   delete the motion for J/Z. Do not call this the static v2 feature version.
   Source video width/height are known, but conversion resizing/cropping,
   handedness convention, tracker version and FPS remain unknown. Avoid an
   unverified aspect correction or anatomical mirror flip. Declare every
   preprocessing assumption and isolate it for later comparison with live
   tracker output.
5. Use the fixed vocabulary a-z plus space, with a separate CTC blank. Keep
   full temporal resolution. Pack/mask variable-length sequences and pass
   correct lengths to CTC. Its minimum is target length plus adjacent repeated
   symbols. Two validation clips ("emmett": 7<8, "all": 3<4) cannot align at
   current time resolution: keep them in CER/exact-transcript metrics, report
   finite CTC validation loss on the 976 alignable cases with the denominator
   explicit. Do not silently hide infinite losses with `zero_infinity=True`.
6. First run shape/mask/blank/repeat/decoder tests and intentionally overfit a
   fixed small **training-only** subset, e.g. 16 short clips, to validate the
   learning path. Report that as a debugging result, not generalization.
   Once functioning, train a modest CPU pilot with recorded seed and config;
   select by validation character error rate, not training accuracy. Keep a
   bounded compute budget and report actual elapsed time and peak memory.
7. Save configuration, vocabulary, split/source hashes, checkpoint and learning
   history under clearly separate sequence paths. Report CER, exact-transcript
   accuracy and empty-prediction rate over all 978 scoped validation clips.
   Separately report performance on clips containing J and Z, with counts;
   presence in the labels does not establish that either letter is recognized.
   No test scoring, no claim of live performance, and no automatic publication.
8. Update `docs/experiments.md` and README with what actually ran. Run relevant
   tests, lint and a load/infer check. Explain the changes in simple Sudanese
   Arabic: input, architecture, training result, limitations and next step.

After the offline pilot, the next distinct milestone is a streaming/live
experiment with measured output delay and whole-word results. Continuous
unsegmented camera operation must not be claimed from pre-segmented clips.
Additional gestures remain a separate task: this dataset has no digit labels.
