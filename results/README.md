# Published experiment evidence

The small numerical reports, predictions and provenance here support the historical
claims in [the experiment log](../docs/experiments.md). They are not runtime settings;
the deployed decoder is [configs/sequence_decoder.json](../configs/sequence_decoder.json).

- `static_*`, `kaggle_*`, `baseline_recheck.*`: original static baseline and extraction coverage.
- `adaptation/`: camera adaptation settings and s03 comparisons.
- `final_static/`: final report, manifest, history and per-letter tables. The report
  retains per-clip before/after evidence; separate repeated exports remain local.
- `s05_diagnostic/`: numerical comparison/per-letter tables only. Original galleries,
  diagnostic PNGs and detailed private-image reports remain local.
- `mn_diagnostic_2026-11-07/`: numerical clip predictions only; imagery stays local.
- `sequence/gru_ctc_pilot_v1/` and `sequence/gru_ctc_long_v1/`: configuration,
  validation predictions, learning history, metrics and source provenance.
- `sequence/ctc_decode_v1/`: decoder comparison, predictions and replay evidence.
- `sequence_audit/`: historical access-attempt evidence; the subsequent immutable
  `asl_sequence_audit/` package is the training source snapshot.

Raw logits, debug runs, redundant training-prediction exports and session logs are
not distributed. Some historical reports name these local artifacts. Scripts retain
the ability to produce them; an existing output is never a license to overwrite it.
Private camera clip IDs in numerical tables are anonymous study IDs, not recordings.
