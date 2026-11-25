# ASL sequence data audit — 21 November 2026

This is a standalone, completed data audit for Omar's fingerspelling project.
It does not replace the current project or change any static checkpoint.
**No GRU/CTC model has been trained by this package.**

Read `sequence_data_audit.md` for the measured findings and `NEXT_CODEX_PROMPT.md`
for the proposed next experiment in the current VS Code repository.

## What was actually checked

- All three pinned public Parquet files: 7,304 sequences, 414,214,372 bytes.
- Exact SHA-256 for every download.
- Original university CSV recovered from **only the first 1 MiB** of its archive.
- Every sequence ID, label, split and frame count against that CSV.
- All hand coordinates, missing frames, character coverage, CTC length limits,
  and signer/ID/exact nonempty hand-sequence overlap.
- The original split remains unchanged; test data has not been used for model
  inference, training or hyperparameter selection.

## Reproduce in a separate audit environment

Use Python 3.12. These commands are run inside this unpacked folder. They do not
install dependencies into the existing fingerspelling project's environment.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-audit.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python fetch_data.py --cache-dir cache
.venv/bin/python audit_data.py --cache-dir cache --output-dir reproduced_results
```

The fetch downloads about 415 MB including the archive prefix. It does not
download the 14-GB video archive, need a Kaggle account, accept terms on your
behalf, or execute any remote code. Cached files are rechecked. A changed source
hash fails explicitly rather than silently producing a new experiment.

`audit_data.py` reads 32 records at a time. It does not load the complete nested
landmark dataset into Python memory. Only audit metadata is accumulated.

## Included artifacts

- `results/summary.json`: complete raw-dataset measurements.
- `results/sequence_manifest.csv`: one audit record per original sequence.
- `results/issues.csv`: the 18 empty-input training clips and 2 CTC-short dev clips.
- `results/pilot_plan.json`: the explicitly scoped alphabet-plus-space pilot.
- `results/pilot_*_ids.txt`: fixed membership; do not randomly resplit frames.
- `results/pilot_exclusions.csv`: whole clips excluded from this pilot and why.
- `provenance/`: hashes, source URLs, pinned repository metadata and schema card.

The bundle contains audit code and results, **not the Parquet data, raw videos,
credentials, existing checkpoints, or a trained sequence model**. Raw downloaded
files belong in the ignored cache, not in a public Git commit.

## Scope caveats

This community conversion has undocumented extraction settings and hand order.
Its structural correspondence with the original dataset is verified; the
semantic correctness of each detected joint and temporal order is not.
Original frame dimensions are available, but the conversion's resizing/cropping
policy is unknown. Do not assume the current static v2 features apply directly.
The recommendation is a controlled offline pilot, followed by a camera/domain
compatibility check, not immediate production adoption.
