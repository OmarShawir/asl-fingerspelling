# Runtime checkpoints

These two small final checkpoints are included in Git. Intermediate checkpoints
and the automatically downloaded MediaPipe tracker stay local. Load only trusted
PyTorch files; the runtime uses `weights_only=True`.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `mlp_static_final.pt` | 214882 | `7821db6c17a04ba0be940fbca10a85fcb40db41b3aaa1dff1b33fdbaec08ce0a` |
| `sequence/gru_ctc_long_v1.pt` | 867068 | `762d11636d38f35ae8590fbef6bfb9edd2687b5f0acdd5c027dba063146973fc` |

Static: 24 letters, features v2, Kaggle training plus private s04/s06;
s03/s05 development selection. See [final static provenance](../results/final_static/manifest.json).
Sequence: lowercase a–z and space, CTC blank 0, space 27; ChicagoFSWild
signer-disjoint training/validation. See [run report](../results/sequence/gru_ctc_long_v1/report.json)
and [hash-bound decoder](../configs/sequence_decoder.json). Test was not inferred.

The root README contains installation/run commands and reproduction limitations.
