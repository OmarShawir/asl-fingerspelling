# Two weeks

| Days | Deliverable |
| --- | --- |
| 1–2 | Confirm camera tracking and record the first eight letters plus some genuine words. Use A, E, H, L, M, N, O and T for the pilot. |
| 3–5 | Split groups, normalize valid hand points, train the from-scratch MLP, and spell HELLO live using the word builder. |
| 6–7 | Expand toward the 24 static letters and tag a stable baseline on GitHub. |
| 8–9 | Train a small GRU with CTC on training clips. Use carefully composed training sequences and genuine cross-letter motion. Include moving J and Z only when examples cover the trajectory. Make an early decision using real validation words. |
| 10–11 | Compare both versions under the same live conditions on their common supported vocabulary. Report extra letters separately. Choose the display model on validation data. |
| 12 | Freeze the chosen model and measure on held-out signers, words and sessions. |
| 13–14 | Reproduce the setup, publish the trained checkpoint, record the demo and finish the public README. |

## Model contracts

The MLP receives 63 centered and scaled values for each valid frame. Preserve hand orientation. Skip missing frames and use the presence signal in the word builder to re-arm repeated letters. Train the weights in PyTorch. Values for hold, release and confidence in configs/default.yaml are initial settings to tune on validation data.

The GRU receives an ordered sequence with hand-shape features, raw wrist displacement, elapsed time and visibility. It predicts character or blank at each timestep. A CTC decoder carries state across live chunks and preserves repeated letters separated by a predicted blank. Train with whole transcripts. Both the word builder and the CTC decoder should emit the current text and its commit timestamp on the same monotonic clock.

Measure whole-word exact match before any manual correction, character error rate, and latency on the display laptop. Compare both models on the same held-out real clips and document any differences in their intended signing speeds. Use validation to select the final model and the untouched test set for the reported result.

## Dependencies when the corresponding stage is implemented

| Stage | Tools |
| --- | --- |
| Training | PyTorch for the MLP and GRU/CTC; select a CPU or compatible GPU build for the training machine. |
| Evaluation | scikit-learn for grouped splits and confusion matrices, jiwer for character error rate, Python equality for full-word matches. |
| Analysis | Matplotlib for figures; add pandas only if the analysis benefits from dataframes. |
| Quality | pytest and Ruff in the dev dependency group. |

Pin new training dependencies after testing the chosen build and update uv.lock. The current requirements.txt and lock cover the implemented camera and collection tools. Recognition modules and training scripts in the scaffold show the planned file names. They do not contain trained classifiers yet.
