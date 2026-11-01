# Model files

The camera downloads Google's hand_landmarker.task into this directory on its first run. It supplies hand coordinates. If you ran the standalone camera script earlier, you can copy its hand_landmarker.task here and run offline after installing the dependencies.

Our planned mlp_static.pt and gru_ctc.pt checkpoints will come from our training scripts. When publishing a trained release, include the supported labels and ordering, feature settings, checkpoint, validation settings, evaluation results, and a reproducible download path. Weights, raw recordings and the virtual environment are excluded from the starter ZIP.

[Official hand tracker and model card](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
