# Format example

format_example.npz contains three computer-generated frames and a dummy transcript A. The middle frame represents a missed detection. This is a file-format fixture, not a recording of the ASL letter A. It is for inspecting the loader and is excluded from the local training and test directories.

From the project root:

```bash
uv run --locked python -c "import numpy as np; d=np.load('data/sample/format_example.npz'); print(d['xyz'].shape, d['t_ms'].tolist(), d['valid'].tolist())"
```

Expected output:

```text
(3, 21, 3) [0, 33, 79] [True, False, True]
```
