---
dataset_info:
  features:
  - name: id
    dtype: string
  - name: label
    dtype: string
  - name: hands
    list:
      list:
        list: float64
  - name: pose
    list:
      list:
        list: float64
  - name: face
    list:
      list:
        list: float64
  splits:
  - name: train
    num_bytes: 474903875
    num_examples: 5455
  - name: validation
    num_bytes: 92497678
    num_examples: 981
  - name: test
    num_bytes: 82492339
    num_examples: 868
  download_size: 414214372
  dataset_size: 649893892
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---
