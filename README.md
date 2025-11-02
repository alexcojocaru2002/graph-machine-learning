# graph-machine-learning

## Loading Data 

Before running the code, please download the [DeepGlobe Land Cover Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset/data) and place its zip contents in a `data` folder at the root of this repository.

## Training: GAT-based Superpixel Area Regression

This repo implements a CNN+GNN pipeline inspired by MLRSSC-CNN-GNN [paper](https://skyearth.org/publication/papers/2020_mrsiscccnnagnn.pdf). We use a remote-sensing ResNet-50 backbone (TorchGeo) to extract per-pixel features, SLIC to form superpixels, and a multi-layer integrated GAT to predict, for each superpixel, the area fraction per class (regression).

Key components:
- `dataset_loader.py`: loads images and RGB masks, handles resizing and normalization
- `feature_extractor.py`: extracts 2048-D features and SLIC superpixels
- `utils/graph_utils.py`: builds superpixel adjacency and per-superpixel area targets
- `datasets/graph_superpixel_dataset.py`: multi-k dataset wrapper, returns graph samples
- `models/gat.py`: multi-layer integrated GAT for node regression
- `train.py`: training loop with CUDA support and YAML/CLI configuration
  - Tips for speed:
    - Set `cache_features: true` to avoid recomputing CNN features.
    - Use `feature_device: cpu` to keep GPU free for GNN; backbone uses AMP.
    - Keep `batch_size: 1`; graphs vary in size and are processed efficiently per-sample.

### Install

```bash
pip install -r requirements.txt
```

Ensure a CUDA-enabled PyTorch if you plan to train on GPU.

### Run training with YAML config

Edit `configs/train_example.yaml` as needed, then:

```bash
python train.py --config configs/train_example.yaml
```

### Override config via CLI

```bash
python train.py --config configs/train_example.yaml --k_values 300 600 900 --device cuda --epochs 40
```

### Notes
- Multiple SLIC `k_values` are supported; dataset includes all (image,k) pairs.
- Targets are area fractions per superpixel (excluding `unknown` class if present in `class_dict.csv`).
- Checkpoints are saved under `artifacts/`.
