# graph-machine-learning

## Loading Data 

Before running the code, please download the [DeepGlobe Land Cover Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset/data) and place its zip contents in a `data` folder at the root of this repository.

## Training

We provide multiple training entrypoints under `training/` for different model families:

- `train_gcn.py`, `train_gcn_mult_k.py`
- `train_gat.py`, `train_gat_mult_k.py`
- `train_transformer.py`, `train_transformer_mult_k.py`

Each script instantiates a model, configures `GeometricTrainConfig`, and calls `training/train_geometric.py`, which handles dataset splitting, graph construction, and the KL-divergence training loop.

### Install

```bash
pip install -r requirements.txt
```

Ensure you have a CUDA-enabled PyTorch build if you plan to train on GPU.

## Evaluation

Use `evaluation/eval_models.py` to compute classification and regression metrics or to plot existing results.

### Compute metrics and generate plots

```bash
python evaluation/eval_models.py --k 60 --seed 42 \
  --ckpt-dir artifacts/eval \
  --viz-image train/6399_sat.jpg
```

This will:
- Locate all `.pt` checkpoints in `artifacts/eval` by default (or any directories passed via `--ckpt-dir`, `--ckpt`).
- Reproduce the train/test split with the given `--seed`.
- Evaluate every checkpoint sequentially on the selected `--k` value.
- Save summary metrics, per-class CSVs, calibrated thresholds, plots, and optional qualitative mask visualisations in `artifacts/eval/`.

Other useful flags:

- `--results-csv path/to/metrics_summary.csv`: skip computation and only regenerate plots from the CSV.
- `--plot-per-class`: also emit per-class plots (CSV always saved).
- `--batch-size`, `--num-workers`, `--device`: control evaluation settings.

### Plot-only mode

```bash
python evaluation/eval_models.py --k 60 --seed 42 \
  --results-csv artifacts/eval/metrics_summary_k60_seed42.csv
```

This reloads the saved CSV and regenerates the metric plots without touching checkpoints.

### Outputs
- `artifacts/eval/metrics_summary_k{K}_seed{seed}.csv`: macro/micro metrics per model.
- `artifacts/eval/per_class/*.csv`: detailed per-class regression scores.
- `artifacts/eval/plots/*.png`: bar charts for each metric.
- `artifacts/eval/thresholds/*.json`: calibrated classification thresholds.
- `artifacts/eval/viz/*.png`: qualitative comparisons (if `--viz-image` supplied).

## Notes
- Superpixel graphs and feature maps are cached under `artifacts/cache/` to speed up repeated runs.
- Targets are area fractions per superpixel (excluding the `unknown` class listed in `class_dict.csv`, if present).
- Checkpoints from training are saved under `artifacts/`; you can copy or symlink them into `artifacts/eval/` for evaluation.
