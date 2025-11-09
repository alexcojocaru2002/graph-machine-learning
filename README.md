# graph-machine-learning

Graph-based land-cover models trained on DeepGlobe superpixel graphs. The code builds superpixel graphs on-the-fly (with caching), trains PyTorch Geometric models, and offers rich evaluation utilities that can also pull public checkpoints from Hugging Face.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate  # or source .venv/bin/activate on Unix
pip install -r requirements.txt
```

PyTorch should match your hardware; install a CUDA build if you plan to train on GPU.

## Data Preparation

1. **Download DeepGlobe**  
   - Automatic: run `python dataset_download.py` (requires `kagglehub` and a configured Kaggle API token).  
   - Manual: download the [DeepGlobe Land Cover dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset/data) and extract the contents into `data/`.

2. **Expected layout**
   ```
   data/
     train/
       6399_sat.jpg
       6399_mask.png
       ...
     valid/  # optional (train/valid/test are treated uniformly)
     test/
     class_dict.csv
     metadata.csv
   ```
   Paths are resolved via `const.py`, so keep the folder names unchanged.

3. **Caching**  
   `datasets/SuperpixelGraphDatasetV2` caches CNN feature maps, graphs, and superpixel targets under `artifacts/cache/`. The first run is slower; subsequent runs reuse the cache.

## Training

All model scripts build a `GeometricTrainConfig`, instantiate a model, and call `training/train_geometric.py`, which handles the dataset split (80/20), superpixel graph construction, KL-divergence training loop, checkpointing, and metric logging.

Available entrypoints:

| Script | Model | Default K values | Notes |
| --- | --- | --- | --- |
| `python training/train_gcn.py` | `models.GCN2` | `[60]` | Baseline 2-layer GCN. |
| `python training/train_gcn_mult_k.py` | `models.GCN2` | `[60,120,300,500]` | Trains across multiple graphs per image. |
| `python training/train_gat.py` | `models.MultiheadGAT` | `[60,120,300,500]` | Uses smaller batch size (32). |
| `python training/train_gat_mult_k.py` | `models.MultiheadGAT` | `[60,120,300,500]` | Same as above, prints CUDA info on start. |
| `python training/train_transformer.py` | `models.Transformer` | `[60]` | Graph-aware transformer over node features. |
| `python training/train_transformer_mult_k.py` | `models.Transformer` | `[60,120,300,500]` | Multi-K variant. |

Key outputs (under `artifacts/`):
- `<model_name>_best.ckpt` / `<model_name>_last.ckpt`: full checkpoints with optimizer state.
- `<model_name>_best_weights.pt`: clean state_dict used by evaluation scripts.
- `<model_name>_metrics.csv`: training logs (`epoch`, `train_loss`, `val_loss`, `val_mse`).

To customise training, edit the per-script `GeometricTrainConfig` (e.g., change `epochs`, `batch_size`, `k_values`, or `hidden_dim`). All scripts seed NumPy/PyTorch for reproducible splits via `config.random_seed`.

### Plot training curves

After training you can visualise learning curves:

```bash
python plot_training_metrics.py artifacts/gat_k60_k120_k300_k500_metrics.csv
```

The script saves a PNG beside the CSV and prints best validation statistics.

## Validation and Testing

### Quick validation on the hold-out split

`test_models.py` reloads best checkpoints, evaluates them on the validation (20%) split, and writes `artifacts/test_summary.csv`. It also regenerates validation plots if the training CSV exists.

```bash
python test_models.py
```

### Full evaluation & plotting

`evaluation/eval_models.py` computes classification and regression metrics, calibrates thresholds, writes per-class CSVs, and generates plots. Typical usage:

```bash
python evaluation/eval_models.py --k 60 --seed 42 \
  --ckpt-dir artifacts \
  --viz-image train/6399_sat.jpg
```

What happens:
- discovers `.pt` files in `artifacts/` plus any extra directories passed via repeated `--ckpt-dir` and `--ckpt`.
- recreates the graph dataset with the same `--seed` and evaluates each checkpoint.
- writes summary CSVs, per-class CSVs, calibrated thresholds, metric plots, and optional qualitative mask visualisations to `artifacts/eval/`.

Useful flags:
- `--batch-size`, `--num-workers`, `--device`: control evaluation performance (`--device auto` chooses CUDA/MPS when available).
- `--results-csv <path>`: skip recomputation and only regenerate plots from a saved metrics CSV.
- `--plot-per-class`: placeholder flag (per-class plots are written as CSVs today).
- `--viz-image <train/relative/path>`: dumps qualitative overlays for the requested image.

### Hugging Face checkpoints

Pass `--use-hf-models` to download the default pretrained checkpoints defined in `evaluation/eval_models.py`:

```bash
python evaluation/eval_models.py --k 60 --seed 42 --use-hf-models
```

When this flag is set, the script pulls weights/configs from the Hugging Face Hub (via `huggingface_hub`), stores them in the local cache, and evaluates them alongside any local checkpoints. No manual download is required.

## Project Structure Highlights

- `dataset_loader.py`: basic image/mask loading, normalisation, and palette conversion.
- `feature_extractor.py`: ResNet50-based feature extraction and SLIC superpixel graph construction.
- `datasets/superpixel_graph_dataset_v2.py`: wraps DeepGlobe into PyTorch Geometric `Data` objects with caching.
- `models/`: implementations for GCN, multi-head GAT, and transformer models. Each exposes `load_model` utilities used during evaluation.
- `utils/metrics.py`: classification/regression metric collectors and scorers.

## Tips

- The first run will populate `artifacts/cache/` with feature maps, graphs, and targets. Delete individual files if you need to regenerate them.
- Ensure `huggingface_hub` is installed if you rely on `--use-hf-models`: `pip install huggingface_hub`.
- GPU memory usage is dominated by the CNN backbone during feature extraction; use smaller `k_values` or batch sizes if you encounter OOMs.
