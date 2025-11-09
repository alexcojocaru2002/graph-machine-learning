from typing import List, Optional, Tuple, Dict

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data  # type: ignore[import]

import const
from dataset_loader import DeepGlobeDataset
from feature_extractor import extract_image_feature_map, get_slic_graph
from utils.graph_utils import compute_superpixel_area_targets

class SuperpixelGraphDatasetV2(Dataset):
    """
    Builds graph samples from DeepGlobe images for multiple K values per image.
    Each sample corresponds to a (image_idx, k) pair and returns a PyG Data object:
      - x: [N, F]
      - edge_index: [2, E]
      - y: [N, C_eff] (area fractions if normalize_targets=True)
    Feature extractor pipeline:
      1) extract_image_feature_map(img_t)
      2) extract_features(feature_map, img_rgb, k) -> (X, edge_index, sp)
      3) compute_superpixel_area_targets(sp, mask)
    """

    def __init__(
        self,
        base: DeepGlobeDataset,
        class_rgb_values: List[Tuple[int, int, int]],
        k_values: List[int],
        unknown_index: Optional[int],
        normalize_targets: bool = True,
        device: str | torch.device = "cpu",
        samples_per_image: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.base = base
        self.class_rgb_values = class_rgb_values
        self.k_values = list(k_values)
        self.unknown_index = unknown_index
        self.normalize_targets = normalize_targets
        self.device = torch.device(device)
        self.samples_per_image = samples_per_image

        # Cache directories
        self.cache_root = Path(const.CACHE_DIR)
        self.fm_dir = self.cache_root / "feature_maps"
        self.graph_dir = self.cache_root / "graphs"
        self.tgt_dir = self.cache_root / "targets"
        for d in (self.fm_dir, self.graph_dir, self.tgt_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Build (image_idx, k) index map: select a subset of K values per image deterministically w.r.t. RNG seed.
        self.index_map: List[Tuple[int, int]] = []
        rng = np.random.default_rng(seed=42)
        if len(self.k_values) == 0:
            raise ValueError("k_values must contain at least one entry.")
        if self.samples_per_image is None:
            default_samples = 2
            self.samples_per_image = default_samples if len(self.k_values) >= default_samples else len(self.k_values)
        if self.samples_per_image <= 0:
            raise ValueError("samples_per_image must be positive.")
        for i in range(len(self.base)):
            if self.samples_per_image >= len(self.k_values):
                k_selected = list(self.k_values)
            else:
                k_selected = rng.choice(
                    self.k_values, size=self.samples_per_image, replace=False
                )
            for k in k_selected:
                self.index_map.append((i, k))

        # Simple in-process cache for per-image CNN feature maps to avoid recomputation across K in the same process
        self._feature_map_cache: Dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Data:
        image_idx, k = self.index_map[idx]
        img_t, img_rgb, mask_t = self.base[image_idx]

        # 1) Feature map (on-disk cached per image)
        Fm = self._load_feature_map(image_idx, img_t)

        # 2) Graph for this K (on-disk cached X, sp and edge_index)
        X, edge_index, sp = self._load_graph(image_idx, Fm, img_rgb, k)

        # 3) Targets per superpixel (on-disk cached per image,k)
        y = self._load_targets(image_idx, k, sp, mask_t)

        data = Data(
            x=X.float(),
            edge_index=edge_index.long(),
            y=y.float(),
            meta={
                "image_idx": int(image_idx),
                "k": int(k),
                "num_nodes": int(X.shape[0]),
            },
        )
        return data

    # ---- Cache helpers ----
    def _orig_image_path(self, image_idx: int) -> str:
        if isinstance(self.base, Subset):
            orig_ds = self.base.dataset
            orig_idx = self.base.indices[image_idx]
            return orig_ds.image_paths[orig_idx]
        else:
            return self.base.image_paths[image_idx]

    def _image_key(self, image_idx: int) -> str:
        p = Path(self._orig_image_path(image_idx))
        return p.stem  # e.g., 'xxx_sat' as unique key

    def _load_feature_map(self, image_idx: int, img_t: torch.Tensor) -> torch.Tensor:
        key = self._image_key(image_idx)
        fm_path = self.fm_dir / f"{key}_fm.pt"
        if fm_path.exists():
            try:
                Fm = torch.load(fm_path, map_location="cpu")
            except (EOFError, RuntimeError, ValueError):
                print(f"Warning: Corrupted feature map cache for {key}, regenerating...")
                Fm = extract_image_feature_map(img_t, device=self.device)
                torch.save(Fm.cpu(), fm_path)
        else:
            Fm = extract_image_feature_map(img_t, device=self.device)
            torch.save(Fm.cpu(), fm_path)
        # Keep also an in-process cache to avoid reloading within same epoch
        self._feature_map_cache[image_idx] = Fm
        return Fm

    def _load_graph(self, image_idx: int, Fm: torch.Tensor, img_rgb: np.ndarray, k: int):
        key = self._image_key(image_idx)
        sp_path = self.graph_dir / f"{key}_k{k}_sp.npy"
        ei_path = self.graph_dir / f"{key}_k{k}_edge_index.pt"
        x_path = self.graph_dir / f"{key}_k{k}_X.pt"
        if sp_path.exists() and ei_path.exists() and x_path.exists():
            try:
                sp = np.load(sp_path)
                edge_index = torch.load(ei_path, map_location="cpu")
                X = torch.load(x_path, map_location="cpu")
                return X, edge_index, sp
            except (EOFError, RuntimeError, ValueError) as e:
                # Corrupted cache file, regenerate
                print(f"Warning: Corrupted cache for {key}_k{k}, regenerating...")
                pass
        # Generate new graph
        X, edge_index, sp = get_slic_graph(Fm, img_rgb, k=k, device=self.device)
        np.save(sp_path, sp)
        torch.save(edge_index.cpu(), ei_path)
        torch.save(X.cpu(), x_path)
        return X, edge_index, sp

    def _load_targets(self, image_idx: int, k: int, sp: np.ndarray, mask_t: torch.Tensor) -> torch.Tensor:
        key = self._image_key(image_idx)
        num_classes = len(self.class_rgb_values)
        norm_flag = 1 if self.normalize_targets else 0
        unk = -1 if self.unknown_index is None else int(self.unknown_index)
        y_path = self.tgt_dir / f"{key}_k{k}_C{num_classes}_unk{unk}_n{norm_flag}.pt"
        if y_path.exists():
            try:
                return torch.load(y_path, map_location="cpu")
            except (EOFError, RuntimeError, ValueError):
                print(f"Warning: Corrupted target cache for {key}_k{k}, regenerating...")
                pass
        y = compute_superpixel_area_targets(
            sp=sp,
            mask=mask_t.numpy().astype(np.int64),
            num_classes=num_classes,
            unknown_index=self.unknown_index,
            normalize=self.normalize_targets,
        )
        torch.save(y.cpu(), y_path)
        return y
