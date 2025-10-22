from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset_loader import DeepGlobeDataset
from feature_extractor import extract_features, compute_backbone_map, pool_from_backbone_map
from utils.graph_utils import compute_edge_index_from_superpixels, compute_superpixel_area_targets
from skimage.segmentation import slic

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


class GraphSuperpixelDataset(Dataset):
    """
    Wraps DeepGlobeDataset to produce graph-based samples per image and per SLIC k.

    Each item corresponds to a (image_idx, k) pair and returns:
      - x:  FloatTensor [N, F] superpixel features
      - edge_index: LongTensor [2, E] adjacency (directed edges)
      - y:  FloatTensor [N, C] per-superpixel area per class (fractions if normalize=True)
      - meta: dict with keys {image_path, mask_path, k, num_nodes}
    """

    def __init__(
        self,
        data_dir: str | Path,
        class_rgb_values: Sequence[Tuple[int, int, int]],
        unknown_index: Optional[int] = None,
        k_values: Sequence[int] = (400,),
        img_size: Optional[Tuple[int, int]] = None,
        device: str | torch.device = "cpu",
        feature_device: str | torch.device = "cpu",
        cache_features: bool = False,
        cache_dir: str | Path = "artifacts/features",
        normalize_targets: bool = True,
        precompute: bool = True,
        slic_compactness: float = 10.0,
        slic_sigma: float = 0.0,
        slic_start_label: int = 0,
        use_amp: bool = True,
    ) -> None:
        super().__init__()
        self.base_ds = DeepGlobeDataset(str(data_dir), class_rgb_values, img_size=img_size)
        self.class_rgb_values = list(class_rgb_values)
        self.unknown_index = unknown_index
        self.k_values = list(k_values)
        self.device = torch.device(device)
        self.feature_device = torch.device(feature_device)
        self.cache_features = cache_features
        self.cache_dir = Path(cache_dir)
        self.normalize_targets = normalize_targets
        self.precompute_flag = precompute
        self.slic_compactness = float(slic_compactness)
        self.slic_sigma = float(slic_sigma)
        self.slic_start_label = int(slic_start_label)
        self.use_amp = bool(use_amp)

        # Build index mapping from linear idx -> (image_idx, k)
        self.index_map: List[Tuple[int, int]] = []
        for i in range(len(self.base_ds)):
            for k in self.k_values:
                self.index_map.append((i, k))

        if self.cache_features:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.num_classes_eff = len(class_rgb_values) - (1 if (unknown_index is not None and 0 <= unknown_index < len(class_rgb_values)) else 0)

        # Optional upfront precompute of all features and SLIC to avoid runtime overhead
        if self.cache_features and self.precompute_flag:
            self.precompute()

    def __len__(self) -> int:
        return len(self.index_map)

    def _cache_key(self, image_path: str, k: int, img_size: Optional[Tuple[int, int]]) -> Path:
        name = Path(image_path).with_suffix("").name  # strip extension
        size_tag = "none" if img_size is None else f"{img_size[0]}x{img_size[1]}"
        fname = f"{name}_k{k}_size{size_tag}.npz"
        return self.cache_dir / fname

    def __getitem__(self, idx: int):
        image_idx, k = self.index_map[idx]
        img_t, img_rgb, mask_t = self.base_ds[image_idx]
        image_path = self.base_ds.image_paths[image_idx]
        mask_path = self.base_ds.mask_paths[image_idx]

        # Try cache
        X: Optional[torch.Tensor] = None
        sp: Optional[np.ndarray] = None
        cache_path = self._cache_key(image_path, k, self.base_ds.img_size)
        if self.cache_features and cache_path.exists():
            data = np.load(cache_path)
            X = torch.from_numpy(data["X"])  # [N, 2048]
            sp = data["sp"].astype(np.int64)
        else:
            # Compute now (slower). Prefer calling precompute() once to populate cache.
            Fm = compute_backbone_map(img_t, device=self.feature_device, backbone=None, use_amp=self.use_amp)
            sp = slic(img_rgb, n_segments=k, compactness=self.slic_compactness, sigma=self.slic_sigma, start_label=self.slic_start_label)
            X = pool_from_backbone_map(Fm, sp, device=self.feature_device)
            if self.cache_features:
                np.savez_compressed(cache_path, X=X.numpy(), sp=sp)

        # Adjacency graph
        edge_index = compute_edge_index_from_superpixels(sp)  # [2, E]

        # Targets from mask
        mask_np = mask_t.numpy().astype(np.int64)
        y = compute_superpixel_area_targets(
            sp=sp,
            mask=mask_np,
            num_classes=len(self.class_rgb_values),
            unknown_index=self.unknown_index,
            normalize=self.normalize_targets,
        )  # [N, C_eff]

        sample = {
            "x": X.float(),
            "edge_index": edge_index.long(),
            "y": y.float(),
            "meta": {
                "image_path": image_path,
                "mask_path": mask_path,
                "k": int(k),
                "num_nodes": int(X.shape[0]),
            },
        }
        return sample

    def precompute(self) -> None:
        """
        Precompute backbone feature maps per image (once) and SLIC+pooled features for each k,
        saving (X, sp) to cache. Skips files already cached.
        """
        if not self.cache_features:
            return
        iterator = range(len(self.base_ds))
        if tqdm is not None:
            iterator = tqdm(iterator, desc="Precompute CNN+SLIC")

        # Reuse one backbone instance via compute_backbone_map (by passing backbone as None here we let it lazily create once per call).
        backbone = None

        for i in iterator:
            img_t, img_rgb, _ = self.base_ds[i]
            image_path = self.base_ds.image_paths[i]
            # Compute backbone map once per image
            Fm = compute_backbone_map(img_t, device=self.feature_device, backbone=backbone, use_amp=self.use_amp)
            # For each k, compute SLIC and pooled features if not cached
            for k in self.k_values:
                cache_path = self._cache_key(image_path, k, self.base_ds.img_size)
                if cache_path.exists():
                    continue
                sp = slic(img_rgb, n_segments=k, compactness=self.slic_compactness, sigma=self.slic_sigma, start_label=self.slic_start_label)
                X = pool_from_backbone_map(Fm, sp, device=self.feature_device)
                np.savez_compressed(cache_path, X=X.numpy(), sp=sp)


