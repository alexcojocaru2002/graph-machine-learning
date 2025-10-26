from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import os
import tempfile
import hashlib

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset_loader import DeepGlobeDataset
from feature_extractor import (
    extract_features,
    compute_backbone_maps_vgg,
    pool_from_backbone_maps_max,
    compute_backbone_maps_vgg_batch,
)
from utils.graph_utils import compute_edge_index_from_superpixels, compute_superpixel_area_targets, slic_labels

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Process-local caches for features and adjacency (available to DataLoader workers)
_GLOBAL_FEATURES = {}
_GLOBAL_ADJ = {}


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
        # Optional: also cache backbone feature maps per image to accelerate multi-k runs and reuse across sessions
        cache_backbone_maps: bool = True,
        backbone_cache_dir: Optional[str | Path] = None,
        normalize_targets: bool = True,
        precompute: bool = True,
        slic_compactness: float = 10.0,
        slic_sigma: float = 0.0,
        slic_start_label: int = 0,
        use_amp: bool = True,
        hsv_threshold: float = 0.2,
        backbone: Optional[torch.nn.Module] = None,
        feature_batch_size: int = 8,
        precompute_workers: int = 0,
        slic_backend: str = "cpu",
    ) -> None:
        super().__init__()
        self.base_ds = DeepGlobeDataset(str(data_dir), class_rgb_values, img_size=img_size)
        self.class_rgb_values = list(class_rgb_values)
        self.unknown_index = unknown_index
        self.k_values = list(k_values)
        self.device = torch.device(device)
        self.feature_device = torch.device(feature_device)
        # Deprecated flag; caching is always enabled
        self.cache_features = True
        self.cache_dir = Path(cache_dir)
        self.normalize_targets = normalize_targets
        self.precompute_flag = precompute
        self.slic_compactness = float(slic_compactness)
        self.slic_sigma = float(slic_sigma)
        self.slic_start_label = int(slic_start_label)
        self.use_amp = bool(use_amp)
        self.hsv_threshold = float(hsv_threshold)
        self.backbone = backbone  # Optional pre-instantiated VGG16 features module placed on feature_device
        self.cache_backbone_maps = bool(cache_backbone_maps)
        if backbone_cache_dir is None:
            self.backbone_cache_dir = Path(cache_dir) / "backbone_maps"
        else:
            self.backbone_cache_dir = Path(backbone_cache_dir)
        self.feature_batch_size = int(max(1, feature_batch_size))
        self.precompute_workers = int(max(0, precompute_workers))
        # Backend is fixed to CPU (skimage)
        self.slic_backend = "cpu"

        # In-memory global caches shared across dataset instances in this process
        global _GLOBAL_FEATURES, _GLOBAL_ADJ
        try:
            _GLOBAL_FEATURES
        except NameError:
            _GLOBAL_FEATURES = {}
        try:
            _GLOBAL_ADJ
        except NameError:
            _GLOBAL_ADJ = {}

        # Build index mapping from linear idx -> (image_idx, k)
        self.index_map: List[Tuple[int, int]] = []
        for i in range(len(self.base_ds)):
            for k in self.k_values:
                self.index_map.append((i, k))

        # Always ensure caches exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.cache_backbone_maps:
            self.backbone_cache_dir.mkdir(parents=True, exist_ok=True)

        self.num_classes_eff = len(class_rgb_values) - (1 if (unknown_index is not None and 0 <= unknown_index < len(class_rgb_values)) else 0)

        # Optional upfront precompute of all features and SLIC to avoid runtime overhead
        if self.precompute_flag:
            print(f"[GraphSuperpixelDataset] Starting precompute: images={len(self.base_ds)}, ks={list(self.k_values)}; feature_batch_size={self.feature_batch_size}")
            self.precompute()

    def __len__(self) -> int:
        return len(self.index_map)

    def _features_cache_key(self, image_path: str, k: int, img_size: Optional[Tuple[int, int]]) -> Path:
        # Features depend on: image identity, resize size, k, SLIC params, backbone arch/weights
        name = Path(image_path).with_suffix("").name
        size_tag = "none" if img_size is None else f"{img_size[0]}x{img_size[1]}"
        slic_tag = f"c{int(self.slic_compactness)}_s{int(self.slic_sigma*10)}_st{self.slic_start_label}"
        bb_tag = "vgg16_i1kv1"
        h = hashlib.sha1(str(Path(image_path)).encode("utf-8")).hexdigest()[:10]
        fname = f"{name}_k{k}_size{size_tag}_slic-{slic_tag}_bb-{bb_tag}_{h}.npz"
        return self.cache_dir / fname

    def _features_cache_paths_npy(self, image_path: str, k: int, img_size: Optional[Tuple[int, int]]) -> Tuple[Path, Path]:
        base_npz = self._features_cache_key(image_path, k, img_size)
        base = str(base_npz.with_suffix(""))
        x_path = Path(base + "_X.npy")
        sp_path = Path(base + "_sp.npy")
        return x_path, sp_path

    def _adj_cache_key(self, image_path: str, k: int, img_size: Optional[Tuple[int, int]], hsv_threshold: float) -> Path:
        # Adjacency depends on sp (image, size, k, slic params) and hsv_threshold
        name = Path(image_path).with_suffix("").name
        size_tag = "none" if img_size is None else f"{img_size[0]}x{img_size[1]}"
        slic_tag = f"c{int(self.slic_compactness)}_s{int(self.slic_sigma*10)}_st{self.slic_start_label}"
        t_tag = f"t{int(hsv_threshold*1000)}"
        h = hashlib.sha1(str(Path(image_path)).encode("utf-8")).hexdigest()[:10]
        fname = f"{name}_k{k}_size{size_tag}_slic-{slic_tag}_{t_tag}_adj_{h}.npz"
        return self.cache_dir / fname

    def _adj_cache_path_npy(self, image_path: str, k: int, img_size: Optional[Tuple[int, int]], hsv_threshold: float) -> Path:
        base_npz = self._adj_cache_key(image_path, k, img_size, hsv_threshold)
        base = str(base_npz.with_suffix(""))
        return Path(base + "_edge_index.npy")

    def _backbone_cache_key(self, image_path: str, img_size: Optional[Tuple[int, int]]) -> Path:
        name = Path(image_path).with_suffix("").name
        size_tag = "none" if img_size is None else f"{img_size[0]}x{img_size[1]}"
        bb_tag = "vgg16_i1kv1"
        h = hashlib.sha1(str(Path(image_path)).encode("utf-8")).hexdigest()[:10]
        fname = f"{name}_size{size_tag}_bb-{bb_tag}_{h}.npz"
        return self.backbone_cache_dir / fname

    def _legacy_features_cache_key(self, image_path: str, k: int, img_size: Optional[Tuple[int, int]]) -> Path:
        # Backward compatibility: older caches included hsv_threshold and omitted SLIC params, backbone tag
        name = Path(image_path).with_suffix("").name
        size_tag = "none" if img_size is None else f"{img_size[0]}x{img_size[1]}"
        h = hashlib.sha1(str(Path(image_path)).encode("utf-8")).hexdigest()[:10]
        fname = f"{name}_k{k}_size{size_tag}_t{int(self.hsv_threshold*1000)}_{h}.npz"
        return self.cache_dir / fname

    def _labels_cache_key(self, image_path: str, k: int, img_size: Optional[Tuple[int, int]]) -> Path:
        # Labels depend on image identity, resize size, k, SLIC params, and unknown handling via num_classes_eff
        name = Path(image_path).with_suffix("").name
        size_tag = "none" if img_size is None else f"{img_size[0]}x{img_size[1]}"
        slic_tag = f"c{int(self.slic_compactness)}_s{int(self.slic_sigma*10)}_st{self.slic_start_label}"
        h = hashlib.sha1(str(Path(image_path)).encode("utf-8")).hexdigest()[:10]
        fname = f"{name}_k{k}_size{size_tag}_slic-{slic_tag}_ycounts_{h}.npz"
        return self.cache_dir / fname

    @staticmethod
    def _atomic_savez(path: Path, **arrays) -> None:
        path = Path(path)
        tmp_dir = path.parent
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=str(tmp_dir), delete=False, suffix=".npz") as tmp:
            tmp_name = tmp.name
        try:
            np.savez_compressed(tmp_name, **arrays)
            os.replace(tmp_name, path)
        finally:
            try:
                if os.path.exists(tmp_name):
                    os.remove(tmp_name)
            except Exception:
                pass

    def __getitem__(self, idx: int):
        image_idx, k = self.index_map[idx]
        img_t, img_rgb, mask_t = self.base_ds[image_idx]
        image_path = self.base_ds.image_paths[image_idx]
        mask_path = self.base_ds.mask_paths[image_idx]

        # Try in-memory cache first
        X: Optional[torch.Tensor] = None
        sp: Optional[np.ndarray] = None
        gkey_feat = (image_path, int(k))
        global _GLOBAL_FEATURES
        if gkey_feat in _GLOBAL_FEATURES:
            X, sp = _GLOBAL_FEATURES[gkey_feat]
        # Disk cache if needed
        feat_cache_path = self._features_cache_key(image_path, k, self.base_ds.img_size)
        x_path, sp_path = self._features_cache_paths_npy(image_path, k, self.base_ds.img_size)
        if (X is None or sp is None):
            # Prefer fast .npy memory-mapped loads
            try:
                if x_path.exists() and sp_path.exists():
                    X = torch.from_numpy(np.load(x_path, mmap_mode='r'))
                    sp = np.load(sp_path, mmap_mode='r').astype(np.int64)
                elif feat_cache_path.exists():
                    data = np.load(feat_cache_path, mmap_mode='r')
                    X = torch.from_numpy(data["X"])  # [N, 1024]
                    sp = data["sp"].astype(np.int64)
            except Exception:
                X, sp = X, sp
        else:
            # Backward-compat: try legacy cache naming
            legacy_path = self._legacy_features_cache_key(image_path, k, self.base_ds.img_size)
            if legacy_path.exists():
                try:
                    data = np.load(legacy_path, mmap_mode='r')
                    X = torch.from_numpy(data["X"])  # [N,1024]
                    sp = data["sp"].astype(np.int64)
                    # Migrate to new cache key for future runs
                    try:
                        self._atomic_savez(feat_cache_path, X=X.numpy(), sp=sp)
                    except Exception:
                        pass
                except Exception:
                    X, sp = None, None
            # If still missing, compute now and persist if caching enabled
            if X is None or sp is None:
                # Try to reuse cached backbone maps
                F4 = None
                F5 = None
                if self.cache_backbone_maps:
                    bb_path = self._backbone_cache_key(image_path, self.base_ds.img_size)
                    if bb_path.exists():
                        try:
                            bb = np.load(bb_path, mmap_mode='r')
                            F4 = torch.from_numpy(bb["F4"])  # [512,h4,w4]
                            F5 = torch.from_numpy(bb["F5"])  # [512,h5,w5]
                        except Exception:
                            F4, F5 = None, None
                if (F4 is None) or (F5 is None):
                    F4, F5 = compute_backbone_maps_vgg(img_t, device=self.feature_device, backbone=self.backbone, use_amp=self.use_amp)
                    if self.cache_backbone_maps:
                        try:
                            self._atomic_savez(self._backbone_cache_key(image_path, self.base_ds.img_size), F4=F4.numpy(), F5=F5.numpy())
                        except Exception:
                            pass
                sp = slic_labels(
                    img_rgb,
                    n_segments=k,
                    compactness=self.slic_compactness,
                    sigma=self.slic_sigma,
                    start_label=self.slic_start_label,
                    backend=self.slic_backend,
                )
                X = pool_from_backbone_maps_max(F4, F5, sp, device=self.feature_device)
                # Persist both legacy .npz and fast .npy for future runs
                try:
                    self._atomic_savez(feat_cache_path, X=X.numpy(), sp=sp)
                except Exception:
                    pass
                try:
                    np.save(x_path, X.numpy())
                    np.save(sp_path, sp)
                except Exception:
                    pass
        # Populate global memory cache
        _GLOBAL_FEATURES[gkey_feat] = (X, sp)

        # Adjacency graph
        # Adjacency graph (cached separately by hsv_threshold)
        adj_cache_path = self._adj_cache_key(image_path, k, self.base_ds.img_size, self.hsv_threshold)
        adj_npy_path = self._adj_cache_path_npy(image_path, k, self.base_ds.img_size, self.hsv_threshold)
        # Try in-memory adjacency cache first
        gkey_adj = (image_path, int(k), float(self.hsv_threshold))
        global _GLOBAL_ADJ
        if gkey_adj in _GLOBAL_ADJ:
            edge_index = _GLOBAL_ADJ[gkey_adj]
        elif adj_npy_path.exists():
            try:
                ei = np.load(adj_npy_path, mmap_mode='r')
                edge_index = torch.from_numpy(ei).long()
            except Exception:
                edge_index = None
        elif adj_cache_path.exists():
            try:
                adj = np.load(adj_cache_path, mmap_mode='r')
                edge_index = torch.from_numpy(adj["edge_index"]).long()
            except Exception:
                edge_index = None
        else:
            edge_index = None
            # Backward-compat: if using legacy features cache that stored edge_index, reuse and migrate
            try:
                legacy_path = self._legacy_features_cache_key(image_path, k, self.base_ds.img_size)
                if legacy_path.exists():
                    data = np.load(legacy_path, mmap_mode='r')
                    if "edge_index" in data.files:
                        edge_index = torch.from_numpy(data["edge_index"]).long()
                        try:
                            self._atomic_savez(adj_cache_path, edge_index=edge_index.numpy())
                        except Exception:
                            pass
            except Exception:
                pass
        if edge_index is None:
            edge_index = compute_edge_index_from_superpixels(
                sp,
                connectivity=8,
                rgb=img_rgb,
                hsv_threshold=self.hsv_threshold,
                add_self_loops=True,
            )  # [2, E]
            # Persist in both formats (npy for speed, npz for backwards compatibility)
            try:
                np.save(adj_npy_path, edge_index.numpy())
            except Exception:
                pass
            try:
                self._atomic_savez(adj_cache_path, edge_index=edge_index.numpy())
            except Exception:
                pass
        # Populate in-memory adjacency cache
        _GLOBAL_ADJ[gkey_adj] = edge_index

        # Targets from mask (cache y counts per (image,k))
        labels_cache_path = self._labels_cache_key(image_path, k, self.base_ds.img_size)
        try:
            if labels_cache_path.exists():
                lab = np.load(labels_cache_path, mmap_mode='r')
                y = torch.from_numpy(lab["y"]).float()
            else:
                mask_np = mask_t.numpy().astype(np.int64)
                y = compute_superpixel_area_targets(
                    sp=sp,
                    mask=mask_np,
                    num_classes=len(self.class_rgb_values),
                    unknown_index=self.unknown_index,
                    normalize=self.normalize_targets,
                )
                self._atomic_savez(labels_cache_path, y=y.numpy())
        except Exception:
            mask_np = mask_t.numpy().astype(np.int64)
            y = compute_superpixel_area_targets(
                sp=sp,
                mask=mask_np,
                num_classes=len(self.class_rgb_values),
                unknown_index=self.unknown_index,
                normalize=self.normalize_targets,
            )

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
        # Always precompute when called

        # Ensure backbone is ready on feature_device
        if self.backbone is None:
            from torchvision.models import vgg16, VGG16_Weights
            self.backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(self.feature_device).eval()

        num_images = len(self.base_ds)
        indices = list(range(num_images))
        rng = range(0, num_images, self.feature_batch_size)
        if tqdm is not None:
            rng = tqdm(rng, desc="Precompute CNN backbone (batched)")

        # 1) Batched backbone forward for images missing backbone cache
        for start in rng:
            end = min(start + self.feature_batch_size, num_images)
            batch_idx = []
            imgs_t: List[torch.Tensor] = []
            for i in range(start, end):
                image_path = self.base_ds.image_paths[i]
                bb_path = self._backbone_cache_key(image_path, self.base_ds.img_size)
                if self.cache_backbone_maps and bb_path.exists():
                    continue
                img_t, _img_rgb, _ = self.base_ds[i]
                batch_idx.append(i)
                imgs_t.append(img_t)
            if len(batch_idx) == 0:
                continue
            F4_list, F5_list = compute_backbone_maps_vgg_batch(imgs_t, device=self.feature_device, backbone=self.backbone, use_amp=self.use_amp)
            for i, F4, F5 in zip(batch_idx, F4_list, F5_list):
                image_path = self.base_ds.image_paths[i]
                try:
                    self._atomic_savez(self._backbone_cache_key(image_path, self.base_ds.img_size), F4=F4.numpy(), F5=F5.numpy())
                except Exception:
                    pass

        # 2) For each image and k, compute or LOAD SLIC + pooled features and adjacency into in-memory caches
        it2 = range(num_images)
        if tqdm is not None:
            it2 = tqdm(it2, desc="Precompute SLIC+pool+adj")
        for i in it2:
            img_t, img_rgb, _ = self.base_ds[i]
            image_path = self.base_ds.image_paths[i]
            # Load backbone maps (should exist now)
            F4: Optional[torch.Tensor] = None
            F5: Optional[torch.Tensor] = None
            if self.cache_backbone_maps:
                bb_path = self._backbone_cache_key(image_path, self.base_ds.img_size)
                if bb_path.exists():
                    try:
                        bb = np.load(bb_path, mmap_mode='r')
                        F4 = torch.from_numpy(bb["F4"])  # [512,h4,w4]
                        F5 = torch.from_numpy(bb["F5"])  # [512,h5,w5]
                    except Exception:
                        F4, F5 = None, None
            if (F4 is None) or (F5 is None):
                F4, F5 = compute_backbone_maps_vgg(img_t, device=self.feature_device, backbone=self.backbone, use_amp=self.use_amp)
                if self.cache_backbone_maps:
                    try:
                        self._atomic_savez(self._backbone_cache_key(image_path, self.base_ds.img_size), F4=F4.numpy(), F5=F5.numpy())
                    except Exception:
                        pass

            for k in self.k_values:
                feat_cache_path = self._features_cache_key(image_path, k, self.base_ds.img_size)
                # Load cached features if available, else compute then save
                try:
                    if feat_cache_path.exists():
                        data = np.load(feat_cache_path, mmap_mode='r')
                        sp = data["sp"].astype(np.int64)
                        X = torch.from_numpy(data["X"])  # [N, 1024]
                    else:
                        sp = slic_labels(
                            img_rgb,
                            n_segments=k,
                            compactness=self.slic_compactness,
                            sigma=self.slic_sigma,
                            start_label=self.slic_start_label,
                        )
                        X = pool_from_backbone_maps_max(F4, F5, sp, device=self.feature_device)
                        self._atomic_savez(feat_cache_path, X=X.numpy(), sp=sp)
                except Exception:
                    # As a fallback, recompute and try to persist
                    sp = slic_labels(
                        img_rgb,
                        n_segments=k,
                        compactness=self.slic_compactness,
                        sigma=self.slic_sigma,
                        start_label=self.slic_start_label,
                    )
                    X = pool_from_backbone_maps_max(F4, F5, sp, device=self.feature_device)
                    try:
                        self._atomic_savez(feat_cache_path, X=X.numpy(), sp=sp)
                    except Exception:
                        pass

                # Store in-process cache for dataloader fast path
                global _GLOBAL_FEATURES
                _GLOBAL_FEATURES[(image_path, int(k))] = (X, sp)

                # Precompute adjacency per (image,k,threshold)
                adj_cache_path = self._adj_cache_key(image_path, k, self.base_ds.img_size, self.hsv_threshold)
                # Load adjacency if available, else compute then save
                try:
                    if adj_cache_path.exists():
                        adj = np.load(adj_cache_path, mmap_mode='r')
                        edge_index = torch.from_numpy(adj["edge_index"]).long()
                    else:
                        edge_index = compute_edge_index_from_superpixels(
                            sp,
                            connectivity=8,
                            rgb=img_rgb,
                            hsv_threshold=self.hsv_threshold,
                            add_self_loops=True,
                        )
                        self._atomic_savez(adj_cache_path, edge_index=edge_index.numpy())
                except Exception:
                    edge_index = compute_edge_index_from_superpixels(
                        sp,
                        connectivity=8,
                        rgb=img_rgb,
                        hsv_threshold=self.hsv_threshold,
                        add_self_loops=True,
                    )
                    try:
                        self._atomic_savez(adj_cache_path, edge_index=edge_index.numpy())
                    except Exception:
                        pass

                # Store in-process adjacency cache
                global _GLOBAL_ADJ
                _GLOBAL_ADJ[(image_path, int(k), float(self.hsv_threshold))] = edge_index


