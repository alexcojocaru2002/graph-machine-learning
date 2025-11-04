from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

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
    ) -> None:
        super().__init__()
        self.base = base
        self.class_rgb_values = class_rgb_values
        self.k_values = list(k_values)
        self.unknown_index = unknown_index
        self.normalize_targets = normalize_targets
        self.device = torch.device(device)

        # Build (image_idx, k) index map, optionally filtered by image indices
        self.index_map: List[Tuple[int, int]] = []
        for i in range(len(self.base)):
            for k in self.k_values:
                self.index_map.append((i, k))

        # Simple in-process cache for per-image CNN feature maps to avoid recomputation across K
        self._feature_map_cache: Dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Data:
        image_idx, k = self.index_map[idx]
        img_t, img_rgb, mask_t = self.base[image_idx]

        # 1) Backbone feature map (reuse per image if multiple Ks)
        if image_idx in self._feature_map_cache:
            Fm = self._feature_map_cache[image_idx]
        else:
            Fm = extract_image_feature_map(img_t, device=self.device)
            self._feature_map_cache[image_idx] = Fm

        # 2) Superpixel features and graph for this K
        X, edge_index, sp = get_slic_graph(Fm, img_rgb, k=k, device=self.device)

        # 3) Area-fraction targets per superpixel
        y = compute_superpixel_area_targets(
            sp=sp,
            mask=mask_t.numpy().astype(np.int64),
            num_classes=len(self.class_rgb_values),
            unknown_index=self.unknown_index,
            normalize=self.normalize_targets,
        )

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
