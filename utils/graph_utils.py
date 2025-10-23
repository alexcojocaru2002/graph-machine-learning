from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from skimage.color import rgb2hsv


def compute_edge_index_from_superpixels(
    sp: np.ndarray,
    connectivity: int = 8,
    rgb: Optional[np.ndarray] = None,
    hsv_threshold: Optional[float] = None,
    add_self_loops: bool = True,
) -> torch.Tensor:
    """
    Build a directed edge_index [2, E] for superpixel adjacency.
    - Two nodes are adjacent if superpixels touch (4- or 8-connectivity).
    - If hsv_threshold is provided and rgb is given, keep edge (i,j) only if ||v_i - v_j||_2 <= hsv_threshold
      where v_i is mean HSV of region i (values in [0,1]).
    - If add_self_loops, include (i,i) for all nodes.
    """
    assert sp.ndim == 2, "superpixel map must be 2D"
    H, W = sp.shape
    max_id = int(sp.max())
    N = max_id + 1

    neighbors = set()

    # 4-connectivity (right and down to avoid duplicates; we'll add symmetric edges later)
    right_diff = sp[:, :-1] != sp[:, 1:]
    ys, xs = np.where(right_diff)
    for y, x in zip(ys, xs):
        a, b = int(sp[y, x]), int(sp[y, x + 1])
        neighbors.add((a, b))
        neighbors.add((b, a))

    down_diff = sp[:-1, :] != sp[1:, :]
    ys, xs = np.where(down_diff)
    for y, x in zip(ys, xs):
        a, b = int(sp[y, x]), int(sp[y + 1, x])
        neighbors.add((a, b))
        neighbors.add((b, a))

    if connectivity == 8:
        # Diagonals
        diag1_diff = sp[:-1, :-1] != sp[1:, 1:]
        ys, xs = np.where(diag1_diff)
        for y, x in zip(ys, xs):
            a, b = int(sp[y, x]), int(sp[y + 1, x + 1])
            neighbors.add((a, b))
            neighbors.add((b, a))

        diag2_diff = sp[:-1, 1:] != sp[1:, :-1]
        ys, xs = np.where(diag2_diff)
        for y, x in zip(ys, xs):
            a, b = int(sp[y, x + 1]), int(sp[y + 1, x])
            neighbors.add((a, b))
            neighbors.add((b, a))

    # Optional HSV similarity filtering
    if hsv_threshold is not None and rgb is not None:
        hsv = rgb2hsv(rgb.astype(np.float32) / 255.0)
        # Vectorized mean HSV per region using bincount
        flat_ids = sp.reshape(-1).astype(np.int64)
        hsv_flat = hsv.reshape(-1, 3).astype(np.float32)
        counts = np.bincount(flat_ids, minlength=N).astype(np.int64)
        # Avoid division by zero
        counts_safe = counts.copy()
        counts_safe[counts_safe == 0] = 1
        sums = np.stack([
            np.bincount(flat_ids, weights=hsv_flat[:, 0], minlength=N),
            np.bincount(flat_ids, weights=hsv_flat[:, 1], minlength=N),
            np.bincount(flat_ids, weights=hsv_flat[:, 2], minlength=N),
        ], axis=1).astype(np.float32)
        means = sums / counts_safe[:, None]

        filtered = set()
        t = float(hsv_threshold)
        for (a, b) in neighbors:
            if a == b:
                filtered.add((a, b))
                continue
            dv = means[a] - means[b]
            dist = float(np.sqrt(np.sum(dv * dv)))
            if dist <= t:
                filtered.add((a, b))
        neighbors = filtered

    if add_self_loops:
        for i in range(N):
            neighbors.add((i, i))

    if len(neighbors) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    edge_index = torch.tensor(list(neighbors), dtype=torch.long).T  # [2, E]
    return edge_index


def compute_superpixel_area_targets(
    sp: np.ndarray,
    mask: np.ndarray,
    num_classes: int,
    unknown_index: Optional[int] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute per-superpixel area per class.

    Returns FloatTensor [N, C_eff] where C_eff excludes unknown if provided.
    If normalize=True, each row sums to 1 over included classes (area fractions).
    """
    assert sp.ndim == 2 and mask.ndim == 2, "sp and mask must be (H, W)"
    assert sp.shape == mask.shape, "sp and mask must have the same shape"
    H, W = sp.shape
    N = int(sp.max()) + 1

    sp_flat = sp.reshape(-1).astype(np.int64)
    mask_flat = mask.reshape(-1).astype(np.int64)

    # If unknown_index is provided, we will exclude it at the end
    C_eff = num_classes - (1 if (unknown_index is not None and 0 <= unknown_index < num_classes) else 0)

    # Pair encoding to count occurrences efficiently
    pair = sp_flat * num_classes + mask_flat  # unique id per (sp_id, class_id)
    counts = np.bincount(pair, minlength=N * num_classes).reshape(N, num_classes)

    if unknown_index is not None and 0 <= unknown_index < num_classes:
        counts = np.delete(counts, unknown_index, axis=1)

    if normalize:
        # Normalize by superpixel size over included classes
        denom = counts.sum(axis=1, keepdims=True).astype(np.float32)
        denom[denom == 0] = 1.0
        targets = (counts.astype(np.float32) / denom).astype(np.float32)
    else:
        targets = counts.astype(np.float32)

    return torch.from_numpy(targets)


