from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import torch


def compute_edge_index_from_superpixels(sp: np.ndarray, connectivity: int = 8) -> torch.Tensor:
    """
    Build a directed edge_index [2, E] for an adjacency graph where nodes are superpixels.
    Two nodes are connected if their superpixels touch (4- or 8-connectivity).
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


