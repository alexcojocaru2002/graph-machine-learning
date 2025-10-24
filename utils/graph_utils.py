from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from skimage.color import rgb2hsv
from typing import Optional

try:
    # Optional GPU SLIC via cuCIM, API similar to skimage
    from cucim.skimage.segmentation import slic as cucim_slic  # type: ignore
    _HAS_CUCIM = True
except Exception:
    _HAS_CUCIM = False


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

    # Build adjacency pairs vectorized
    pairs = []
    # 4-connectivity
    right_diff = sp[:, :-1] != sp[:, 1:]
    ys, xs = np.where(right_diff)
    if ys.size:
        a = sp[ys, xs]
        b = sp[ys, xs + 1]
        pairs.append(np.stack([a, b], axis=1))
        pairs.append(np.stack([b, a], axis=1))

    down_diff = sp[:-1, :] != sp[1:, :]
    ys, xs = np.where(down_diff)
    if ys.size:
        a = sp[ys, xs]
        b = sp[ys + 1, xs]
        pairs.append(np.stack([a, b], axis=1))
        pairs.append(np.stack([b, a], axis=1))

    if connectivity == 8:
        diag1_diff = sp[:-1, :-1] != sp[1:, 1:]
        ys, xs = np.where(diag1_diff)
        if ys.size:
            a = sp[ys, xs]
            b = sp[ys + 1, xs + 1]
            pairs.append(np.stack([a, b], axis=1))
            pairs.append(np.stack([b, a], axis=1))

        diag2_diff = sp[:-1, 1:] != sp[1:, :-1]
        ys, xs = np.where(diag2_diff)
        if ys.size:
            a = sp[ys, xs + 1]
            b = sp[ys + 1, xs]
            pairs.append(np.stack([a, b], axis=1))
            pairs.append(np.stack([b, a], axis=1))

    if len(pairs) == 0:
        base_pairs = np.empty((0, 2), dtype=np.int64)
    else:
        base_pairs = np.concatenate(pairs, axis=0).astype(np.int64)
        # Remove self duplicates from this stage; we'll add explicit self-loops later if requested
        # Also unique the edges to shrink work for HSV filtering
        if base_pairs.size:
            base_pairs = np.unique(base_pairs, axis=0)

    # Optional HSV similarity filtering
    neighbors = base_pairs

    if hsv_threshold is not None and rgb is not None and neighbors.size:
        hsv = rgb2hsv(rgb.astype(np.float32) / 255.0)
        flat_ids = sp.reshape(-1).astype(np.int64)
        hsv_flat = hsv.reshape(-1, 3).astype(np.float32)
        counts = np.bincount(flat_ids, minlength=N).astype(np.int64)
        counts_safe = counts.copy()
        counts_safe[counts_safe == 0] = 1
        sums = np.stack([
            np.bincount(flat_ids, weights=hsv_flat[:, 0], minlength=N),
            np.bincount(flat_ids, weights=hsv_flat[:, 1], minlength=N),
            np.bincount(flat_ids, weights=hsv_flat[:, 2], minlength=N),
        ], axis=1).astype(np.float32)
        means = sums / counts_safe[:, None]

        a_idx = neighbors[:, 0]
        b_idx = neighbors[:, 1]
        dv = means[a_idx] - means[b_idx]
        dist = np.sqrt(np.sum(dv * dv, axis=1))
        mask = (dist <= float(hsv_threshold)) | (a_idx == b_idx)
        neighbors = neighbors[mask]

    if add_self_loops:
        diag = np.arange(N, dtype=np.int64)
        self_loops = np.stack([diag, diag], axis=1)
        if neighbors.size:
            neighbors = np.concatenate([neighbors, self_loops], axis=0)
        else:
            neighbors = self_loops

    if neighbors.size == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    # Unique one last time to avoid duplicates
    neighbors = np.unique(neighbors, axis=0)
    edge_index = torch.from_numpy(neighbors.T.astype(np.int64))  # [2, E]
    return edge_index


def slic_labels(
    img_rgb: np.ndarray,
    n_segments: int,
    compactness: float,
    sigma: float,
    start_label: int,
    backend: str = "auto",
) -> np.ndarray:
    """
    Compute SLIC superpixels with selectable backend. Returns labels [H,W] int64 starting at start_label.
    backends: 'cpu' (skimage), 'cucim' (GPU via cuCIM), 'auto' (prefer cucim when available)
    """
    if backend not in ("cpu", "cucim", "auto"):
        backend = "auto"
    if backend in ("cucim", "auto") and _HAS_CUCIM:
        try:
            # cuCIM aims for skimage API compatibility
            labels = cucim_slic(
                img_rgb,
                n_segments=int(n_segments),
                compactness=float(compactness),
                sigma=float(sigma),
                start_label=int(start_label),
                channel_axis=-1,
            )
            # Ensure numpy on CPU, contiguous int64
            if hasattr(labels, "get"):
                labels = labels.get()
            labels = np.asarray(labels)
            return labels.astype(np.int64, copy=False)
        except Exception:
            # Fallback to CPU implementation
            pass
    # CPU skimage fallback
    from skimage.segmentation import slic as sk_slic
    labels = sk_slic(
        img_rgb,
        n_segments=int(n_segments),
        compactness=float(compactness),
        sigma=float(sigma),
        start_label=int(start_label),
    )
    return labels.astype(np.int64, copy=False)


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


