from typing import Sequence, Tuple, List
import torch
import torch.nn.functional as F
from torchgeo.models import resnet50, ResNet50_Weights
from skimage.segmentation import slic
from skimage.graph import rag_mean_color
import numpy as np

@torch.inference_mode()
def extract_features(
    img_t: torch.Tensor,          # [3,H,W] or [1,3,H,W], normalized
    img_rgb: np.ndarray | None,   # [H,W,3] uint8; if None, we’ll reconstruct from img_t
    k: int = 500,
    device: str | torch.device = "cpu",
):
    """
    Returns:
        X:  torch.FloatTensor [N, 2048]  superpixel features
        sp: np.ndarray [H, W]            superpixel labels (0..N-1)
    """
    if img_t.ndim == 3:
        x = img_t.unsqueeze(0)                  # [1,3,H,W]
    elif img_t.ndim == 4:
        x = img_t                               # [N,3,H,W] (assume N==1 for this function)

    # Move to device & get H,W
    x = x.to(device)
    _, _, H, W = x.shape

    # 1) Backbone
    backbone = resnet50(
        weights=ResNet50_Weights.FMOW_RGB_GASSL,
        num_classes=0,
        global_pool="",
    ).to(device).eval()

    Fm = backbone(x)               # [1, 2048, h, w]
    _, C, h, w = Fm.shape
    Fm = Fm[0]                     # [2048, h, w]

    sp = slic(img_rgb, n_segments=k, compactness=10, start_label=0)  # [H,W]
    N = int(sp.max()) + 1

    # 3) Downsample superpixel map to (h,w) and masked mean pooling
    sp_small = torch.from_numpy(sp)[None, None].float().to(device)   # [1,1,H,W]
    sp_small = F.interpolate(sp_small, size=(h, w), mode="nearest")[0, 0].long()  # [h,w]

    F_flat = Fm.reshape(C, -1)             # [C, h*w]
    sp_flat = sp_small.reshape(-1)         # [h*w]

    ones = torch.ones(h * w, dtype=F_flat.dtype, device=device)
    denom = torch.zeros(N, dtype=F_flat.dtype, device=device).scatter_add_(0, sp_flat, ones) + 1e-6  # [N]

    X = torch.zeros(N, C, dtype=F_flat.dtype, device=device)
    idx = sp_flat.unsqueeze(0).expand(C, -1)           # [C, h*w]
    X.scatter_add_(0, idx.T, F_flat.T)                 # sum features per SP
    X = (X.T / denom).T                                # mean -> [N, 2048]

    rag = rag_mean_color(img_rgb, sp)   # Region Adjacency Graph over Superpixels

    return X.cpu(), sp, rag