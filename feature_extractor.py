from typing import Sequence, Tuple, List, Optional

import torch
import torch.nn.functional as F
from torchgeo.models import resnet50, ResNet50_Weights
from skimage.segmentation import slic
import numpy as np

try:
    from torch.amp import autocast as torch_autocast
except Exception:
    from torch.cuda.amp import autocast as torch_autocast


@torch.inference_mode()
def extract_features(
    img_t: torch.Tensor,          # [3,H,W] or [1,3,H,W], normalized
    img_rgb: np.ndarray | None,   # [H,W,3] uint8; if None, we’ll reconstruct from img_t
    k: int = 500,
    device: str | torch.device = "cpu",
    backbone: Optional[torch.nn.Module] = None,
    use_amp: bool = True,
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
    local_backbone = backbone
    if local_backbone is None:
        local_backbone = resnet50(
            weights=ResNet50_Weights.FMOW_RGB_GASSL,
            num_classes=0,
            global_pool="",
        ).to(device).eval()

    with torch_autocast(device_type=str(device), enabled=(use_amp and torch.device(device).type == "cuda")):
        Fm = local_backbone(x)               # [1, 2048, h, w]
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

    return X.cpu(), sp


@torch.inference_mode()
def compute_backbone_map(
    img_t: torch.Tensor,
    device: str | torch.device = "cpu",
    backbone: Optional[torch.nn.Module] = None,
    use_amp: bool = True,
) -> torch.Tensor:
    """
    Return feature map Fm: FloatTensor [C, h, w]
    """
    if img_t.ndim == 3:
        x = img_t.unsqueeze(0)
    else:
        x = img_t
    x = x.to(device)
    local_backbone = backbone
    if local_backbone is None:
        local_backbone = resnet50(
            weights=ResNet50_Weights.FMOW_RGB_GASSL,
            num_classes=0,
            global_pool="",
        ).to(device).eval()
    with torch_autocast(device_type=str(device), enabled=(use_amp and torch.device(device).type == "cuda")):
        Fm = local_backbone(x)  # [1, C, h, w]
    return Fm[0].detach().to("cpu")  # [C, h, w] on CPU


@torch.inference_mode()
def pool_from_backbone_map(
    Fm: torch.Tensor,  # [C, h, w] on CPU or device
    sp: np.ndarray,    # [H, W] int labels
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Aggregate backbone feature map to superpixel features using mean pooling.
    Returns X: FloatTensor [N, C]
    """
    if isinstance(device, str):
        device = torch.device(device)
    C, h, w = Fm.shape
    # Move to device
    Fm = Fm.to(device)
    # Downsample sp to (h, w)
    sp_small = torch.from_numpy(sp)[None, None].float().to(device)
    sp_small = F.interpolate(sp_small, size=(h, w), mode="nearest")[0, 0].long()  # [h,w]

    N = int(sp.max()) + 1
    F_flat = Fm.reshape(C, -1)             # [C, h*w]
    sp_flat = sp_small.reshape(-1)         # [h*w]
    ones = torch.ones(h * w, dtype=F_flat.dtype, device=device)
    denom = torch.zeros(N, dtype=F_flat.dtype, device=device).scatter_add_(0, sp_flat, ones) + 1e-6
    X = torch.zeros(N, C, dtype=F_flat.dtype, device=device)
    idx = sp_flat.unsqueeze(0).expand(C, -1)           # [C, h*w]
    X.scatter_add_(0, idx.T, F_flat.T)                 # sum features per SP
    X = (X.T / denom).T                                # [N, C]
    return X.detach().to("cpu")