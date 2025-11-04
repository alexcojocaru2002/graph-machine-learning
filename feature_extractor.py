from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch.amp import autocast as torch_autocast
from torchgeo.models import resnet50, ResNet50_Weights
from skimage.segmentation import slic
import numpy as np
from torchvision.models import vgg16, VGG16_Weights

from utils.graph_utils import compute_edge_index_from_superpixels

@torch.inference_mode()
def extract_image_feature_map(
    img_t: torch.Tensor,
    device: str | torch.device = "cpu",
):
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

    return backbone(x)               # [1, 2048, h, w]

@torch.inference_mode()
def get_slic_graph(
    feature_map: np.ndarray, # [3,H,W] or [1,3,H,W], normalized
    img_rgb: np.ndarray,   # [H,W,3] uint8;
    k: int = 500,
    device: str | torch.device = "cpu",
):
    """
    Returns:
        X:  torch.FloatTensor [N, 2048]  superpixel features
        sp: np.ndarray [H, W]            superpixel labels (0..N-1)
    """

    _, C, h, w = feature_map.shape
    Fm = feature_map[0]

    sp = slic(img_rgb, n_segments=k, compactness=10, start_label=0)  # [H,W]
    N = int(sp.max()) + 1

    # 3) Downsample superpixel map to (h,w) and masked MAX pooling
    sp_small = torch.from_numpy(sp)[None, None].float().to(device)   # [1,1,H,W]
    sp_small = F.interpolate(sp_small, size=(h, w), mode="nearest")[0, 0].long()  # [h,w]

    F_flat = Fm.reshape(C, -1).transpose(0, 1)   # [P, C]
    sp_flat = sp_small.reshape(-1)               # [P]
    P = int(sp_flat.numel())
    index = sp_flat.unsqueeze(1).expand(P, C)    # [P, C]
    X = torch.full((N, C), -float("inf"), dtype=F_flat.dtype, device=device)
    X.scatter_reduce_(0, index, F_flat, reduce='amax', include_self=True)
    X[~torch.isfinite(X)] = 0.0                  # handle empty segments if any

    return X.cpu(), compute_edge_index_from_superpixels(sp, rgb=img_rgb), sp

@torch.inference_mode()
def compute_backbone_maps_vgg(
    img_t: torch.Tensor,
    device: str | torch.device = "cpu",
    backbone: Optional[torch.nn.Module] = None,
    use_amp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return VGG16 conv4_3 and conv5_3 feature maps as (F4, F5): each FloatTensor [512, h, w] on CPU.
    """
    if img_t.ndim == 3:
        x = img_t.unsqueeze(0)
    else:
        x = img_t
    x = x.to(device)
    local_backbone = backbone
    if local_backbone is None:
        local_backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
    feats = []
    with torch_autocast(device_type=str(device), enabled=(use_amp and torch.device(device).type == "cuda")):
        out = x
        for i, layer in enumerate(local_backbone):
            out = layer(out)
            if i in (22, 29):
                feats.append(out)
            if i >= 29 and len(feats) == 2:
                break
    F4, F5 = feats
    return F4[0].detach().to("cpu"), F5[0].detach().to("cpu")


@torch.inference_mode()
def pool_from_backbone_maps_max(
    F4: torch.Tensor,  # [512, h4, w4]
    F5: torch.Tensor,  # [512, h5, w5]
    sp: np.ndarray,    # [H, W]
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Aggregate VGG conv4_3 and conv5_3 feature maps to superpixel features using MAX pooling.
    Returns X: FloatTensor [N, 1024]
    """
    if isinstance(device, str):
        device = torch.device(device)
    H, W = sp.shape
    F4 = F4.to(device)
    F5 = F5.to(device)
    F4_up = F.interpolate(F4.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
    F5_up = F.interpolate(F5.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
    Fcat = torch.cat([F4_up, F5_up], dim=0)  # [1024,H,W]
    return _pool_from_feature_tensor_max(Fcat, sp, device=device)


@torch.inference_mode()
def _pool_from_feature_tensor_max(
    Fcat: torch.Tensor,  # [C, H, W]
    sp: np.ndarray,      # [H, W]
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Vectorized MAX pooling over superpixel regions without per-channel Python loops.
    Returns X: FloatTensor [N, C] on CPU.
    """
    if isinstance(device, str):
        device = torch.device(device)
    C, H, W = Fcat.shape
    sp_t = torch.from_numpy(sp).to(device)
    sp_flat = sp_t.reshape(-1)
    C = Fcat.shape[0]
    idx = sp_t.reshape(-1).long()  # [P]
    P = int(idx.numel())
    N = int(sp.max()) + 1
    X = torch.full((N, C), -float("inf"), dtype=Fcat.dtype, device=device)
    F_flat = Fcat.reshape(C, -1)
    for c in range(C):
        vals = F_flat[c]
        X[:, c].scatter_reduce_(0, sp_flat, vals, reduce='amax', include_self=True)
    F_flat = Fcat.reshape(C, -1).transpose(0, 1)  # [P, C]
    X = torch.full((N, C), -float("inf"), dtype=F_flat.dtype, device=device)
    index = idx.unsqueeze(1).expand(P, C)
    X.scatter_reduce_(0, index, F_flat, reduce='amax', include_self=True)
    X[~torch.isfinite(X)] = 0.0
    return X.detach().to("cpu")


@torch.inference_mode()
def compute_backbone_maps_vgg_batch(
    imgs_t: List[torch.Tensor],
    device: str | torch.device = "cpu",
    backbone: Optional[torch.nn.Module] = None,
    use_amp: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Batched VGG16 forward that returns lists of conv4_3 and conv5_3 feature maps on CPU.
    Each element in the returned lists is a FloatTensor [512, h, w].
    """
    if len(imgs_t) == 0:
        return [], []
    if isinstance(device, str):
        device = torch.device(device)
    x = torch.stack([(t if t.ndim == 3 else t.squeeze(0)) for t in imgs_t], dim=0).to(device)
    local_backbone = backbone
    if local_backbone is None:
        local_backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
    feats4 = None
    feats5 = None
    with torch_autocast(device_type=str(device), enabled=(use_amp and device.type == "cuda")):
        out = x
        for i, layer in enumerate(local_backbone):
            out = layer(out)
            if i == 22:
                feats4 = out  # [B,512,h4,w4]
            if i == 29:
                feats5 = out  # [B,512,h5,w5]
    return X.cpu(), compute_edge_index_from_superpixels(sp, rgb=img_rgb), sp

@torch.inference_mode()
def compute_backbone_maps_vgg(
    img_t: torch.Tensor,
    device: str | torch.device = "cpu",
    backbone: Optional[torch.nn.Module] = None,
    use_amp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return VGG16 conv4_3 and conv5_3 feature maps as (F4, F5): each FloatTensor [512, h, w] on CPU.
    """
    if img_t.ndim == 3:
        x = img_t.unsqueeze(0)
    else:
        x = img_t
    x = x.to(device)
    local_backbone = backbone
    if local_backbone is None:
        local_backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
    feats = []
    with torch_autocast(device_type=str(device), enabled=(use_amp and torch.device(device).type == "cuda")):
        out = x
        for i, layer in enumerate(local_backbone):
            out = layer(out)
            if i in (22, 29):
                feats.append(out)
            if i >= 29 and len(feats) == 2:
                break
    F4, F5 = feats
    return F4[0].detach().to("cpu"), F5[0].detach().to("cpu")


@torch.inference_mode()
def pool_from_backbone_maps_max(
    F4: torch.Tensor,  # [512, h4, w4]
    F5: torch.Tensor,  # [512, h5, w5]
    sp: np.ndarray,    # [H, W]
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Aggregate VGG conv4_3 and conv5_3 feature maps to superpixel features using MAX pooling.
    Returns X: FloatTensor [N, 1024]
    """
    if isinstance(device, str):
        device = torch.device(device)
    H, W = sp.shape
    F4 = F4.to(device)
    F5 = F5.to(device)
    F4_up = F.interpolate(F4.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
    F5_up = F.interpolate(F5.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
    Fcat = torch.cat([F4_up, F5_up], dim=0)  # [1024,H,W]
    return _pool_from_feature_tensor_max(Fcat, sp, device=device)


@torch.inference_mode()
def _pool_from_feature_tensor_max(
    Fcat: torch.Tensor,  # [C, H, W]
    sp: np.ndarray,      # [H, W]
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Vectorized MAX pooling over superpixel regions without per-channel Python loops.
    Returns X: FloatTensor [N, C] on CPU.
    """
    if isinstance(device, str):
        device = torch.device(device)
    C, H, W = Fcat.shape
    sp_t = torch.from_numpy(sp).to(device)
    sp_flat = sp_t.reshape(-1)
    C = Fcat.shape[0]
    idx = sp_t.reshape(-1).long()  # [P]
    P = int(idx.numel())
    N = int(sp.max()) + 1
    X = torch.full((N, C), -float("inf"), dtype=Fcat.dtype, device=device)
    F_flat = Fcat.reshape(C, -1)
    for c in range(C):
        vals = F_flat[c]
        X[:, c].scatter_reduce_(0, sp_flat, vals, reduce='amax', include_self=True)
    F_flat = Fcat.reshape(C, -1).transpose(0, 1)  # [P, C]
    X = torch.full((N, C), -float("inf"), dtype=F_flat.dtype, device=device)
    index = idx.unsqueeze(1).expand(P, C)
    X.scatter_reduce_(0, index, F_flat, reduce='amax', include_self=True)
    X[~torch.isfinite(X)] = 0.0
    return X.detach().to("cpu")


@torch.inference_mode()
def compute_backbone_maps_vgg_batch(
    imgs_t: List[torch.Tensor],
    device: str | torch.device = "cpu",
    backbone: Optional[torch.nn.Module] = None,
    use_amp: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Batched VGG16 forward that returns lists of conv4_3 and conv5_3 feature maps on CPU.
    Each element in the returned lists is a FloatTensor [512, h, w].
    """
    if len(imgs_t) == 0:
        return [], []
    if isinstance(device, str):
        device = torch.device(device)
    x = torch.stack([(t if t.ndim == 3 else t.squeeze(0)) for t in imgs_t], dim=0).to(device)
    local_backbone = backbone
    if local_backbone is None:
        local_backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
    feats4 = None
    feats5 = None
    with torch_autocast(device_type=str(device), enabled=(use_amp and device.type == "cuda")):
        out = x
        for i, layer in enumerate(local_backbone):
            out = layer(out)
            if i == 22:
                feats4 = out  # [B,512,h4,w4]
            if i == 29:
                feats5 = out  # [B,512,h5,w5]
                break
    assert feats4 is not None and feats5 is not None
    F4_list = [feats4[b].detach().to("cpu") for b in range(feats4.shape[0])]

                break
    assert feats4 is not None and feats5 is not None
    F4_list = [feats4[b].detach().to("cpu") for b in range(feats4.shape[0])]
    F5_list = [feats5[b].detach().to("cpu") for b in range(feats5.shape[0])]
    return F4_list, F5_list
