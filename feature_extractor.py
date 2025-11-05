import torch
import torch.nn.functional as F
from torchgeo.models import resnet50, ResNet50_Weights
from skimage.segmentation import slic
import numpy as np

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

    x = backbone.conv1(x)
    x = backbone.bn1(x)
    x = backbone.act1(x)
    x = backbone.maxpool(x)
    x = backbone.layer1(x)
    x = backbone.layer2(x)
    x = backbone.layer3(x)
    return x  # [1, 1024, h, w]
    # return backbone(x)               # [1, 2048, h, w]

@torch.inference_mode()
def get_slic_graph(
    feature_map: np.ndarray, # torch.Tensor [1, C, h, w]
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

    # 3) Downsample superpixel map to (h,w) and masked MAX pooling on CPU (avoid MPS scatter_reduce issues)
    cpu = torch.device("cpu")
    sp_small = torch.from_numpy(sp)[None, None].float().to(cpu)  # [1,1,H,W]
    sp_small = F.interpolate(sp_small, size=(h, w), mode="nearest")[0, 0].long()  # [h,w]

    F_flat = Fm.detach().to(cpu).reshape(C, -1).transpose(0, 1)  # [P, C]
    sp_flat = sp_small.reshape(-1)  # [P]
    P = int(sp_flat.numel())
    index = sp_flat.unsqueeze(1).expand(P, C)  # [P, C]
    X = torch.full((N, C), -float("inf"), dtype=F_flat.dtype, device=cpu)
    X.scatter_reduce_(0, index, F_flat, reduce='amax', include_self=True)
    X[~torch.isfinite(X)] = 0.0  # handle empty segments if any

    return X, compute_edge_index_from_superpixels(sp, rgb=img_rgb), sp