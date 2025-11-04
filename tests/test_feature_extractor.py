from skimage.measure import regionprops
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from feature_extractor import extract_features       # the version that accepts img_t (+ optionally img_rgb)
from dataset_loader import DeepGlobeDataset                  # adjust import to your path
from load_palette import load_class_palette           # if you use the CSV palette
import pandas as pd

# --- Config ---
data_dir = "../data/train"
k = 250

# --- Load class palette (optional, if your dataset needs it) ---
# CSV: name,r,g,b
names, class_rgb_values, _unknown = load_class_palette("../data/class_dict.csv")

# --- Build dataset & dataloader ---
ds = DeepGlobeDataset(data_dir, class_rgb_values)

target_name = "6399_sat.jpg"
idx = next(i for i, p in enumerate(ds.image_paths) if target_name in p)

img_t, img_rgb, mask_t = ds[idx]
X, sp = extract_features(img_t, img_rgb, k=k)
# ------------- Inspect -------------
print(f"Superpixels: {sp.max() + 1}")
print(f"Feature matrix shape: {tuple(X.shape)}")
print(f"First feature vector (first 10 dims):\n{X[0, :10]}")

# ------------- Visualization with labels -------------
plt.figure(figsize=(10, 10))
plt.imshow(mark_boundaries(img_rgb, sp, color=(1, 0, 0)))
plt.title("SLIC Superpixels with IDs")
plt.axis("off")

regions = regionprops(sp + 1)
for r in regions:
    cy, cx = r.centroid
    sp_id = r.label - 1
    plt.text(
        cx, cy, str(sp_id),
        color="yellow", fontsize=8,
        ha="center", va="center",
        bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2")
    )

plt.show()

# ------------- Cosine similarities -------------
pick_ids = [52, 38, 151, 103]

Xn = F.normalize(X.float(), dim=1)

# Build the similarity matrix
sim_matrix = torch.zeros((len(pick_ids), len(pick_ids)))
for i_idx, i in enumerate(pick_ids):
    for j_idx, j in enumerate(pick_ids):
        if i == j:
            sim_matrix[i_idx, j_idx] = 1.0  # self-similarity
        else:
            sim_matrix[i_idx, j_idx] = torch.dot(Xn[i], Xn[j]).item()

df = pd.DataFrame(sim_matrix.numpy(), index=pick_ids, columns=pick_ids)
df = df.round(3)

print("\nPairwise cosine similarity matrix:")
print(df.to_string())

# VGG - Pairwise cosine similarity matrix:
#        52     38     151    103
# 52   1.000  0.833  0.592  0.774
# 38   0.833  1.000  0.514  0.725
# 151  0.592  0.514  1.000  0.565
# 103  0.774  0.725  0.565  1.000

# ResNet  52     38     151    103
# 52   1.000  0.801  0.223  0.273
# 38   0.801  1.000  0.315  0.218
# 151  0.223  0.315  1.000  0.256
# 103  0.273  0.218  0.256  1.000