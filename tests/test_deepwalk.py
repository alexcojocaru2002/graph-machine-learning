import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import graph
from skimage.color import label2rgb
from feature_extractor import extract_features       # the version that accepts img_t (+ optionally img_rgb)
from dataset_loader import DeepGlobeDataset                  # adjust import to your path
from karateclub import DeepWalk
from load_palette import load_class_palette           # if you use the CSV palette
import pandas as pd


# --- Config ---
data_dir = os.path.join(project_root, "data", "train")
k = 250

# --- Load class palette (optional, if your dataset needs it) ---
# CSV: name,r,g,b
csv_path = os.path.join(project_root, "data", "class_dict.csv")
names, class_rgb_values, _unknown = load_class_palette(csv_path)

# --- Build dataset & dataloader ---
ds = DeepGlobeDataset(data_dir, class_rgb_values)

target_name = "6399_sat.jpg"
idx = next(i for i, p in enumerate(ds.image_paths) if target_name in p)

img_t, img_rgb, mask_t = ds[idx]
X, sp, rag = extract_features(img_t, img_rgb, k=k)

# ------------- Inspect -------------
print(f"Superpixels: {sp.max() + 1}")
print(f"RAG: {rag.number_of_edges()} edges, {rag.number_of_nodes()} nodes")
out = label2rgb(sp, img_rgb, kind='avg', bg_label=0)

# Draw RAG edges on top
fig, ax = plt.subplots(figsize=(8, 8))
graph.show_rag(sp, rag, out, ax=ax)
ax.set_title("Region Adjacency Graph (RAG) over Superpixels")
plt.show()

model = DeepWalk(
    walk_number=10,
    walk_length=80,
    dimensions=128,
    workers=4,
    window_size=5,
    epochs=1,
    learning_rate=0.05,
    min_count=1,
    seed=42
)

model.fit(rag)

# --- Get embeddings ---
embeddings = model.get_embedding()  # shape: (num_nodes, 128)
print("Embeddings shape:", embeddings.shape)

node_ids = list(rag.nodes())
print("Embedding for superpixel 0 (first 10):", embeddings[0][0:10])

# ------------- Cosine similarities -------------
pick_ids = [52, 38, 151, 103]

emb_n = F.normalize(torch.tensor(embeddings), dim=1)

sim_matrix = torch.zeros((len(pick_ids), len(pick_ids)))
for i_idx, i in enumerate(pick_ids):
    for j_idx, j in enumerate(pick_ids):
        if i == j:
            sim_matrix[i_idx, j_idx] = 1.0  # self-similarity
        else:
            sim_matrix[i_idx, j_idx] = torch.dot(emb_n[i], emb_n[j]).item()

df = pd.DataFrame(sim_matrix.numpy(), index=pick_ids, columns=pick_ids)
df = df.round(3)

print("\nDeepWalk embedding similarity matrix:")
print(df)

# --- Regression on embeddings ---
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

mask_np = mask_t.numpy()
num_nodes = sp.max() + 1
num_classes = len(class_rgb_values)

fractions = np.zeros((num_nodes, num_classes), dtype=float)
for sid in range(num_nodes):
    region_mask = (sp == sid)
    region_classes = mask_np[region_mask]
    if len(region_classes) == 0:
        continue
    unique, counts = np.unique(region_classes, return_counts=True)
    fractions[sid, unique] = counts
fractions = fractions / fractions.sum(axis=1, keepdims=True)
fractions[np.isnan(fractions)] = 0.0

X = embeddings
y = fractions
mask_valid = np.any(y > 0, axis=1)
X, y = X[mask_valid], y[mask_valid]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=300, max_depth=25, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred, multioutput='variance_weighted')

print("\n=== Regression Results ===")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score (weighted): {r2:.4f}")

r2_per_class = [r2_score(y_test[:, c], y_pred[:, c]) for c in range(y.shape[1])]
r2_df = pd.Series(r2_per_class, index=names)
print("\nPer-class R² scores:")
print(r2_df.round(3))

num_classes = len(class_rgb_values)

for cls_id in range(num_classes):
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test[:, cls_id], y_pred[:, cls_id], alpha=0.6)
    plt.xlabel("True fraction")
    plt.ylabel("Predicted fraction")
    plt.title(f"Class {names[cls_id]} regression (DeepWalk → RandomForest)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()