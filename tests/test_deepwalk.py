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