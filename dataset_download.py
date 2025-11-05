# ...existing code...
import os
import kagglehub

out_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(out_dir, exist_ok=True)

_cwd = os.getcwd()
try:
    os.chdir(out_dir)
    print(out_dir)
    # Download latest version into data/
    path = kagglehub.dataset_download("balraj98/deepglobe-land-cover-classification-dataset")
finally:
    os.chdir(_cwd)

print("Path to dataset files:", path)
# ...existing code...