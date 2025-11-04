from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
def slash_data(subpath: str) -> str:
    return str(PROJECT_ROOT / "data" / subpath)

CLASS_CSV = slash_data("class_dict.csv")
IMG_SIZE = (512, 512)
TRAIN_DATA_DIR = slash_data("train")
TEST_DATA_DIR = slash_data("test")
BATCH_SIZE = 1
LR = 2e-4
EPOCHS = 20
HIDDEN_DIM = 512
K_VALUES_GCN = [100]