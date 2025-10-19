import csv
from pathlib import Path

def load_class_palette(csv_path: str):
    """Return (names, class_rgb_values, unknown_index)."""
    names, class_rgb_values = [], []
    unknown_index = None
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            names.append(row["name"])
            rgb = (int(row["r"]), int(row["g"]), int(row["b"]))
            class_rgb_values.append(rgb)
            if row["name"].lower() == "unknown":
                unknown_index = i
    return names, class_rgb_values, unknown_index