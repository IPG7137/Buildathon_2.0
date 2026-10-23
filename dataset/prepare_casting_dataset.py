"""
Prepare Casting Defect Dataset (dataset2.zip)
=============================================
Maps casting_data:
  ok_front  -> GOOD (0)
  def_front -> DEFECT (1)

Creates:
  dataset/casting_labels.csv
  dataset/casting_class_map.json
"""

import os
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "dataset" / "casting_dataset"
OUT_CSV = ROOT / "dataset" / "casting_labels.csv"
OUT_MAP = ROOT / "dataset" / "casting_class_map.json"

CLASS_MAP = {
    "0": "GOOD",
    "1": "DEFECT",
    "GOOD": 0,
    "DEFECT": 1,
    "class_descriptions": {
        "GOOD": "Defect-free metal casting component (ok_front)",
        "DEFECT": "Casting surface defect (porosity, pinhole, blowhole, burr, mould drop)"
    }
}

def main():
    random.seed(42)
    rows = []

    # 1. Test set (pre-split from dataset2)
    for cls_folder, label, cls_name in [("ok_front", 0, "GOOD"), ("def_front", 1, "DEFECT")]:
        folder = DATA_DIR / "test" / cls_folder
        for f in folder.glob("*.jpeg"):
            rows.append({
                "filepath": str(f.relative_to(ROOT)).replace("\\", "/"),
                "filename": f.name,
                "binary_label": label,
                "class_name": cls_name,
                "split": "test"
            })

    # 2. Train and Val split
    train_pool = []
    for cls_folder, label, cls_name in [("ok_front", 0, "GOOD"), ("def_front", 1, "DEFECT")]:
        folder = DATA_DIR / "train" / cls_folder
        files = list(folder.glob("*.jpeg"))
        random.shuffle(files)
        # 15% validation
        val_count = int(len(files) * 0.15)
        val_files = files[:val_count]
        tr_files = files[val_count:]

        for f in tr_files:
            train_pool.append({
                "filepath": str(f.relative_to(ROOT)).replace("\\", "/"),
                "filename": f.name,
                "binary_label": label,
                "class_name": cls_name,
                "split": "train"
            })
        for f in val_files:
            train_pool.append({
                "filepath": str(f.relative_to(ROOT)).replace("\\", "/"),
                "filename": f.name,
                "binary_label": label,
                "class_name": cls_name,
                "split": "val"
            })

    rows.extend(train_pool)

    # Save CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "filename", "binary_label", "class_name", "split"])
        writer.writeheader()
        writer.writerows(rows)

    with open(OUT_MAP, "w", encoding="utf-8") as f:
        json.dump(CLASS_MAP, f, indent=2)

    train_c = sum(1 for r in rows if r["split"] == "train")
    val_c = sum(1 for r in rows if r["split"] == "val")
    test_c = sum(1 for r in rows if r["split"] == "test")

    print(f"Casting dataset prepared:")
    print(f"  Total: {len(rows):,} images")
    print(f"  Train: {train_c:,}")
    print(f"  Val:   {val_c:,}")
    print(f"  Test:  {test_c:,}")
    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_MAP}")

if __name__ == "__main__":
    main()
