"""
Dataset Preparation Script for PS01 - Metal Casting (Steel) Defect Inspection
==============================================================================
Dataset: Severstal Steel Defect Detection (Kaggle)
  - train.csv: ImageId, ClassId (1-4), EncodedPixels (RLE segmentation)
  - train_images/: ~12,568 training images (256x1600 steel strip images)
  - 4 defect classes; images can have 0, 1 or multiple defect classes

Approach: Convert segmentation -> multi-label classification
  - An image is "DEFECT" if ANY defect class is present
  - Also supports per-class multi-label (class 1,2,3,4 independently)
  - We train a binary (GOOD vs DEFECT) MobileNetV3-Small classifier for the app
  - And also save per-class labels for the multi-class variant

Steps:
1. Extract train_images from zip (sample for prototype speed)
2. Build label CSV
3. Create train/val/test splits (no leakage)
4. Save class mapping and split info
"""

import os
import sys
import csv
import json
import random
import zipfile
import shutil
from pathlib import Path
from collections import defaultdict, Counter

ZIP_PATH = r"C:\Users\ishaa\Downloads\severstal-steel-defect-detection.zip"
DATASET_DIR = Path(__file__).parent
IMAGES_DIR = DATASET_DIR / "train_images"
LABELS_CSV = DATASET_DIR / "labels.csv"
SPLIT_CSV = DATASET_DIR / "splits.csv"
CLASS_MAP_PATH = DATASET_DIR / "class_map.json"

# For prototype: extract up to N images per label balance
# Full dataset has 12568 train images; we use all but cap extraction for speed
MAX_IMAGES = 3000  # extract up to 3000 images for reasonable training time

random.seed(42)

def extract_train_images(zip_path, dest_dir, max_images=MAX_IMAGES):
    """Extract train_images from zip, stopping at max_images."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Opening zip: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        train_entries = [e for e in zf.infolist() if e.filename.startswith('train_images/') and e.filename.endswith('.jpg')]
        print(f"Total train images in zip: {len(train_entries)}")
        
        # Shuffle for representative sampling
        random.shuffle(train_entries)
        train_entries = train_entries[:max_images]
        
        extracted = 0
        for entry in train_entries:
            fname = Path(entry.filename).name
            dest_path = dest_dir / fname
            if dest_path.exists():
                extracted += 1
                continue
            with zf.open(entry) as src, open(dest_path, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1
            if extracted % 100 == 0:
                print(f"  Extracted {extracted}/{len(train_entries)} images...")
        
        print(f"Done. Extracted {extracted} train images to {dest_dir}")
        return extracted


def read_train_csv(zip_path):
    """Read train.csv from inside zip, return dict: image_id -> list of class_ids."""
    image_classes = defaultdict(set)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open('train.csv') as f:
            reader = csv.reader(line.decode('utf-8') for line in f)
            header = next(reader)
            print(f"train.csv header: {header}")
            for row in reader:
                if len(row) >= 2:
                    img_id = row[0].strip()
                    class_id = int(row[1].strip())
                    image_classes[img_id].add(class_id)
    return image_classes


def build_labels(image_classes, images_dir):
    """Build label CSV for available images."""
    images_dir = Path(images_dir)
    available = {p.name for p in images_dir.glob("*.jpg")}
    print(f"Available extracted images: {len(available)}")
    
    rows = []
    for img_name in available:
        classes = image_classes.get(img_name, set())
        # Binary label: 0 = good (no defect), 1 = defect
        binary_label = 1 if len(classes) > 0 else 0
        # Multi-label: one-hot for classes 1,2,3,4
        c1 = 1 if 1 in classes else 0
        c2 = 1 if 2 in classes else 0
        c3 = 1 if 3 in classes else 0
        c4 = 1 if 4 in classes else 0
        # Primary defect class (for display; use most severe if multiple)
        if len(classes) > 0:
            # Priority: class 2 (rarest/most severe looking) > 4 > 1 > 3
            primary = sorted(classes, key=lambda x: {2:0,4:1,1:2,3:3}.get(x,4))[0]
        else:
            primary = 0
        rows.append({
            'image_id': img_name,
            'binary_label': binary_label,
            'primary_class': primary,
            'has_class1': c1,
            'has_class2': c2,
            'has_class3': c3,
            'has_class4': c4,
            'defect_classes': ','.join(str(c) for c in sorted(classes)) if classes else ''
        })
    
    # Stats
    total = len(rows)
    good = sum(1 for r in rows if r['binary_label'] == 0)
    defect = sum(1 for r in rows if r['binary_label'] == 1)
    print(f"\nLabel summary ({total} images):")
    print(f"  GOOD (no defect): {good} ({good/total*100:.1f}%)")
    print(f"  DEFECT (any class): {defect} ({defect/total*100:.1f}%)")
    for c in [1,2,3,4]:
        cnt = sum(1 for r in rows if r[f'has_class{c}'])
        print(f"  Class {c} defects: {cnt}")
    
    return rows


def create_splits(rows, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """Create stratified train/val/test splits."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    good = [r for r in rows if r['binary_label'] == 0]
    defect = [r for r in rows if r['binary_label'] == 1]
    
    random.shuffle(good)
    random.shuffle(defect)
    
    def split_list(lst):
        n = len(lst)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return lst[:n_train], lst[n_train:n_train+n_val], lst[n_train+n_val:]
    
    g_train, g_val, g_test = split_list(good)
    d_train, d_val, d_test = split_list(defect)
    
    train = g_train + d_train
    val = g_val + d_val
    test = g_test + d_test
    
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    for r in train: r['split'] = 'train'
    for r in val: r['split'] = 'val'
    for r in test: r['split'] = 'test'
    
    all_rows = train + val + test
    
    print(f"\nSplit summary:")
    print(f"  Train: {len(train)} (good={len(g_train)}, defect={len(d_train)})")
    print(f"  Val:   {len(val)} (good={len(g_val)}, defect={len(d_val)})")
    print(f"  Test:  {len(test)} (good={len(g_test)}, defect={len(d_test)})")
    
    return all_rows


def save_class_map():
    """Save the class mapping JSON for the model and backend."""
    class_map = {
        "binary": {
            "0": "GOOD",
            "1": "DEFECT"
        },
        "defect_classes": {
            "1": "Class 1 - Scratches",
            "2": "Class 2 - Inclusions",
            "3": "Class 3 - Patches",
            "4": "Class 4 - Pitted Surface"
        },
        "label_names": ["GOOD", "DEFECT"],
        "num_classes": 2,
        "model_type": "binary_classification",
        "input_size": [224, 224],
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "model_arch": "mobilenet_v3_small",
        "dataset": "Severstal Steel Defect Detection",
        "description": "Binary classifier: GOOD (no defect) vs DEFECT (any defect class present)",
        "defect_class_names": {
            "0": "GOOD",
            "1": "DEFECT_CLASS1_SCRATCHES",
            "2": "DEFECT_CLASS2_INCLUSIONS",
            "3": "DEFECT_CLASS3_PATCHES",
            "4": "DEFECT_CLASS4_PITTED"
        }
    }
    with open(CLASS_MAP_PATH, 'w') as f:
        json.dump(class_map, f, indent=2)
    print(f"\nClass map saved to {CLASS_MAP_PATH}")
    return class_map


def main():
    print("=" * 60)
    print("PS01 Dataset Preparation")
    print("=" * 60)
    
    # Step 1: Read labels from zip CSV
    print("\n[1/4] Reading train.csv from zip...")
    image_classes = read_train_csv(ZIP_PATH)
    print(f"Images with defect labels: {len(image_classes)}")
    
    # Class distribution in full train.csv
    class_dist = Counter()
    for classes in image_classes.values():
        for c in classes:
            class_dist[c] += 1
    print(f"Full dataset defect class distribution: {dict(sorted(class_dist.items()))}")
    
    # Total images (including those with no defects - not in train.csv)
    # The full training set has 12,568 images; those not in train.csv have no defects
    
    # Step 2: Extract train images
    print(f"\n[2/4] Extracting up to {MAX_IMAGES} train images from zip...")
    if IMAGES_DIR.exists() and len(list(IMAGES_DIR.glob("*.jpg"))) >= 100:
        print(f"  Images already extracted ({len(list(IMAGES_DIR.glob('*.jpg')))} found). Skipping.")
    else:
        extract_train_images(ZIP_PATH, IMAGES_DIR, MAX_IMAGES)
    
    # Step 3: Build labels
    print("\n[3/4] Building label CSV...")
    rows = build_labels(image_classes, IMAGES_DIR)
    
    # Step 4: Create splits
    print("\n[4/4] Creating train/val/test splits...")
    all_rows = create_splits(rows)
    
    # Save labels CSV
    fieldnames = ['image_id', 'binary_label', 'primary_class', 'has_class1', 
                  'has_class2', 'has_class3', 'has_class4', 'defect_classes', 'split']
    with open(LABELS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nLabels CSV saved: {LABELS_CSV}")
    
    # Save split CSV separately
    with open(SPLIT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    
    # Save class map
    save_class_map()
    
    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print(f"  Labels CSV: {LABELS_CSV}")
    print(f"  Class map:  {CLASS_MAP_PATH}")
    print("="*60)


if __name__ == "__main__":
    main()
