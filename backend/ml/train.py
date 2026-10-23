"""
ML Training Script for PS01 - Steel Defect Binary Classifier
=============================================================
Architecture: MobileNetV3-Small (pretrained on ImageNet)
Task: Binary classification - GOOD (0) vs DEFECT (1)
Dataset: Severstal Steel Defect Detection (converted from segmentation)

Usage:
    python backend/ml/train.py
    python backend/ml/train.py --epochs 15 --batch-size 32

Outputs:
    models/model_v1.pth          - trained weights
    models/class_map.json        - class mapping (copied from dataset/)
    models/training_config.json  - preprocessing config
    models/eval_report.json      - evaluation metrics
    models/confusion_matrix.txt  - confusion matrix
"""

import os
import sys
import csv
import json
import time
import random
import argparse
import warnings
from pathlib import Path
from copy import deepcopy

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings('ignore')

# ─── Paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
DATASET_DIR = ROOT / "dataset"
STEEL_IMAGES_DIR = DATASET_DIR / "train_images"
STEEL_LABELS_CSV = DATASET_DIR / "labels.csv"
STEEL_CLASS_MAP_PATH = DATASET_DIR / "class_map.json"
CASTING_LABELS_CSV = DATASET_DIR / "casting_labels.csv"
CASTING_CLASS_MAP_PATH = DATASET_DIR / "casting_class_map.json"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ─── Config ─────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "model_arch": "mobilenet_v3_small",
    "num_classes": 2,
    "input_size": 224,
    "batch_size": 64,
    "epochs": 5,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "label_smoothing": 0.1,
    "use_amp": True,
    "num_workers": 0,   # 0 for Windows compatibility
    "seed": 42,
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
    "model_version": "1.0.0",
}

# ─── Dataset ────────────────────────────────────────────────────────────────
class DefectDataset(Dataset):
    """
    Binary classification: GOOD (0) vs DEFECT (1)
    Supports both Casting Defect dataset (dataset2.zip) and Steel Surface dataset.
    """
    
    def __init__(self, rows, images_dir=None, root_dir=ROOT, transform=None, augment=False):
        self.rows = rows
        self.images_dir = Path(images_dir) if images_dir else None
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.augment = augment
        self._setup_transforms()
    
    def _setup_transforms(self):
        try:
            import torchvision.transforms as T
            mean = DEFAULT_CONFIG["normalize_mean"]
            std = DEFAULT_CONFIG["normalize_std"]
            size = DEFAULT_CONFIG["input_size"]
            
            if self.augment:
                self.transform = T.Compose([
                    T.Resize((size + 32, size + 32)),
                    T.RandomCrop(size),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                    T.RandomRotation(10),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std),
                ])
            else:
                self.transform = T.Compose([
                    T.Resize((size, size)),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std),
                ])
        except ImportError:
            raise RuntimeError("torchvision not available. Run: pip install torchvision")
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        row = self.rows[idx]
        
        # Resolve image path
        if "filepath" in row and row["filepath"]:
            p = Path(row["filepath"])
            img_path = p if p.is_absolute() else (self.root_dir / p)
        elif self.images_dir and "image_id" in row and row["image_id"]:
            img_path = self.images_dir / row["image_id"]
        elif self.images_dir and "filename" in row and row["filename"]:
            img_path = self.images_dir / row["filename"]
        elif "filename" in row and row["filename"]:
            img_path = Path(row["filename"])
        else:
            img_path = Path(row.get("image_id", ""))
            
        label = int(row["binary_label"])
        
        try:
            img = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, UnidentifiedImageError, Exception) as e:
            # Return a black image on missing/corrupt image
            img = Image.new('RGB', (DEFAULT_CONFIG["input_size"], DEFAULT_CONFIG["input_size"]), (0, 0, 0))
        
        if self.transform:
            img = self.transform(img)
        
        img_id = row.get("filename") or row.get("image_id") or Path(str(img_path)).name
        return img, label, img_id

# Maintain backwards compatibility
SteelDefectDataset = DefectDataset


def load_labels(labels_csv):
    """Load labels CSV and return split dicts."""
    rows = []
    with open(labels_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    rnd = random.Random(42)
    train = [r for r in rows if r['split'] == 'train']
    val = [r for r in rows if r['split'] == 'val']
    test = [r for r in rows if r['split'] == 'test']
    rnd.shuffle(train)
    rnd.shuffle(val)
    rnd.shuffle(test)
    
    print(f"Loaded: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def compute_class_weights(train_rows):
    """Compute class weights for imbalanced data."""
    labels = [int(r['binary_label']) for r in train_rows]
    n_total = len(labels)
    n_good = labels.count(0)
    n_defect = labels.count(1)
    
    w_good = n_total / (2 * n_good) if n_good > 0 else 1.0
    w_defect = n_total / (2 * n_defect) if n_defect > 0 else 1.0
    
    print(f"Class weights: GOOD={w_good:.3f}, DEFECT={w_defect:.3f}")
    return torch.tensor([w_good, w_defect], dtype=torch.float32)


def build_model(config, device, checkpoint_path=None):
    """Build MobileNetV3-Small with custom head, optionally resuming from checkpoint."""
    try:
        import torchvision.models as models
        
        print(f"Loading MobileNetV3-Small architecture...")
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        
        # Replace classifier
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=config["dropout"]),
            nn.Linear(256, config["num_classes"]),
        )
        
        if checkpoint_path and str(checkpoint_path).lower() != "none":
            ckpt_p = Path(checkpoint_path)
            if ckpt_p.exists():
                print(f"  Continuing training from checkpoint: {ckpt_p}")
                state = torch.load(ckpt_p, map_location=device)
                try:
                    model.load_state_dict(state)
                    print("  Successfully loaded all checkpoint weights.")
                except Exception as ex:
                    print(f"  Loaded with strict=False ({ex})")
                    model.load_state_dict(state, strict=False)
            else:
                print(f"  Checkpoint {ckpt_p} not found, initializing from ImageNet.")
        
        return model.to(device)
        
    except Exception as e:
        print(f"Error building model: {e}")
        raise


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, epoch=0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels, _) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] loss={total_loss/total:.4f} acc={correct/total:.4f}")
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on a dataloader."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for images, labels, _ in loader:
        images = images.to(device)
        labels_dev = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels_dev)
        
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        
        total_loss += loss.item() * images.size(0)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs[:, 1].cpu().numpy().tolist())  # prob of DEFECT
    
    n = len(all_labels)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    
    return {
        "loss": total_loss / n,
        "accuracy": accuracy,
        "preds": all_preds,
        "labels": all_labels,
        "probs": all_probs,
    }


def compute_metrics(result, class_names=["GOOD", "DEFECT"]):
    """Compute per-class precision, recall, F1 and confusion matrix."""
    preds = result["preds"]
    labels = result["labels"]
    n_classes = len(class_names)
    
    # Confusion matrix
    cm = [[0] * n_classes for _ in range(n_classes)]
    for p, l in zip(preds, labels):
        cm[l][p] += 1
    
    # Per-class metrics
    metrics = {}
    for c in range(n_classes):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(n_classes) if r != c)
        fn = sum(cm[c][r] for r in range(n_classes) if r != c)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[class_names[c]] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    
    # False acceptance (defect predicted as good) and false rejection (good predicted as defect)
    false_acceptance = cm[1][0]  # true=DEFECT, pred=GOOD
    false_rejection = cm[0][1]   # true=GOOD, pred=DEFECT
    
    return {
        "accuracy": result["accuracy"],
        "confusion_matrix": cm,
        "class_names": class_names,
        "per_class": metrics,
        "false_acceptance_count": false_acceptance,  # dangerous: defect missed
        "false_rejection_count": false_rejection,     # costly: good item rejected
        "false_acceptance_rate": false_acceptance / sum(cm[1]) if sum(cm[1]) > 0 else 0.0,
        "false_rejection_rate": false_rejection / sum(cm[0]) if sum(cm[0]) > 0 else 0.0,
    }


def print_report(metrics, split_name="Test"):
    """Print evaluation report."""
    print(f"\n{'='*60}")
    print(f"  {split_name} Evaluation Report")
    print(f"{'='*60}")
    print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"\n  Per-Class Metrics:")
    for cls, m in metrics["per_class"].items():
        print(f"    {cls:10s}: P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  Support={m['support']}")
    print(f"\n  SAFETY METRICS (critical for QC):")
    print(f"    False Acceptance (defect missed): {metrics['false_acceptance_count']} ({metrics['false_acceptance_rate']*100:.1f}%)")
    print(f"    False Rejection (good rejected):  {metrics['false_rejection_count']} ({metrics['false_rejection_rate']*100:.1f}%)")
    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    cn = metrics["class_names"]
    header = "        " + "  ".join(f"{c:10s}" for c in cn)
    print(f"  {header}")
    for i, row in enumerate(metrics["confusion_matrix"]):
        row_str = "  ".join(f"{v:10d}" for v in row)
        print(f"  {cn[i]:6s}  {row_str}")
    print(f"{'='*60}")


def save_model(model, config, metrics, class_map, output_name="model_v1.pth", class_map_src=None, dataset_type="casting"):
    """Save model weights and all metadata."""
    model_path = MODELS_DIR / output_name
    config_path = MODELS_DIR / "training_config.json"
    eval_path = MODELS_DIR / "eval_report.json"
    cm_path = MODELS_DIR / "confusion_matrix.txt"
    class_map_dest = MODELS_DIR / "class_map.json"
    
    # Backup existing model_v1.pth if not already backed up
    if (MODELS_DIR / "model_v1.pth").exists() and not (MODELS_DIR / "model_steel_v1.pth").exists():
        import shutil
        shutil.copy(MODELS_DIR / "model_v1.pth", MODELS_DIR / "model_steel_v1.pth")
    
    # Save weights to primary model path
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved: {model_path}")
    
    # Also save dataset-specific artifacts if casting
    if dataset_type == "casting":
        casting_pth = MODELS_DIR / "model_casting_v1.pth"
        torch.save(model.state_dict(), casting_pth)
        with open(MODELS_DIR / "casting_eval_report.json", "w") as f:
            eval_copy = {k: v for k, v in metrics.items() if k not in ("preds", "labels", "probs")}
            json.dump(eval_copy, f, indent=2)
        print(f"Dataset-specific artifact saved: {casting_pth}")
    
    # Save training config
    config_out = deepcopy(config)
    config_out["trained_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    config_out["dataset_type"] = dataset_type
    with open(config_path, 'w') as f:
        json.dump(config_out, f, indent=2)
    
    # Save eval report (remove large arrays)
    eval_out = {k: v for k, v in metrics.items() if k not in ("preds", "labels", "probs")}
    with open(eval_path, 'w') as f:
        json.dump(eval_out, f, indent=2)
    print(f"Eval report saved: {eval_path}")
    
    # Save confusion matrix text
    with open(cm_path, 'w') as f:
        cn = metrics["class_names"]
        f.write(f"Confusion Matrix (rows=true, cols=pred)\n")
        f.write(f"Dataset: {dataset_type}\n")
        f.write(f"Classes: {cn}\n\n")
        for i, row in enumerate(metrics["confusion_matrix"]):
            f.write(f"  {cn[i]:10s}: {row}\n")
        f.write(f"\nAccuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"False Acceptance Rate: {metrics['false_acceptance_rate']:.4f}\n")
        f.write(f"False Rejection Rate: {metrics['false_rejection_rate']:.4f}\n")
    
    # Copy class map
    import shutil
    if class_map_src and Path(class_map_src).exists():
        shutil.copy(class_map_src, class_map_dest)
        if dataset_type == "casting":
            shutil.copy(class_map_src, MODELS_DIR / "casting_class_map.json")
    else:
        with open(class_map_dest, 'w') as f:
            json.dump(class_map, f, indent=2)
    
    print(f"All model artifacts saved in: {MODELS_DIR}")
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train Defect Binary Classifier (Casting / Steel)")
    parser.add_argument("--dataset", type=str, default="casting", choices=["casting", "steel"],
                        help="Dataset: 'casting' (dataset2.zip) or 'steel'")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit total samples for quick test (0=all)")
    parser.add_argument("--checkpoint", type=str, default="models/model_v1.pth",
                        help="Path to checkpoint to continue/fine-tune from (or 'none' for scratch)")
    parser.add_argument("--output-name", type=str, default="model_v1.pth", help="Model filename in models/")
    args = parser.parse_args()
    
    config = deepcopy(DEFAULT_CONFIG)
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["lr"] = args.lr
    if args.no_amp:
        config["use_amp"] = False
    
    # Set PyTorch intra-op threads
    try:
        torch.set_num_threads(8)
    except Exception:
        pass

    # Seed
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  CPU Threads: {torch.get_num_threads()}")
    
    # Select dataset
    if args.dataset == "casting":
        labels_csv = CASTING_LABELS_CSV
        class_map_path = CASTING_CLASS_MAP_PATH
        images_dir = None
        
        # Prepare casting dataset if CSV not generated yet
        if not labels_csv.exists():
            print(f"Casting labels CSV not found. Running dataset/prepare_casting_dataset.py...")
            import subprocess
            subprocess.run([sys.executable, str(ROOT / "dataset" / "prepare_casting_dataset.py")], check=True)
            
        print("Dataset: Casting Defect Inspection (dataset2.zip)")
    else:
        labels_csv = STEEL_LABELS_CSV
        class_map_path = STEEL_CLASS_MAP_PATH
        images_dir = STEEL_IMAGES_DIR
        
        if not labels_csv.exists():
            print(f"ERROR: Labels CSV not found at {labels_csv}")
            print("  Run: python dataset/prepare_dataset.py first")
            sys.exit(1)
        print("Dataset: Steel Surface Defect Inspection")
    
    # Load class map
    class_map = {}
    if class_map_path.exists():
        with open(class_map_path) as f:
            class_map = json.load(f)
    
    # Load data
    print("\n[1/5] Loading dataset splits...")
    train_rows, val_rows, test_rows = load_labels(labels_csv)
    
    if args.max_samples > 0:
        print(f"  Limiting training to {args.max_samples} samples for bounded run (full held-out test set evaluated)")
        train_rows = train_rows[:int(args.max_samples * 0.85)]
        val_rows = val_rows[:int(args.max_samples * 0.15)]
    
    # Class weights
    class_weights = compute_class_weights(train_rows).to(device)
    
    # Datasets
    train_ds = DefectDataset(train_rows, images_dir=images_dir, augment=True)
    val_ds = DefectDataset(val_rows, images_dir=images_dir, augment=False)
    test_ds = DefectDataset(test_rows, images_dir=images_dir, augment=False)
    
    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["num_workers"], pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
                            num_workers=config["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False,
                             num_workers=config["num_workers"])
    
    print(f"\n  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Build model
    print("\n[2/5] Building model...")
    checkpoint_to_load = args.checkpoint if args.checkpoint and str(args.checkpoint).lower() != "none" else None
    model = build_model(config, device, checkpoint_path=checkpoint_to_load)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config["label_smoothing"])
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)
    
    # AMP scaler
    scaler = GradScaler() if (config["use_amp"] and device.type == "cuda") else None
    if scaler:
        print("  Using mixed precision training (AMP)")
    
    # Training loop
    print(f"\n[3/5] Training for {config['epochs']} epochs on {args.dataset}...")
    best_val_acc = 0.0
    best_model_state = None
    history = []
    
    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, epoch)
        val_result = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        elapsed = time.time() - t0
        print(f"\nEpoch {epoch}/{config['epochs']}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_result['loss']:.4f}  val_acc={val_result['accuracy']:.4f}  "
              f"({elapsed:.1f}s)")
        
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_result["loss"], "val_acc": val_result["accuracy"]
        })
        
        if val_result["accuracy"] > best_val_acc:
            best_val_acc = val_result["accuracy"]
            best_model_state = deepcopy(model.state_dict())
            print(f"  * New best val_acc: {best_val_acc:.4f} -- checkpoint saved")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best checkpoint (val_acc={best_val_acc:.4f})")
    
    # Evaluate on test set
    print("\n[4/5] Evaluating on held-out test set...")
    test_result = evaluate(model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_result)
    print_report(test_metrics, "Test (held-out)")
    
    # Also report val metrics
    val_result_full = evaluate(model, val_loader, criterion, device)
    val_metrics = compute_metrics(val_result_full)
    print_report(val_metrics, "Validation")
    
    # Save everything
    print("\n[5/5] Saving model and artifacts...")
    full_metrics = {
        **test_metrics,
        "val_accuracy": val_metrics["accuracy"],
        "best_val_accuracy": best_val_acc,
        "training_history": history,
        "config": config,
    }
    model_path = save_model(
        model=model,
        config=config,
        metrics=full_metrics,
        class_map=class_map,
        output_name=args.output_name,
        class_map_src=class_map_path,
        dataset_type=args.dataset
    )
    
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Dataset:           {args.dataset.upper()} (dataset2.zip)")
    print(f"  Best val accuracy: {best_val_acc*100:.2f}%")
    print(f"  Test accuracy:     {test_metrics['accuracy']*100:.2f}%")
    print(f"  Model saved:       {model_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
