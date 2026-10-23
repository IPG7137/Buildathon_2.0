"""
ML Inference Engine for PS01 - Steel Defect Classification
===========================================================
Loads the trained MobileNetV3-Small model and provides predict() function.
Used by the FastAPI backend. Thread-safe singleton pattern.
"""

import io
import json
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn

# ─── Paths ─────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).parent.parent
ROOT_DIR = BACKEND_DIR.parent
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "model_v1.pth"
CLASS_MAP_PATH = MODELS_DIR / "class_map.json"
CONFIG_PATH = MODELS_DIR / "training_config.json"

# ─── Constants ──────────────────────────────────────────────────────────────
MAX_IMAGE_BYTES = 20 * 1024 * 1024   # 20 MB
MIN_IMAGE_BYTES = 1024               # 1 KB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
INFERENCE_TIMEOUT_S = 30.0

# ─── Threshold configuration ─────────────────────────────────────────────────
# These should be calibrated on validation data. Defaults set conservatively.
DEFECT_THRESHOLD = 0.45          # prob(DEFECT) >= this -> DEFECT_SUSPECTED
UNCERTAINTY_LOW = 0.35           # prob(DEFECT) in [UNCERTAINTY_LOW, DEFECT_THRESHOLD] -> NEEDS_MANUAL_REVIEW
# prob(DEFECT) < UNCERTAINTY_LOW -> GOOD_PREDICTION


class InferenceEngine:
    """
    Singleton inference engine. Thread-safe model loading and inference.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._model = None
        self._class_map = None
        self._config = None
        self._device = None
        self._model_lock = threading.Lock()
        self._load_error = None
        self._load_time = None
        self._model_ready = False
        self._transform = None
    
    def load(self) -> bool:
        """Load model from disk. Returns True on success."""
        with self._model_lock:
            try:
                self._load_error = None
                t0 = time.time()
                
                # Load class map
                if not CLASS_MAP_PATH.exists():
                    self._load_error = f"class_map.json not found at {CLASS_MAP_PATH}"
                    return False
                with open(CLASS_MAP_PATH) as f:
                    self._class_map = json.load(f)
                
                # Load config
                if CONFIG_PATH.exists():
                    with open(CONFIG_PATH) as f:
                        self._config = json.load(f)
                else:
                    self._config = {
                        "model_arch": "mobilenet_v3_small",
                        "num_classes": 2,
                        "input_size": 224,
                        "normalize_mean": [0.485, 0.456, 0.406],
                        "normalize_std": [0.229, 0.224, 0.225],
                        "dropout": 0.3,
                        "model_version": "1.0.0",
                    }
                
                # Determine device
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Build model
                self._model = self._build_model()
                
                # Load weights
                if not MODEL_PATH.exists():
                    self._load_error = f"Model weights not found at {MODEL_PATH}. Run: python backend/ml/train.py"
                    return False
                
                state_dict = torch.load(MODEL_PATH, map_location=self._device, weights_only=True)
                self._model.load_state_dict(state_dict)
                self._model.eval()
                
                # Build transform
                self._transform = self._build_transform()
                
                self._load_time = time.time() - t0
                self._model_ready = True
                print(f"[InferenceEngine] Model loaded successfully in {self._load_time:.2f}s on {self._device}")
                return True
                
            except Exception as e:
                self._load_error = str(e)
                self._model_ready = False
                print(f"[InferenceEngine] Model load FAILED: {e}")
                return False
    
    def _build_model(self):
        """Build MobileNetV3-Small with same head as training."""
        import torchvision.models as models
        
        cfg = self._config
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[0].in_features
        dropout = cfg.get("dropout", 0.3)
        num_classes = cfg.get("num_classes", 2)
        
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )
        return model.to(self._device)
    
    def _build_transform(self):
        """Build the same preprocessing as used during inference (eval mode)."""
        import torchvision.transforms as T
        cfg = self._config
        size = cfg.get("input_size", 224)
        mean = cfg.get("normalize_mean", [0.485, 0.456, 0.406])
        std = cfg.get("normalize_std", [0.229, 0.224, 0.225])
        
        return T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    
    @property
    def is_ready(self) -> bool:
        return self._model_ready
    
    @property
    def load_error(self) -> Optional[str]:
        return self._load_error
    
    @property
    def model_version(self) -> str:
        if self._config:
            return self._config.get("model_version", "1.0.0")
        return "unknown"
    
    def validate_image(self, image_bytes: bytes, filename: str = "") -> Tuple[bool, str, Optional[Image.Image]]:
        """
        Validate image bytes. Returns (ok, error_message, pil_image).
        """
        # Size check
        nbytes = len(image_bytes)
        if nbytes < MIN_IMAGE_BYTES:
            return False, f"Image too small ({nbytes} bytes). Minimum is {MIN_IMAGE_BYTES} bytes.", None
        if nbytes > MAX_IMAGE_BYTES:
            return False, f"Image too large ({nbytes/1024/1024:.1f} MB). Maximum is {MAX_IMAGE_BYTES/1024/1024:.0f} MB.", None
        
        # Extension check
        if filename:
            ext = Path(filename).suffix.lower()
            if ext and ext not in ALLOWED_EXTENSIONS:
                return False, f"File type '{ext}' not allowed. Allowed: {ALLOWED_EXTENSIONS}", None
        
        # Decode check
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img.verify()  # Checks for corruption
            # Re-open after verify (PIL closes after verify)
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            w, h = img.size
            if w < 16 or h < 16:
                return False, f"Image too small ({w}x{h} pixels). Minimum 16x16.", None
            return True, "", img
        except UnidentifiedImageError:
            return False, "Cannot identify image format. Upload a valid JPG, PNG or WebP.", None
        except Exception as e:
            return False, f"Image decode error: {e}", None
    
    @torch.no_grad()
    def predict(self, image_bytes: bytes, filename: str = "") -> Dict[str, Any]:
        """
        Run inference on image bytes.
        Returns dict with prediction, confidence, status, etc.
        """
        result = {
            "success": False,
            "error": None,
            "prediction": None,
            "predicted_class_id": None,
            "defect_score": None,        # prob(DEFECT class)
            "good_score": None,          # prob(GOOD class)
            "inspection_status": None,   # GOOD_PREDICTION / DEFECT_SUSPECTED / NEEDS_MANUAL_REVIEW / INSPECTION_FAILED
            "model_version": self.model_version,
            "inference_time_ms": None,
            "image_valid": False,
            "recommended_action": None,
        }
        
        # Check model readiness
        if not self._model_ready:
            result["error"] = f"Model not ready: {self._load_error or 'Not loaded'}"
            result["inspection_status"] = "INSPECTION_FAILED"
            result["recommended_action"] = "Backend model unavailable. Contact system administrator."
            return result
        
        # Validate image
        ok, err_msg, img = self.validate_image(image_bytes, filename)
        if not ok:
            result["error"] = err_msg
            result["inspection_status"] = "INSPECTION_FAILED"
            result["recommended_action"] = "Upload a valid image file (JPG/PNG/WebP) and retry."
            return result
        
        result["image_valid"] = True
        
        # Inference
        try:
            t0 = time.time()
            tensor = self._transform(img).unsqueeze(0).to(self._device)
            
            with torch.no_grad():
                logits = self._model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            
            elapsed_ms = (time.time() - t0) * 1000
            
            good_score = float(probs[0])
            defect_score = float(probs[1])
            pred_class_id = int(np.argmax(probs))
            
            # Determine inspection status based on calibrated thresholds
            if defect_score >= DEFECT_THRESHOLD:
                status = "DEFECT_SUSPECTED"
                pred_label = "DEFECT"
                action = ("Segregate or flag this item per your authorised quality procedure. "
                         "Request human quality review. Do not ship until reviewed.")
            elif defect_score >= UNCERTAINTY_LOW:
                status = "NEEDS_MANUAL_REVIEW"
                pred_label = "UNCERTAIN"
                action = ("Confidence is borderline. Capture a clearer image or send to an authorised reviewer. "
                         "Do not make a pass/fail decision based on this result alone.")
            else:
                status = "GOOD_PREDICTION"
                pred_label = "GOOD"
                action = ("Record this inspection and continue per your authorised quality procedure. "
                         "Note: AI prediction is NOT a release certificate. Authorised QC sign-off is required.")
            
            result.update({
                "success": True,
                "prediction": pred_label,
                "predicted_class_id": pred_class_id,
                "defect_score": round(defect_score, 4),
                "good_score": round(good_score, 4),
                "inspection_status": status,
                "inference_time_ms": round(elapsed_ms, 1),
                "recommended_action": action,
                "raw_scores": {"GOOD": round(good_score, 4), "DEFECT": round(defect_score, 4)},
                "threshold_used": {"defect_threshold": DEFECT_THRESHOLD, "uncertainty_low": UNCERTAINTY_LOW},
            })
            
        except torch.cuda.OutOfMemoryError:
            result["error"] = "GPU out of memory. Try a smaller image."
            result["inspection_status"] = "INSPECTION_FAILED"
            result["recommended_action"] = "Retry with a smaller image, or contact system administrator."
        except Exception as e:
            result["error"] = f"Inference error: {str(e)}"
            result["inspection_status"] = "INSPECTION_FAILED"
            result["recommended_action"] = "Retry. If error persists, capture a new image or contact system administrator."
        
        return result


# Global singleton
engine = InferenceEngine()
