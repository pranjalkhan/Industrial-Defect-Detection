"""
Inference module for production use.

Supports multi-category defect detection across three model architectures:
- patchcore (default, highest accuracy: WideResNet50-2 + coreset memory bank)
- resnet    (ResNet18 intermediate feature distance)
- autoencoder (ConvAutoencoder reconstruction baseline)
"""

import base64
import io
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import create_heatmap_overlay

# Cache detectors by (category, model_type)
_detectors: Dict[Tuple[str, str], object] = {}


def list_available_categories(models_root: str = "models") -> List[str]:
    """Return sorted list of categories that have at least one trained model on disk."""
    categories = []
    if os.path.exists(models_root):
        for entry in sorted(os.listdir(models_root)):
            cat_dir = os.path.join(models_root, entry)
            if os.path.isdir(cat_dir) and any(
                os.path.exists(os.path.join(cat_dir, f))
                for f in ("patchcore_bank.pt", "resnet_detector.pkl", "autoencoder_best.pth", "autoencoder.pth")
            ):
                categories.append(entry)
    if not categories and (
        os.path.exists(os.path.join(models_root, "autoencoder_best.pth"))
        or os.path.exists(os.path.join(models_root, "autoencoder.pth"))
    ):
        categories.append("bottle")
    return categories


def get_default_threshold(category: str = "bottle", model_type: str = "patchcore") -> Optional[float]:
    """Read the calibrated threshold for a (category, model_type) pair."""
    if model_type == "patchcore":
        for p in (
            os.path.join("models", category, "patchcore_threshold.txt"),
            os.path.join("artifacts", category, "evaluation_patchcore", "metrics.json"),
        ):
            if os.path.exists(p):
                if p.endswith(".json"):
                    with open(p, "r") as f:
                        return float(json.load(f)["threshold"])
                with open(p, "r") as f:
                    return float(f.read().strip())
    elif model_type == "resnet":
        p = os.path.join("artifacts", category, "evaluation_resnet", "metrics.json")
        if os.path.exists(p):
            with open(p, "r") as f:
                return float(json.load(f)["threshold"])
    else:
        for p in (
            os.path.join("models", category, "threshold.txt"),
            "models/threshold.txt",
        ):
            if os.path.exists(p):
                with open(p, "r") as f:
                    return float(f.read().strip())
    return None


class _FeatureDetectorAdapter:
    """Unified inference adapter for PatchCore and ResNet18 detectors."""

    def __init__(self, detector, threshold: Optional[float], image_size: int = 256):
        self.detector = detector
        self.threshold = threshold
        self.image_size = image_size
        self.resize_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.norm_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def detect(self, image_input, return_visuals: bool = True) -> Dict:
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        norm_tensor = self.norm_transform(image)
        anomaly_score, anomaly_map = self.detector.predict(norm_tensor)

        if self.threshold is not None:
            prediction = "DEFECTIVE" if anomaly_score > self.threshold else "NORMAL"
        else:
            prediction = "UNKNOWN (no threshold calibrated)"

        result = {
            "prediction": prediction,
            "anomaly_score": float(anomaly_score),
            "threshold": self.threshold,
            "anomaly_map": anomaly_map,
        }

        if return_visuals:
            orig_tensor = self.resize_transform(image)
            original_np = (np.transpose(orig_tensor.numpy(), (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)

            if anomaly_map.max() > anomaly_map.min():
                norm_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
            else:
                norm_map = np.zeros_like(anomaly_map)
            heatmap = (norm_map * 255).astype(np.uint8)
            overlay = create_heatmap_overlay(original_np, anomaly_map)

            result["original"] = original_np
            result["heatmap"] = heatmap
            result["overlay"] = overlay

        return result


def get_detector(
    category: str = "bottle",
    model_type: str = "patchcore",
    model_path: Optional[str] = None,
    threshold: Optional[float] = None,
):
    """Get or create a cached detector for (category, model_type)."""
    global _detectors
    key = (category, model_type)
    if key not in _detectors or model_path is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_type == "patchcore":
            bank_path = model_path or os.path.join("models", category, "patchcore_bank.pt")
            if not os.path.exists(bank_path):
                # Fallback to autoencoder if patchcore isn't trained for this category
                return get_detector(category=category, model_type="autoencoder", threshold=threshold)

            from src.patchcore import PatchCoreDetector

            pc = PatchCoreDetector(device=device, layers=["layer2", "layer3"], image_size=256)
            pc.memory_bank = torch.load(bank_path, map_location=device, weights_only=True).to(device)
            resolved_thresh = threshold if threshold is not None else get_default_threshold(category, "patchcore")
            _detectors[key] = _FeatureDetectorAdapter(pc, resolved_thresh)

        elif model_type == "resnet":
            pkl_path = model_path or os.path.join("models", category, "resnet_detector.pkl")
            if not os.path.exists(pkl_path):
                return get_detector(category=category, model_type="autoencoder", threshold=threshold)

            from src.model import ResNetAnomalyDetector

            rn = ResNetAnomalyDetector(device=device, layers=["layer2", "layer3"])
            with open(pkl_path, "rb") as f:
                state = pickle.load(f)
            rn.mean_features = state["mean_features"]
            rn.std_features = state["std_features"]
            resolved_thresh = threshold if threshold is not None else get_default_threshold(category, "resnet")
            _detectors[key] = _FeatureDetectorAdapter(rn, resolved_thresh)

        else:
            from src.detect import DefectDetector

            resolved_path = model_path
            if resolved_path is None:
                for c in (
                    os.path.join("models", category, "autoencoder_best.pth"),
                    os.path.join("models", category, "autoencoder.pth"),
                    "models/autoencoder_best.pth",
                    "models/autoencoder.pth",
                ):
                    if os.path.exists(c):
                        resolved_path = c
                        break
                if resolved_path is None:
                    raise FileNotFoundError(
                        f"No trained model found for category '{category}'. Train the model first."
                    )

            detector = DefectDetector(resolved_path)
            detector.threshold = threshold if threshold is not None else get_default_threshold(category, "autoencoder")
            _detectors[key] = detector

    return _detectors[key]


def predict_image(
    image: Image.Image,
    category: str = "bottle",
    model_type: str = "patchcore",
    model_path: Optional[str] = None,
) -> Dict:
    """
    Run defect detection on a PIL image for a given product category and model type.
    """
    detector = get_detector(category=category, model_type=model_type, model_path=model_path)
    result = detector.detect(image, return_visuals=True)

    heatmap_img = Image.fromarray(result["heatmap"])
    buf = io.BytesIO()
    heatmap_img.save(buf, format="PNG")
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    overlay_img = Image.fromarray(result["overlay"])
    buf2 = io.BytesIO()
    overlay_img.save(buf2, format="PNG")
    overlay_b64 = base64.b64encode(buf2.getvalue()).decode("utf-8")

    return {
        "category": category,
        "model_type": model_type,
        "prediction": result["prediction"],
        "anomaly_score": round(result["anomaly_score"], 6),
        "threshold": round(result["threshold"], 6) if result["threshold"] else None,
        "heatmap_base64": heatmap_b64,
        "overlay_base64": overlay_b64,
    }
