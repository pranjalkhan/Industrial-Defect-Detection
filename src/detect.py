"""
Defect detection and heatmap generation.

Takes a trained model and an input image, produces:
- Anomaly score (image-level)
- Anomaly map (pixel-level)
- Heatmap visualization
- Normal/Defective classification
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import ConvAutoencoder
from src.preprocessing import create_heatmap_overlay, tensor_to_image


class DefectDetector:
    """
    Defect detection engine using a trained autoencoder.
    
    Usage:
        detector = DefectDetector("models/autoencoder_best.pth")
        detector.calibrate_threshold(val_loader)
        result = detector.detect("path/to/image.png")
    """

    def __init__(
        self,
        model_path: str,
        image_size: int = 256,
        score_top_k: float = 0.01,
        device: Optional[torch.device] = None,
    ):
        self.model_path = model_path
        self.image_size = image_size
        self.score_top_k = score_top_k
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = None

        # Load model
        self.model = ConvAutoencoder().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def _compute_anomaly(self, image_tensor: torch.Tensor) -> Tuple[float, np.ndarray]:
        """
        Compute anomaly score and map for a single image tensor.
        
        Returns:
            anomaly_score: float
            anomaly_map: np.ndarray of shape (H, W)
        """
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device) if image_tensor.dim() == 3 else image_tensor.to(self.device)
            reconstructed = self.model(image_tensor)

            original = image_tensor[0].cpu().numpy()
            recon = reconstructed[0].cpu().numpy()

            # Pixel-level error: mean absolute error across channels
            anomaly_map = np.mean(np.abs(original - recon), axis=0)  # (H, W)

            # Image-level score: mean of top-k% pixel errors
            flat = anomaly_map.flatten()
            k = max(1, int(len(flat) * self.score_top_k))
            anomaly_score = float(np.sort(flat)[-k:].mean())

        return anomaly_score, anomaly_map

    def calibrate_threshold(self, normal_loader, percentile: float = 99.0) -> float:
        """
        Calibrate detection threshold using normal validation images.
        
        Uses the given percentile of normal image anomaly scores.
        """
        scores = []
        for images in normal_loader:
            if isinstance(images, (list, tuple)):
                images = images[0]
            for i in range(images.shape[0]):
                score, _ = self._compute_anomaly(images[i])
                scores.append(score)

        self.threshold = float(np.percentile(scores, percentile))
        print(f"Calibrated threshold: {self.threshold:.6f} (p{percentile})")
        print(f"  Normal score range: [{min(scores):.6f}, {max(scores):.6f}]")
        print(f"  Normal score mean:  {np.mean(scores):.6f}")
        print(f"  Normal score std:   {np.std(scores):.6f}")
        return self.threshold

    def detect(self, image_input, return_visuals: bool = True) -> Dict:
        """
        Detect defects in a single image.
        
        Args:
            image_input: Path string or PIL Image
            return_visuals: Whether to generate visualization images
            
        Returns:
            Dict with keys:
                - prediction: "NORMAL" or "DEFECTIVE"
                - anomaly_score: float
                - threshold: float or None
                - anomaly_map: np.ndarray (H, W)
                - heatmap: np.ndarray (H, W, 3) if return_visuals
                - overlay: np.ndarray (H, W, 3) if return_visuals
                - reconstructed: np.ndarray (H, W, 3) if return_visuals
        """
        # Load image
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")
        
        original_size = image.size

        # Preprocess
        tensor = self.transform(image)

        # Compute anomaly
        anomaly_score, anomaly_map = self._compute_anomaly(tensor)

        # Classify
        if self.threshold is not None:
            prediction = "DEFECTIVE" if anomaly_score > self.threshold else "NORMAL"
        else:
            prediction = "UNKNOWN (no threshold calibrated)"

        result = {
            "prediction": prediction,
            "anomaly_score": anomaly_score,
            "threshold": self.threshold,
            "anomaly_map": anomaly_map,
        }

        if return_visuals:
            # Get reconstructed image
            with torch.no_grad():
                tensor_in = tensor.unsqueeze(0).to(self.device)
                recon_tensor = self.model(tensor_in)
            
            original_np = tensor_to_image(tensor)
            recon_np = tensor_to_image(recon_tensor)

            # Normalize anomaly map for visualization
            if anomaly_map.max() > anomaly_map.min():
                norm_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
            else:
                norm_map = np.zeros_like(anomaly_map)
            heatmap = (norm_map * 255).astype(np.uint8)

            # Create overlay
            overlay = create_heatmap_overlay(original_np, anomaly_map)

            result["original"] = original_np
            result["reconstructed"] = recon_np
            result["heatmap"] = heatmap
            result["overlay"] = overlay

        return result

    def batch_detect(self, image_paths: list) -> list:
        """Run detection on multiple images."""
        results = []
        for path in image_paths:
            result = self.detect(path, return_visuals=False)
            result["path"] = path
            results.append(result)
        return results
