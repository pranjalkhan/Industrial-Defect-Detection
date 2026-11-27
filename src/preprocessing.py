"""
Image preprocessing utilities for the defect detection pipeline.

Handles:
- Image loading and validation
- Resizing and normalization
- Data augmentation
- Batch preprocessing for inference
"""

import os
from typing import Tuple, Optional
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms


def validate_image(image_path: str) -> bool:
    """Check if an image file is valid and not corrupted."""
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except Exception:
        return False


def scan_dataset(data_dir: str) -> dict:
    """
    Scan dataset directory and return statistics.
    
    Returns dict with:
        - total_images: int
        - categories: dict mapping category name to image count  
        - image_sizes: list of (width, height) tuples
        - corrupted: list of corrupted file paths
    """
    stats = {
        "total_images": 0,
        "categories": {},
        "image_sizes": set(),
        "corrupted": [],
        "modes": set(),
    }

    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            path = os.path.join(root, f)
            rel = os.path.relpath(root, data_dir)
            category = rel.split(os.sep)[0] if os.sep in rel else rel

            if not validate_image(path):
                stats["corrupted"].append(path)
                continue

            stats["total_images"] += 1
            stats["categories"][category] = stats["categories"].get(category, 0) + 1

            img = Image.open(path)
            stats["image_sizes"].add(img.size)
            stats["modes"].add(img.mode)

    stats["image_sizes"] = list(stats["image_sizes"])
    stats["modes"] = list(stats["modes"])
    return stats


def preprocess_image(
    image: Image.Image,
    image_size: int = 256,
) -> torch.Tensor:
    """
    Preprocess a single PIL image for model inference.
    
    Args:
        image: PIL Image (RGB)
        image_size: Target size for resizing
        
    Returns:
        Tensor of shape (1, 3, image_size, image_size)
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    tensor = transform(image)
    return tensor.unsqueeze(0)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor back to a numpy image (H, W, C) in [0, 255] range.
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = tensor.cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def create_heatmap_overlay(
    original: np.ndarray,
    anomaly_map: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Create a heatmap overlay on the original image.
    
    Args:
        original: Original image (H, W, 3) in [0, 255]
        anomaly_map: Anomaly map (H, W) with arbitrary range
        alpha: Overlay transparency
        colormap: OpenCV colormap
        
    Returns:
        Overlay image (H, W, 3) in [0, 255]
    """
    # Normalize anomaly map to [0, 255]
    if anomaly_map.max() > anomaly_map.min():
        normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
    else:
        normalized = np.zeros_like(anomaly_map)
    
    heatmap_uint8 = (normalized * 255).astype(np.uint8)
    
    # Resize heatmap to match original
    if heatmap_uint8.shape[:2] != original.shape[:2]:
        heatmap_uint8 = cv2.resize(heatmap_uint8, (original.shape[1], original.shape[0]))
    
    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = (1 - alpha) * original.astype(np.float32) + alpha * heatmap_color.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return overlay
