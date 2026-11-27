"""
Tests for preprocessing utilities.
"""

import os
import sys
import pytest
import numpy as np
import torch
from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    validate_image,
    preprocess_image,
    tensor_to_image,
    create_heatmap_overlay,
)
from src.dataset import NormalDataset, TestDataset, get_transforms


class TestPreprocessing:
    """Test image preprocessing functions."""

    def test_preprocess_image_shape(self):
        """Test that preprocessing produces correct tensor shape."""
        img = Image.new("RGB", (640, 480), color="blue")
        tensor = preprocess_image(img, image_size=256)
        assert tensor.shape == (1, 3, 256, 256)

    def test_preprocess_image_range(self):
        """Test that tensor values are in [0, 1] range."""
        img = Image.new("RGB", (100, 100), color="red")
        tensor = preprocess_image(img, image_size=256)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_tensor_to_image(self):
        """Test tensor to numpy image conversion."""
        tensor = torch.rand(3, 256, 256)
        img = tensor_to_image(tensor)
        assert img.shape == (256, 256, 3)
        assert img.dtype == np.uint8
        assert img.min() >= 0
        assert img.max() <= 255

    def test_tensor_to_image_batch(self):
        """Test tensor to image with batch dimension."""
        tensor = torch.rand(1, 3, 128, 128)
        img = tensor_to_image(tensor)
        assert img.shape == (128, 128, 3)

    def test_heatmap_overlay(self):
        """Test heatmap overlay generation."""
        original = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        anomaly_map = np.random.rand(256, 256).astype(np.float32)
        overlay = create_heatmap_overlay(original, anomaly_map)
        assert overlay.shape == (256, 256, 3)
        assert overlay.dtype == np.uint8

    def test_heatmap_overlay_different_sizes(self):
        """Test heatmap overlay with mismatched sizes."""
        original = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        anomaly_map = np.random.rand(256, 256).astype(np.float32)
        overlay = create_heatmap_overlay(original, anomaly_map)
        assert overlay.shape == (512, 512, 3)


class TestTransforms:
    """Test data transforms."""

    def test_basic_transforms(self):
        """Test basic transform pipeline."""
        transform = get_transforms(256, augment=False)
        img = Image.new("RGB", (640, 480))
        tensor = transform(img)
        assert tensor.shape == (3, 256, 256)

    def test_augmented_transforms(self):
        """Test augmented transform pipeline."""
        transform = get_transforms(256, augment=True)
        img = Image.new("RGB", (640, 480))
        tensor = transform(img)
        assert tensor.shape == (3, 256, 256)


class TestNormalDataset:
    """Test NormalDataset class."""

    def test_dataset_with_temp_images(self, tmp_path):
        """Test dataset loading with temporary images."""
        # Create temporary images
        for i in range(5):
            img = Image.new("RGB", (100, 100), color=(i * 50, 0, 0))
            img.save(tmp_path / f"image_{i:03d}.png")

        transform = get_transforms(256)
        dataset = NormalDataset(str(tmp_path), transform=transform)
        
        assert len(dataset) == 5
        sample = dataset[0]
        assert sample.shape == (3, 256, 256)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
