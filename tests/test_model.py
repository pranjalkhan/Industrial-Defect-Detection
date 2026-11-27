"""
Tests for model architecture and functionality.
"""

import os
import sys
import pytest
import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import ConvAutoencoder, ResNetFeatureExtractor


class TestConvAutoencoder:
    """Test the Convolutional Autoencoder model."""

    def test_model_creation(self):
        """Test that model can be instantiated."""
        model = ConvAutoencoder()
        assert model is not None

    def test_forward_pass(self):
        """Test forward pass with correct input shape."""
        model = ConvAutoencoder()
        model.eval()
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (2, 3, 256, 256), f"Expected (2,3,256,256), got {output.shape}"

    def test_output_range(self):
        """Test that output is in [0, 1] range (Sigmoid activation)."""
        model = ConvAutoencoder()
        model.eval()
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model(x)
        assert output.min() >= 0.0, f"Min value {output.min()} is below 0"
        assert output.max() <= 1.0, f"Max value {output.max()} is above 1"

    def test_encoder(self):
        """Test encoder produces correct latent shape."""
        model = ConvAutoencoder()
        model.eval()
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            latent = model.encode(x)
        assert latent.shape == (1, 256, 16, 16), f"Expected (1,256,16,16), got {latent.shape}"

    def test_parameter_count(self):
        """Test model has expected parameter count."""
        model = ConvAutoencoder()
        total = sum(p.numel() for p in model.parameters())
        assert total > 100_000, f"Model seems too small: {total} parameters"
        assert total < 50_000_000, f"Model seems too large: {total} parameters"

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = ConvAutoencoder()
        x = torch.randn(1, 3, 256, 256, requires_grad=True)
        output = model(x)
        loss = output.mean()
        loss.backward()
        assert x.grad is not None, "Gradients not flowing to input"


class TestResNetFeatureExtractor:
    """Test the ResNet feature extractor."""

    def test_creation(self):
        """Test extractor can be created."""
        extractor = ResNetFeatureExtractor()
        assert extractor is not None

    def test_feature_extraction(self):
        """Test feature extraction returns expected layers."""
        extractor = ResNetFeatureExtractor(layers=["layer2", "layer3"])
        extractor.eval()
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            features = extractor(x)
        assert "layer2" in features
        assert "layer3" in features
        assert features["layer2"].dim() == 4
        assert features["layer3"].dim() == 4

    def test_frozen_parameters(self):
        """Test that parameters are frozen."""
        extractor = ResNetFeatureExtractor()
        for param in extractor.model.parameters():
            assert not param.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
