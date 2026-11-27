"""
Deep Learning Models for Industrial Defect Detection.

Models:
1. ConvAutoencoder - Baseline reconstruction-based anomaly detection
2. ResNetFeatureExtractor - Pretrained feature-based detection
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Dict


# ============================================================
# Model 1: Convolutional Autoencoder
# ============================================================

class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for anomaly detection.
    
    Learns to reconstruct normal product images. At inference time,
    defective regions produce higher reconstruction error.
    
    Architecture:
        Encoder: 4 conv layers (3 -> 32 -> 64 -> 128 -> 256)
        Decoder: 4 transposed conv layers (256 -> 128 -> 64 -> 32 -> 3)
    
    Input:  (B, 3, 256, 256)
    Output: (B, 3, 256, 256)
    """

    def __init__(self):
        super().__init__()

        # Encoder: progressively downsample and extract features
        self.encoder = nn.Sequential(
            # (B, 3, 256, 256) -> (B, 32, 128, 128)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # (B, 32, 128, 128) -> (B, 64, 64, 64)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # (B, 64, 64, 64) -> (B, 128, 32, 32)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # (B, 128, 32, 32) -> (B, 256, 16, 16)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Decoder: progressively upsample and reconstruct
        self.decoder = nn.Sequential(
            # (B, 256, 16, 16) -> (B, 128, 32, 32)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # (B, 128, 32, 32) -> (B, 64, 64, 64)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # (B, 64, 64, 64) -> (B, 32, 128, 128)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # (B, 32, 128, 128) -> (B, 3, 256, 256)
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Output in [0, 1] range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent representation."""
        return self.encoder(x)


# ============================================================
# Model 2: ResNet Feature Extractor
# ============================================================

class ResNetFeatureExtractor(nn.Module):
    """
    Feature extractor using pretrained ResNet18.
    
    Extracts intermediate feature maps from specified layers.
    These features capture semantic information useful for
    comparing normal vs anomalous patterns.
    """

    def __init__(self, layers: List[str] = None):
        super().__init__()
        self.layers = layers or ["layer2", "layer3"]
        
        # Load pretrained ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Register hooks to capture intermediate features
        self._features: Dict[str, torch.Tensor] = {}
        for layer_name in self.layers:
            layer = dict(self.model.named_children())[layer_name]
            layer.register_forward_hook(self._hook_fn(layer_name))

    def _hook_fn(self, layer_name: str):
        def hook(module, input, output):
            self._features[layer_name] = output
        return hook

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from specified layers."""
        self._features = {}
        _ = self.model(x)
        return dict(self._features)


class ResNetAnomalyDetector:
    """
    Anomaly detection using ResNet features.
    
    During fitting: collects feature statistics (mean, std) from normal images.
    During inference: compares test features against normal statistics.
    """

    def __init__(self, device: torch.device, layers: List[str] = None):
        self.device = device
        self.feature_extractor = ResNetFeatureExtractor(layers).to(device)
        self.feature_extractor.eval()
        self.mean_features: Dict[str, torch.Tensor] = {}
        self.std_features: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def fit(self, dataloader):
        """Compute feature statistics from normal training data."""
        all_features: Dict[str, List[torch.Tensor]] = {
            layer: [] for layer in self.feature_extractor.layers
        }

        for images in dataloader:
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(self.device)
            features = self.feature_extractor(images)
            for layer_name, feat in features.items():
                all_features[layer_name].append(feat.cpu())

        for layer_name, feat_list in all_features.items():
            stacked = torch.cat(feat_list, dim=0)
            self.mean_features[layer_name] = stacked.mean(dim=0)
            self.std_features[layer_name] = stacked.std(dim=0) + 1e-8

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor):
        """
        Compute anomaly score and map for a single image.
        
        Returns:
            anomaly_score: float, image-level anomaly score
            anomaly_map: numpy array (H, W), pixel-level anomaly map
        """
        import numpy as np
        import torch.nn.functional as F

        image_tensor = image_tensor.unsqueeze(0).to(self.device) if image_tensor.dim() == 3 else image_tensor.to(self.device)
        features = self.feature_extractor(image_tensor)

        anomaly_maps = []
        for layer_name, feat in features.items():
            feat = feat[0].cpu()
            mean = self.mean_features[layer_name]
            std = self.std_features[layer_name]

            # Mahalanobis-like distance per channel
            diff = ((feat - mean) / std).pow(2).mean(dim=0)  # (H, W)
            
            # Upsample to original image size
            diff_up = F.interpolate(
                diff.unsqueeze(0).unsqueeze(0),
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            ).squeeze().numpy()
            
            anomaly_maps.append(diff_up)

        # Combine maps from different layers
        anomaly_map = np.mean(anomaly_maps, axis=0)
        
        # Score: mean of top 1% pixels
        flat = anomaly_map.flatten()
        k = max(1, int(len(flat) * 0.01))
        anomaly_score = float(np.sort(flat)[-k:].mean())

        return anomaly_score, anomaly_map
