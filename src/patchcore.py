"""
PatchCore Anomaly Detection implementation in PyTorch.

Follows the PatchCore algorithm (Roth et al., CVPR 2022):
1. Extract locally aware patch features from intermediate layers of a pretrained WideResNet50-2.
2. Build a memory bank of normal patch features using greedy coreset subsampling.
3. Score test images via nearest-neighbor distance to the memory bank.
4. Produce image-level anomaly scores and upsampled pixel-level anomaly maps.
"""

import time
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PatchCoreDetector:
    """PatchCore anomaly detector using WideResNet50-2 patch embeddings and a coreset memory bank."""

    def __init__(
        self,
        device: torch.device,
        layers: List[str] = None,
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
        image_size: int = 256,
    ):
        self.device = device
        self.layers = layers or ["layer2", "layer3"]
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.num_neighbors = num_neighbors
        self.image_size = image_size

        self.backbone = models.wide_resnet50_2(
            weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        ).to(device)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self._features: Dict[str, torch.Tensor] = {}
        for layer_name in self.layers:
            layer = dict(self.backbone.named_children())[layer_name]
            layer.register_forward_hook(self._make_hook(layer_name))

        self.memory_bank: torch.Tensor = None

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            self._features[name] = out
        return hook

    @torch.no_grad()
    def _extract_patch_embeddings(self, images: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Extract locally-averaged multi-scale patch embeddings.

        Returns:
            embeddings: Tensor of shape (B * H * W, D)
            h, w: spatial dimensions of the feature map
        """
        self._features = {}
        _ = self.backbone(images.to(self.device))

        target_h, target_w = self._features[self.layers[0]].shape[2:]
        pooled_maps = []
        for layer_name in self.layers:
            feat = self._features[layer_name]
            # Local neighborhood averaging (3x3) for locally aware patch features
            feat = F.avg_pool2d(feat, kernel_size=3, stride=1, padding=1)
            if feat.shape[2:] != (target_h, target_w):
                feat = F.interpolate(
                    feat, size=(target_h, target_w), mode="bilinear", align_corners=False
                )
            pooled_maps.append(feat)

        combined = torch.cat(pooled_maps, dim=1)  # (B, D, H, W)
        b, d, h, w = combined.shape
        embeddings = combined.permute(0, 2, 3, 1).reshape(b * h * w, d)
        return embeddings, h, w

    @staticmethod
    def _greedy_coreset_subsample(
        embeddings: torch.Tensor, ratio: float, seed: int = 42
    ) -> torch.Tensor:
        """Select representative normal patch embeddings via greedy k-center coreset."""
        n_samples = embeddings.shape[0]
        n_select = max(1, int(n_samples * ratio))
        if n_select >= n_samples:
            return embeddings

        rng = torch.Generator(device=embeddings.device)
        rng.manual_seed(seed)

        # Random projection to 128D for fast distance computation during coreset selection
        if embeddings.shape[1] > 128:
            proj = torch.randn(embeddings.shape[1], 128, device=embeddings.device, generator=rng)
            proj_emb = embeddings @ proj
        else:
            proj_emb = embeddings

        selected_indices = []
        first_idx = int(torch.randint(0, n_samples, (1,), generator=rng, device=embeddings.device).item())
        selected_indices.append(first_idx)

        min_distances = torch.cdist(proj_emb, proj_emb[first_idx : first_idx + 1]).squeeze(1)

        for _ in range(1, n_select):
            next_idx = int(torch.argmax(min_distances).item())
            selected_indices.append(next_idx)
            dist_to_new = torch.cdist(proj_emb, proj_emb[next_idx : next_idx + 1]).squeeze(1)
            min_distances = torch.minimum(min_distances, dist_to_new)

        return embeddings[selected_indices]

    @torch.no_grad()
    def fit(self, dataloader) -> None:
        """Build the coreset memory bank from normal training images."""
        all_embeddings = []
        for images in dataloader:
            if isinstance(images, (list, tuple)):
                images = images[0]
            emb, _, _ = self._extract_patch_embeddings(images)
            all_embeddings.append(emb)

        full_bank = torch.cat(all_embeddings, dim=0)
        print(f"PatchCore total normal patches: {full_bank.shape[0]} (dim={full_bank.shape[1]})")
        self.memory_bank = self._greedy_coreset_subsample(
            full_bank, ratio=self.coreset_sampling_ratio
        )
        print(f"PatchCore coreset memory bank size: {self.memory_bank.shape[0]} patches")

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> Tuple[float, np.ndarray]:
        """
        Compute anomaly score and pixel-level anomaly map for a single image.

        Returns:
            anomaly_score: float
            anomaly_map: np.ndarray of shape (image_size, image_size)
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        patch_emb, h, w = self._extract_patch_embeddings(image_tensor)
        # Compute Euclidean distances to memory bank
        dists = torch.cdist(patch_emb, self.memory_bank)  # (H*W, M)
        knn_dists, _ = torch.topk(dists, k=min(self.num_neighbors, self.memory_bank.shape[0]), dim=1, largest=False)

        # Nearest neighbor distance per patch
        patch_scores = knn_dists[:, 0]
        max_idx = int(torch.argmax(patch_scores).item())

        # Re-weighted image-level anomaly score (PatchCore Eq. 7)
        s_star = patch_scores[max_idx]
        nearest_k_dists = knn_dists[max_idx]
        weights = torch.softmax(nearest_k_dists, dim=0)
        image_score = float((1.0 - weights[0]) * s_star + (weights * nearest_k_dists).sum())
        # Fallback simple max if single neighbor
        if self.num_neighbors <= 1:
            image_score = float(s_star)

        # Reshape to spatial grid and upsample + Gaussian smooth
        score_map = patch_scores.reshape(1, 1, h, w)
        score_map_up = F.interpolate(
            score_map, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()

        # Gaussian smoothing (sigma=4) as in PatchCore
        score_map_up = cv2.GaussianBlur(score_map_up.astype(np.float32), (0, 0), sigmaX=4, sigmaY=4)
        return image_score, score_map_up
