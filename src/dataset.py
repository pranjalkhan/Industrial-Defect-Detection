"""
Dataset classes for MVTec Anomaly Detection.
Handles loading of normal training images, test images (normal + defective),
and ground-truth masks for evaluation.
"""

import os
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class NormalDataset(Dataset):
    """Dataset for loading normal (good) product images for training."""

    def __init__(self, image_dir: str, transform: Optional[transforms.Compose] = None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


class TestDataset(Dataset):
    """
    Dataset for loading test images with labels and optional ground-truth masks.
    
    Returns:
        image: Preprocessed image tensor
        label: 0 = normal, 1 = defective
        mask: Ground-truth mask tensor (zeros if normal)
        path: Original file path
    """

    def __init__(
        self,
        test_dir: str,
        ground_truth_dir: str,
        transform: Optional[transforms.Compose] = None,
        mask_transform: Optional[transforms.Compose] = None,
        image_size: int = 256,
    ):
        self.test_dir = test_dir
        self.ground_truth_dir = ground_truth_dir
        self.transform = transform
        self.image_size = image_size

        if mask_transform is None:
            self.mask_transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ])
        else:
            self.mask_transform = mask_transform

        self.samples: List[Tuple[str, int, Optional[str]]] = []
        self._load_samples()

    def _load_samples(self):
        """Scan test directory to build sample list with labels and mask paths."""
        for defect_type in sorted(os.listdir(self.test_dir)):
            type_dir = os.path.join(self.test_dir, defect_type)
            if not os.path.isdir(type_dir):
                continue

            is_normal = defect_type == "good"
            label = 0 if is_normal else 1

            for img_name in sorted(os.listdir(type_dir)):
                if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                img_path = os.path.join(type_dir, img_name)
                mask_path = None

                if not is_normal:
                    # Ground truth mask: same name but with _mask suffix
                    mask_name = os.path.splitext(img_name)[0] + "_mask.png"
                    candidate = os.path.join(self.ground_truth_dir, defect_type, mask_name)
                    if os.path.exists(candidate):
                        mask_path = candidate

                self.samples.append((img_path, label, mask_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label, mask_path = self.samples[idx]

        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load or create mask
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mask)
            # Binarize mask (threshold at 0.5)
            mask = (mask > 0.5).float()
        else:
            # Normal image: all-zero mask
            mask = np.zeros((1, self.image_size, self.image_size), dtype=np.float32)
            import torch
            mask = torch.from_numpy(mask)

        return image, label, mask, img_path


def get_transforms(image_size: int = 256, augment: bool = False) -> transforms.Compose:
    """
    Get image transforms for training or inference.
    
    For anomaly detection autoencoders, we keep transforms minimal:
    - Resize to consistent dimensions
    - Convert to tensor (scales to [0, 1])
    - Optional light augmentation for training
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
    ]

    if augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])

    transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)
