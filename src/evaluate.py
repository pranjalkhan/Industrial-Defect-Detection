"""
Model evaluation pipeline.

Evaluates trained models on the MVTec AD test set using:
- Image-level metrics: ROC-AUC, F1, Precision, Recall
- Pixel-level metrics: ROC-AUC, F1, IoU, PRO
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import TestDataset, get_transforms
from src.detect import DefectDetector


def evaluate_model(
    detector: DefectDetector,
    test_dir: str,
    ground_truth_dir: str,
    image_size: int = 256,
    output_dir: str = "artifacts/evaluation",
) -> Dict:
    """
    Full evaluation pipeline.
    
    Returns dict with all computed metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    transform = get_transforms(image_size, augment=False)
    test_dataset = TestDataset(
        test_dir=test_dir,
        ground_truth_dir=ground_truth_dir,
        transform=transform,
        image_size=image_size,
    )
    
    print(f"Test set: {len(test_dataset)} images")
    
    # Collect predictions
    all_labels = []
    all_scores = []
    all_pixel_labels = []
    all_pixel_scores = []

    import time
    t0 = time.time()
    for i in range(len(test_dataset)):
        image, label, mask, path = test_dataset[i]
        
        score, anomaly_map = detector._compute_anomaly(image)
        
        all_labels.append(label)
        all_scores.append(score)
        
        # Pixel-level: flatten mask and anomaly map
        mask_flat = mask.numpy().flatten()
        map_flat = anomaly_map.flatten()
        
        # Resize anomaly map if needed
        if len(map_flat) != len(mask_flat):
            import cv2
            h, w = mask.shape[1], mask.shape[2]
            anomaly_map_resized = cv2.resize(anomaly_map, (w, h))
            map_flat = anomaly_map_resized.flatten()
        
        all_pixel_labels.append(mask_flat)
        all_pixel_scores.append(map_flat)

        if (i + 1) % 20 == 0:
            print(f"  Evaluated {i+1}/{len(test_dataset)} images...")

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    all_pixel_labels = np.concatenate(all_pixel_labels)
    all_pixel_scores = np.concatenate(all_pixel_scores)
    inference_ms = ((time.time() - t0) / len(test_dataset)) * 1000.0
    model_path = getattr(detector, "model_path", "models/autoencoder_best.pth")
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0.0

    # ============================================================
    # Image-level metrics
    # ============================================================
    image_auroc = roc_auc_score(all_labels, all_scores)
    
    # Binary predictions using threshold
    if detector.threshold is not None:
        threshold = detector.threshold
    else:
        # Use optimal threshold from ROC curve
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds[optimal_idx]
    
    all_preds = (all_scores > threshold).astype(int)
    
    image_f1 = f1_score(all_labels, all_preds)
    image_precision = precision_score(all_labels, all_preds, zero_division=0)
    image_recall = recall_score(all_labels, all_preds, zero_division=0)
    image_ap = average_precision_score(all_labels, all_scores)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # ============================================================
    # Pixel-level metrics
    # ============================================================
    # Only compute if there are actual defective pixels
    has_defective_pixels = all_pixel_labels.sum() > 0
    
    if has_defective_pixels:
        pixel_auroc = roc_auc_score(all_pixel_labels, all_pixel_scores)
        
        # Optimal pixel threshold
        from sklearn.metrics import roc_curve
        pfpr, ptpr, pthresholds = roc_curve(all_pixel_labels, all_pixel_scores)
        pixel_opt_idx = np.argmax(ptpr - pfpr)
        pixel_threshold = pthresholds[pixel_opt_idx]
        pixel_preds = (all_pixel_scores > pixel_threshold).astype(int)
        
        pixel_f1 = f1_score(all_pixel_labels, pixel_preds)
        pixel_precision = precision_score(all_pixel_labels, pixel_preds, zero_division=0)
        pixel_recall = recall_score(all_pixel_labels, pixel_preds, zero_division=0)
        
        # IoU
        intersection = np.logical_and(pixel_preds, all_pixel_labels).sum()
        union = np.logical_or(pixel_preds, all_pixel_labels).sum()
        pixel_iou = intersection / max(union, 1)
        pixel_ap = average_precision_score(all_pixel_labels, all_pixel_scores)
    else:
        pixel_auroc = pixel_f1 = pixel_precision = pixel_recall = pixel_iou = pixel_ap = 0.0

    metrics = {
        # Image-level
        "image_auroc": float(image_auroc),
        "image_f1": float(image_f1),
        "image_precision": float(image_precision),
        "image_recall": float(image_recall),
        "image_ap": float(image_ap),
        "threshold": float(threshold),
        "confusion_matrix": conf_matrix.tolist(),
        # Pixel-level
        "pixel_auroc": float(pixel_auroc),
        "pixel_f1": float(pixel_f1),
        "pixel_precision": float(pixel_precision),
        "pixel_recall": float(pixel_recall),
        "pixel_iou": float(pixel_iou),
        "pixel_ap": float(pixel_ap),
        # Sample counts & performance
        "total_test_images": len(all_labels),
        "normal_count": int((all_labels == 0).sum()),
        "defective_count": int((all_labels == 1).sum()),
        "inference_ms": round(inference_ms, 2),
        "model_size_mb": round(model_size_mb, 2),
    }

    # ============================================================
    # Print results
    # ============================================================
    print(f"\n{'='*60}")
    print("MODEL EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"\nTest set: {metrics['total_test_images']} images "
          f"({metrics['normal_count']} normal, {metrics['defective_count']} defective)")
    
    print(f"\n--- Image-Level Metrics ---")
    print(f"  ROC-AUC:   {metrics['image_auroc']:.4f}")
    print(f"  F1-Score:  {metrics['image_f1']:.4f}")
    print(f"  Precision: {metrics['image_precision']:.4f}")
    print(f"  Recall:    {metrics['image_recall']:.4f}")
    print(f"  AP:        {metrics['image_ap']:.4f}")
    print(f"  Threshold: {metrics['threshold']:.6f}")
    
    print(f"\n--- Pixel-Level Metrics ---")
    print(f"  ROC-AUC:   {metrics['pixel_auroc']:.4f}")
    print(f"  F1-Score:  {metrics['pixel_f1']:.4f}")
    print(f"  Precision: {metrics['pixel_precision']:.4f}")
    print(f"  Recall:    {metrics['pixel_recall']:.4f}")
    print(f"  IoU:       {metrics['pixel_iou']:.4f}")
    
    print(f"\n--- Confusion Matrix ---")
    print(f"  {conf_matrix}")
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=["Normal", "Defective"])
    print(f"\n{report}")

    # ============================================================
    # Generate plots
    # ============================================================
    _plot_evaluation(all_labels, all_scores, all_preds, metrics, output_dir)
    
    # Save metrics
    import json
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def _plot_evaluation(
    labels: np.ndarray,
    scores: np.ndarray,
    preds: np.ndarray,
    metrics: Dict,
    output_dir: str,
):
    """Generate evaluation plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    axes[0, 0].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {metrics["image_auroc"]:.4f}')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('Image-Level ROC Curve')
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Score Distribution
    normal_scores = scores[labels == 0]
    defective_scores = scores[labels == 1]
    axes[0, 1].hist(normal_scores, bins=30, alpha=0.6, label='Normal', color='green')
    axes[0, 1].hist(defective_scores, bins=30, alpha=0.6, label='Defective', color='red')
    if metrics["threshold"] is not None:
        axes[0, 1].axvline(x=metrics["threshold"], color='blue', linestyle='--', label=f'Threshold = {metrics["threshold"]:.4f}')
    axes[0, 1].set_xlabel('Anomaly Score')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Anomaly Score Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Confusion Matrix
    cm = np.array(metrics["confusion_matrix"])
    im = axes[1, 0].imshow(cm, cmap='Blues')
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_xticklabels(['Normal', 'Defective'])
    axes[1, 0].set_yticklabels(['Normal', 'Defective'])
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_title('Confusion Matrix')
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, str(cm[i][j]), ha='center', va='center', fontsize=16, fontweight='bold')
    plt.colorbar(im, ax=axes[1, 0])
    
    # 4. Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(labels, scores)
    axes[1, 1].plot(recall_vals, precision_vals, 'r-', linewidth=2, label=f'AP = {metrics["image_ap"]:.4f}')
    axes[1, 1].set_xlabel('Recall')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].set_title('Precision-Recall Curve')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation_plots.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nEvaluation plots saved to: {output_dir}/evaluation_plots.png")


def evaluate_sample_visualizations(
    detector: DefectDetector,
    test_dir: str,
    output_dir: str = "artifacts/evaluation/samples",
    num_samples: int = 5,
):
    """Generate visual samples showing detection results."""
    os.makedirs(output_dir, exist_ok=True)
    
    sample_count = 0
    for defect_type in sorted(os.listdir(test_dir)):
        type_dir = os.path.join(test_dir, defect_type)
        if not os.path.isdir(type_dir):
            continue
        
        images = sorted([
            f for f in os.listdir(type_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        
        for img_name in images[:num_samples]:
            img_path = os.path.join(type_dir, img_name)
            result = detector.detect(img_path)
            
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            axes[0].imshow(result["original"])
            axes[0].set_title(f"Original ({defect_type})")
            axes[0].axis("off")
            
            axes[1].imshow(result["reconstructed"])
            axes[1].set_title("Reconstructed")
            axes[1].axis("off")
            
            axes[2].imshow(result["heatmap"], cmap="jet")
            axes[2].set_title(f"Anomaly Map\nScore: {result['anomaly_score']:.4f}")
            axes[2].axis("off")
            
            axes[3].imshow(result["overlay"])
            axes[3].set_title(f"Result: {result['prediction']}")
            axes[3].axis("off")
            
            plt.tight_layout()
            save_name = f"{defect_type}_{img_name.replace('.png', '')}.png"
            plt.savefig(os.path.join(output_dir, save_name), dpi=150, bbox_inches='tight')
            plt.close()
            
            sample_count += 1
    
    print(f"Saved {sample_count} sample visualizations to {output_dir}/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate defect detection model")
    parser.add_argument("--model-path", type=str, default="models/autoencoder_best.pth")
    parser.add_argument("--test-dir", type=str, default="data/mvtec_anomaly_detection/bottle/test")
    parser.add_argument("--gt-dir", type=str, default="data/mvtec_anomaly_detection/bottle/ground_truth")
    parser.add_argument("--output-dir", type=str, default="artifacts/evaluation")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    
    detector = DefectDetector(args.model_path)
    
    if args.threshold:
        detector.threshold = args.threshold
    else:
        # Calibrate from validation data
        from src.dataset import NormalDataset, get_transforms
        from torch.utils.data import DataLoader
        
        val_transform = get_transforms(256, augment=False)
        val_dir = args.test_dir.replace("/test", "/train/good")
        val_dataset = NormalDataset(val_dir, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        detector.calibrate_threshold(val_loader, percentile=99.0)
    
    metrics = evaluate_model(
        detector=detector,
        test_dir=args.test_dir,
        ground_truth_dir=args.gt_dir,
        output_dir=args.output_dir,
    )
    
    evaluate_sample_visualizations(
        detector=detector,
        test_dir=args.test_dir,
        output_dir=os.path.join(args.output_dir, "samples"),
    )
