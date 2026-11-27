"""
Master pipeline script - Run the complete defect detection workflow across one or more categories.

Usage:
    python run_pipeline.py --all --category bottle
    python run_pipeline.py --all --categories bottle screw metal_nut cable capsule
    python run_pipeline.py --download --category screw
    python run_pipeline.py --train --calibrate --evaluate --category screw
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CATEGORIES = ["bottle", "screw", "metal_nut", "cable", "capsule"]


def step_download(category="bottle", data_dir="data/mvtec_anomaly_detection"):
    """Step 1: Download MVTec AD dataset."""
    print("\n" + "=" * 60)
    print(f"STEP 1: Download Dataset ({category})")
    print("=" * 60)
    from src.download_dataset import download_mvtec_category
    return download_mvtec_category(category=category, data_dir=data_dir)


def step_train_autoencoder(
    data_dir,
    output_dir,
    epochs=50,
    batch_size=16,
    lr=0.001,
):
    """Step 2: Train the Convolutional Autoencoder."""
    print("\n" + "=" * 60)
    print(f"STEP 2: Train Autoencoder -> {output_dir}")
    print("=" * 60)
    from src.train import train
    model, _, _ = train(
        data_dir=data_dir,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        use_mlflow=True,
        augment=True,
    )
    return model


def step_calibrate_threshold(
    model_path,
    val_dir,
    output_dir,
    percentile=99.0,
):
    """Step 3: Calibrate detection threshold from normal validation data."""
    print("\n" + "=" * 60)
    print(f"STEP 3: Calibrate Threshold -> {output_dir}")
    print("=" * 60)
    from src.detect import DefectDetector
    from src.dataset import NormalDataset, get_transforms

    detector = DefectDetector(model_path)
    transform = get_transforms(256, augment=False)
    dataset = NormalDataset(val_dir, transform=transform)

    val_size = max(1, int(0.15 * len(dataset)))
    train_size = len(dataset) - val_size
    _, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    threshold = detector.calibrate_threshold(val_loader, percentile=percentile)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "threshold.txt"), "w") as f:
        f.write(str(threshold))
    print(f"Threshold saved to {output_dir}/threshold.txt")
    return threshold


def step_evaluate(
    model_path,
    threshold_path,
    val_dir,
    test_dir,
    gt_dir,
    output_dir,
):
    """Step 4: Evaluate Autoencoder on test set."""
    print("\n" + "=" * 60)
    print(f"STEP 4: Evaluate Autoencoder -> {output_dir}")
    print("=" * 60)
    from src.detect import DefectDetector
    from src.evaluate import evaluate_model, evaluate_sample_visualizations

    detector = DefectDetector(model_path)

    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            detector.threshold = float(f.read().strip())
        print(f"Loaded threshold: {detector.threshold}")
    else:
        detector.threshold = step_calibrate_threshold(
            model_path, val_dir, os.path.dirname(threshold_path)
        )

    metrics = evaluate_model(
        detector=detector,
        test_dir=test_dir,
        ground_truth_dir=gt_dir,
        output_dir=output_dir,
    )
    evaluate_sample_visualizations(
        detector=detector,
        test_dir=test_dir,
        output_dir=os.path.join(output_dir, "samples"),
        num_samples=3,
    )
    return metrics


def step_train_resnet(train_dir, output_dir):
    """Step 5: Train ResNet-based anomaly detector."""
    print("\n" + "=" * 60)
    print(f"STEP 5: Train ResNet Feature Detector -> {output_dir}")
    print("=" * 60)
    from src.model import ResNetAnomalyDetector
    from src.dataset import NormalDataset
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = NormalDataset(train_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    detector = ResNetAnomalyDetector(device, layers=["layer2", "layer3"])
    detector.fit(loader)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "resnet_detector.pkl")
    state = {
        "mean_features": {k: v.cpu() for k, v in detector.mean_features.items()},
        "std_features": {k: v.cpu() for k, v in detector.std_features.items()},
    }
    with open(save_path, "wb") as f:
        pickle.dump(state, f)
    print(f"ResNet detector saved to {save_path}")
    return detector


def step_evaluate_resnet(train_dir, test_dir, gt_dir, model_dir, output_dir):
    """Step 6: Evaluate ResNet detector."""
    print("\n" + "=" * 60)
    print(f"STEP 6: Evaluate ResNet Detector -> {output_dir}")
    print("=" * 60)
    from src.model import ResNetAnomalyDetector
    from src.dataset import NormalDataset, TestDataset
    from torchvision import transforms
    from sklearn.metrics import roc_auc_score, f1_score, classification_report
    import cv2

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    detector_path = os.path.join(model_dir, "resnet_detector.pkl")
    detector = ResNetAnomalyDetector(device, layers=["layer2", "layer3"])

    if os.path.exists(detector_path):
        with open(detector_path, "rb") as f:
            state = pickle.load(f)
        detector.mean_features = state["mean_features"]
        detector.std_features = state["std_features"]
        print("Loaded ResNet detector from disk")
    else:
        train_dataset = NormalDataset(train_dir, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
        detector.fit(train_loader)

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    test_dataset = TestDataset(test_dir, gt_dir, transform=transform, mask_transform=mask_transform)

    all_labels, all_scores = [], []
    all_pixel_labels, all_pixel_scores = [], []

    t0 = time.time()
    for i in range(len(test_dataset)):
        image, label, mask, _ = test_dataset[i]
        score, anomaly_map = detector.predict(image)
        all_labels.append(label)
        all_scores.append(score)

        mask_flat = mask.numpy().flatten()
        if anomaly_map.shape != (256, 256):
            anomaly_map = cv2.resize(anomaly_map, (256, 256))
        all_pixel_labels.append(mask_flat)
        all_pixel_scores.append(anomaly_map.flatten())
    inference_ms = ((time.time() - t0) / len(test_dataset)) * 1000.0

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    all_pixel_labels = np.concatenate(all_pixel_labels)
    all_pixel_scores = np.concatenate(all_pixel_scores)

    image_auroc = roc_auc_score(all_labels, all_scores)
    pixel_auroc = roc_auc_score(all_pixel_labels, all_pixel_scores) if all_pixel_labels.sum() > 0 else 0.0

    val_dataset = NormalDataset(train_dir, transform=transform)
    val_size = max(1, int(0.15 * len(val_dataset)))
    train_size = len(val_dataset) - val_size
    _, val_subset = random_split(
        val_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    val_scores = [detector.predict(val_subset[i])[0] for i in range(len(val_subset))]
    threshold = float(np.percentile(val_scores, 99))
    preds = (all_scores > threshold).astype(int)
    image_f1 = f1_score(all_labels, preds)
    model_size_mb = os.path.getsize(detector_path) / (1024 * 1024) if os.path.exists(detector_path) else 0.0

    metrics = {
        "model": "ResNet18 Feature Extractor",
        "image_auroc": float(image_auroc),
        "pixel_auroc": float(pixel_auroc),
        "image_f1": float(image_f1),
        "threshold": float(threshold),
        "inference_ms": round(inference_ms, 2),
        "model_size_mb": round(model_size_mb, 2),
    }

    print(f"\n--- ResNet Feature Detector Results ---")
    print(f"  Image AUROC: {image_auroc:.4f} | Pixel AUROC: {pixel_auroc:.4f} | F1: {image_f1:.4f} | {inference_ms:.2f} ms/img")
    print(classification_report(all_labels, preds, target_names=["Normal", "Defective"], zero_division=0))

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def step_patchcore(train_dir, test_dir, gt_dir, model_dir, output_dir):
    """Step 6b: Train and evaluate PatchCore detector."""
    print("\n" + "=" * 60)
    print(f"STEP 6b: Train & Evaluate PatchCore -> {output_dir}")
    print("=" * 60)
    from src.patchcore import PatchCoreDetector
    from src.dataset import NormalDataset, TestDataset
    from torchvision import transforms
    from sklearn.metrics import roc_auc_score, f1_score, classification_report
    import cv2

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = NormalDataset(train_dir, transform=transform)
    val_size = max(1, int(0.15 * len(train_dataset)))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=False)

    detector = PatchCoreDetector(
        device=device,
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        image_size=256,
    )
    detector.fit(train_loader)

    bank_path = os.path.join(model_dir, "patchcore_bank.pt")
    torch.save(detector.memory_bank.cpu(), bank_path)
    model_size_mb = os.path.getsize(bank_path) / (1024 * 1024)

    val_scores = [detector.predict(val_subset[i])[0] for i in range(len(val_subset))]
    threshold = float(np.percentile(val_scores, 99))

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    test_dataset = TestDataset(test_dir, gt_dir, transform=transform, mask_transform=mask_transform)

    all_labels, all_scores = [], []
    all_pixel_labels, all_pixel_scores = [], []

    t0 = time.time()
    for i in range(len(test_dataset)):
        image, label, mask, _ = test_dataset[i]
        score, anomaly_map = detector.predict(image)
        all_labels.append(label)
        all_scores.append(score)

        mask_flat = mask.numpy().flatten()
        if anomaly_map.shape != (256, 256):
            anomaly_map = cv2.resize(anomaly_map, (256, 256))
        all_pixel_labels.append(mask_flat)
        all_pixel_scores.append(anomaly_map.flatten())
    inference_ms = ((time.time() - t0) / len(test_dataset)) * 1000.0

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    all_pixel_labels = np.concatenate(all_pixel_labels)
    all_pixel_scores = np.concatenate(all_pixel_scores)

    image_auroc = roc_auc_score(all_labels, all_scores)
    pixel_auroc = roc_auc_score(all_pixel_labels, all_pixel_scores) if all_pixel_labels.sum() > 0 else 0.0
    preds = (all_scores > threshold).astype(int)
    image_f1 = f1_score(all_labels, preds)

    metrics = {
        "model": "PatchCore (WideResNet50-2)",
        "image_auroc": float(image_auroc),
        "pixel_auroc": float(pixel_auroc),
        "image_f1": float(image_f1),
        "threshold": float(threshold),
        "inference_ms": round(inference_ms, 2),
        "model_size_mb": round(model_size_mb, 2),
    }

    print(f"\n--- PatchCore Results ---")
    print(f"  Image AUROC: {image_auroc:.4f} | Pixel AUROC: {pixel_auroc:.4f} | F1: {image_f1:.4f} | {inference_ms:.2f} ms/img")
    print(classification_report(all_labels, preds, target_names=["Normal", "Defective"], zero_division=0))

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def step_compare_models(output_dir):
    """Step 7: Compare all model results for a category."""
    print("\n" + "=" * 60)
    print(f"STEP 7: Model Comparison ({output_dir})")
    print("=" * 60)

    results = {}
    for label, sub in [
        ("Autoencoder", "evaluation"),
        ("ResNet Features", "evaluation_resnet"),
        ("PatchCore", "evaluation_patchcore"),
    ]:
        p = os.path.join(output_dir, sub, "metrics.json")
        if os.path.exists(p):
            with open(p) as f:
                results[label] = json.load(f)

    if not results:
        print("No evaluation results found. Run evaluation first.")
        return None

    print(f"\n{'Model':<20} {'Image AUROC':<13} {'Pixel AUROC':<13} {'Image F1':<11} {'Infer (ms)':<12} {'Size (MB)':<10}")
    print("-" * 80)
    for name, m in results.items():
        infer = f"{m['inference_ms']:.2f}" if "inference_ms" in m else "N/A"
        size = f"{m['model_size_mb']:.2f}" if "model_size_mb" in m else "N/A"
        print(
            f"{name:<20} "
            f"{m.get('image_auroc', 0):<13.4f} "
            f"{m.get('pixel_auroc', 0):<13.4f} "
            f"{m.get('image_f1', 0):<11.4f} "
            f"{infer:<12} "
            f"{size:<10}"
        )

    with open(os.path.join(output_dir, "model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser(description="Industrial Defect Detection Pipeline")
    parser.add_argument("--all", action="store_true", help="Run complete pipeline")
    parser.add_argument("--download", action="store_true", help="Download dataset")
    parser.add_argument("--train", action="store_true", help="Train autoencoder")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate threshold")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate autoencoder")
    parser.add_argument("--train-resnet", action="store_true", help="Train ResNet detector")
    parser.add_argument("--eval-resnet", action="store_true", help="Evaluate ResNet detector")
    parser.add_argument("--patchcore", action="store_true", help="Train and evaluate PatchCore")
    parser.add_argument("--compare", action="store_true", help="Compare models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--categories", nargs="+", default=None, help="Run across multiple categories")
    parser.add_argument("--all-categories", action="store_true", help=f"Run across {DEFAULT_CATEGORIES}")
    args = parser.parse_args()

    cats = args.categories or (DEFAULT_CATEGORIES if args.all_categories else [args.category])
    data_base = "data/mvtec_anomaly_detection"
    start = time.time()

    all_summary = {}
    for cat in cats:
        print(f"\n\n{'#' * 60}")
        print(f"# CATEGORY: {cat.upper()}")
        print(f"{'#' * 60}")

        train_dir = f"{data_base}/{cat}/train/good"
        test_dir = f"{data_base}/{cat}/test"
        gt_dir = f"{data_base}/{cat}/ground_truth"
        model_dir = f"models/{cat}"
        art_dir = f"artifacts/{cat}"

        if args.all or args.download:
            step_download(category=cat, data_dir=data_base)

        if args.all or args.train:
            step_train_autoencoder(train_dir, output_dir=model_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

        if args.all or args.calibrate:
            step_calibrate_threshold(
                model_path=os.path.join(model_dir, "autoencoder_best.pth"),
                val_dir=train_dir,
                output_dir=model_dir,
            )

        if args.all or args.evaluate:
            step_evaluate(
                model_path=os.path.join(model_dir, "autoencoder_best.pth"),
                threshold_path=os.path.join(model_dir, "threshold.txt"),
                val_dir=train_dir,
                test_dir=test_dir,
                gt_dir=gt_dir,
                output_dir=os.path.join(art_dir, "evaluation"),
            )

        if args.all or args.train_resnet:
            step_train_resnet(train_dir=train_dir, output_dir=model_dir)

        if args.all or args.eval_resnet:
            step_evaluate_resnet(
                train_dir=train_dir,
                test_dir=test_dir,
                gt_dir=gt_dir,
                model_dir=model_dir,
                output_dir=os.path.join(art_dir, "evaluation_resnet"),
            )

        if args.all or args.patchcore:
            step_patchcore(
                train_dir=train_dir,
                test_dir=test_dir,
                gt_dir=gt_dir,
                model_dir=model_dir,
                output_dir=os.path.join(art_dir, "evaluation_patchcore"),
            )

        if args.all or args.compare:
            comp = step_compare_models(output_dir=art_dir)
            if comp:
                all_summary[cat] = comp

    if len(all_summary) > 1:
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/all_categories_comparison.json", "w") as f:
            json.dump(all_summary, f, indent=2)
        print("\nSaved multi-category summary to artifacts/all_categories_comparison.json")

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Pipeline completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
