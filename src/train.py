"""
Training script for the Convolutional Autoencoder.

Trains on normal product images and monitors reconstruction loss.
Supports MLflow experiment tracking.
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import NormalDataset, get_transforms
from src.model import ConvAutoencoder


def train(
    data_dir: str,
    output_dir: str = "models",
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    image_size: int = 256,
    val_split: float = 0.15,
    use_mlflow: bool = True,
    augment: bool = True,
    seed: int = 42,
):
    """Train the autoencoder on normal images."""
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Transforms
    train_transform = get_transforms(image_size, augment=augment)
    val_transform = get_transforms(image_size, augment=False)
    
    # Dataset
    full_dataset = NormalDataset(data_dir, transform=train_transform)
    print(f"Total normal images: {len(full_dataset)}")
    
    # Train/val split
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    # Override val transform (no augmentation)
    val_dataset.dataset = NormalDataset(data_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Train: {train_size}, Validation: {val_size}")
    
    # Model
    model = ConvAutoencoder().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # MLflow
    mlflow_run = None
    if use_mlflow:
        try:
            import mlflow
            import mlflow.pytorch
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment("industrial-defect-detection")
            mlflow_run = mlflow.start_run(run_name=f"autoencoder_lr{learning_rate}_bs{batch_size}_ep{epochs}")
            mlflow.log_params({
                "architecture": "ConvAutoencoder",
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "image_size": image_size,
                "augment": augment,
                "total_params": total_params,
                "device": str(device),
            })
            print("MLflow tracking enabled")
        except ImportError:
            print("MLflow not available, skipping tracking")
            use_mlflow = False
    
    # Training
    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    
    print(f"\n{'='*60}")
    print(f"Training Convolutional Autoencoder")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        
        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch+1:3d}/{epochs}] "
            f"Train: {avg_train_loss:.6f} | "
            f"Val: {avg_val_loss:.6f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {elapsed:.1f}s"
        )
        
        # MLflow logging
        if use_mlflow:
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "learning_rate": current_lr,
            }, step=epoch)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            }, os.path.join(output_dir, "autoencoder_best.pth"))
            print(f"  -> Saved best model (val_loss: {best_val_loss:.6f})")
        
        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            }, os.path.join(output_dir, f"autoencoder_epoch{epoch+1}.pth"))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, "autoencoder.pth"))
    
    # Save training history
    np.savez(
        os.path.join(output_dir, "training_history.npz"),
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
    )
    
    # Log final model to MLflow
    if use_mlflow:
        mlflow.log_metrics({
            "best_val_loss": best_val_loss,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
        })
        mlflow.pytorch.log_model(model, name="autoencoder", serialization_format="pickle")
        mlflow.end_run()
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {output_dir}/")
    print(f"{'='*60}")
    
    return model, train_losses, val_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Autoencoder for Anomaly Detection")
    parser.add_argument("--data-dir", type=str, default="data/mvtec_anomaly_detection/bottle/train/good")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        use_mlflow=not args.no_mlflow,
        augment=not args.no_augment,
    )
