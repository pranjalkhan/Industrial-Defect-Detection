"""
Download MVTec Anomaly Detection Dataset.

Downloads the requested category (default: 'bottle') from the Hugging Face
mirror (foersben/mvtec-ad), with fallback instructions for the official MVTec AD portal.
"""

import os
import argparse


def download_mvtec_category(
    category: str = "bottle",
    data_dir: str = "data/mvtec_anomaly_detection",
    force: bool = False,
):
    """
    Download and extract a specific MVTec AD category.

    Args:
        category: Category name (e.g., 'bottle', 'cable', 'capsule')
        data_dir: Root directory for the dataset
        force: Whether to re-download if already exists
    """
    os.makedirs(data_dir, exist_ok=True)
    category_dir = os.path.join(data_dir, category)

    if os.path.exists(category_dir) and not force:
        num_files = sum(1 for _, _, files in os.walk(category_dir) for f in files)
        if num_files > 10:
            print(f"Category '{category}' already exists with {num_files} files.")
            print("Use --force to re-download.")
            return category_dir

    print(f"Downloading MVTec AD category '{category}' from Hugging Face (foersben/mvtec-ad)...")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="foersben/mvtec-ad",
            repo_type="dataset",
            allow_patterns=f"{category}/**",
            local_dir=data_dir,
        )
    except Exception as e:
        print(f"\nAutomatic download failed: {e}")
        print("\nPlease download manually from:")
        print("  https://www.mvtec.com/company/research/datasets/mvtec-ad")
        print(f"\nExtract the '{category}' category to: {data_dir}/{category}/")
        return None

    if os.path.exists(category_dir):
        num_files = sum(1 for _, _, files in os.walk(category_dir) for f in files)
        print(f"\nDataset ready: {category_dir}")
        print(f"  Total files: {num_files}")

        for split in ["train", "test", "ground_truth"]:
            split_dir = os.path.join(category_dir, split)
            if os.path.exists(split_dir):
                subdirs = sorted([
                    d for d in os.listdir(split_dir)
                    if os.path.isdir(os.path.join(split_dir, d))
                ])
                for sd in subdirs:
                    sd_path = os.path.join(split_dir, sd)
                    count = len([f for f in os.listdir(sd_path) if f.lower().endswith((".png", ".jpg"))])
                    print(f"  {split}/{sd}: {count} images")
    else:
        print(f"Warning: Expected directory {category_dir} was not created.")
        return None

    return category_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MVTec AD Dataset")
    parser.add_argument("--category", type=str, default="bottle", help="Category to download")
    parser.add_argument("--data-dir", type=str, default="data/mvtec_anomaly_detection")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    download_mvtec_category(
        category=args.category,
        data_dir=args.data_dir,
        force=args.force,
    )
