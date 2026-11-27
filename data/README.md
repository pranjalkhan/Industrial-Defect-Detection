# MVTec AD Dataset

This directory is where the MVTec Anomaly Detection dataset is stored.

**Do NOT commit the dataset to Git.** It is listed in `.gitignore`.

## Download

Run the download script:

```bash
python src/download_dataset.py --category bottle
```

Or download manually from: https://www.mvtec.com/company/research/datasets/mvtec-ad

Extract the `bottle` category into:
```
data/mvtec_anomaly_detection/bottle/
```

## Expected Structure

```
data/mvtec_anomaly_detection/
└── bottle/
    ├── train/
    │   └── good/          # Normal training images
    ├── test/
    │   ├── good/          # Normal test images
    │   ├── broken_large/  # Defective test images
    │   ├── broken_small/  # Defective test images
    │   └── contamination/ # Defective test images
    └── ground_truth/
        ├── broken_large/  # Pixel-level defect masks
        ├── broken_small/
        └── contamination/
```
