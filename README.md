# 🏭 Industrial Defect Detection System

> AI-powered quality inspection system for manufacturing defect detection using deep learning.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Overview

This project implements an end-to-end industrial anomaly detection system that inspects product images and identifies manufacturing defects. The system learns what normal products look like and flags deviations — making it suitable for real-world scenarios where defective samples are rare or unknown.

### Key Features

- **Anomaly Detection**: Trained only on normal images; detects unseen defect types
- **Defect Localization**: Pixel-level heatmaps showing suspected defective regions
- **Multiple Models**: Autoencoder baseline, ResNet feature extractor, and PatchCore comparison
- **Experiment Tracking**: MLflow integration for reproducible experiments
- **REST API**: FastAPI endpoint for real-time predictions
- **Interactive Dashboard**: Streamlit-based quality inspection interface
- **Docker**: Containerized deployment

## 🏗️ Architecture

```
Product Image → Preprocessing → Deep Learning Model → Anomaly Scoring → Classification
                                                     ↓
                                               Heatmap Generation → Dashboard / API
```

### Models Implemented

| Model | Approach | Key Idea |
|-------|----------|----------|
| **Convolutional Autoencoder** | Reconstruction | Learns to reconstruct normal images; defects cause high reconstruction error |
| **ResNet18 Feature Extractor** | Feature comparison | Compares pretrained feature statistics against normal reference |
| **PatchCore** (via Anomalib) | Memory bank | Stores representative normal patch features; detects deviations |

## 📊 Dataset

**MVTec Anomaly Detection (MVTec AD)** — Bottle category

- **Training**: Normal product images only
- **Testing**: Normal + defective images (broken_large, broken_small, contamination)
- **Ground Truth**: Pixel-level defect masks

Download: https://www.mvtec.com/company/research/datasets/mvtec-ad

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/YOUR_USERNAME/industrial-defect-detection.git
cd industrial-defect-detection
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA (adjust for your GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install remaining dependencies
pip install -r requirements.txt
```

### 3. Download Dataset

```bash
python src/download_dataset.py --category bottle
```

### 4. Run Complete Pipeline

```bash
# Run everything: download → train → evaluate → compare
python run_pipeline.py --all

# Or run individual steps:
python run_pipeline.py --download
python run_pipeline.py --train --epochs 50
python run_pipeline.py --calibrate
python run_pipeline.py --evaluate
python run_pipeline.py --train-resnet
python run_pipeline.py --eval-resnet
python run_pipeline.py --compare
```

### 5. Launch API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

### 6. Launch Dashboard

```bash
streamlit run app/dashboard.py
```

Dashboard: http://localhost:8501

## 📁 Project Structure

```
industrial-defect-detection/
├── data/                       # Dataset (not in Git)
├── notebooks/                  # Jupyter notebooks for EDA
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_autoencoder_training.ipynb
│   └── 04_model_evaluation.ipynb
├── src/                        # Core source code
│   ├── dataset.py              # PyTorch dataset classes
│   ├── preprocessing.py        # Image preprocessing utilities
│   ├── model.py                # Model architectures
│   ├── train.py                # Training script
│   ├── detect.py               # Detection engine
│   ├── evaluate.py             # Evaluation pipeline
│   ├── inference.py            # Production inference
│   └── download_dataset.py     # Dataset downloader
├── models/                     # Saved model weights (not in Git)
├── app/                        # Application layer
│   ├── main.py                 # FastAPI application
│   └── dashboard.py            # Streamlit dashboard
├── tests/                      # Unit tests
│   ├── test_model.py
│   ├── test_preprocessing.py
│   └── test_api.py
├── configs/
│   └── config.yaml             # Configuration file
├── artifacts/                  # Evaluation results and plots
├── run_pipeline.py             # Master pipeline script
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

## 🧪 Evaluation Metrics

### Image-Level
- **ROC-AUC**: Overall detection performance
- **F1-Score**: Balance between precision and recall
- **Precision**: Fraction of predicted defects that are actual defects
- **Recall**: Fraction of actual defects detected

### Pixel-Level
- **Pixel AUROC**: Localization accuracy
- **Pixel F1 / IoU**: Defect region overlap quality

## 🐳 Docker

```bash
# Build
docker build -t defect-detection .

# Run
docker run -p 8000:8000 defect-detection
```

## 📈 MLflow Experiment Tracking

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 to compare experiments.

## 🛠️ Technology Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.11+ | Core language |
| PyTorch | Deep learning framework |
| NumPy / Pandas | Data processing |
| OpenCV / Pillow | Image processing |
| Scikit-learn | Evaluation metrics |
| Matplotlib / Seaborn | Visualization |
| MLflow | Experiment tracking |
| FastAPI | REST API |
| Streamlit | Dashboard |
| Docker | Containerization |

## 📝 License

This project is for educational and portfolio purposes.
The MVTec AD dataset has its own license terms — see https://www.mvtec.com/company/research/datasets/mvtec-ad.
