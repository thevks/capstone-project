# GAN-Manipulated Medical Image Detection (LBP + U-Net + SVM)

This project implements the approach from **A. S. & S. Narayan (2024, AMATHE)**  
using Local Binary Patterns (LBP) preprocessing, U-Net feature extraction, and an SVM classifier.

---

## Components
- **synthetic_dataset.py** → creates a toy medical-like dataset (real vs fake).
- **unet2d.py** → 2D U-Net model for embedding extraction.
- **train_2d.py** → training pipeline (U-Net + embeddings + SVM).
- **optuna_search.py** → Optuna tuning for SVM.
- **gan_detection_notebook.ipynb** → notebook demo (Google Colab friendly).

---

## Usage

1. Install dependencies:
pip install torch torchvision scikit-learn scikit-image optuna matplotlib

2. Run synthetic demo training:
python train_2d.py

3. Tune SVM with Optuna:
python optuna_search.py

4. Open Colab Notebook:
jupyter notebook gan_detection_notebook.ipynb




