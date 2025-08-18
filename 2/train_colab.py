# train_colab.py
"""
Colab-ready end-to-end runner:
 - Mount Google Drive (if using Colab)
 - Install dependencies (in Colab cell run)
 - Load dataset via LIDCSliceDataset (point to your drive path)
 - Compute LBP maps for each sample (skimage)
 - Train 3D U-Net for a few epochs (proxy classifier or segmentation if labels exist)
 - Extract embeddings, train SVM, evaluate
 - Visualize embeddings via PCA & t-SNE
"""

# Example of the commands to run at top of Colab:
# !pip install -q torch torchvision scikit-learn scikit-image SimpleITK pydicom umap-learn

import os
import argparse
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import joblib
import torch
from torch.utils.data import DataLoader
from skimage.feature import local_binary_pattern

from lidc_dataset import LIDCSliceDataset
from unet3d import UNet3D

def compute_lbp_map(img: np.ndarray, P=8, R=1, method='uniform'):
    return local_binary_pattern(img, P, R, method).astype(np.float32)

def collate_for_unet3d(batch):
    # batch: list of (vol3d, meta)
    vols = [torch.from_numpy(item[0]).unsqueeze(0) for item in batch]  # (C=1,D,H,W)
    vols = torch.stack(vols, dim=0).float()
    return vols  # you can return labels if available

def extract_embeddings(model, dataloader, device='cpu'):
    model = model.to(device)
    model.eval()
    feats = []
    metas = []
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            _, emb = model(x)
            feats.append(emb.cpu().numpy())
    feats = np.vstack(feats)
    return feats

def visualize_embeddings(X, y, savepath=None, method='pca'):
    if method == 'pca':
        p = PCA(n_components=2).fit_transform(X)
    else:
        p = TSNE(n_components=2, perplexity=30, init='pca', n_iter=1000).fit_transform(X)
    plt.figure(figsize=(6,6))
    for lab in np.unique(y):
        mask = y == lab
        plt.scatter(p[mask,0], p[mask,1], label=str(int(lab)), alpha=0.7, s=12)
    plt.legend()
    plt.title(f"Embedding visualization ({method.upper()})")
    if savepath:
        plt.savefig(savepath, dpi=150)
    plt.show()

def main():
    # Example: point to a folder in Drive containing multiple study folders (each a DICOM series)
    data_root = "/content/drive/MyDrive/LIDC_data"  # change to your path
    mode = '3d'
    n_slices = 9
    batch = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = LIDCSliceDataset(data_root, mode=mode, n_slices_3d=n_slices)
    print("Dataset size:", len(ds))
    # Build dataloader (for embeddings extraction, we just want the volumes)
    def simple_collate(batch):
        vols = []
        for v, meta in batch:
            # v shape: (D, H, W) normalized
            vols.append(torch.from_numpy(v).unsqueeze(0))  # (1,D,H,W)
        vols = torch.stack(vols, 0).float()  # (B,1,D,H,W)
        return vols

    loader = DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=simple_collate)

    # Build model
    model = UNet3D(in_channels=1, n_classes=2, base_filters=16)
    # Option: load pretrained weights or do a quick proxy training (skipped here for brevity)
    # Quick hack: run a forward pass to warm up
    sample = next(iter(loader))
    out, emb = model(sample.to(device))
    print("Sample embedding shape:", emb.shape)

    # For production: train model (with labels) or use supervised proxy training (train small classifier on top of embedding)
    # Here we extract embeddings and save them; you can then train SVM externally after collecting labeled embeddings.
    embeddings = extract_embeddings(model, loader, device=device)

    # Placeholders: create fake labels and train SVM to demonstrate pipeline. Replace with real labels.
    y_dummy = np.random.randint(0, 2, size=len(embeddings))

    scaler = StandardScaler().fit(embeddings)
    Xs = scaler.transform(embeddings)
    svm = SVC(kernel='rbf', probability=True, class_weight='balanced')
    svm.fit(Xs, y_dummy)

    y_pred = svm.predict(Xs)
    print("Dummy SVM acc:", accuracy_score(y_dummy, y_pred))
    print(classification_report(y_dummy, y_pred))

    # Visualize embeddings
    visualize_embeddings(embeddings, y_dummy, method='pca')

    # Save pipeline
    joblib.dump({'svm': svm, 'scaler': scaler}, "/content/drive/MyDrive/svm_pipeline.joblib")
    torch.save(model.state_dict(), "/content/drive/MyDrive/unet3d_final.pth")
    print("Saved models to Drive")

if __name__ == "__main__":
    main()

'''
Colab notes:

Run a cell at top to install dependencies:

!pip install -q torch torchvision scikit-learn scikit-image SimpleITK pydicom umap-learn


Mount Drive:

from google.colab import drive
drive.mount('/content/drive')


Upload lidc_dataset.py and unet3d.py to the Colab session (or push them into your Drive and add to sys.path).
'''

'''
4) PCA / t-SNE / UMAP visualization & analysis code

Use in Colab or locally; code snippet (already in train_colab.py) does PCA/t-SNE. For larger embeddings use UMAP for speed:

# UMAP visualization (optional, faster than TSNE)
import umap
um = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
proj = um.fit_transform(X_embeddings)
plt.scatter(proj[:,0], proj[:,1], c=y_labels, cmap='tab10', s=6)
plt.title("UMAP projection of embeddings")
'''
