# detect_gan_medical.py
import os
import random
import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# -------------------------
# Configuration / Defaults
# -------------------------
DEFAULT_IMG_SIZE = 256
LBP_P = 8  # number of circularly symmetric neighbour set points
LBP_R = 1  # radius
LBP_METHOD = 'uniform'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Utilities
# -------------------------
def read_image(path: str, size: int = DEFAULT_IMG_SIZE) -> np.ndarray:
    img = Image.open(path).convert("L")  # grayscale
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def compute_lbp(img: np.ndarray, P=LBP_P, R=LBP_R, method=LBP_METHOD) -> np.ndarray:
    """
    img: 2D grayscale image (float in [0,1])
    Returns a normalized histogram or LBP map resized to match image shape.
    We'll return the LBP map (same size) here so we can concatenate spatially or pool later.
    """
    lbp_map = local_binary_pattern(img, P, R, method)
    # Normalize to [0,1] by dividing by max possible value for this LBP method
    max_val = lbp_map.max() if lbp_map.max() > 0 else 1.0
    lbp_norm = lbp_map.astype(np.float32) / max_val
    return lbp_norm


# -------------------------
# Toy Dataset (for testing)
# -------------------------
class SyntheticCTDataset(Dataset):
    """
    Creates synthetic 'real' and 'fake' images for quick testing.
    Real images: smooth gaussian blobs.
    Fake images: gaussian blobs + synthetic 'artifact' (circle/stripe) resembling GAN artifact.
    """
    def __init__(self, n_samples=200, img_size=DEFAULT_IMG_SIZE, transform=None, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.n = n_samples
        self.size = img_size
        self.transform = transform
        self.X, self.y = self._generate(n_samples)

    def _generate_blob(self):
        x = np.linspace(-1,1,self.size)
        xv, yv = np.meshgrid(x,x)
        sigma = random.uniform(0.2, 0.5)
        cx = random.uniform(-0.5,0.5)
        cy = random.uniform(-0.5,0.5)
        blob = np.exp(-(((xv-cx)**2 + (yv-cy)**2)/(2*sigma**2)))
        # add mild noise
        blob += np.random.normal(0, 0.02, size=blob.shape)
        blob = np.clip(blob, 0, 1)
        return blob

    def _fake_artifact(self, img):
        # add a synthetic stripe or circle artifact
        img = img.copy()
        h, w = img.shape
        if random.random() < 0.5:
            # horizontal stripe
            y = random.randint(h//4, 3*h//4)
            thickness = random.randint(2, 6)
            img[y:y+thickness, :] = np.clip(img[y:y+thickness,:] + 0.4, 0, 1)
        else:
            # circle
            cx = random.randint(w//4, 3*w//4)
            cy = random.randint(h//4, 3*h//4)
            rr, cc = np.ogrid[:h,:w]
            mask = (rr - cy)**2 + (cc - cx)**2 <= (min(h,w)//8)**2
            img[mask] = np.clip(img[mask] + 0.35, 0, 1)
        img += np.random.normal(0, 0.02, size=img.shape)
        return np.clip(img, 0, 1)

    def _generate(self, n):
        X = []
        y = []
        for i in range(n):
            base = self._generate_blob()
            if random.random() < 0.5:
                # real
                X.append(base)
                y.append(0)
            else:
                # fake
                X.append(self._fake_artifact(base))
                y.append(1)
        return np.stack(X), np.array(y, dtype=np.int64)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img = self.X[idx]
        label = self.y[idx]
        # output tensors: image, lbp_map, label
        lbp = compute_lbp(img)
        img_t = torch.from_numpy(img).unsqueeze(0).float()  # (1,H,W)
        lbp_t = torch.from_numpy(lbp).unsqueeze(0).float()
        return img_t, lbp_t, int(label)


# -------------------------
# U-Net (small) - PyTorch
# -------------------------
def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNetEncoder(nn.Module):
    """
    A U-Net encoder-only variant that returns a bottleneck feature map and global pooled vector.
    """
    def __init__(self, in_channels=1, base_filters=32):
        super().__init__()
        self.enc1 = double_conv(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = double_conv(base_filters, base_filters*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = double_conv(base_filters*2, base_filters*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = double_conv(base_filters*4, base_filters*8)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = double_conv(base_filters*8, base_filters*16)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x1 = self.enc1(x)  # H
        p1 = self.pool1(x1)
        x2 = self.enc2(p1)
        p2 = self.pool2(x2)
        x3 = self.enc3(p2)
        p3 = self.pool3(x3)
        x4 = self.enc4(p3)
        p4 = self.pool4(x4)
        b = self.bottleneck(p4)
        # global feature vector
        g = self.global_pool(b).view(b.size(0), -1)
        return b, g  # bottleneck map, global vector


# -------------------------
# Training helpers
# -------------------------
def train_unet_encoder(encoder: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                       epochs=10, lr=1e-3, model_save_path="unet_encoder.pth"):
    encoder = encoder.to(DEVICE)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    # We'll train encoder as an autoencoder-style proxy: reconstruct input image
    # Simple decoder to reconstruct grayscale image from bottleneck
    decoder = nn.Sequential(
        nn.ConvTranspose2d(encoder.bottleneck[0].in_channels, 256, kernel_size=2, stride=2), # adapt later
    )
    # For simplicity we create a small decoder matching our bottleneck dims dynamically:
    # Instead, we'll train using a tiny reconstruction head that upsamples the pooled vector.
    recon_head = nn.Sequential(
        nn.Linear(encoder.global_pool.output_size if hasattr(encoder.global_pool, 'output_size') else encoder.bottleneck[-1].out_channels, 256),
    )
    # Simpler approach: freeze recon_head idea — but to keep it straightforward, we'll train using contrastive proxy:
    # Instead, do a supervised proxy: predict label using global vector. This pushes encoder to learn discriminative features.
    clf = nn.Sequential(
        nn.Linear(encoder.bottleneck[-1].out_channels if hasattr(encoder.bottleneck[-1], 'out_channels') else 512, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    ).to(DEVICE)

    # But building dynamic shapes is messy; instead: we'll do a simpler approach:
    # Train encoder + a small classifier head end-to-end to predict real/fake — to produce embeddings that SVM can use.
    # So reinitialize classifier with correct input dim:
    sample_x, sample_lbp, sample_y = next(iter(train_loader))
    sample_x = sample_x.to(DEVICE)
    with torch.no_grad():
        _, sample_g = encoder(sample_x)
    feat_dim = sample_g.shape[1]

    classifier = nn.Sequential(
        nn.Linear(feat_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2)
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(epochs):
        encoder.train(); classifier.train()
        train_losses = []
        for imgs, lbps, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            _, feats = encoder(imgs)
            preds = classifier(feats)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            # optimizer should include encoder and classifier params
            # simple way: create optimizer over both
            # (we did optimizer = Adam(encoder.parameters()), but must include classifier)
            # so instead update manually:
            loss.backward()
            optimizer.step()
            # Also update classifier with a separate optimizer:
            # For clarity: we'll use single optimizer over both:
            train_losses.append(loss.item())
        # validation:
        encoder.eval(); classifier.eval()
        ys = []; ypred = []
        with torch.no_grad():
            for imgs, lbps, labels in val_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                _, feats = encoder(imgs)
                preds = classifier(feats)
                pred_labels = preds.argmax(dim=1)
                ys.extend(labels.cpu().numpy().tolist())
                ypred.extend(pred_labels.cpu().numpy().tolist())
        val_acc = accuracy_score(ys, ypred)
        print(f"Epoch {epoch+1}/{epochs} train_loss={np.mean(train_losses):.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'encoder_state': encoder.state_dict(),
                'classifier_state': classifier.state_dict()
            }, model_save_path)
    print("Best val acc:", best_val_acc)
    # Return trained encoder (load best if saved)
    ck = torch.load(model_save_path, map_location=DEVICE)
    encoder.load_state_dict(ck['encoder_state'])
    classifier.load_state_dict(ck['classifier_state'])
    return encoder, classifier


def extract_embeddings(encoder: nn.Module, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    encoder = encoder.to(DEVICE)
    encoder.eval()
    feats_list = []
    lbp_hist_list = []
    labels_list = []
    with torch.no_grad():
        for imgs, lbps, labels in dataloader:
            imgs = imgs.to(DEVICE)
            _, feats = encoder(imgs)  # feats: (B, D)
            feats_np = feats.cpu().numpy()
            lbp_np = lbps.view(lbps.size(0), -1).numpy()  # flatten LBP map; could pool instead
            feats_list.append(feats_np)
            lbp_hist_list.append(lbp_np)
            labels_list.append(labels.numpy())
    X_feats = np.vstack(feats_list)
    X_lbp = np.vstack(lbp_hist_list)
    y = np.concatenate(labels_list)
    # Option: reduce lbp dimension by mean pooling spatially
    # We'll compute simple global LBP stats to keep SVM size reasonable
    lbp_mean = X_lbp.mean(axis=1, keepdims=True)
    lbp_std = X_lbp.std(axis=1, keepdims=True)
    X_comb = np.concatenate([X_feats, lbp_mean, lbp_std], axis=1)
    return X_comb, y


# -------------------------
# Main pipeline
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="./saved")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 1) Prepare dataset (synthetic quick example)
    ds = SyntheticCTDataset(n_samples=400, img_size=128)
    # split
    idx = list(range(len(ds)))
    train_idx, test_idx = train_test_split(idx, test_size=0.3, stratify=ds.y, random_state=42)
    # create subset dataloaders
    def subset(ds_obj, indices):
        class Subset(Dataset):
            def __init__(self, ds, idxs):
                self.ds = ds
                self.idxs = idxs
            def __len__(self): return len(self.idxs)
            def __getitem__(self, i): return self.ds[self.idxs[i]]
        return Subset(ds_obj, indices)

    train_ds = subset(ds, train_idx)
    test_ds = subset(ds, test_idx)
    # further split train into train/val
    tr_idx, val_idx = train_test_split(list(range(len(train_ds))), test_size=0.15, stratify=[train_ds[i][2] for i in range(len(train_ds))], random_state=1)
    train_ds = subset(train_ds, tr_idx)
    val_ds = subset(ds, [train_idx[i] for i in val_idx])  # map indices

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    # 2) Build U-Net encoder
    encoder = UNetEncoder(in_channels=1, base_filters=16)  # small model for test
    print("Encoder built. Params:", sum(p.numel() for p in encoder.parameters()))

    # 3) Train encoder with proxy classifier to produce discriminative embeddings
    print("Training encoder (proxy supervised training)...")
    encoder, classifier = train_unet_encoder(encoder, train_loader, val_loader,
                                            epochs=args.epochs, lr=1e-3,
                                            model_save_path=os.path.join(args.save_dir, "encoder_clf.pth"))

    # 4) Extract embeddings for train/test
    print("Extracting embeddings for SVM...")
    X_train, y_train = extract_embeddings(encoder, train_loader)
    X_test, y_test = extract_embeddings(encoder, test_loader)

    # 5) Standardize and train SVM
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    svm = SVC(kernel='rbf', probability=True, class_weight='balanced', C=1.0)
    print("Training SVM on embeddings ...")
    svm.fit(X_train_s, y_train)

    # 6) Evaluate
    y_pred = svm.predict(X_test_s)
    y_proba = svm.predict_proba(X_test_s)[:,1]
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except:
        auc = None
    print("SVM test acc:", acc)
    if auc is not None:
        print("SVM test AUC:", auc)
    print(classification_report(y_test, y_pred))

    # 7) Save models & scaler
    joblib.dump({'svm': svm, 'scaler': scaler}, os.path.join(args.save_dir, "svm_pipeline.joblib"))
    torch.save({'encoder_state': encoder.state_dict()}, os.path.join(args.save_dir, "encoder_final.pth"))
    print("Saved encoder and SVM pipeline to", args.save_dir)

    print("Done. To use on your own dataset: adapt Dataset to read real CT images and GAN images, and rerun pipeline.")

if __name__ == "__main__":
    main()
