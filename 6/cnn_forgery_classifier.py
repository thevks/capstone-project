"""
cnn_forgery_classifier.py

CNN-based End-to-End Medical Image Forgery Classifier
- Detects GAN-manipulated CT scans
- Classes: TB, TM, FB, FM

Author: Adapted for user
"""

import os, glob, random
import numpy as np
import cv2, pydicom
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# --------------------------
# Config
# --------------------------
DATA_DIR = "/path/to/dataset"  # <-- change to your dataset
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 4   # TB, TM, FB, FM
EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["FB", "FM", "TB", "TM"]

# --------------------------
# Utils: read CT image (DICOM/PNG)
# --------------------------
def read_image(path, size=(IMG_SIZE, IMG_SIZE)):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dcm":
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max()
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    arr = cv2.resize(arr, size, interpolation=cv2.INTER_AREA)
    return arr

# --------------------------
# Dataset
# --------------------------
class ForgeryDataset(Dataset):
    def __init__(self, data_dir, file_list=None, train=True):
        self.data_dir = data_dir
        if file_list is None:
            self.paths = sorted(glob.glob(os.path.join(data_dir, "images", "*")))
        else:
            self.paths = file_list

        self.class_names = CLASS_NAMES
        self.le = LabelEncoder()
        self.le.fit(self.class_names)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip() if train else T.Lambda(lambda x: x),
            T.RandomRotation(10) if train else T.Lambda(lambda x: x),
            T.ColorJitter(brightness=0.1, contrast=0.1) if train else T.Lambda(lambda x: x),
            T.ToTensor(),
            T.Normalize(mean=[0.485], std=[0.229])  # grayscale normalized
        ])

    def _infer_label(self, fname):
        for cname in self.class_names:
            if cname.lower() in fname.lower():
                return int(self.le.transform([cname])[0])
        return 2  # default TB

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = read_image(p, size=(IMG_SIZE, IMG_SIZE))
        img3 = np.stack([img, img, img], axis=-1)  # 3-channel for CNN
        img_t = self.transform(img3)
        label = self._infer_label(os.path.basename(p))
        return img_t, label, os.path.basename(p)

# --------------------------
# Model: Pretrained CNN
# --------------------------
class ForgeryClassifier(nn.Module):
    def __init__(self, backbone="efficientnet_b0", num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)

# --------------------------
# Grad-CAM
# --------------------------
class GradCAM:
    def __init__(self, model, target_layer="blocks.6.2.conv_pw"):  
        self.model = model
        self.target_layer = dict([*self.model.backbone.named_modules()])[target_layer]
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        loss = logits[:, class_idx]
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        return cam

# --------------------------
# Train / Eval
# --------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total, correct, running_loss = 0, 0, 0
    for imgs, labels, _ in tqdm(loader, desc="Train"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return running_loss/total, correct/total

def eval_model(model, loader, criterion):
    model.eval()
    total, correct, running_loss = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels, _ in tqdm(loader, desc="Eval"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss/total, correct/total, all_preds, all_labels

# --------------------------
# Main
# --------------------------
def main():
    # load dataset
    ds = ForgeryDataset(DATA_DIR)
    paths = ds.paths
    random.shuffle(paths)
    n = len(paths)
    train_paths, val_paths, test_paths = paths[:int(0.7*n)], paths[int(0.7*n):int(0.85*n)], paths[int(0.85*n):]

    train_ds = ForgeryDataset(DATA_DIR, file_list=train_paths, train=True)
    val_ds   = ForgeryDataset(DATA_DIR, file_list=val_paths, train=False)
    test_ds  = ForgeryDataset(DATA_DIR, file_list=test_paths, train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # model
    model = ForgeryClassifier("efficientnet_b0", num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # training loop
    best_val_acc = 0
    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _ = eval_model(model, val_loader, criterion)
        print(f"Epoch {epoch}: Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f}")
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), "cnn_best.pth")
            best_val_acc = val_acc
            print("✅ Saved new best model")

    # evaluation on test set
    model.load_state_dict(torch.load("cnn_best.pth"))
    _, test_acc, preds, labels = eval_model(model, test_loader, criterion)
    print("Test Accuracy:", test_acc)
    print("Classification Report:\n", classification_report(labels, preds, target_names=CLASS_NAMES))
    print("Confusion Matrix:\n", confusion_matrix(labels, preds))

    # Grad-CAM visualization
    gradcam = GradCAM(model)
    sample_img, _, fname = test_ds[0]
    input_tensor = sample_img.unsqueeze(0).to(DEVICE)
    cam = gradcam.generate(input_tensor)
    img = sample_img[0].numpy()  # first channel
    plt.imshow(img, cmap="gray")
    plt.imshow(cam, cmap="jet", alpha=0.5)
    plt.title(f"Grad-CAM: {fname}")
    plt.show()

if __name__ == "__main__":
    main()
