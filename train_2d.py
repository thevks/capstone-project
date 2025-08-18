import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

from unet2d import UNet2D
from synthetic_dataset import SyntheticMedicalDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
dataset = SyntheticMedicalDataset(n=200)
train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.3, stratify=dataset.labels)
train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=16, shuffle=True)
test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=16)

# Model
model = UNet2D().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Train autoencoder-style to learn embeddings
print("Training U-Net encoder...")
for epoch in range(5):
    model.train()
    for x, _ in train_loader:
        x = x.to(device)
        emb = model(x)
        loss = torch.mean(emb**2)  # proxy loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: loss {loss.item():.4f}")

# Extract embeddings
def get_embeddings(loader):
    model.eval()
    embs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            embs.append(model(x).cpu())
            labels.append(y)
    return torch.cat(embs), torch.cat(labels)

X_train, y_train = get_embeddings(train_loader)
X_test, y_test = get_embeddings(test_loader)

# SVM
print("Training SVM...")
svm = SVC(kernel="rbf", C=1, gamma="scale")
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(svm, "svm_model.pkl")
torch.save(model.state_dict(), "unet_encoder.pth")
print("Models saved!")
