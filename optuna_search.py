import optuna
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from synthetic_dataset import SyntheticMedicalDataset
import torch

dataset = SyntheticMedicalDataset(n=200)
X = dataset.data.view(len(dataset), -1).numpy()
y = dataset.labels.numpy()

def objective(trial):
    C = trial.suggest_loguniform("C", 1e-2, 1e2)
    gamma = trial.suggest_loguniform("gamma", 1e-3, 1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y)
    clf = SVC(kernel="rbf", C=C, gamma=gamma)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    return accuracy_score(y_val, y_pred)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Best params:", study.best_params)
