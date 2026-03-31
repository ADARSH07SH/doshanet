import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
Training pipeline for DoshaNet (no torchvision dependency).
  - Loads data.json dataset
  - Trains multimodal model (40 epochs, label smoothing, early stopping)
  - Also trains a Random Forest baseline on questionnaire features only
  - Saves best model to model/saved/dosha_model.pt
  - Saves RF baseline to model/saved/rf_baseline.pkl
"""

import json
import sys
import time

import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in [_ROOT, _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model.model import DoshaNet, CLASSES  # noqa (path set above)

# ── Config ────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(__file__))
DATA_JSON  = os.path.join(ROOT, "dataset", "data.json")
SAVE_DIR   = os.path.join(ROOT, "model", "saved")
IMG_ROOT   = os.path.join(ROOT, "dataset")

BATCH_SIZE  = 128
EPOCHS      = 15
LR          = 1e-3
PATIENCE    = 4
LABEL_SMOOTH= 0.1
IMG_SIZE    = 64   # Smaller size for lightweight CNN
DEVICE      = "cpu"

IMG_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
IMG_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)

os.makedirs(SAVE_DIR, exist_ok=True)


# ── Image helpers (pure PIL + numpy, no torchvision) ─────────────────────────
def load_image_tensor(path: str, augment: bool = False) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)

    if augment:
        # Random horizontal flip
        if np.random.rand() > 0.5:
            arr = arr[:, ::-1, :].copy()
        # Color jitter
        brightness = np.random.uniform(0.8, 1.2)
        arr = np.clip(arr * brightness, 0, 1)

    arr = (arr - IMG_MEAN) / IMG_STD
    return torch.tensor(arr).permute(2, 0, 1)  # (3, H, W)


# ── Dataset ───────────────────────────────────────────────────────────────────
class DoshaDataset(Dataset):
    def __init__(self, records, augment=False):
        self.records = records
        self.augment = augment
        self.le      = LabelEncoder()
        self.le.fit(CLASSES)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec      = self.records[idx]
        img_path = os.path.join(IMG_ROOT, rec["image"])
        image    = load_image_tensor(img_path, self.augment)
        features = torch.tensor(rec["features"], dtype=torch.float32)
        label    = torch.tensor(self.le.transform([rec["label"]])[0], dtype=torch.long)
        return image, features, label


def load_splits():
    with open(DATA_JSON) as f:
        data = json.load(f)
    train = [r for r in data if r["split"] == "train"]
    val   = [r for r in data if r["split"] == "val"]
    test  = [r for r in data if r["split"] == "test"]
    return train, val, test


# ── Baseline RF ───────────────────────────────────────────────────────────────
def train_rf_baseline(train, val):
    print("\n-- Random Forest Baseline --")
    X_train = np.array([r["features"] for r in train])
    y_train = [r["label"] for r in train]
    X_val   = np.array([r["features"] for r in val])
    y_val   = [r["label"] for r in val]

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)

    val_acc = accuracy_score(y_val, rf.predict(X_val))
    print(f"   RF Val Accuracy : {val_acc*100:.1f}%")

    out = os.path.join(SAVE_DIR, "rf_baseline.pkl")
    joblib.dump(rf, out)
    print(f"   Saved -> {out}")
    return rf


# ── DoshaNet Training ─────────────────────────────────────────────────────────
def train_model(train_records, val_records):
    print(f"\n-- DoshaNet Training (device={DEVICE}) --")
    train_ds = DoshaDataset(train_records, augment=True)
    val_ds   = DoshaDataset(val_records,   augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = DoshaNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")
    patience_cnt  = 0
    best_state    = None

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss, train_correct, n = 0.0, 0, 0
        for imgs, feats, labels in train_dl:
            optimizer.zero_grad()
            logits = model(imgs, feats)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * len(labels)
            train_correct += int((logits.argmax(1) == labels).sum())
            n             += len(labels)
        scheduler.step()
        train_loss /= n
        train_acc   = train_correct / n

        # Validate
        model.eval()
        val_loss, val_correct, nv = 0.0, 0, 0
        with torch.no_grad():
            for imgs, feats, labels in val_dl:
                logits   = model(imgs, feats)
                loss     = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                val_correct += int((logits.argmax(1) == labels).sum())
                nv += len(labels)
        val_loss /= nv
        val_acc   = val_correct / nv

        print(
            f"  Epoch {epoch:02d}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc*100:.1f}%  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.1f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    out = os.path.join(SAVE_DIR, "dosha_model.pt")
    torch.save(model.state_dict(), out)
    print(f"[OK] Best model saved -> {out}  (val_loss={best_val_loss:.4f})")
    return model


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    train_r, val_r, test_r = load_splits()
    print(f"Dataset: train={len(train_r)}  val={len(val_r)}  test={len(test_r)}")

    train_rf_baseline(train_r, val_r)
    train_model(train_r, val_r)

    elapsed = time.time() - t0
    print(f"\n⏱  Total training time: {elapsed:.1f}s")
    print("Run model/evaluate.py for test-set metrics.")
