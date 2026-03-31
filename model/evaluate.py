import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
Evaluation script for DoshaNet on the test split.
No torchvision dependency — pure PIL + numpy.
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.preprocessing import LabelEncoder

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in [_ROOT, _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model.model import DoshaNet, CLASSES  # noqa

ROOT      = os.path.dirname(os.path.dirname(__file__))
DATA_JSON = os.path.join(ROOT, "dataset", "data.json")
IMG_ROOT  = os.path.join(ROOT, "dataset")
MODEL_PT  = os.path.join(ROOT, "model", "saved", "dosha_model.pt")
SAVE_DIR  = os.path.join(ROOT, "model", "saved")
DEVICE    = "cpu"

IMG_SIZE = 64
IMG_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
IMG_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)

le = LabelEncoder()
le.fit(CLASSES)


def load_image_tensor(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMG_MEAN) / IMG_STD
    return torch.tensor(arr).permute(2, 0, 1)


def load_model():
    model = DoshaNet()
    model.load_state_dict(torch.load(MODEL_PT, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def run_inference(model, records):
    y_true, y_pred, y_proba = [], [], []
    with torch.no_grad():
        for rec in records:
            img  = load_image_tensor(os.path.join(IMG_ROOT, rec["image"])).unsqueeze(0)
            feat = torch.tensor([rec["features"]], dtype=torch.float32)
            prob = model.predict_proba(img, feat).numpy()[0]
            pred_idx = int(np.argmax(prob))
            y_true.append(rec["true_label"])
            y_pred.append(CLASSES[pred_idx])
            y_proba.append(prob)
    return y_true, y_pred, y_proba


def plot_confusion_matrix(y_true, y_pred):
    cm  = confusion_matrix(y_true, y_pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES, ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — DoshaNet (Test Set)", fontsize=13)
    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=120)
    print(f"   Saved → {out}")
    plt.close()


if __name__ == "__main__":
    with open(DATA_JSON) as f:
        data = json.load(f)
    test_records = [r for r in data if r["split"] == "test"]
    print(f"Test samples: {len(test_records)}")

    model = load_model()
    y_true, y_pred, _ = run_inference(model, test_records)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")

    print(f"\n-- Test Metrics --")
    print(f"  Accuracy  : {acc*100:.1f}%")
    print(f"  F1 (macro): {f1:.3f}")
    print(f"\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    plot_confusion_matrix(y_true, y_pred)
    print("\nEvaluation complete.")
