import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""Evaluate from project root."""
import sys
sys.path.insert(0, os.path.dirname(__file__))
from model.evaluate import load_model, run_inference, plot_confusion_matrix, DATA_JSON, CLASSES
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json

if __name__ == "__main__":
    with open(DATA_JSON) as f:
        data = json.load(f)
    test_records = [r for r in data if r["split"] == "test"]
    print(f"Test samples: {len(test_records)}")
    model = load_model()
    y_true, y_pred, _ = run_inference(model, test_records)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    print(f"\n── Test Metrics ─────────────────────────────")
    print(f"  Accuracy  : {acc*100:.1f}%")
    print(f"  F1 (macro): {f1:.3f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=CLASSES)}")
    plot_confusion_matrix(y_true, y_pred)
    print("\nDone.")
