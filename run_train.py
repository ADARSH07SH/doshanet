import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""Wrapper script to run training from project root."""
import sys
sys.path.insert(0, os.path.dirname(__file__))
from model.train import load_splits, train_rf_baseline, train_model
import time

if __name__ == "__main__":
    t0 = time.time()
    train_r, val_r, test_r = load_splits()
    print(f"Dataset: train={len(train_r)}  val={len(val_r)}  test={len(test_r)}")
    train_rf_baseline(train_r, val_r)
    train_model(train_r, val_r)
    print(f"\n⏱  Total: {time.time()-t0:.1f}s")
    print("Run evaluate.py next.")
