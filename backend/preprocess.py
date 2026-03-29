import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
Preprocessing helpers — pure PIL + numpy, no torchvision.
"""

import io
import numpy as np
import torch
from PIL import Image

IMG_SIZE = 64
IMG_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
IMG_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def preprocess_image(img_bytes: bytes) -> torch.Tensor:
    """Decode image bytes → normalised tensor of shape (1, 3, 64, 64)."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMG_MEAN) / IMG_STD
    return torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)


def preprocess_features(raw: list) -> torch.Tensor:
    """Clip feature values to [0, 1] and return tensor of shape (1, 10)."""
    arr = np.clip(np.array(raw, dtype=np.float32), 0.0, 1.0)
    return torch.tensor(arr).unsqueeze(0)
