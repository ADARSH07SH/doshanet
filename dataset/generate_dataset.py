"""
DoshaNet v2 Dataset Generator

Improvements over v1:
  - 300 samples (100 per class, vs 90 total before)
  - Face geometry varies meaningfully by dosha:
      Vata  → narrow oval  (width/height ≈ 0.55-0.65)
      Pitta → medium oval  (width/height ≈ 0.65-0.75)
      Kapha → wide/round   (width/height ≈ 0.78-0.90)
  - Realistic skin gradient shading (not solid fill)
  - Proper facial features with per-dosha variation
  - Stores face_aspect_ratio metadata per sample
  - 15% label noise (simulates weak annotation)
"""

import json
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

random.seed(42)
np.random.seed(42)

LABELS     = ["Vata", "Pitta", "Kapha"]
N_SAMPLES  = 300   # 100 per class
NOISE_RATE = 0.12
IMG_SIZE   = 64

FEATURE_NAMES = [
    "body_frame", "skin_moisture", "skin_temperature",
    "digestion_speed", "energy_level", "sleep_quality",
    "stress_tendency", "appetite", "memory_type", "face_width_ratio",
]

# Feature prototypes grounded in Ayurvedic literature
PROTOTYPES = {
    "Vata":  np.array([0.18, 0.12, 0.28, 0.35, 0.42, 0.32, 0.78, 0.28, 0.22, 0.28]),
    "Pitta": np.array([0.50, 0.50, 0.82, 0.88, 0.82, 0.55, 0.62, 0.88, 0.60, 0.52]),
    "Kapha": np.array([0.82, 0.88, 0.22, 0.18, 0.28, 0.88, 0.22, 0.52, 0.88, 0.80]),
}

# Std devs for each feature (how much natural variation)
FEATURE_STD = np.array([0.08, 0.07, 0.07, 0.08, 0.07, 0.08, 0.07, 0.08, 0.07, 0.06])

# Skin tone base (R, G, B) per dosha
SKIN_TONES = {
    "Vata":  (220, 195, 178),   # pale, slightly cool
    "Pitta": (205, 148, 108),   # warm, reddish-tan
    "Kapha": (162, 128, 98),    # olive, slightly oily-looking
}

# Face aspect ratio ranges: width/height (Vata narrower, Kapha wider)
FACE_RATIOS = {
    "Vata":  (0.52, 0.64),
    "Pitta": (0.64, 0.76),
    "Kapha": (0.76, 0.92),
}


def make_face_image(label: str, idx: int, out_dir: str) -> tuple[str, float]:
    """
    Generates a procedural synthetic face with dosha-specific geometry.
    Returns (filename, face_width_ratio).
    """
    img_size = IMG_SIZE
    bg_color = (18, 18, 24)
    img = Image.new("RGB", (img_size, img_size), bg_color)
    draw = ImageDraw.Draw(img)

    # Skin tone with per-sample variation
    base = SKIN_TONES[label]
    r = int(np.clip(base[0] + np.random.randint(-18, 18), 60, 255))
    g = int(np.clip(base[1] + np.random.randint(-15, 15), 50, 220))
    b = int(np.clip(base[2] + np.random.randint(-15, 15), 40, 210))

    # Face oval geometry (dosha-specific aspect ratio)
    ratio_lo, ratio_hi = FACE_RATIOS[label]
    face_ratio = float(np.random.uniform(ratio_lo, ratio_hi))  # width/height

    # Face bounding box centered in image
    margin = 6
    face_h = img_size - 2 * margin
    face_w = int(face_h * face_ratio)
    x0 = (img_size - face_w) // 2
    y0 = margin
    x1 = x0 + face_w
    y1 = y0 + face_h

    # Draw face oval
    draw.ellipse([x0, y0, x1, y1], fill=(r, g, b))

    # Shade gradient (darker at edges for 3D look)
    for layer in range(5):
        alpha = int(20 + layer * 8)
        shrink = layer * 1
        draw.ellipse(
            [x0 + shrink, y0 + shrink, x1 - shrink, y1 - shrink],
            outline=(
                min(r + layer * 4, 255),
                min(g + layer * 3, 255),
                min(b + layer * 3, 255),
            )
        )

    # Center of face
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2

    # Eyes (dosha-specific size)
    eye_size = {"Vata": 4, "Pitta": 5, "Kapha": 5}[label]
    eye_sep  = int(face_w * 0.28)
    eye_y    = cy - int(face_h * 0.10)
    eye_col  = (int(r * 0.35), int(g * 0.3), int(b * 0.25))
    draw.ellipse([cx - eye_sep - eye_size, eye_y - eye_size,
                  cx - eye_sep + eye_size, eye_y + eye_size], fill=eye_col)
    draw.ellipse([cx + eye_sep - eye_size, eye_y - eye_size,
                  cx + eye_sep + eye_size, eye_y + eye_size], fill=eye_col)

    # Eyebrows
    brow_y = eye_y - eye_size - 2
    brow_col = (max(r - 50, 0), max(g - 50, 0), max(b - 50, 0))
    draw.line([(cx-eye_sep-eye_size-1, brow_y), (cx-eye_sep+eye_size+1, brow_y-1)],
              fill=brow_col, width=1)
    draw.line([(cx+eye_sep-eye_size-1, brow_y-1), (cx+eye_sep+eye_size+1, brow_y)],
              fill=brow_col, width=1)

    # Nose
    nose_y0 = cy
    nose_y1 = cy + int(face_h * 0.12)
    nose_col = (max(r - 25, 0), max(g - 25, 0), max(b - 25, 0))
    draw.line([(cx, nose_y0), (cx, nose_y1)], fill=nose_col, width=1)
    draw.line([(cx - 2, nose_y1), (cx + 2, nose_y1)], fill=nose_col, width=1)

    # Mouth (dosha-specific expression)
    mouth_y = cy + int(face_h * 0.22)
    mouth_w = int(face_w * 0.25)
    mouth_col = (int(r * 0.65), int(g * 0.45), int(b * 0.45))
    if label == "Pitta":    # wider, more defined
        draw.arc([cx-mouth_w, mouth_y-3, cx+mouth_w, mouth_y+5],
                 start=0, end=180, fill=mouth_col, width=2)
    elif label == "Kapha":  # fuller, relaxed
        draw.arc([cx-mouth_w+2, mouth_y-2, cx+mouth_w-2, mouth_y+6],
                 start=0, end=180, fill=mouth_col, width=2)
    else:                   # Vata: thinner
        draw.arc([cx-mouth_w+3, mouth_y, cx+mouth_w-3, mouth_y+4],
                 start=5, end=175, fill=mouth_col, width=1)

    # Facial hair / texture noise for realism
    for _ in range(60):
        px = np.random.randint(x0, x1)
        py = np.random.randint(y0, y1)
        noise = np.random.randint(-8, 8)
        try:
            img.putpixel((px, py), (
                int(np.clip(r + noise, 0, 255)),
                int(np.clip(g + noise, 0, 255)),
                int(np.clip(b + noise, 0, 255)),
            ))
        except IndexError:
            pass

    # Soft blur
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))

    fname = f"{label.lower()}_{idx:04d}.jpg"
    fpath = os.path.join(out_dir, fname)
    img.save(fpath, "JPEG", quality=90)
    return fname, face_ratio


def generate_features(label: str, face_ratio: float) -> list:
    """Generate questionnaire features; face_width_ratio is set from geometry."""
    proto = PROTOTYPES[label].copy()
    proto[9] = face_ratio          # actual geometric width ratio
    noise    = np.random.normal(0, FEATURE_STD)
    feats    = np.clip(proto + noise, 0.0, 1.0)
    return [round(float(f), 4) for f in feats]


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(out_dir, exist_ok=True)

    per_class = N_SAMPLES // len(LABELS)
    records   = []
    idx       = 0

    for label in LABELS:
        for _ in range(per_class):
            true_label = label
            noisy_label = (
                random.choice([l for l in LABELS if l != label])
                if random.random() < NOISE_RATE else true_label
            )
            img_fname, face_ratio = make_face_image(true_label, idx, out_dir)
            features              = generate_features(true_label, face_ratio)

            records.append({
                "id":            idx,
                "image":         f"images/{img_fname}",
                "features":      features,
                "feature_names": FEATURE_NAMES,
                "true_label":    true_label,
                "label":         noisy_label,
                "face_ratio":    round(face_ratio, 4),
            })
            idx += 1

    random.shuffle(records)

    n      = len(records)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    for i, rec in enumerate(records):
        rec["split"] = "train" if i < n_train else "val" if i < n_train + n_val else "test"

    out_path = os.path.join(os.path.dirname(__file__), "data.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    from collections import Counter
    splits = Counter(r["split"] for r in records)
    labels = Counter(r["label"] for r in records)
    noisy  = sum(1 for r in records if r["label"] != r["true_label"])
    print(f"✅ Dataset v2 generated: {out_path}")
    print(f"   Samples : {n}  ({per_class} per class)")
    print(f"   Splits  : {dict(splits)}")
    print(f"   Labels  : {dict(labels)}")
    print(f"   Noise   : {noisy}/{n} ({noisy/n*100:.1f}%)")
    print(f"   Images  : {out_dir}/")


if __name__ == "__main__":
    main()
