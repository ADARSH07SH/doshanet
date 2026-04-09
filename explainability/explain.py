import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
SHAP-based explainability for the questionnaire branch of DoshaNet.
Returns top-3 feature contributions for each prediction.
"""

import os
import sys

import numpy as np
import shap
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model.model import DoshaNet, CLASSES

FEATURE_NAMES = [
    "body_frame",
    "skin_moisture",
    "skin_temperature",
    "digestion_speed",
    "energy_level",
    "sleep_quality",
    "stress_tendency",
    "appetite",
    "memory_type",
    "face_width_ratio",
]

FEATURE_LABELS = {
    "body_frame":       ("Thin frame", "Heavy frame"),
    "skin_moisture":    ("Dry skin", "Oily skin"),
    "skin_temperature": ("Cool skin", "Warm skin"),
    "digestion_speed":  ("Irregular digestion", "Slow-steady digestion"),
    "energy_level":     ("Low energy", "High energy"),
    "sleep_quality":    ("Poor sleep", "Deep sleep"),
    "stress_tendency":  ("Relaxed", "High stress"),
    "appetite":         ("Variable appetite", "Strong appetite"),
    "memory_type":      ("Quick but forgetful", "Slow but retentive"),
    "face_width_ratio": ("Narrow face", "Wide face"),
}

ROOT     = os.path.dirname(os.path.dirname(__file__))
MODEL_PT = os.path.join(ROOT, "model", "saved", "dosha_model.pt")
DEVICE   = "cpu"


class SHAPExplainer:
    """
    Wraps DoshaNet's query branch + a fixed/dummy image feature vector
    to produce SHAP explanations for questionnaire inputs only.
    """

    def __init__(self, model: DoshaNet, background_features: np.ndarray):
        self.model = model
        self.model.eval()

        # Wrapper: only features vary; image is a neutral zero tensor
        def _predict(feat_array: np.ndarray) -> np.ndarray:
            out = []
            for row in feat_array:
                feat_t    = torch.tensor(row, dtype=torch.float32).unsqueeze(0)
                dummy_img = torch.zeros(1, 3, 64, 64)
                with torch.no_grad():
                    prob = model.predict_proba(dummy_img, feat_t).numpy()[0]
                out.append(prob)
            return np.array(out)

        self._predict_fn = _predict
        self.explainer = shap.KernelExplainer(
            _predict,
            shap.kmeans(background_features, 10),
        )

    def explain(self, features: list, pred_class_idx: int, n_top: int = 3) -> list:
        """
        Returns list of dicts: [{feature, human_label, impact, direction}, ...]
        """
        feat_arr = np.array([features])
        shap_vals = self.explainer.shap_values(feat_arr, nsamples=50, silent=True)
        # shap_vals: list of [1 x n_features] per class
        vals = shap_vals[pred_class_idx][0]
        top_idx = np.argsort(np.abs(vals))[::-1][:n_top]

        result = []
        for i in top_idx:
            fname = FEATURE_NAMES[i]
            low_label, high_label = FEATURE_LABELS[fname]
            val   = float(features[i])
            sv    = float(vals[i])
            direction = "supports" if sv > 0 else "opposes"
            descriptor = high_label if val > 0.5 else low_label
            result.append({
                "feature": fname,
                "value": round(val, 3),
                "shap": round(sv, 4),
                "direction": direction,
                "description": descriptor,
            })
        return result


def load_explainer(background_features: np.ndarray) -> SHAPExplainer:
    model = DoshaNet()
    model.load_state_dict(torch.load(MODEL_PT, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return SHAPExplainer(model, background_features)
