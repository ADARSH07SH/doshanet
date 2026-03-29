"""
GradCAM — Gradient-weighted Class Activation Mapping
Selvaraju et al., ICCV 2017

Algorithm:
  1. Register forward hook on target conv layer → save activations A^k [C, H, W]
  2. Register backward hook → save gradients ∂y^c/∂A^k
  3. α^c_k = (1/Z) Σ_ij  ∂y^c/∂A^k_ij   (global average pool of gradients)
  4. L^c_GradCAM = ReLU( Σ_k  α^c_k · A^k )
  5. Normalize → [0, 1], resize to input resolution
  6. Overlay jet colormap onto original image

The result shows WHICH spatial regions of the face most influenced
the predicted dosha class — a visually interpretable saliency map.
"""

import base64
import io
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model.model import DoshaNet, CLASSES


def _jet_colormap(x: np.ndarray) -> np.ndarray:
    """Pure-numpy jet colormap. x ∈ [0,1] → RGB [0,255]."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


class GradCAM:
    """
    GradCAM for DoshaNet image branch.
    Hooks into image_branch.block4 (last conv before global avg pool).
    """

    def __init__(self, model: DoshaNet):
        self.model       = model
        self._acts       = None
        self._grads      = None
        self._fwd_hook   = None
        self._bwd_hook   = None
        self._register_hooks()

    def _register_hooks(self):
        target = self.model.image_branch.block4

        def fwd_hook(module, inp, out):
            self._acts = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._grads = grad_out[0].detach()

        self._fwd_hook = target.register_forward_hook(fwd_hook)
        self._bwd_hook = target.register_full_backward_hook(bwd_hook)

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def compute_cam(self, image_tensor: torch.Tensor,
                    features_tensor: torch.Tensor,
                    target_class: int) -> np.ndarray:
        """
        Returns CAM array [4, 4] normalized to [0, 1].
        target_class: 0=Vata, 1=Pitta, 2=Kapha
        """
        self.model.eval()
        image_tensor   = image_tensor.requires_grad_(False)
        features_tensor = features_tensor.requires_grad_(False)

        # Forward
        logits = self.model(image_tensor, features_tensor)

        # Backward for target class
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # α^c_k = mean of gradients over spatial dims
        grads = self._grads       # [1, C, H, W]
        acts  = self._acts        # [1, C, H, W]
        alpha = grads.mean(dim=[2, 3], keepdim=True)   # [1, C, 1, 1]

        # Weighted combination + ReLU (only positive influence)
        cam = F.relu((alpha * acts).sum(dim=1, keepdim=True))  # [1, 1, H, W]
        cam = cam.squeeze().cpu().numpy()                       # [H, W]

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay_on_image(self, image_bytes: bytes, cam: np.ndarray,
                         alpha: float = 0.45) -> str:
        """
        Overlays jet-colormap CAM onto original image.
        Returns base64-encoded JPEG string.
        """
        img    = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img).astype(np.float32)
        H, W   = img_np.shape[:2]

        # Resize CAM to image resolution
        cam_pil     = PILImage.fromarray((cam * 255).astype(np.uint8))
        cam_resized = np.array(cam_pil.resize((W, H), PILImage.BILINEAR)) / 255.0

        # Apply jet colormap
        heatmap = _jet_colormap(cam_resized).astype(np.float32)

        # Alpha blend
        blended = (img_np * (1 - alpha) + heatmap * alpha).clip(0, 255).astype(np.uint8)

        # Encode to base64
        out = PILImage.fromarray(blended)
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=88)
        return base64.b64encode(buf.getvalue()).decode()


def load_gradcam(model_path: str) -> GradCAM:
    model = DoshaNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return GradCAM(model)
