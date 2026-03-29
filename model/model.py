import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
DoshaNet v2 — Research-Quality Multimodal Architecture

Innovations vs v1:
  1. Cross-Modal Attention Fusion (Transformer-style)
       Q = questionnaire features attend over image spatial tokens (K, V)
       Enables genuine modality interaction rather than blind concatenation
       Complexity: O(S·d) per head where S=16 spatial tokens
  2. Monte Carlo Dropout (Gal & Ghahramani, NeurIPS 2016)
       Keeps dropout ON at inference → samples from approximate Bayesian posterior
       q*(θ) = Bernoulli(p) ≈ p(θ|X,Y) via variational inference
       Separates epistemic (model) vs aleatoric (data) uncertainty
  3. GradCAM-compatible architecture
       Returns spatial feature maps → gradient-based saliency heatmaps

Architecture:
  ImageBranch  : 4 ConvBlocks → [B, 256, 4, 4]
                  spatial_tokens [B, 16, 256] + global_feat [B, 256]
  QueryBranch  : Dense(10→64→128) with LayerNorm + GELU
  CrossAttn    : MultiHeadAttn(Q=query, K=V=img_tokens) → [B, 128]
  Fusion       : cat(attended[128], proj(global)[128], query[128]) → [B, 384]
  Classifier   : Linear→GELU→MC-Drop→Linear→GELU→MC-Drop→Linear(3)
"""

import torch
import torch.nn as nn

CLASSES    = ["Vata", "Pitta", "Kapha"]
N_FEATURES = 10
IMG_DIM    = 256
QUERY_DIM  = 128
FUSED_DIM  = 128
N_HEADS    = 4
MC_T       = 50   # Monte Carlo samples at inference


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ImageBranch(nn.Module):
    """
    CNN producing spatial feature map for cross-attention + GradCAM.
    Input: [B, 3, 64, 64]
    block1 → [B, 32,  32, 32]
    block2 → [B, 64,  16, 16]
    block3 → [B, 128,  8,  8]
    block4 → [B, 256,  4,  4]  ← GradCAM hook target
    Returns: spatial_tokens [B, 16, 256], global_feat [B, 256]
    """
    def __init__(self):
        super().__init__()
        self.block1 = ConvBlock(3,   32)
        self.block2 = ConvBlock(32,  64)
        self.block3 = ConvBlock(64,  128)
        self.block4 = ConvBlock(128, 256)   # GradCAM target
        self.out_dim = IMG_DIM

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)                           # [B, 256, 4, 4]
        spatial_tokens = x.flatten(2).permute(0,2,1) # [B, 16, 256]
        global_feat    = x.mean(dim=[2,3])            # [B, 256]
        return spatial_tokens, global_feat


class QueryBranch(nn.Module):
    """Encodes 10 questionnaire values → QUERY_DIM. LayerNorm for small batches."""
    def __init__(self, n_features=N_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, QUERY_DIM),
            nn.LayerNorm(QUERY_DIM),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class CrossModalAttention(nn.Module):
    """
    Scaled multi-head cross-attention between modalities.
      Q = questionnaire features (what we want to know from the face)
      K = V = image spatial tokens (spatial evidence)

    Uses PyTorch nn.MultiheadAttention with kdim/vdim ≠ embed_dim.
    Residual connection + LayerNorm for training stability.
    """
    def __init__(self, query_dim=QUERY_DIM, key_dim=IMG_DIM,
                 out_dim=FUSED_DIM, n_heads=N_HEADS, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=n_heads,
            kdim=key_dim,
            vdim=key_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.norm     = nn.LayerNorm(query_dim)
        self.out_proj = nn.Linear(query_dim, out_dim)

    def forward(self, query_feat, img_tokens):
        """
        query_feat : [B, QUERY_DIM]
        img_tokens : [B, S, IMG_DIM]   S=16
        Returns    : attended [B, out_dim],  attn_weights [B, S]
        """
        q = query_feat.unsqueeze(1)                            # [B, 1, QUERY_DIM]
        attended, attn_w = self.attn(
            query=q, key=img_tokens, value=img_tokens,
            need_weights=True, average_attn_weights=True,
        )                                                      # [B, 1, QUERY_DIM], [B, 1, S]
        attended = self.norm(attended + q).squeeze(1)          # [B, QUERY_DIM]  residual
        return self.out_proj(attended), attn_w.squeeze(1)      # [B, FUSED_DIM], [B, S]


class DoshaNet(nn.Module):
    """
    DoshaNet v2 — Bayesian Cross-Attention Multimodal Classifier.

    Fusion:
      img_tokens, global = ImageBranch(image)
      query              = QueryBranch(features)
      attended, attn_w   = CrossModalAttention(query, img_tokens)
      fused = cat(attended[128], proj(global)[128], query[128]) → [384]
      logits = classifier_with_mc_dropout(fused)

    Uncertainty (MC-Dropout):
      T=50 stochastic forward passes → mean=prediction, var=epistemic
      entropy(mean) = aleatoric uncertainty
    """
    def __init__(self, n_classes=3, n_features=N_FEATURES, mc_dropout=0.3):
        super().__init__()
        self.image_branch = ImageBranch()
        self.query_branch = QueryBranch(n_features)
        self.cross_attn   = CrossModalAttention(
            query_dim=QUERY_DIM, key_dim=IMG_DIM,
            out_dim=FUSED_DIM,   n_heads=N_HEADS,
        )
        self.global_proj = nn.Sequential(
            nn.Linear(IMG_DIM, FUSED_DIM),
            nn.LayerNorm(FUSED_DIM),
            nn.GELU(),
        )
        # MC-Dropout classifier: Dropout stays ON during inference
        self.classifier = nn.Sequential(
            nn.Linear(FUSED_DIM * 3, 256),
            nn.GELU(),
            nn.Dropout(mc_dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(mc_dropout),
            nn.Linear(128, n_classes),
        )
        self._mc_dropout = mc_dropout

    # ── forward helpers ──────────────────────────────────────────────────────
    def _fuse(self, image, features):
        img_tokens, global_feat = self.image_branch(image)
        query_feat              = self.query_branch(features)
        attended, attn_w        = self.cross_attn(query_feat, img_tokens)
        global_proj             = self.global_proj(global_feat)
        fused = torch.cat([attended, global_proj, query_feat], dim=1)
        return fused, attn_w

    def forward(self, image, features):
        fused, _ = self._fuse(image, features)
        return self.classifier(fused)

    def forward_with_attn(self, image, features):
        fused, attn_w = self._fuse(image, features)
        return self.classifier(fused), attn_w   # attn_w: [B, 16]

    # ── prediction API ───────────────────────────────────────────────────────
    def predict_proba(self, image, features):
        """Single deterministic pass (eval mode)."""
        self.eval()
        with torch.no_grad():
            return torch.softmax(self.forward(image, features), dim=1)

    def predict_with_uncertainty(self, image, features, T=MC_T):
        """
        MC-Dropout Bayesian inference.
        Sets model to train() so dropout is active, runs T forward passes.

        Returns:
          mean_proba   [B, C]  — point estimate
          epistemic    float   — model uncertainty (↓ with more data)
          aleatoric    float   — data uncertainty (irreducible noise)
          attn_weights [16]    — spatial attention on last pass
        """
        self.train()          # keep dropout ON
        probs_list = []
        attn_last  = None

        with torch.no_grad():
            for _ in range(T):
                logits, attn = self.forward_with_attn(image, features)
                probs_list.append(torch.softmax(logits, dim=1))
                attn_last = attn

        probs = torch.stack(probs_list)      # [T, B, C]
        mean_proba = probs.mean(0)           # [B, C]
        variance   = probs.var(0)            # [B, C]

        epistemic = float(variance.mean(-1).squeeze().item())
        p = mean_proba.clamp(min=1e-8)
        aleatoric = float(-(p * p.log()).sum(-1).squeeze().item())

        self.eval()
        return mean_proba, epistemic, aleatoric, attn_last
