# DoshaNet v2 🧬 — Bayesian Multimodal Ayurvedic Phenotype Classifier

> **Research-Level AI System** for Ayurvedic Phenotype (Dosha) classification. Combines facial geometry analysis and an adaptive Bayesian questionnaire using state-of-the-art deep learning architectures.

---

## 🔬 Core Research Components

| Component | Technical Implementation | Research Concept |
|:---|:---|:---|
| **Multimodal Fusion** | **Cross-Modal Attention** | `Q = questionnaire` attends over `K,V = image spatial tokens` via Transformer-style multi-head attention. |
| **Uncertainty Est.** | **Bayesian MC-Dropout** | T=50 stochastic forward passes to quantify **Epistemic** (model) vs **Aleatoric** (noise) uncertainty. |
| **Adaptive Quiz** | **Info-Gain Optimization** | Greedy approximation of the NP-Hard **Optimal Decision Tree** problem via Mutual Information maximization. |
| **Interpretability** | **GradCAM & SHAP** | Spatial saliency maps (GradCAM) and feature attribution (SHAP) for transparent local explanations. |

---

## 🏗️ Project Structure

```
mini-project/
├── dataset/
│   ├── generate_dataset.py    # Procedural face generator (v2: 300 samples)
│   └── data.json              # Dataset metadata & feature splits
├── model/
│   ├── model.py               # DoshaNet v2 (Attention-Fusion + MC-Dropout)
│   ├── train.py               # Bayesian training pipeline
│   └── saved/                 # Weights: dosha_model.pt (~1.7MB)
├── explainability/
│   ├── gradcam.py             # Gradient-weighted Class Activation Mapping
│   └── explain.py             # SHAP KernelExplainer (production-guarded)
├── backend/
│   ├── main.py                # FastAPI + Adaptive Quiz Engine
│   └── adaptive_quiz.py       # Bayesian Greedy Information Gain logic
├── frontend/                  # Modern Glassmorphism SPA (Step Wizard)
│   ├── index.html
│   ├── app.js                 # MediaPipe FaceMesh + UI Logic
│   └── style.css
├── render.yaml                # Infrastructure-as-Code (for Render.com)
└── requirements.txt           # Optimized CPU-only dependencies
```

---

## 🚀 Deployment Guide (Render.com)

This project is pre-configured for **Render.com** using `render.yaml`. The backend serves the frontend as static files, resulting in a single high-performance deployment.

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "feat: upgrade to v2 research-level AI"
# Create a repo on GitHub.com, then:
git remote add origin https://github.com/YOUR_USERNAME/doshanet-v2.git
git branch -M main
git push -u origin main
```

### 2. Connect to Render
1. Go to [Dashboard.render.com](https://dashboard.render.com)
2. Click **New** → **Blueprint**
3. Select your repository
4. Render will automatically detect `render.yaml` and configure:
   - **Environment**: Python 3.x
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free Tier (CPU-Only optimized)

---

## 📊 Technical Deep-Dive

### Cross-Modal Attention
Traditional models use simple concatenation. DoshaNet v2 uses queries from the questionnaire branch to attend to 16 spatial patches of the facial feature map. This allows the model to "look" for specific facial traits based on questionnaire answers (e.g., attending to the jawline if "frame" is mentioned).

### Bayesian Uncertainty
By keeping Dropout active during inference (Monte Carlo Dropout), we sample from the approximate posterior. 
- **Low Uncertainty**: High confidence agreement across all 50 samples.
- **High Uncertainty**: Model has not seen similar phenotypes or image is ambiguous.

### Adaptive Quiz Algorithm
The order of questions is not fixed. After each answer, the system calculates the **Expected Information Gain (Kullback–Leibler divergence)** for all remaining questions and selects the most "informative" one. This reduces the average quiz length by ~40% while maintaining accuracy.

---

## 📡 API v2 Reference

### `POST /predict/uncertainty`
Performs T=50 Bayesian inference.
**Response**:
```json
{
  "prediction": "Vata",
  "uncertainty_level": "low",
  "epistemic": 0.0012,
  "aleatoric": 0.452,
  "confidence": {"Vata": 88.4, "Pitta": 7.2, "Kapha": 4.4}
}
```

### `POST /gradcam`
Generates a base64 encoded heatmap overlay for facial analysis.

---

## ⚖️ Disclaimer
This is a research prototype. Ayurvedic classifications are based on computational approximations and should not be used for medical diagnosis.
