# DoshaNet: Multimodal Ayurvedic Phenotype Classifier 🧬

![DoshaNet Banner](https://img.shields.io/badge/Status-Active-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)

An end-to-end multimodal machine learning system designed to learn and classify complex human phenotypes based on noisy, subjective clinical labels. DoshaNet merges **facial image feature extraction** with **structured clinical questionnaire data** to predict a person's Ayurvedic phenotype (*Dosha*).

---

## 📖 The Science: What are the Doshas?

In Ayurvedic science, a person's phenotype—their physiological and psychological constitution—is categorized into three primary "Doshas": **Vata**, **Pitta**, and **Kapha**. 

This system acts as an ancient classification of human biological variance. Our model learns to map morphological and behavioral traits to these classes:

| Phenotype (Dosha) | Key Element | Morphological Traits (Facial) | Physiological Traits | Psychological Traits |
| :--- | :--- | :--- | :--- | :--- |
| **Vata** | Air/Space | Narrow oval face, thin frame, dry skin, cooler skin | Fast but irregular digestion, variable appetite | Quick memory but forgetful, prone to anxiety |
| **Pitta** | Fire/Water | Medium proportional face, warm/reddish skin tone | Strong digestion, high body heat, strong appetite | Fast retentive memory, prone to stress/anger |
| **Kapha** | Earth/Water | Wide/round face, heavier frame, oily/smooth skin | Slow/steady digestion, low body heat | Slow but deep memory, highly relaxed/calm |

The inherent difficulty of this project is that in real life, doctors make **subjective** assessments of these phenotypes. Thus, DoshaNet acts as a study in learning robust representations under **noisy supervision** (~15% label noise).

---

## 📊 Dataset: Large-Scale Stochastic Simulation

Real-world medical datasets of this specific niche are guarded by heavy privacy constraints. To prove ML capabilities, we engineered a custom dataset generator (`dataset/generate_dataset.py`) simulating a robust **10,000 patient clinical trial**.

### Dataset Specifications:
- **Total Records:** 10,000 samples.
- **Image Architecture:** 300 algorithmically generated base facial geometries. We vary properties like `face_width_ratio` (Vata: 0.52-0.64 | Kapha: 0.76-0.92) and RGB skin tones.
- **Covariant Feature Noise:** Clinical features (Sleep, Digestion, Moisture) are not randomly assigned. They are generated using **Statistical Covariance Matrices** anchored to the underlying facial geometry.
- **Label Noise Constraint:** 15% of the data has intentionally corrupted target labels to simulate human subjectivity and forcing the model to prevent overfitting.

---

## 🧠 Multimodal Architecture

DoshaNet uses a sophisticated dual-branch neural network designed in PyTorch without heavy `torchvision` dependencies (making it perfectly lightweight for edge deployment).

### 1. Vision Branch (Custom CNN)
- A highly optimized, custom 4-block Convolutional Neural Network (CNN).
- Transcribes $64 \times 64$ images into 16 spatial tokens and a global average pool.
- Includes hook registries for **Grad-CAM Saliency Maps**.

### 2. Clinical Branch (Query MLP)
- Formats 10 continuous questionnaire variables.
- Uses LayerNorm + GELU dense layers to map into a shared embedding space.

### 3. Fusion & Inference
- **Cross-Modal Attention**: The model treats the questionnaire data as a "Query" (what we want to know) and the facial spatial tokens as "Keys/Values", allowing the network to explicitly look at regions of the face that confirm the questionnaire inputs.
- **Monte Carlo (MC) Dropout**: Dropout stays active during inference (`/predict/uncertainty`). By running 50 stochastic forward passes, the model calculates both **Epistemic** (model) and **Aleatoric** (data) uncertainty.

---

## 🔍 Explainable AI (XAI) Report Card

Because DoshaNet processes medical/biometric approximations, it ships with an enterprise-grade Explainable AI suite to prevent "black box" syndrome.

1. **SHAP (SHapley Additive exPlanations)**: KernelExplainer measures exactly how much each subjective questionnaire answer (e.g. "Thin frame") shifted the probability towards a specific Dosha.
2. **Grad-CAM**: Gradient-weighted Class Activation Mapping computes how strongly the gradients of the target class flow into the final CNN layer, overlaying a visual heatmap on the face to show the user exactly where the model looked.

---

## 🚀 How to Run (Local Development)

### 1. Start the System
On Windows simply run the batch script which will manage the API server:
```cmd
start.bat
```

### 2. Train the Model
If you want to re-run the 10,000 sample dataset generation and train the PyTorch model from scratch:
```cmd
python dataset/generate_dataset.py
python run_train.py
python run_evaluate.py
```

### 3. View the Frontend
The frontend requires no build steps (Vanilla JS + Chart.js + CSS).
Open `/frontend/index.html` in your browser. It automatically routes requests to the FastAPI backend running on port 8000.

---

## ☁️ Cloud Deployment Configuration

This repository is optimized for deployment on PAAS providers like Render. 
The `Dockerfile` handles the installation of `uvicorn`, `fastapi`, `torch` (CPU optimized version), and automatically binds to the cloud `$PORT` variable.

```dockerfile
# Start command inside Docker:
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WEB_CONCURRENCY:-1}
```
