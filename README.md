# DoshaNet 🌿 — Multimodal AI Ayurvedic Dosha Classifier

> **An end-to-end ML system** combining facial image analysis and questionnaire data to classify Ayurvedic doshas (Vata / Pitta / Kapha) using a multimodal deep learning model, with SHAP-based explainability and a FastAPI + HTML/JS web interface.

---

## 🏗️ Project Structure

```
mini-project/
├── dataset/
│   ├── generate_dataset.py    # Synthetic dataset generator
│   ├── data.json              # Dataset manifest (90 samples)
│   └── images/                # Placeholder face images
├── model/
│   ├── model.py               # DoshaNet (MobileNetV2 + Dense fusion)
│   ├── train.py               # Training pipeline
│   ├── evaluate.py            # Evaluation + confusion matrix
│   └── saved/                 # Model weights
├── explainability/
│   └── explain.py             # SHAP KernelExplainer
├── backend/
│   ├── main.py                # FastAPI app
│   ├── preprocess.py          # Image/feature preprocessing
│   └── schemas.py             # Pydantic schemas
├── frontend/
│   ├── index.html             # Web UI
│   ├── style.css              # Dark glassmorphism styles
│   └── app.js                 # JS logic + Chart.js
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Local Setup

### 1. Install Dependencies

```powershell
cd e:\project1\mini-project
pip install -r requirements.txt
```

### 2. Generate Dataset

```powershell
python dataset/generate_dataset.py
# → Creates dataset/data.json and dataset/images/ (90 samples)
```

### 3. Train Model

```powershell
python model/train.py
# → Trains DoshaNet + RF baseline
# → Saves model/saved/dosha_model.pt
```

### 4. Evaluate Model

```powershell
python model/evaluate.py
# → Prints accuracy, F1, classification report
# → Saves model/saved/confusion_matrix.png
```

### 5. Start Backend

```powershell
uvicorn backend.main:app --reload --port 8000
```

### 6. Open Frontend

Open `frontend/index.html` directly in your browser, OR serve it:
```powershell
cd frontend
python -m http.server 3000
# → Visit http://localhost:3000
```

---

## 🌐 Cloud Deployment

### Backend — Render.com (Free Tier)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Settings:
   - **Build Command**: `pip install -r requirements.txt && python dataset/generate_dataset.py && python model/train.py`
   - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Python 3.11
5. Click **Deploy**
6. Copy the public URL (e.g., `https://doshanet.onrender.com`)

### Frontend — Netlify (Free)

1. Update `API_BASE` in `frontend/app.js` to your Render URL
2. Go to [netlify.com](https://netlify.com) → **Sites → Add new site → Deploy manually**
3. Drag and drop the `frontend/` folder
4. Your app is live instantly!

---

## 📊 Model Architecture

```
Image (224×224×3)
      ↓
MobileNetV2 (frozen backbone)
      ↓
1280-d feature vector
      ↓ ←────────────────────────────────── Questionnaire (10 features)
                                                   ↓
                                            Dense(10→64→64)
                                                   ↓
                                            64-d feature vector
Fusion: Concat(1280 + 64 = 1344)
      ↓
Linear(1344→256) → BatchNorm → ReLU → Dropout(0.3)
      ↓
Linear(256→64) → ReLU
      ↓
Linear(64→3) → Softmax
      ↓
[Vata, Pitta, Kapha] probabilities
```

### Noise Handling
- **Label smoothing** (ε=0.1) in CrossEntropyLoss
- **15% synthetic label noise** in dataset (weak supervision simulation)
- **Data augmentation**: horizontal flip, color jitter
- **Cosine LR scheduling** + **early stopping** (patience=8)

---

## 🔍 Explainability

Uses **SHAP KernelExplainer** on the questionnaire branch:
- Explains which features pushed the prediction toward the winning dosha
- Returns top-3 feature contributions with `supports` / `opposes` direction
- Displayed as explanation cards in the UI

---

## 📡 API Reference

### `GET /health`
```json
{"status": "ok", "model_loaded": true}
```

### `POST /predict`
**Form data:**
- `image`: image file (JPG/PNG)
- `features`: JSON string — array of 10 floats `[0.0 – 1.0]`

**Response:**
```json
{
  "prediction": "Pitta",
  "confidence": {"Vata": 10.5, "Pitta": 62.3, "Kapha": 27.2},
  "explanation": [
    {"feature": "skin_temperature", "description": "Warm skin", "direction": "supports", "value": 0.82, "shap": 0.031},
    {"feature": "energy_level",     "description": "High energy",     "direction": "supports", "value": 0.75, "shap": 0.021},
    {"feature": "stress_tendency",  "description": "High stress",     "direction": "supports", "value": 0.68, "shap": 0.018}
  ],
  "model_version": "1.0.0"
}
```

---

## ⚖️ Disclaimer

Dosha predictions are for educational and research purposes only. This is not medical advice.
