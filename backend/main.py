import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"]       = "1"

"""
DoshaNet v2 FastAPI Backend

Endpoints:
  GET  /health
  POST /predict              → single-pass prediction + SHAP
  POST /predict/uncertainty  → MC-Dropout prediction + epistemic/aleatoric
  POST /gradcam              → GradCAM saliency heatmap (base64 JPEG)
  POST /quiz/start           → first adaptive question
  POST /quiz/next            → next question OR final prediction
  GET  /quiz/questions       → all 10 question definitions
"""

import io, json, sys, traceback, base64, sqlite3, secrets
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from model.model        import DoshaNet, CLASSES
from backend.preprocess import preprocess_image, preprocess_features
from backend.schemas import (
    GradCAMResponse,
    PredictResponse,
    QuizNextRequest,
    QuizNextResponse,
    QuizQuestion,
    QuizStartRequest,
    QuizStartResponse,
    QuizState,
    UncertaintyResponse,
    ProfileSaveRequest,
    ProfileSaveResponse
)
from backend.adaptive_quiz import AdaptiveQuizEngine, QUESTIONS as QUIZ_QUESTIONS
from explainability.explain import SHAPExplainer
from explainability.gradcam import GradCAM

MODEL_PT  = os.path.join(ROOT, "model", "saved", "dosha_model.pt")
DATA_JSON = os.path.join(ROOT, "dataset", "data.json")
DEVICE    = "cpu"

_model:   Optional[DoshaNet]        = None
_shap:    Optional[SHAPExplainer]   = None
_gradcam: Optional[GradCAM]         = None
_quiz:    Optional[AdaptiveQuizEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _shap, _gradcam, _quiz

    print("🔄  Loading DoshaNet v2 model…")
    _model = DoshaNet()
    if os.path.exists(MODEL_PT):
        _model.load_state_dict(torch.load(MODEL_PT, map_location=DEVICE))
    _model.to(DEVICE).eval()
    print("✅  Model loaded.")

    try:
        with open(DATA_JSON) as f:
            data = json.load(f)
        train_recs = [r for r in data if r["split"] == "train"]
        bg_feats   = np.array([r["features"] for r in train_recs])

        # Skip SHAP on Render free tier to save ~150MB RAM
        is_production = os.environ.get("DOSHANET_ENV", "").lower() == "production"
        if not is_production:
            _shap = SHAPExplainer(_model, bg_feats)
            print("✅  SHAP explainer ready.")
        else:
            print("ℹ️   SHAP skipped (production mode) — using rule-based fallback.")
    except Exception as e:
        print(f"⚠️  SHAP unavailable: {e}")


    try:
        _gradcam = GradCAM(_model)
        print("✅  GradCAM ready.")
    except Exception as e:
        print(f"⚠️  GradCAM unavailable: {e}")

    try:
        _quiz = AdaptiveQuizEngine(DATA_JSON)
        print("✅  Adaptive quiz engine ready.")
    except Exception as e:
        print(f"⚠️  Quiz engine unavailable: {e}")

    yield
    if _gradcam:
        _gradcam.remove_hooks()
    print("🛑  Shutting down.")


app = FastAPI(
    title="DoshaNet API v2",
    description="Bayesian Multimodal Ayurvedic Phenotype Classifier",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(ROOT, "frontend")
app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/app/index.html")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded":  _model  is not None,
        "shap_ready":    _shap   is not None,
        "gradcam_ready": _gradcam is not None,
        "quiz_ready":    _quiz   is not None,
        "version": "2.0.0",
    }


# ── Shared helpers ────────────────────────────────────────────────────────────
def _parse_features(features_str: str) -> list:
    feats = json.loads(features_str)
    if len(feats) != 10:
        raise HTTPException(422, "features must have exactly 10 values")
    return feats


def _build_explanation(feat_list: list, pred_idx: int) -> list:
    """Try SHAP; fall back to rule-based top-3 deviation."""
    if _shap:
        try:
            return [FeatureExplanation(**e) for e in _shap.explain(feat_list, pred_idx)]
        except Exception:
            pass

    FEAT_NAMES = [
        "body_frame","skin_moisture","skin_temperature","digestion_speed",
        "energy_level","sleep_quality","stress_tendency","appetite",
        "memory_type","face_width_ratio",
    ]
    DESCS = {
        "body_frame":       ("Thin frame","Heavy frame"),
        "skin_moisture":    ("Dry skin","Oily skin"),
        "skin_temperature": ("Cool skin","Warm skin"),
        "digestion_speed":  ("Irregular digestion","Slow-steady digestion"),
        "energy_level":     ("Low energy","High energy"),
        "sleep_quality":    ("Poor sleep","Deep sleep"),
        "stress_tendency":  ("Relaxed","High stress"),
        "appetite":         ("Variable appetite","Strong appetite"),
        "memory_type":      ("Quick but forgetful","Slow but retentive"),
        "face_width_ratio": ("Narrow face","Wide face"),
    }
    top3 = sorted(range(10), key=lambda i: abs(feat_list[i] - 0.5), reverse=True)[:3]
    return [
        FeatureExplanation(
            feature=FEAT_NAMES[i],
            description=DESCS[FEAT_NAMES[i]][1 if feat_list[i] >= 0.5 else 0],
            direction="supports",
            value=round(feat_list[i], 3),
            shap=0.0,
        )
        for i in top3
    ]


# ── Predict (single pass) ─────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
async def predict(
    image:    UploadFile = File(...),
    features: str        = Form(...),
):
    try:
        feat_list   = _parse_features(features)
        img_bytes   = await image.read()
        img_t       = preprocess_image(img_bytes).to(DEVICE)
        feat_t      = preprocess_features(feat_list).to(DEVICE)

        proba       = _model.predict_proba(img_t, feat_t).cpu().numpy()[0]
        pred_idx    = int(np.argmax(proba))
        confidence  = {c: round(float(p) * 100, 1) for c, p in zip(CLASSES, proba)}
        explanation = _build_explanation(feat_list, pred_idx)

        return PredictResponse(
            prediction=CLASSES[pred_idx],
            confidence=confidence,
            explanation=explanation,
        )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


# ── Predict with MC-Dropout Uncertainty ──────────────────────────────────────
@app.post("/predict/uncertainty", response_model=UncertaintyResponse)
async def predict_uncertainty(
    image:    UploadFile = File(...),
    features: str        = Form(...),
):
    try:
        feat_list = _parse_features(features)
        img_bytes = await image.read()
        img_t     = preprocess_image(img_bytes).to(DEVICE)
        feat_t    = preprocess_features(feat_list).to(DEVICE)

        mean_proba, epistemic, aleatoric, attn_weights = \
            _model.predict_with_uncertainty(img_t, feat_t, T=50)

        proba    = mean_proba.cpu().numpy()[0]
        pred_idx = int(np.argmax(proba))

        # Classify uncertainty level
        if epistemic < 0.005:
            unc_level = "low"
        elif epistemic < 0.015:
            unc_level = "medium"
        else:
            unc_level = "high"

        attn_list = attn_weights.cpu().numpy()[0].tolist() if attn_weights is not None else []
        return UncertaintyResponse(
            prediction=CLASSES[pred_idx],
            confidence={c: round(float(p)*100,1) for c,p in zip(CLASSES, proba)},
            epistemic=round(epistemic, 6),
            aleatoric=round(aleatoric, 4),
            uncertainty_level=unc_level,
            attn_weights=[round(float(w), 4) for w in attn_list],
            explanation=_build_explanation(feat_list, pred_idx),
        )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


# ── GradCAM ───────────────────────────────────────────────────────────────────
@app.post("/gradcam", response_model=GradCAMResponse)
async def gradcam(
    image:        UploadFile = File(...),
    features:     str        = Form(...),
    target_class: int        = Form(default=-1),  # -1 = use predicted class
):
    if not _gradcam:
        raise HTTPException(503, "GradCAM not available")
    try:
        feat_list = _parse_features(features)
        img_bytes = await image.read()
        img_t     = preprocess_image(img_bytes).to(DEVICE)
        feat_t    = preprocess_features(feat_list).to(DEVICE)

        if target_class < 0:
            proba       = _model.predict_proba(img_t, feat_t).cpu().numpy()[0]
            target_class = int(np.argmax(proba))

        cam      = _gradcam.compute_cam(img_t, feat_t, target_class)
        heatmap  = _gradcam.overlay_on_image(img_bytes, cam, alpha=0.45)

        return GradCAMResponse(
            heatmap_b64=heatmap,
            target_class=CLASSES[target_class],
        )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


# ── Adaptive Quiz ─────────────────────────────────────────────────────────────
@app.get("/quiz/questions")
def quiz_questions():
    return QUIZ_QUESTIONS


@app.post("/quiz/next", response_model=QuizNextResponse)
def quiz_next(req: QuizNextRequest):
    if not _quiz:
        raise HTTPException(503, "Quiz engine not available")
    
    try:
        answered   = {int(k): float(v) for k, v in req.state.answered.items()}
        posterior  = list(req.state.posterior)
        q_idx      = int(req.question_idx)
        answer     = float(req.answer)

        # Update state with this answer
        answered[q_idx] = answer
        posterior       = _quiz.bayes_update(posterior, q_idx, answer)
        n_answered      = len(answered)
        entropy         = _quiz.entropy(posterior)

        new_state = QuizState(
            answered=answered,
            posterior=posterior,
            n_answered=n_answered,
            entropy=round(entropy, 4),
        )

        # Check stopping condition (Confidence > 82% or Max Questions reached)
        if _quiz.should_stop(posterior, n_answered):
            pred, conf = _quiz.get_prediction(posterior)
            # Fill unknowns with 0.5 for explanation generation
            feat_list = [answered.get(i, 0.5) for i in range(10)]
            pred_idx  = CLASSES.index(pred)
            explanation = _build_explanation(feat_list, pred_idx)
            return QuizNextResponse(
                done=True, state=new_state,
                prediction=pred, confidence=conf,
                explanation=explanation,
            )

        # Pick next question via Information Gain
        next_q = _quiz.select_next_question(posterior, set(answered.keys()))
        if next_q is None:
            pred, conf = _quiz.get_prediction(posterior)
            feat_list = [answered.get(i, 0.5) for i in range(10)]
            explanation = _build_explanation(feat_list, CLASSES.index(pred))
            return QuizNextResponse(
                done=True, state=new_state, 
                prediction=pred, confidence=conf, 
                explanation=explanation
            )

        return QuizNextResponse(
            done=False,
            state=new_state,
            question=QuizQuestion(**_quiz.get_question(next_q)),
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, detail=str(e))

@app.post("/quiz/start", response_model=QuizStartResponse)
def quiz_start(req: QuizStartRequest):
    if not _quiz:
        raise HTTPException(503, "Quiz engine not available")
    
    pre = {int(k): float(v) for k, v in (req.pre_answered or {}).items()}
    posterior = _quiz.initial_posterior()
    
    # Pre-apply any known answers (e.g. face_ratio from webcam)
    for idx, ans in pre.items():
        posterior = _quiz.bayes_update(posterior, idx, ans)

    # Pick the best question
    next_q = _quiz.select_next_question(posterior, set(pre.keys()))
    if next_q is None:
        raise HTTPException(400, "No questions available to start.")

    entropy = _quiz.entropy(posterior)
    state = QuizState(
        answered=pre,
        posterior=posterior,
        n_answered=len(pre),
        entropy=round(entropy, 4)
    )

    return QuizStartResponse(
        question=QuizQuestion(**_quiz.get_question(next_q)),
        state=state
    )

# ── Profiles ──────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(ROOT, "profiles.db")

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS profiles (id TEXT PRIMARY KEY, payload TEXT)")

init_db()

@app.post("/profile/save", response_model=ProfileSaveResponse)
def profile_save(req: ProfileSaveRequest):
    short_id = secrets.token_hex(3)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO profiles (id, payload) VALUES (?, ?)", 
                         (short_id, json.dumps(req.payload)))
        return ProfileSaveResponse(id=short_id)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/profile/{short_id}")
def profile_get(short_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("SELECT payload FROM profiles WHERE id = ?", (short_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(404, "Profile not found")
        return json.loads(row[0])
