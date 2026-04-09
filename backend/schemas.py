from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
class FeatureExplanation(BaseModel):
    feature:     str
    description: str
    direction:   str    # "supports" | "opposes"
    value:       float
    shap:        float


class PredictResponse(BaseModel):
    prediction:  str
    confidence:  Dict[str, float]
    explanation: List[FeatureExplanation]


class UncertaintyResponse(BaseModel):
    prediction:       str
    confidence:       Dict[str, float]
    epistemic:        float   # model uncertainty (var across MC passes)
    aleatoric:        float   # data uncertainty (entropy of mean prediction)
    uncertainty_level:str     # "low" | "medium" | "high"
    attn_weights:     List[float]   # 16 spatial attention weights
    explanation:      List[FeatureExplanation]


class GradCAMResponse(BaseModel):
    heatmap_b64:  str   # base64-encoded JPEG overlay
    target_class: str


class QuizQuestion(BaseModel):
    idx:   int
    key:   str
    label: str
    low:   str
    high:  str


class QuizState(BaseModel):
    answered:   Dict[int, float]   # {feat_idx: answer_value}
    posterior:  List[float]        # [P(Vata), P(Pitta), P(Kapha)]
    n_answered: int
    entropy:    float


class QuizStartRequest(BaseModel):
    pre_answered: Optional[Dict[int, float]] = None  # e.g. {9: 0.65} from webcam


class QuizStartResponse(BaseModel):
    question: QuizQuestion
    state:    QuizState


class QuizNextRequest(BaseModel):
    state:        QuizState
    question_idx: int
    answer:       float   # 0.0 – 1.0


class QuizNextResponse(BaseModel):
    done:       bool
    state:      QuizState
    question:   Optional[QuizQuestion] = None   # None if done
    # If done:
    prediction: Optional[str]                   = None
    confidence: Optional[Dict[str, float]]      = None
    explanation:Optional[List[FeatureExplanation]] = None
