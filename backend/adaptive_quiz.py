"""
Adaptive Questionnaire Engine — DoshaNet v2

Algorithm: Greedy Approximation of NP-Hard Optimal Decision Tree

Background:
  The problem of finding the optimal question ordering to classify with
  minimum expected questions = Optimal Identification Code problem.
  Proven NP-complete by Hyafil & Rivest (1976).

  Our greedy approach (analogous to ID3 algorithm):
    At each step, pick the question q* that minimizes expected
    posterior entropy after observing the answer:

      q* = argmin_q  E_a[ H( P(dosha | answers ∪ {q=a}) ) ]
         = argmax_q  Information Gain(q | current_posterior)

  This is equivalent to maximizing mutual information I(Q; Dosha | seen_answers).
  Greedy submodular maximization achieves (1-1/e) ≈ 63% of optimal.
  Complexity: O(Q² · A) where Q=num questions, A=answer hypotheticals.

Usage:
  engine = AdaptiveQuizEngine(data_json_path)
  # State: dict of {feature_idx: answer_value}, posterior [P(V), P(Pi), P(K)]
  q_idx  = engine.first_question()
  # After user answers:
  new_posterior = engine.bayes_update(posterior, q_idx, answer)
  next_q_idx    = engine.select_next_question(new_posterior, answered_set)
  if engine.should_stop(new_posterior, len(answered_set)):
      label, confidence = engine.get_prediction(new_posterior)
"""

import json
import math
import os

import numpy as np

CLASSES        = ["Vata", "Pitta", "Kapha"]
STOP_THRESHOLD = 0.82   # stop if max posterior > 82%
MAX_QUESTIONS  = 7      # never ask more than 7 of 10

QUESTIONS = [
    {"idx": 0, "key": "body_frame",       "label": "Body Frame",
     "low": "Very thin — lightweight build",          "high": "Heavy — large, solid build"},
    {"idx": 1, "key": "skin_moisture",    "label": "Skin Moisture",
     "low": "Very dry — cracks or flakes",            "high": "Very oily — tends to shine"},
    {"idx": 2, "key": "skin_temperature", "label": "Skin Temperature",
     "low": "Cool or cold to touch",                  "high": "Warm or hot to touch"},
    {"idx": 3, "key": "digestion_speed",  "label": "Digestion",
     "low": "Irregular & unpredictable",              "high": "Slow but consistent & steady"},
    {"idx": 4, "key": "energy_level",     "label": "Energy Pattern",
     "low": "Bursts of energy, tires quickly",        "high": "Sustained, steady high energy"},
    {"idx": 5, "key": "sleep_quality",    "label": "Sleep Quality",
     "low": "Light — frequently interrupted",         "high": "Deep, long, heavy sleep"},
    {"idx": 6, "key": "stress_tendency",  "label": "Stress Response",
     "low": "Rarely stressed — naturally calm",       "high": "Highly stressed or anxious"},
    {"idx": 7, "key": "appetite",         "label": "Appetite",
     "low": "Variable — often skip meals",            "high": "Strong — rarely fully satisfied"},
    {"idx": 8, "key": "memory_type",      "label": "Memory Style",
     "low": "Quick to learn, quick to forget",        "high": "Slow to absorb, never forgets"},
    {"idx": 9, "key": "face_width_ratio", "label": "Face Shape",
     "low": "Narrow or oval face",                    "high": "Wide or round face"},
]


class AdaptiveQuizEngine:
    """
    Stateless Bayesian adaptive questionnaire engine.
    All state is passed in / returned per call (RESTful design).
    """

    def __init__(self, data_json_path: str):
        self._load_statistics(data_json_path)

    def _load_statistics(self, path: str):
        """Compute per-class per-feature μ and σ from training data."""
        with open(path) as f:
            data = json.load(f)
        train = [r for r in data if r["split"] == "train"]

        self.mu    = {}
        self.sigma = {}
        for c in CLASSES:
            feats = np.array([r["features"] for r in train if r["label"] == c])
            self.mu[c]    = feats.mean(axis=0)
            self.sigma[c] = feats.std(axis=0) + 0.04   # smoothing prior

    # ── Core Bayesian logic ───────────────────────────────────────────────────

    def _likelihood(self, answer: float, class_name: str, feat_idx: int) -> float:
        """Gaussian likelihood: P(answer | feature, class)."""
        mu  = self.mu[class_name][feat_idx]
        sig = self.sigma[class_name][feat_idx]
        return math.exp(-0.5 * ((answer - mu) / sig) ** 2) / (sig * math.sqrt(2 * math.pi) + 1e-9)

    def bayes_update(self, posterior: list, feat_idx: int, answer: float) -> list:
        """
        Bayesian update: P(c | new_answer) ∝ P(new_answer | c) · P(c)
        Returns normalized new posterior.
        """
        likelihoods = np.array([self._likelihood(answer, c, feat_idx) for c in CLASSES])
        updated     = np.array(posterior) * likelihoods
        total       = updated.sum()
        if total < 1e-12:
            updated = np.ones(3) / 3
        else:
            updated /= total
        return [round(float(p), 6) for p in updated]

    def entropy(self, posterior: list) -> float:
        """Shannon entropy H(P) in bits."""
        p = np.array(posterior)
        p = p[p > 1e-9]
        return float(-np.sum(p * np.log2(p + 1e-12)))

    def expected_entropy_after(self, posterior: list, feat_idx: int,
                                n_hypotheticals: int = 7) -> float:
        """
        E_a[H(P(dosha | a))] averaged over hypothetical answers ∈ [0,1].
        Lower = more informative question.
        """
        answers = np.linspace(0.0, 1.0, n_hypotheticals)
        total_h = 0.0
        for a in answers:
            updated = self.bayes_update(posterior, feat_idx, float(a))
            total_h += self.entropy(updated)
        return total_h / n_hypotheticals

    def select_next_question(self, posterior: list, answered: set) -> int | None:
        """
        Greedy information-gain selection (approximates NP-hard opt. decision tree).
        Returns feat_idx of the most informative unanswered question.
        """
        unanswered = [i for i in range(len(QUESTIONS)) if i not in answered]
        if not unanswered:
            return None
        best_q = min(unanswered,
                     key=lambda q: self.expected_entropy_after(posterior, q))
        return best_q

    def first_question(self, pre_answered: dict | None = None) -> int:
        """
        Start: pick first question greedily from uniform prior.
        If some features are pre-answered (e.g. from webcam), skip those.
        """
        answered = set(pre_answered.keys()) if pre_answered else set()
        posterior = [1/3, 1/3, 1/3]

        if pre_answered:
            for idx, ans in pre_answered.items():
                posterior = self.bayes_update(posterior, idx, ans)

        return self.select_next_question(posterior, answered)

    def should_stop(self, posterior: list, n_answered: int) -> bool:
        return max(posterior) >= STOP_THRESHOLD or n_answered >= MAX_QUESTIONS

    def get_prediction(self, posterior: list) -> tuple[str, dict]:
        idx   = int(np.argmax(posterior))
        label = CLASSES[idx]
        conf  = {CLASSES[i]: round(posterior[i] * 100, 1) for i in range(3)}
        return label, conf

    def get_question(self, idx: int) -> dict:
        return QUESTIONS[idx]

    def initial_posterior(self) -> list:
        return [round(1/3, 6)] * 3
