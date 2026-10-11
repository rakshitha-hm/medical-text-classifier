import os
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models" / "biomedbert"))
TFIDF_PATH = Path(os.getenv("TFIDF_PATH", ROOT / "models" / "tfidf_baseline.joblib"))
BACKEND = os.getenv("BACKEND", "auto")  # auto | transformer | tfidf

MAX_LENGTH = 512
CONFIDENCE_THRESHOLD = 0.95  # transformer: flag if top probability is below this
MARGIN_THRESHOLD = 0.25      # tfidf: flag if top-2 SVM margin gap is below this


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class SpecialtyClassifier:
    def __init__(self):
        use_transformer = BACKEND == "transformer" or (
            BACKEND == "auto" and torch.cuda.is_available() and MODEL_DIR.exists()
        )
        if use_transformer:
            self.backend = "biomedbert"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            self.model = self.model.to(self.device).eval()
            self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        else:
            self.backend = "tfidf_svm"
            self.device = "cpu"
            self.pipeline = joblib.load(TFIDF_PATH)
            self.labels = list(self.pipeline.classes_)

    @torch.no_grad()
    def _scores(self, texts):
        if self.backend == "biomedbert":
            batch = self.tokenizer(texts, truncation=True, max_length=MAX_LENGTH,
                                   padding=True, return_tensors="pt").to(self.device)
            return torch.softmax(self.model(**batch).logits, dim=-1).cpu().numpy()
        return self.pipeline.decision_function(texts)

    def predict(self, texts, top_k=3):
        results = []
        for row in self._scores(texts):
            order = np.argsort(row)[::-1]
            if self.backend == "biomedbert":
                probs = row
                needs_review = row[order[0]] < CONFIDENCE_THRESHOLD
            else:
                probs = softmax(row)
                needs_review = (row[order[0]] - row[order[1]]) < MARGIN_THRESHOLD
            top = [{"label": self.labels[i], "score": round(float(probs[i]), 4)}
                   for i in order[:top_k]]
            results.append({"top": top, "needs_review": bool(needs_review)})
        return results