from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
SEED = 42


def load_split(name):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    return df["text"], df["label"]


def make_tfidf():
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )


def build_models():
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "complement_nb": ComplementNB(alpha=0.3),
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED
        ),
        "linear_svm": LinearSVC(class_weight="balanced", random_state=SEED),
    }


def evaluate(pipe, X, y):
    pred = pipe.predict(X)
    scores = {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
    }
    return scores, pred


def main():
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    rows, fitted, preds = [], {}, {}
    for name, clf in build_models().items():
        pipe = Pipeline([("tfidf", make_tfidf()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        scores, pred = evaluate(pipe, X_val, y_val)
        rows.append({"model": name, **scores})
        fitted[name], preds[name] = pipe, pred

    vocab_size = len(fitted["logistic_regression"].named_steps["tfidf"].vocabulary_)
    print(f"TF-IDF vocabulary size: {vocab_size}\n")

    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    print(results.round(3).to_string(index=False))

    best = results.iloc[0]["model"]
    print(f"\nBest model: {best}\n")
    print(classification_report(y_val, preds[best], zero_division=0))

    REPORT_DIR.mkdir(exist_ok=True)
    results.to_csv(REPORT_DIR / "day3_baselines.csv", index=False)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(fitted[best], MODEL_DIR / "tfidf_baseline.joblib")
    print(f"Saved best pipeline to {MODEL_DIR / 'tfidf_baseline.joblib'}")


if __name__ == "__main__":
    main()