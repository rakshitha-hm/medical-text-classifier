import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
REPORT_DIR = ROOT / "reports"
TFIDF_PATH = ROOT / "models" / "tfidf_baseline.joblib"
TRANSFORMERS = {
    "distilbert": ROOT / "models" / "distilbert",
    "biomedbert": ROOT / "models" / "biomedbert",
}

MAX_LENGTH = 512
SEED = 42
N_BOOT = 1000
N_LATENCY = 50


def load_transformer(path, device):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    return tokenizer, model


@torch.no_grad()
def predict_transformer(tokenizer, model, texts, device, batch_size=16):
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(texts[i:i + batch_size], truncation=True, max_length=MAX_LENGTH,
                          padding=True, return_tensors="pt").to(device)
        logits = model(**batch).logits
        preds.extend(logits.argmax(dim=-1).tolist())
    return [model.config.id2label[p] for p in preds]


def latency_ms(predict_one, texts):
    predict_one(texts[0])  # warm-up
    times = []
    for text in texts[:N_LATENCY]:
        start = time.perf_counter()
        predict_one(text)
        times.append((time.perf_counter() - start) * 1000)
    return round(float(np.median(times)), 1)


def bootstrap_ci(y_true, y_pred):
    rng = np.random.default_rng(SEED)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    scores = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    return np.percentile(scores, [2.5, 97.5])


def paired_bootstrap(y_true, pred_a, pred_b):
    rng = np.random.default_rng(SEED)
    y_true, pred_a, pred_b = np.array(y_true), np.array(pred_a), np.array(pred_b)
    diffs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y_true), len(y_true))
        f1_a = f1_score(y_true[idx], pred_a[idx], average="macro", zero_division=0)
        f1_b = f1_score(y_true[idx], pred_b[idx], average="macro", zero_division=0)
        diffs.append(f1_a - f1_b)
    diffs = np.array(diffs)
    return np.percentile(diffs, [2.5, 97.5]), float(np.mean(diffs > 0))


def make_row(name, y_true, y_pred, cpu_ms, gpu_ms):
    low, high = bootstrap_ci(y_true, y_pred)
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "ci_low": low,
        "ci_high": high,
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "cpu_ms": cpu_ms,
        "gpu_ms": gpu_ms,
    }


def main():
    test = pd.read_csv(TEST_PATH)
    texts, y_true = test["text"].tolist(), test["label"].tolist()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Test notes: {len(texts)} | device: {device}")

    rows, all_preds = [], {}

    tfidf = joblib.load(TFIDF_PATH)
    pred = list(tfidf.predict(texts))
    all_preds["tfidf_svm"] = pred
    rows.append(make_row("tfidf_svm", y_true, pred,
                         cpu_ms=latency_ms(lambda t: tfidf.predict([t]), texts), gpu_ms=None))

    for name, path in TRANSFORMERS.items():
        if not path.exists():
            print(f"Skipping {name}: {path} not found")
            continue
        print(f"Evaluating {name} ...")
        tokenizer, model = load_transformer(path, device)
        pred = predict_transformer(tokenizer, model, texts, device)
        all_preds[name] = pred

        gpu_ms = None
        if device == "cuda":
            gpu_ms = latency_ms(lambda t: predict_transformer(tokenizer, model, [t], "cuda"), texts)
        model = model.to("cpu")
        cpu_ms = latency_ms(lambda t: predict_transformer(tokenizer, model, [t], "cpu"), texts)

        rows.append(make_row(name, y_true, pred, cpu_ms, gpu_ms))
        del model
        torch.cuda.empty_cache()

    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    print("\n" + results.round(3).to_string(index=False))

    best = results.iloc[0]["model"]
    print(f"\nBest on test: {best}\n")
    print(classification_report(y_true, all_preds[best], zero_division=0))

    if "biomedbert" in all_preds:
        (low, high), share = paired_bootstrap(y_true, all_preds["biomedbert"], all_preds["tfidf_svm"])
        print(f"BiomedBERT minus TF-IDF macro-F1, 95% CI: [{low:.3f}, {high:.3f}] | "
              f"BiomedBERT better in {share:.0%} of resamples")

    REPORT_DIR.mkdir(exist_ok=True)
    results.to_csv(REPORT_DIR / "day6_test_results.csv", index=False)
    pd.DataFrame({"text": texts, "true": y_true, **all_preds}).to_csv(
        REPORT_DIR / "day6_test_predictions.csv", index=False)


if __name__ == "__main__":
    main()