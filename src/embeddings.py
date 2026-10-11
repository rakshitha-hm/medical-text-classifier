import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
EMB_DIR = ROOT / "data" / "embeddings"
REPORT_DIR = ROOT / "reports"
SEED = 42
CHUNK_WORDS = 150
TFIDF_SVM_MACRO_F1 = 0.874

MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "pubmedbert": "NeuML/pubmedbert-base-embeddings",
}
MODES = ("truncate", "chunk")


def chunk_words(text, size=CHUNK_WORDS):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def encode(model, texts, mode):
    texts = list(texts)
    if mode == "truncate":
        return model.encode(texts, batch_size=32, normalize_embeddings=True,
                            show_progress_bar=True)

    chunks, owners = [], []
    for i, text in enumerate(texts):
        for chunk in chunk_words(text):
            chunks.append(chunk)
            owners.append(i)

    chunk_emb = model.encode(chunks, batch_size=32, normalize_embeddings=True,
                             show_progress_bar=True)
    owners = np.array(owners)

    note_emb = np.zeros((len(texts), chunk_emb.shape[1]), dtype=np.float32)
    np.add.at(note_emb, owners, chunk_emb)
    note_emb /= np.bincount(owners)[:, None]
    note_emb /= np.linalg.norm(note_emb, axis=1, keepdims=True)
    return note_emb


def get_embeddings(model, model_key, mode, split, texts):
    path = EMB_DIR / f"{model_key}_{mode}_{split}.npy"
    if path.exists():
        return np.load(path)
    emb = encode(model, texts, mode)
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb


def share_over_limit(model, texts):
    lengths = [len(model.tokenizer.tokenize(t)) for t in texts]
    return np.mean(np.array(lengths) > model.max_seq_length), int(np.median(lengths))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")

    rows = []
    for key, name in MODELS.items():
        model = SentenceTransformer(name, device=device)
        over, median_tokens = share_over_limit(model, train["text"])
        print(f"\n{key}: dim={model.get_sentence_embedding_dimension()}, "
              f"max_seq_length={model.max_seq_length}, median tokens={median_tokens}, "
              f"notes over limit={over:.0%}")

        for mode in MODES:
            start = time.perf_counter()
            X_train = get_embeddings(model, key, mode, "train", train["text"])
            X_val = get_embeddings(model, key, mode, "val", val["text"])
            encode_sec = time.perf_counter() - start

            clf = LogisticRegression(max_iter=3000, C=10, class_weight="balanced",
                                     random_state=SEED)
            clf.fit(X_train, train["label"])
            pred = clf.predict(X_val)

            rows.append({
                "model": key,
                "mode": mode,
                "dim": X_train.shape[1],
                "accuracy": accuracy_score(val["label"], pred),
                "macro_f1": f1_score(val["label"], pred, average="macro", zero_division=0),
                "encode_sec": round(encode_sec, 1),
            })

    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    print("\n" + results.round(3).to_string(index=False))
    print(f"\nTF-IDF + SVM baseline macro-F1: {TFIDF_SVM_MACRO_F1}")

    REPORT_DIR.mkdir(exist_ok=True)
    results.to_csv(REPORT_DIR / "day4_embeddings.csv", index=False)


if __name__ == "__main__":
    main()