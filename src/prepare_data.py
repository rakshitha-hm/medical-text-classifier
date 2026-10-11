import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "mtsamples.csv"
OUT_DIR = ROOT / "data" / "processed"
SEED = 42
MIN_SAMPLES = 50

GENERIC_CLASSES = {"Surgery", "Radiology", "General Medicine"}


def pick_label(labels):
    if len(labels) == 1:
        return next(iter(labels))
    specific = labels - GENERIC_CLASSES
    if len(specific) == 1:
        return next(iter(specific))
    return None  # ambiguous: two or more specific specialties, or only generic ones

DOC_TYPE_CLASSES = {
    "Consult - History and Phy.",
    "SOAP / Chart / Progress Notes",
    "Discharge Summary",
    "Emergency Room Reports",
    "Office Notes",
    "Letters",
    "IME-QME-Work Comp etc.",
    "Autopsy",
}


def clean_text(text):
    text = re.sub(r"\s*,\s*,", ",", text)   # ", ," -> ","
    text = text.replace(":,", ":")           # "DIAGNOSIS:," -> "DIAGNOSIS:"
    text = re.sub(r"\s+", " ", text)         # many spaces/newlines -> one space
    return text.strip()

def load_and_clean():
    report = {}
    df = pd.read_csv(RAW_PATH, usecols=["transcription", "medical_specialty"])
    df = df.rename(columns={"transcription": "text", "medical_specialty": "label"})
    report["raw"] = len(df)

    df = df.dropna(subset=["text"])
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0]
    report["after_dropping_empty"] = len(df)

    df["label"] = df["label"].str.strip()
    df = df[~df["label"].isin(DOC_TYPE_CLASSES)]
    report["after_removing_doc_types"] = len(df)

    # put specific specialties before "Surgery", so drop_duplicates keeps them
    resolved = df.groupby("text")["label"].apply(set).map(pick_label)
    report["ambiguous_dropped"] = int(resolved.isna().sum())
    resolved = resolved.dropna()
    df = pd.DataFrame({"text": resolved.index, "label": resolved.values})
    report["after_dedup"] = len(df)

    counts = df["label"].value_counts()
    keep = counts[counts >= MIN_SAMPLES].index
    df = df[df["label"].isin(keep)].reset_index(drop=True)
    report["after_min_class_size"] = len(df)
    report["classes"] = df["label"].nunique()

    for step, n in report.items():
        print(f"{step:28s} {n}")
    return df

def split_and_save(df):
    train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=SEED)

    tr, va, te = set(train["text"]), set(val["text"]), set(test["text"])
    assert not (tr & va or tr & te or va & te), "Leakage: same note in two splits"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(OUT_DIR / f"{name}.csv", index=False)

    summary = pd.DataFrame({
        "train": train["label"].value_counts(),
        "val": val["label"].value_counts(),
        "test": test["label"].value_counts(),
    })
    print(summary)
    print("Totals:", len(train), len(val), len(test))


if __name__ == "__main__":
    split_and_save(load_and_clean())