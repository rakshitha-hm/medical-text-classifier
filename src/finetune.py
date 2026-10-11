import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="distilbert-base-uncased")
parser.add_argument("--name", default="distilbert")
ARGS = parser.parse_args()

MODEL_NAME = ARGS.model
OUTPUT_DIR = ROOT / "outputs" / ARGS.name
MODEL_DIR = ROOT / "models" / ARGS.name

MAX_LENGTH = 512
SEED = 42


class NotesDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(list(texts), truncation=True, max_length=MAX_LENGTH)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: values[idx] for key, values in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def main():
    set_seed(SEED)
    print(f"Model: {MODEL_NAME} | run name: {ARGS.name}")

    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")

    labels = sorted(train["label"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = NotesDataset(train["text"], train["label"].map(label2id), tokenizer)
    val_ds = NotesDataset(val["text"], val["label"].map(label2id), tokenizer)

    lengths = [len(ids) for ids in tokenizer(list(train["text"]))["input_ids"]]
    print(f"Median tokens per note: {int(np.median(lengths))}")
    print(f"Notes longer than {MAX_LENGTH} tokens: {np.mean(np.array(lengths) > MAX_LENGTH):.0%}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        warmup_steps=150,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    metrics = trainer.evaluate()
    print(f"\nBest checkpoint on validation: {metrics}")

    pred = trainer.predict(val_ds).predictions.argmax(axis=-1)
    print(classification_report(val_ds.labels, pred, target_names=labels, zero_division=0))

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    REPORT_DIR.mkdir(exist_ok=True)
    with open(REPORT_DIR / f"day5_{ARGS.name}.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()