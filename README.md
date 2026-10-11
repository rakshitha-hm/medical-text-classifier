# Medical Text Classifier

![CI](https://github.com/rakshitha-hm/medical-text-classifier/actions/workflows/ci.yml/badge.svg)

Predicts the medical specialty of a clinical transcription note (12 classes) with a fine-tuned BiomedBERT, served by FastAPI with top-3 suggestions, a confidence score and a human-review flag.

![Model comparison](docs/model_comparison.png)

## Results (held-out test set, 250 notes)

| Model | Macro-F1 (95% CI) | CPU ms/note |
| --- | --- | --- |
| TF-IDF + Linear SVM | 0.818 (0.75 to 0.87) | 0.4 |
| DistilBERT, fine-tuned | 0.843 (0.78 to 0.89) | 89 |
| **BiomedBERT, fine-tuned** | **0.874 (0.83 to 0.91)** | 175 (15 on GPU) |

## Quick start

```bash
git clone https://github.com/rakshitha-hm/medical-text-classifier.git
cd medical-text-classifier
docker build -t medical-text-classifier .
docker run -p 8000:8000 medical-text-classifier
```

Open http://localhost:8000/docs. The image uses the TF-IDF model. To serve BiomedBERT, train it with `src/finetune.py`, then run `docker compose up --build`.

## Stack

Hugging Face Transformers, PyTorch, scikit-learn, FastAPI, Docker, pytest, GitHub Actions.

Dataset: [Medical Transcriptions (Kaggle)](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions), not included in the repo. See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for limitations. Not for clinical use.