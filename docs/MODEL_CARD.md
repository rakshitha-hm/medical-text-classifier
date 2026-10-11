# Model Card: Medical Specialty Classifier (BiomedBERT)

## Model details

- **Task:** single-label text classification of medical transcription notes into 12 medical specialties.
- **Primary model:** `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`, fine-tuned end to end (12 transformer layers, about 110M parameters).
- **Fallback model:** TF-IDF (unigrams + bigrams) + Linear SVM, used automatically when no GPU is available.
- **Training setup:** AdamW, learning rate 3e-5, batch size 8, warmup, up to 10 epochs with early stopping on validation macro-F1, max input length 512 tokens, fp16, seed 42.
- **Serving:** FastAPI, Docker (CPU image), returns the top-k specialties, a confidence score and a `needs_review` flag.

## Intended use

- Routing or pre-sorting transcription notes by specialty in a **human-in-the-loop** workflow, where a person confirms or corrects the suggestion.
- Portfolio and research demonstration of an end-to-end clinical NLP pipeline.

## Out-of-scope use

- **Not for clinical decision-making, diagnosis or treatment.**
- Not validated on real electronic health records, other languages, other institutions or other note formats.
- Not for fully automated routing without human review or monitoring.

## Data

- **Source:** MTSamples medical transcriptions (Kaggle, `tboyle10/medicaltranscriptions`), 4,999 sample notes in English.
- **Cleaning:** empty notes removed; document-type labels removed (e.g. Discharge Summary, SOAP / Chart / Progress Notes, Letters), because they describe the note format, not a specialty; duplicate notes resolved with an explicit label-priority rule (generic classes such as Surgery, Radiology and General Medicine yield to a specific specialty); 319 notes with two conflicting specific labels dropped; classes with fewer than 50 notes dropped.
- **Final dataset:** 1,665 unique notes, 12 classes, stratified split 1,165 train / 250 validation / 250 test, with no note in more than one split.
- **Classes:** Cardiovascular / Pulmonary, ENT - Otolaryngology, Gastroenterology, General Medicine, Hematology - Oncology, Nephrology, Neurology, Obstetrics / Gynecology, Ophthalmology, Orthopedic, Pain Management, Urology.
- **Not covered:** Surgery, Radiology, Neurosurgery and other specialties removed during cleaning.

## Evaluation (held-out test set, 250 notes, evaluated once)

| Model | Accuracy | Macro-F1 | 95% CI (bootstrap) | CPU ms/note | GPU ms/note |
| --- | --- | --- | --- | --- | --- |
| **BiomedBERT** | **0.852** | **0.874** | 0.827 to 0.913 | 175 | 15 |
| DistilBERT | 0.832 | 0.843 | 0.781 to 0.891 | 89 | 9 |
| TF-IDF + Linear SVM (fallback) | 0.820 | 0.818 | 0.752 to 0.865 | 0.4 | n/a |

- Paired bootstrap, BiomedBERT minus TF-IDF + SVM: 95% CI -0.004 to 0.127; BiomedBERT better in 97% of 1,000 resamples.
- Model selection used validation data only; the test set was used once.

**Per-class F1 (BiomedBERT, test):** strongest on Urology (1.00), Pain Management (1.00), Obstetrics / Gynecology (0.98), Nephrology (0.92) and Ophthalmology (0.91); weakest on **General Medicine (0.63)**, with precision 0.57.

## Review threshold

- `needs_review` is set when the top probability is below **0.95**, chosen on the validation set.
- On validation, this sends about 5% of notes to review and catches about a third of the model's errors; accuracy on auto-accepted notes rises from 0.912 to 0.937.
- For the TF-IDF fallback, review is triggered when the gap between the top two SVM margins is below 0.25.

## Limitations

- **Over-confidence:** most errors are made with confidence above 0.95, so the review flag catches only about a third of them. Probabilities are not calibrated; temperature scaling is the recommended next step.
- **General Medicine is a catch-all class:** it overlaps with many specialties and is the most common source of errors for every model tested. This is a label-definition problem as much as a modelling one.
- **Truncation:** input is cut at 512 tokens, and many notes are longer, so the end of a long note (often the assessment and plan) may be ignored.
- **Small evaluation set:** 250 test notes, with 7 to 12 notes in several classes, so per-class scores are noisy and confidence intervals are wide.
- **Single seed:** results come from one training run; variance across seeds was not measured.
- **Dataset:** MTSamples are publicly shared sample transcriptions and may not reflect real hospital notes in style, length or class balance.

## Ethical considerations

- Clinical notes can contain protected health information. The API never logs note text, only metadata (backend, latency, predicted class). A production deployment would also need authentication, HTTPS and a data-retention policy.
- Mistakes could delay a note reaching the right specialist, so the model should support, not replace, human routing.

## Recommendations

- Keep a human in the loop and monitor the error rate on randomly sampled auto-accepted notes.
- Calibrate probabilities (temperature scaling) before relying on the confidence score.
- Revisit the General Medicine label definition with domain experts.
- Re-evaluate on data from the target institution before any real use.