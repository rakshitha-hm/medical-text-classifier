from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
pipe = joblib.load(ROOT / "models" / "tfidf_baseline.joblib")
val = pd.read_csv(ROOT / "data" / "processed" / "val.csv")

pred = pipe.predict(val["text"])
labels = pipe.classes_

# 1. Where do the mistakes go?
cm = pd.DataFrame(confusion_matrix(val["label"], pred, labels=labels),
                  index=labels, columns=labels)
errors = [(t, p, cm.loc[t, p]) for t in labels for p in labels
          if t != p and cm.loc[t, p] > 0]
errors_df = pd.DataFrame(errors, columns=["true", "predicted", "count"])
print("Misclassifications (validation):")
print(errors_df.sort_values("count", ascending=False).to_string(index=False))

# 2. Which words drive each class?
vec = pipe.named_steps["tfidf"]
clf = pipe.named_steps["clf"]
words = np.array(vec.get_feature_names_out())
print("\nTop 10 words per class:")
for i, label in enumerate(clf.classes_):
    top = words[np.argsort(clf.coef_[i])[-10:][::-1]]
    print(f"{label}: {', '.join(top)}")