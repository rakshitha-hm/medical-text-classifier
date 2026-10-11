from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
API_URL = "http://localhost:8000"

print(requests.get(f"{API_URL}/health").json())

test = pd.read_csv(ROOT / "data" / "processed" / "test.csv").sample(5, random_state=1)
for _, row in test.iterrows():
    r = requests.post(f"{API_URL}/predict", json={"text": row["text"], "top_k": 3}).json()
    print(f"\nTrue: {row['label']}")
    print(f"Pred: {r['specialty']} ({r['confidence']}) | review: {r['needs_review']} | "
          f"{r['backend']} | {r['latency_ms']} ms")
    print("Top-3:", [(t["label"], t["score"]) for t in r["top_k"]])

bad = requests.post(f"{API_URL}/predict", json={"text": "     too short     "})
print(f"\nInvalid input -> HTTP {bad.status_code}")