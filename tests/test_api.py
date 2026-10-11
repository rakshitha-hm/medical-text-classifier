import pytest

VALID = "Patient underwent cystoscopy; the bladder and prostate were examined."
RESPONSE_KEYS = {"specialty", "confidence", "needs_review", "top_k", "backend", "latency_ms"}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["backend"] == "tfidf_svm"


def test_classes(client):
    response = client.get("/classes")
    assert response.status_code == 200
    assert len(response.json()["classes"]) == 3


def test_predict_valid(client):
    response = client.post("/predict", json={"text": VALID, "top_k": 2})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == RESPONSE_KEYS
    assert len(body["top_k"]) == 2
    assert body["specialty"] == body["top_k"][0]["label"]
    assert isinstance(body["needs_review"], bool)


@pytest.mark.parametrize("payload", [
    {"text": "     too short     "},
    {"text": ""},
    {},
    {"text": VALID, "top_k": 0},
    {"text": VALID, "top_k": 13},
    {"text": "a" * 50_001},
])
def test_predict_invalid(client, payload):
    assert client.post("/predict", json=payload).status_code == 422