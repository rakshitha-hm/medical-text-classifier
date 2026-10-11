import numpy as np


def test_softmax_sums_to_one(classifier_module):
    probs = classifier_module.softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.allclose(probs[1], 1 / 3)


def test_softmax_handles_large_values(classifier_module):
    probs = classifier_module.softmax(np.array([[1000.0, 1001.0, 999.0]]))
    assert np.isfinite(probs).all()


def test_backend_is_tfidf(classifier_module):
    clf = classifier_module.SpecialtyClassifier()
    assert clf.backend == "tfidf_svm"
    assert clf.device == "cpu"
    assert len(clf.labels) == 3


def test_predict_top_k_sorted(classifier_module):
    clf = classifier_module.SpecialtyClassifier()
    result = clf.predict(["cystoscopy of the bladder and prostate"], top_k=2)[0]
    scores = [item["score"] for item in result["top"]]
    assert len(result["top"]) == 2
    assert scores == sorted(scores, reverse=True)
    assert result["top"][0]["label"] == "Urology"
    assert isinstance(result["needs_review"], bool)