import os

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

TRAIN = [
    ("patient with chest pain and coronary artery disease, echocardiogram reviewed", "Cardiovascular / Pulmonary"),
    ("shortness of breath, pulmonary function test and chest x-ray reviewed", "Cardiovascular / Pulmonary"),
    ("cystoscopy performed, bladder and prostate examined, urethra normal", "Urology"),
    ("prostate biopsy and bladder irrigation, urine culture sent", "Urology"),
    ("cataract extraction of the right eye, lens placed in the anterior chamber", "Ophthalmology"),
    ("eyelid surgery, visual acuity checked in both eyes", "Ophthalmology"),
]


@pytest.fixture(scope="session")
def tiny_model_path(tmp_path_factory):
    texts, labels = zip(*TRAIN)
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LinearSVC())])
    pipe.fit(texts, labels)
    path = tmp_path_factory.mktemp("models") / "tiny_tfidf.joblib"
    joblib.dump(pipe, path)
    return path


@pytest.fixture(scope="session")
def classifier_module(tiny_model_path):
    os.environ["BACKEND"] = "tfidf"
    os.environ["TFIDF_PATH"] = str(tiny_model_path)
    import classifier
    return classifier


@pytest.fixture(scope="session")
def client(classifier_module):
    from fastapi.testclient import TestClient
    from app import app
    with TestClient(app) as test_client:
        yield test_client