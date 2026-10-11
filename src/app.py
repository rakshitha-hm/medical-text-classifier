import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from fastapi.responses import RedirectResponse
from classifier import CONFIDENCE_THRESHOLD, SpecialtyClassifier

from classifier import SpecialtyClassifier

state = {}


@asynccontextmanager
async def lifespan(app):
    state["clf"] = SpecialtyClassifier()
    state["clf"].predict(["Warm-up note for the model. " * 10])  # first-call cost paid at start-up
    yield
    state.clear()


app = FastAPI(title="Medical Text Classifier", version="1.0", lifespan=lifespan)


class NoteIn(BaseModel):
    text: str = Field(..., max_length=50_000, description="Clinical note text")
    top_k: int = Field(3, ge=1, le=12)

    @field_validator("text")
    @classmethod
    def check_text(cls, value):
        value = value.strip()
        if len(value) < 20:
            raise ValueError("text must contain at least 20 non-space characters")
        return value


class LabelScore(BaseModel):
    label: str
    score: float


class PredictionOut(BaseModel):
    specialty: str
    confidence: float
    needs_review: bool
    top_k: list[LabelScore]
    backend: str
    latency_ms: float


@app.get("/health")
def health():
    clf = state["clf"]
    return {"status": "ok", "backend": clf.backend, "device": clf.device,
            "classes": len(clf.labels), "review_threshold": CONFIDENCE_THRESHOLD}


@app.get("/classes")
def classes():
    return {"classes": state["clf"].labels}


@app.post("/predict", response_model=PredictionOut)
def predict(note: NoteIn):
    clf = state["clf"]
    start = time.perf_counter()
    result = clf.predict([note.text], top_k=note.top_k)[0]
    latency = (time.perf_counter() - start) * 1000
    best = result["top"][0]
    return PredictionOut(
        specialty=best["label"],
        confidence=best["score"],
        needs_review=result["needs_review"],
        top_k=result["top"],
        backend=clf.backend,
        latency_ms=round(latency, 1),
    )

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")