import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from transformers import AutoTokenizer


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_PATH = "benchmarks/distilbert_int8.onnx"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"],
)

app = FastAPI(title="InferX Inference API")

Instrumentator().instrument(app).expose(app)


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class PredictionResponse(BaseModel):
    label: str
    score: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def readiness():
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    inputs = tokenizer(
        request.text,
        return_tensors="np",
        truncation=True,
    )

    try:
        outputs = session.run(
            None,
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Inference failed") from exc

    logits = outputs[0][0]
    probabilities = np.exp(logits - np.max(logits))
    probabilities = probabilities / probabilities.sum()

    label_id = int(probabilities.argmax())
    labels = ["NEGATIVE", "POSITIVE"]

    return {
        "label": labels[label_id],
        "score": float(probabilities[label_id]),
    }
