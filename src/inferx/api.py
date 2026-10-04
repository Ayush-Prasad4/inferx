import onnxruntime as ort
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
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
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    inputs = tokenizer(
        request.text,
        return_tensors="np",
        truncation=True,
    )

    outputs = session.run(
        None,
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        },
    )

    logits = outputs[0][0]

    label_id = int(logits.argmax())
    labels = ["NEGATIVE", "POSITIVE"]

    return {
        "label": labels[label_id],
        "score": float(logits[label_id]),
    }
