import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    model.eval()

    return tokenizer, model


def run_text_inference(tokenizer, model, text: str):
    inputs = tokenizer(text, return_tensors="pt")

    with torch.inference_mode():
        outputs = model(**inputs)

    return outputs
