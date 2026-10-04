import torch

from inferx.huggingface import load_model


def export_model(output_path: str = "benchmarks/distilbert.onnx"):
    tokenizer, model = load_model()
    model.eval()

    text = "InferX is working correctly."
    inputs = tokenizer(text, return_tensors="pt")

    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    return output_path
