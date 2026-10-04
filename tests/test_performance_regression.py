import time

import torch

from inferx.huggingface import load_model


MAX_AVERAGE_LATENCY_MS = 30.0


def test_transformer_inference_latency():
    tokenizer, model = load_model()
    model.eval()

    inputs = tokenizer(
        "InferX performance regression test.",
        return_tensors="pt",
    )

    for _ in range(3):
        with torch.inference_mode():
            model(**inputs)

    latencies = []

    for _ in range(10):
        start = time.perf_counter()

        with torch.inference_mode():
            model(**inputs)

        latencies.append((time.perf_counter() - start) * 1000)

    average_latency = sum(latencies) / len(latencies)

    assert average_latency < MAX_AVERAGE_LATENCY_MS, (
        f"Average inference latency {average_latency:.2f} ms "
        f"exceeded limit of {MAX_AVERAGE_LATENCY_MS:.2f} ms"
    )
