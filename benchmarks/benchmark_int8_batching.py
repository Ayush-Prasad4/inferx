import time

import onnxruntime as ort
from inferx.huggingface import load_model


tokenizer, _ = load_model()

session = ort.InferenceSession(
    "benchmarks/distilbert_int8.onnx",
    providers=["CPUExecutionProvider"],
)

text = "InferX is working correctly."
batch_sizes = [1, 8, 32, 64]

for batch_size in batch_sizes:
    texts = [text] * batch_size
    inputs = tokenizer(
        texts,
        return_tensors="np",
        padding=True,
        truncation=True,
    )

    onnx_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }

    for _ in range(10):
        session.run(None, onnx_inputs)

    start = time.perf_counter()

    for _ in range(100):
        session.run(None, onnx_inputs)

    end = time.perf_counter()

    total_time = end - start
    latency_ms = (total_time / 100) * 1000
    throughput = (batch_size * 100) / total_time

    print(
        f"Batch size: {batch_size} | "
        f"Latency: {latency_ms:.3f} ms | "
        f"Throughput: {throughput:.2f} samples/sec"
    )
