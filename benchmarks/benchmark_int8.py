import time

import onnxruntime as ort
from inferx.huggingface import load_model


tokenizer, _ = load_model()

text = "InferX is working correctly."
inputs = tokenizer(text, return_tensors="pt")

session = ort.InferenceSession(
    "benchmarks/distilbert_int8.onnx",
    providers=["CPUExecutionProvider"],
)

onnx_inputs = {
    "input_ids": inputs["input_ids"].numpy(),
    "attention_mask": inputs["attention_mask"].numpy(),
}

for _ in range(10):
    session.run(None, onnx_inputs)

latencies = []

for _ in range(100):
    start = time.perf_counter()

    session.run(None, onnx_inputs)

    end = time.perf_counter()

    latencies.append((end - start) * 1000)

latencies.sort()

average_latency = sum(latencies) / len(latencies)
p50 = latencies[49]
p95 = latencies[94]
p99 = latencies[98]

print(f"Runs: {len(latencies)}")
print(f"Average latency: {average_latency:.3f} ms")
print(f"P50 latency: {p50:.3f} ms")
print(f"P95 latency: {p95:.3f} ms")
print(f"P99 latency: {p99:.3f} ms")
print(f"Maximum latency: {latencies[-1]:.3f} ms")
