import time

import torch

from inferx.huggingface import load_model


tokenizer, model = load_model()
model.eval()

text = "InferX is working correctly."
inputs = tokenizer(text, return_tensors="pt")

for _ in range(10):
    with torch.inference_mode():
        model(**inputs)

latencies = []

for _ in range(100):
    start = time.perf_counter()

    with torch.inference_mode():
        model(**inputs)

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
