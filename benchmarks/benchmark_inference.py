import time

import torch

from inferx.inference import run_inference
from inferx.model import SimpleModel


model = SimpleModel()
model.eval()

input_data = torch.tensor([[1.0, 2.0]])

for _ in range(100):
    run_inference(model, input_data)

latencies = []

start_total = time.perf_counter()

for _ in range(1000):
    start = time.perf_counter()

    run_inference(model, input_data)

    end = time.perf_counter()

    latencies.append((end - start) * 1000)

end_total = time.perf_counter()

latencies.sort()

average_latency = sum(latencies) / len(latencies)
minimum_latency = latencies[0]
maximum_latency = latencies[-1]

p50 = latencies[int(len(latencies) * 0.50)]
p95 = latencies[int(len(latencies) * 0.95)]
p99 = latencies[int(len(latencies) * 0.99)]

total_time = end_total - start_total
throughput = len(latencies) / total_time

print(f"Warm-up runs: 100")
print(f"Measured runs: {len(latencies)}")
print(f"Average latency: {average_latency:.3f} ms")
print(f"Minimum latency: {minimum_latency:.3f} ms")
print(f"Maximum latency: {maximum_latency:.3f} ms")
print(f"P50 latency: {p50:.3f} ms")
print(f"P95 latency: {p95:.3f} ms")
print(f"P99 latency: {p99:.3f} ms")
print(f"Throughput: {throughput:.2f} requests/sec")
