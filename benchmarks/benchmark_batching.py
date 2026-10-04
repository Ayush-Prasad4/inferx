import time

import torch

from inferx.inference import run_inference
from inferx.model import SimpleModel


model = SimpleModel()
model.eval()

batch_sizes = [1, 8, 32, 128]

for batch_size in batch_sizes:
    input_data = torch.randn(batch_size, 2)

    for _ in range(100):
        run_inference(model, input_data)

    start = time.perf_counter()

    for _ in range(1000):
        run_inference(model, input_data)

    end = time.perf_counter()

    total_time = end - start
    latency_ms = (total_time / 1000) * 1000
    throughput = (batch_size * 1000) / total_time

    print(
        f"Batch size: {batch_size} | "
        f"Latency: {latency_ms:.3f} ms | "
        f"Throughput: {throughput:.2f} samples/sec"
    )
