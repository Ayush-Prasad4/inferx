import time

from inferx.huggingface import load_model, run_text_inference


tokenizer, model = load_model()

text = "InferX is working correctly."

for _ in range(10):
    run_text_inference(tokenizer, model, text)

latencies = []

for _ in range(100):
    start = time.perf_counter()

    run_text_inference(tokenizer, model, text)

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
