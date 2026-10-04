import time
import torch
from inferx.huggingface import load_model

def benchmark(model, tokenizer, text, runs=100, warmup=10):
    inputs = tokenizer(text, return_tensors="pt")

    for _ in range(warmup):
        with torch.inference_mode():
            model(**inputs)

    latencies = []

    for _ in range(runs):
        start = time.perf_counter()

        with torch.inference_mode():
            model(**inputs)

        latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()

    avg = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]

    return avg, p50, p95, p99


def main():
    tokenizer, model = load_model()
    text = "InferX is testing optimized Transformer inference."

    print("FP32 benchmark")
    fp32 = benchmark(model, tokenizer, text)

    print(f"Average: {fp32[0]:.3f} ms")
    print(f"P50:     {fp32[1]:.3f} ms")
    print(f"P95:     {fp32[2]:.3f} ms")
    print(f"P99:     {fp32[3]:.3f} ms")

    print("\nFP16 benchmark")

    fp16_model = model.half()
    fp16 = benchmark(fp16_model, tokenizer, text)

    print(f"Average: {fp16[0]:.3f} ms")
    print(f"P50:     {fp16[1]:.3f} ms")
    print(f"P95:     {fp16[2]:.3f} ms")
    print(f"P99:     {fp16[3]:.3f} ms")


if __name__ == "__main__":
    main()
