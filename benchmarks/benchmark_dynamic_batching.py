import time

import onnxruntime as ort
from transformers import AutoTokenizer


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_PATH = "benchmarks/distilbert_int8.onnx"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )

    return tokenizer, session


def run_batch(tokenizer, session, batch_size):
    texts = [
        "InferX is testing dynamic batching."
        for _ in range(batch_size)
    ]

    inputs = tokenizer(
        texts,
        return_tensors="np",
        padding=True,
        truncation=True,
    )

    session.run(
        None,
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        },
    )


def benchmark(batch_size, runs=50, warmup=10):
    tokenizer, session = load_model()

    for _ in range(warmup):
        run_batch(tokenizer, session, batch_size)

    latencies = []

    for _ in range(runs):
        start = time.perf_counter()

        run_batch(tokenizer, session, batch_size)

        latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()

    avg = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    throughput = (batch_size * 1000) / avg

    return avg, p50, p95, throughput


def main():
    print("InferX Dynamic Batching Benchmark")
    print("=" * 45)

    for batch_size in [1, 2, 4, 8, 16, 32]:
        avg, p50, p95, throughput = benchmark(batch_size)

        print(f"\nBatch size: {batch_size}")
        print(f"Average latency: {avg:.3f} ms")
        print(f"P50 latency:     {p50:.3f} ms")
        print(f"P95 latency:     {p95:.3f} ms")
        print(f"Throughput:      {throughput:.2f} samples/s")


if __name__ == "__main__":
    main()
