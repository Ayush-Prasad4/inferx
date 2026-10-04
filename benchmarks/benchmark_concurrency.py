import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import onnxruntime as ort
from transformers import AutoTokenizer


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_PATH = "benchmarks/distilbert_int8.onnx"


def load_inference():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )

    return tokenizer, session


def run_request(tokenizer, session, text):
    inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
    )

    session.run(
        None,
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        },
    )


def benchmark(concurrency, requests=100):
    tokenizer, session = load_inference()

    text = "InferX is testing concurrent Transformer inference."

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(run_request, tokenizer, session, text)
            for _ in range(requests)
        ]

        for future in as_completed(futures):
            future.result()

    elapsed = time.perf_counter() - start

    throughput = requests / elapsed

    return elapsed, throughput


def main():
    print("InferX Concurrency Benchmark")
    print("=" * 40)

    for concurrency in [1, 2, 4, 8]:
        elapsed, throughput = benchmark(concurrency)

        print(f"\nConcurrency: {concurrency}")
        print(f"Total time: {elapsed:.3f} s")
        print(f"Throughput: {throughput:.2f} requests/s")


if __name__ == "__main__":
    main()
