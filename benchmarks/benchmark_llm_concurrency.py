import time
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
    )

    model.eval()
    return tokenizer, model


def run_request(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=32,
        )


def benchmark(model, tokenizer, concurrency, requests=16):
    prompt = "Explain why inference optimization matters for production AI systems."

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(run_request, model, tokenizer, prompt)
            for _ in range(requests)
        ]

        for future in futures:
            future.result()

    elapsed = time.perf_counter() - start
    throughput = requests / elapsed

    return elapsed, throughput


def main():
    print("InferX LLM Concurrency Benchmark")
    print("=" * 45)
    print(f"Model: {MODEL_NAME}")

    tokenizer, model = load_model()

    for concurrency in [1, 2, 4, 8]:
        elapsed, throughput = benchmark(
            model,
            tokenizer,
            concurrency,
        )

        print(f"\nConcurrency: {concurrency}")
        print(f"Total time: {elapsed:.3f} s")
        print(f"Throughput: {throughput:.2f} requests/s")


if __name__ == "__main__":
    main()