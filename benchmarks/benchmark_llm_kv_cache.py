import time

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


def benchmark(model, tokenizer, prompt, use_cache, runs=20, warmup=5):
    inputs = tokenizer(prompt, return_tensors="pt")

    for _ in range(warmup):
        with torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens=32,
                use_cache=use_cache,
            )

    latencies = []

    for _ in range(runs):
        start = time.perf_counter()

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=32,
                use_cache=use_cache,
            )

        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    latencies.sort()

    avg = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    input_tokens = inputs["input_ids"].shape[1]
    output_tokens = output.shape[1] - input_tokens
    tokens_per_second = output_tokens / (avg / 1000)

    return avg, p50, p95, tokens_per_second


def main():
    print("InferX LLM KV Cache Benchmark")
    print("=" * 45)
    print(f"Model: {MODEL_NAME}")

    tokenizer, model = load_model()

    prompt = "Explain why inference optimization matters for production AI systems."

    for use_cache in [False, True]:
        label = "KV Cache ENABLED" if use_cache else "KV Cache DISABLED"

        avg, p50, p95, tokens_per_second = benchmark(
            model,
            tokenizer,
            prompt,
            use_cache,
        )

        print(f"\n{label}")
        print(f"Average latency: {avg:.2f} ms")
        print(f"P50 latency:     {p50:.2f} ms")
        print(f"P95 latency:     {p95:.2f} ms")
        print(f"Generation rate: {tokens_per_second:.2f} tokens/s")


if __name__ == "__main__":
    main()
