import time


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def benchmark():
    print("InferX vLLM Benchmark")
    print("=" * 40)
    print(f"Model: {MODEL_NAME}")
    print("Benchmark setup ready.")


if __name__ == "__main__":
    benchmark()
