import time
from concurrent.futures import ThreadPoolExecutor

from inferx.batching import DynamicBatcher


def benchmark(concurrency, requests=100):
    batcher = DynamicBatcher(
        max_batch_size=8,
        max_wait_ms=10,
    )

    text = "InferX is testing dynamic request batching."

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(batcher.submit, text)
            for _ in range(requests)
        ]

        request_futures = [future.result() for future in futures]

        for future in request_futures:
            future.result(timeout=10)

    elapsed = time.perf_counter() - start
    throughput = requests / elapsed

    return elapsed, throughput


def main():
    print("InferX Dynamic Scheduler Benchmark")
    print("=" * 45)

    for concurrency in [1, 2, 4, 8, 16]:
        elapsed, throughput = benchmark(concurrency)

        print(f"\nConcurrency: {concurrency}")
        print(f"Total time: {elapsed:.3f} s")
        print(f"Throughput: {throughput:.2f} requests/s")


if __name__ == "__main__":
    main()
