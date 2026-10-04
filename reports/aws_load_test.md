# InferX AWS Load Test

## Environment

- Platform: AWS EC2
- Instance: c7i-flex.large
- Architecture: x86_64
- Region: eu-central-1
- Deployment: Docker
- API: FastAPI
- Inference runtime: ONNX Runtime
- Model: DistilBERT SST-2
- Quantization: INT8

## Load Test

- Tool: Locust
- Duration: 60 seconds
- Concurrent users: 10
- Spawn rate: 2 users/second
- Endpoint: POST /predict

## Results

| Metric | Result |
|---|---:|
| Total requests | 1,674 |
| Failures | 0 |
| Failure rate | 0% |
| Average latency | 47 ms |
| P50 latency | 29 ms |
| P95 latency | 110 ms |
| P99 latency | 320 ms |
| Maximum observed latency | ~1.1 s |
| Throughput | 28.01 req/s |

## Analysis

The AWS deployment successfully handled 1,674 inference requests during the 60-second load test with zero failures.

Median latency remained at approximately 29 ms. P95 latency was 110 ms and P99 latency was 320 ms, showing that most requests completed quickly while a small tail of requests experienced substantially higher latency.

The observed throughput was approximately 28 requests per second with 10 concurrent users. The tail-latency behavior provides a useful baseline for further optimization of ONNX Runtime threading, CPU scheduling, container resources, request queuing, and application-level concurrency.

## Reproduction

locust -f benchmarks/locustfile.py --headless -u 10 -r 2 -t 60s --host http://<AWS_HOST>:8001 --csv=reports/aws_load_test
