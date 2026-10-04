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
| Total requests | 616 |
| Failures | 0 |
| Failure rate | 0% |
| Average latency | 48 ms |
| Median latency | 27 ms |
| Minimum latency | 20 ms |
| Maximum latency | 2043 ms |
| Peak observed throughput | 28.6 req/s |

## Analysis

The AWS deployment successfully handled 616 inference requests with zero failures during the 60-second load test.

Median latency remained around 27 ms, while average latency was 48 ms. A small number of high-latency requests increased the maximum latency to approximately 2.04 seconds, indicating tail-latency behavior under concurrent load.

Further optimization can investigate ONNX Runtime threading, CPU scheduling, container resource limits, request queuing, and application-level concurrency.

## Reproduction

locust -f benchmarks/locustfile.py --headless -u 10 -r 2 -t 60s --host http://<AWS_HOST>:8001
