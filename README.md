# InferX — AI Inference & Model Optimization Platform

> Production-oriented AI inference platform for optimizing, serving, benchmarking, and monitoring Transformer and LLM workloads.

InferX is an inference engineering project focused on understanding and improving the performance of AI models under realistic serving workloads.

Instead of building another AI chatbot, InferX explores the engineering problems behind fast, efficient, observable, and scalable model inference.

---

## What InferX Does

InferX investigates the complete inference optimization lifecycle:

- PyTorch model inference
- Hugging Face Transformers
- ONNX model export
- ONNX Runtime inference
- INT8 quantization
- FP16 / BF16 optimization
- Batch-size optimization
- Dynamic batching
- Concurrent inference
- LLM inference optimization
- KV-cache analysis
- FastAPI model serving
- Prometheus metrics
- Grafana observability
- Locust load testing
- Docker containerization
- AWS deployment
- GitHub Actions CI/CD
- Amazon ECR image publishing

---

## Architecture

```text
                         ┌─────────────────┐
                         │     Client      │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │    FastAPI      │
                         │ Inference API   │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  ONNX Runtime   │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ INT8 DistilBERT │
                         │   SST-2 Model   │
                         └────────┬────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
             ┌─────────────┐             ┌─────────────┐
             │ Prometheus  │             │   Locust    │
             │  Metrics    │             │ Load Tests  │
             └──────┬──────┘             └─────────────┘
                    │
                    ▼
             ┌─────────────┐
             │   Grafana   │
             │ Dashboards  │
             └─────────────┘
Optimization Pipeline
PyTorch
   │
   ▼
Hugging Face Transformer
   │
   ▼
ONNX Export
   │
   ▼
ONNX Runtime FP32
   │
   ▼
INT8 Quantization
   │
   ▼
Batching / Concurrency
   │
   ▼
FastAPI Serving
   │
   ▼
Docker
   │
   ▼
AWS
Project Structure
inferx/
│
├── src/
│   └── inferx/
│       ├── __init__.py
│       ├── model.py
│       ├── inference.py
│       ├── huggingface.py
│       ├── onnx.py
│       ├── quantization.py
│       ├── batching.py
│       └── api.py
│
├── benchmarks/
│   ├── benchmark_inference.py
│   ├── benchmark_batching.py
│   ├── benchmark_fp16.py
│   ├── benchmark_bf16.py
│   ├── benchmark_concurrency.py
│   ├── benchmark_dynamic_batching.py
│   ├── benchmark_dynamic_scheduler.py
│   ├── benchmark_llm_baseline.py
│   ├── benchmark_llm_fp16.py
│   ├── benchmark_llm_concurrency.py
│   ├── benchmark_llm_kv_cache.py
│   ├── benchmark_llm_memory.py
│   ├── benchmark_int8.py
│   ├── benchmark_int8_batching.py
│   ├── benchmark_onnx.py
│   ├── benchmark_transformer.py
│   ├── benchmark_transformer_fair.py
│   ├── benchmark_compile.py
│   ├── benchmark_vllm.py
│   └── locustfile.py
│
├── tests/
├── monitoring/
│   └── prometheus.yml
├── reports/
│   ├── benchmark_summary.md
│   └── aws_load_test.md
├── configs/
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── pyproject.toml
└── README.md
Benchmark Results

Benchmarks were measured on the available development and AWS environments. Results are workload- and hardware-specific and should not be interpreted as universal performance guarantees.

Transformer Inference

The main optimization pipeline was evaluated using a DistilBERT sentiment-classification model.

Runtime	Avg Latency
PyTorch Eager	14.962 ms
ONNX Runtime FP32	4.347 ms
ONNX Runtime INT8	1.531 ms

ONNX Runtime FP32 reduced average latency by approximately 71% compared with the measured PyTorch eager benchmark.

INT8 reduced average latency by a further approximately 65% compared with ONNX FP32 in the tested workload.

INT8 Batch Scaling
Batch Size	Latency	Throughput
1	1.804 ms	554 req/s
8	7.693 ms	1,040 req/s
32	26.199 ms	1,221 req/s
64	52.896 ms	1,209 req/s

Batch size 32 produced the highest observed throughput in this benchmark. Increasing the batch size to 64 increased latency without improving throughput.

Dynamic Batching

InferX includes an event-driven dynamic batching scheduler that collects incoming requests for a bounded time window before executing inference.

Concurrency	Throughput
1	762.87 req/s
2	853.06 req/s
4	807.88 req/s
8	665.31 req/s
16	892.99 req/s

The scheduler improved observed throughput compared with the earlier implementation while maintaining asynchronous request handling through futures and a worker thread.

LLM Optimization

InferX also includes experiments using:

Qwen/Qwen2.5-0.5B-Instruct

FP32 vs FP16
Configuration	Avg Latency	Generation Throughput
FP32	1257.91 ms	25.44 tok/s
FP16	843.13 ms	37.95 tok/s

In the tested environment, FP16 reduced average latency by approximately 33% and increased generation throughput by approximately 49%.

KV Cache
Configuration	Avg Latency	Throughput
KV Cache Disabled	3901.52 ms	8.20 tok/s
KV Cache Enabled	869.83 ms	36.79 tok/s

Enabling KV-cache reuse reduced average latency by approximately 78% and increased generation throughput by approximately 4.5× in the tested workload.

LLM Memory
Configuration	Memory
FP32	1884.59 MB
FP16	942.29 MB

FP16 reduced measured model memory usage by approximately 50%.

Concurrent Inference

Concurrency experiments were performed to understand throughput scaling under parallel inference requests.

Concurrency	Throughput
1	544.94 req/s
2	807.22 req/s
4	1,113.55 req/s
8	1,407.22 req/s

The results show increasing throughput within the tested workload, while also demonstrating why concurrency needs to be tuned against available compute resources.

API

InferX exposes a FastAPI inference service with:

Endpoint	Purpose
GET /health	Liveness check
GET /ready	Readiness check
POST /predict	Sentiment inference
GET /metrics	Prometheus metrics
Health Check
GET /health

Response:

{
  "status": "ok"
}
Readiness
GET /ready

Response:

{
  "status": "ready"
}
Prediction
POST /predict
Content-Type: application/json

Request:

{
  "text": "I really enjoyed this product."
}

Response:

{
  "label": "POSITIVE",
  "score": 0.9998691082000732
}

The API validates input length between 1 and 2000 characters.

Observability

InferX exposes Prometheus-compatible metrics through:

/metrics

The monitoring stack is:

FastAPI
   │
   ▼
Prometheus
   │
   ▼
Grafana

The Grafana dashboard tracks:

AWS request rate
P95 latency
Total inference requests
Error rate

This provides visibility into inference behavior beyond application logs.

AWS Deployment

InferX was deployed as a Dockerized inference service on AWS for end-to-end validation.

Environment
AWS EC2
x86_64
Ubuntu 24.04
c7i-flex.large
Docker
Amazon ECR
FastAPI
ONNX Runtime
INT8 DistilBERT
AWS Region: eu-central-1

The deployment exposed:

GET  /health
GET  /ready
POST /predict
GET  /metrics

The AWS deployment was validated with health, readiness, prediction, metrics, and load-testing requests.

AWS Load Test

Load testing was performed with Locust against the deployed API.

Configuration
Duration:       60 seconds
Users:          10
Spawn rate:     2 users/second
Endpoint:       POST /predict
Results
Metric	Result
Total Requests	1,674
Failures	0
Failure Rate	0%
Average Latency	47 ms
P50	29 ms
P95	110 ms
P99	320 ms
Maximum	~1.1 s
Throughput	28.01 req/s

The deployment processed all 1,674 requests with zero failures during the test.

The observed tail latency provides a baseline for future optimization of CPU scheduling, ONNX Runtime threading, application concurrency, request queuing, and container resources.

Detailed results:

reports/aws_load_test.md
Docker

InferX uses a CPU-focused Docker image designed to avoid unnecessary CUDA dependencies for CPU deployments.

Build:

docker build -t inferx .

Run:

docker run --rm -p 8000:8000 inferx

The optimized image significantly reduced Docker storage usage compared with the initial CUDA-heavy configuration.

CI/CD

GitHub Actions provides automated testing and Docker image publishing.

Git Push
   │
   ▼
GitHub Actions
   │
   ├── Run Tests
   │
   ├── Configure AWS OIDC
   │
   ├── Authenticate with Amazon ECR
   │
   ├── Build linux/amd64 image
   │
   └── Push image to Amazon ECR

AWS authentication uses GitHub OIDC rather than long-lived AWS access keys stored in GitHub.

The Docker publishing stage runs only after the test job succeeds.

Testing

Run the complete test suite:

pytest -q

The current test suite covers:

package functionality
batching behavior
performance regression protection

The performance regression test maintains a defined latency threshold for the transformer inference workload.

vLLM

InferX includes a vLLM benchmark entry point for future LLM serving experiments.

A vLLM CPU runtime compatibility experiment was performed on the x86 AWS environment using the official CPU container image.

A production-quality vLLM performance benchmark is not claimed because the available development environment did not provide suitable GPU resources for a meaningful comparison.

Future vLLM experiments will target appropriate NVIDIA/CUDA hardware.

Engineering Focus

InferX is intentionally focused on AI inference engineering rather than another generic AI application.

The project investigates the relationship between:

Model
  +
Runtime
  +
Precision
  +
Quantization
  +
Batching
  +
Concurrency
  +
Hardware
  +
Serving
  +
Observability

Key engineering areas include:

Latency optimization
Throughput optimization
Memory efficiency
Quantization
Precision optimization
Batch-size tuning
Dynamic batching
Concurrent inference
LLM generation
KV-cache behavior
API serving
Load testing
Monitoring
Containerization
Cloud deployment
CI/CD
Limitations

Current experiments are primarily CPU-based.

Therefore:

GPU-specific optimization has not been fully evaluated.
Distributed inference has not been implemented.
vLLM performance results are not included.
Benchmark results are hardware- and workload-specific.
AWS load-test results should not be interpreted as universal production capacity.

These limitations are intentionally documented rather than presenting development benchmarks as universal production guarantees.

Future Work

Potential extensions include:

NVIDIA GPU inference benchmarking
vLLM performance evaluation
Continuous batching
Advanced quantization methods
TensorRT / TensorRT-LLM experiments
Autoscaling
Distributed inference
Model routing
Adaptive batching
GPU memory profiling
Larger LLM workloads
Automated benchmark dashboards
Production-style deployment orchestration
Tech Stack
ML / AI
PyTorch
Hugging Face Transformers
Inference
ONNX
ONNX Runtime
vLLM experimentation
Optimization
INT8
FP16
BF16
KV Cache
Dynamic Batching
Concurrent Inference
Serving
FastAPI
Uvicorn
Observability
Prometheus
Grafana
Testing / Benchmarking
Pytest
Locust
Custom Python benchmarks
Infrastructure
Docker
AWS EC2
Amazon ECR
GitHub Actions
GitHub OIDC
Project Status

InferX currently includes:

Transformer inference pipeline
ONNX optimization
INT8 quantization
FP16 / BF16 experiments
Dynamic batching
Concurrent inference
LLM optimization experiments
KV-cache analysis
FastAPI serving
Prometheus metrics
Grafana dashboards
Locust load testing
Docker deployment
AWS deployment validation
GitHub Actions CI/CD
Automated testing

The project is developed as a practical AI inference and model optimization engineering portfolio.

License

This project is intended as an engineering portfolio and research project.


### Bas is baar

1. **Ctrl+A** in GitHub README editor
2. Delete everything
3. Upar wala **poora block ek saath paste**
4. Preview check
5. **Commit changes**
6. Commit message:
   `Document InferX architecture and benchmark results`

