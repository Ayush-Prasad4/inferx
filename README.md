# InferX — AI Inference & Model Optimization Platform

> Production-oriented platform for optimizing, serving, benchmarking, and monitoring Transformer and LLM inference workloads.

## Overview

InferX is an AI inference engineering project focused on reducing inference latency, improving throughput, evaluating optimization techniques, and serving optimized models through production-style APIs.

The platform explores the complete inference lifecycle:

* PyTorch model inference
* Hugging Face Transformers
* ONNX Runtime optimization
* INT8 quantization
* FP16 / BF16 inference
* Dynamic batching
* Concurrent inference
* LLM optimization
* KV-cache analysis
* FastAPI model serving
* Prometheus monitoring
* Grafana observability
* Locust load testing
* Docker containerization
* AWS deployment
* GitHub Actions CI/CD
* Amazon ECR image publishing

## Architecture

```text
                         ┌─────────────────────┐
                         │      Clients        │
                         │  HTTP / Load Tests  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      FastAPI        │
                         │   /health /ready    │
                         │      /predict       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   ONNX Runtime      │
                         │    INT8 Model       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ DistilBERT SST-2    │
                         │ Sequence Classifier │
                         └─────────────────────┘

          ┌──────────────────┐       ┌──────────────────┐
          │    Prometheus    │──────▶│     Grafana      │
          │     Metrics      │       │   Dashboards     │
          └──────────────────┘       └──────────────────┘

                         ┌─────────────────────┐
                         │       Locust        │
                         │    Load Testing     │
                         └─────────────────────┘
```

## Key Engineering Results

The project evaluates inference optimization across multiple runtimes, precisions, batch sizes, concurrency levels, and deployment environments.

### Transformer inference

| Configuration     | Avg latency |
| ----------------- | ----------: |
| PyTorch Eager     |   14.962 ms |
| ONNX Runtime FP32 |    4.347 ms |
| ONNX Runtime INT8 |    1.531 ms |

ONNX Runtime reduced average latency by approximately 71% compared with the representative PyTorch Eager benchmark.

Dynamic INT8 batching showed a throughput improvement up to approximately 1,221 samples/sec around batch size 32, while larger batches showed diminishing returns.

### Precision experiments

| Configuration | Avg latency |
| ------------- | ----------: |
| FP32          |   13.371 ms |
| FP16          |    6.191 ms |
| BF16          |    9.761 ms |

On the development Apple Silicon environment, FP16 produced the lowest measured latency among these precision experiments.

### LLM optimization

The project also evaluates `Qwen/Qwen2.5-0.5B-Instruct`.

| Configuration | Avg latency |  Throughput |
| ------------- | ----------: | ----------: |
| FP32          |  1257.91 ms | 25.44 tok/s |
| FP16          |   843.13 ms | 37.95 tok/s |

FP16 reduced measured average latency by approximately 33% and increased generation throughput by approximately 49%.

### KV-cache experiment

| Configuration     | Avg latency |  Throughput |
| ----------------- | ----------: | ----------: |
| KV cache disabled |  3901.52 ms |  8.20 tok/s |
| KV cache enabled  |   869.83 ms | 36.79 tok/s |

The experiment demonstrates the substantial effect of KV-cache reuse on autoregressive generation performance.

## Dynamic Batching

InferX implements an event-driven dynamic batching scheduler.

Requests are placed into a queue and grouped according to:

* Maximum batch size
* Maximum waiting time
* Request arrival patterns

Each request receives its own `Future`, allowing the scheduler to combine requests into a batch while returning individual results to callers.

The scheduler experiment improved measured throughput from approximately 382 requests/sec in the earlier implementation to approximately 893 requests/sec under the tested workload.

## API

InferX exposes a FastAPI inference service with:

```text
GET  /health
GET  /ready
POST /predict
GET  /metrics
```

The prediction endpoint validates input length, performs ONNX Runtime inference, converts logits into normalized probabilities, and returns:

```json
{
  "label": "POSITIVE",
  "score": 0.9998691082000732
}
```

The API also includes readiness and health endpoints suitable for containerized deployments.

## Monitoring

InferX exposes Prometheus-compatible metrics through the FastAPI application.

Grafana dashboards track:

* AWS request rate
* P95 inference latency
* Total inference requests
* Error rate

Prometheus configuration is available in:

```text
monitoring/prometheus.yml
```

## Load Testing

Locust was used to evaluate the deployed API under concurrent traffic.

Representative AWS load test:

* Platform: AWS EC2
* Instance: c7i-flex.large
* Region: eu-central-1
* Concurrent users: 10
* Spawn rate: 2 users/sec
* Duration: 60 seconds
* Endpoint: `/predict`

Results:

| Metric                   |      Result |
| ------------------------ | ----------: |
| Total requests           |       1,674 |
| Failures                 |           0 |
| Failure rate             |          0% |
| Average latency          |       47 ms |
| P50 latency              |       29 ms |
| P95 latency              |      110 ms |
| P99 latency              |      320 ms |
| Maximum observed latency |      ~1.1 s |
| Throughput               | 28.01 req/s |

These results represent a measured test environment and are not presented as production capacity guarantees.

## Docker

InferX is containerized using a CPU-oriented Docker image.

The optimized Docker image uses:

* Python 3.11
* CPU-only PyTorch
* Transformers
* ONNX Runtime
* FastAPI
* Pydantic
* Prometheus instrumentation

The CPU-only dependency strategy substantially reduced image storage compared with the initial CUDA-containing image.

## AWS Deployment

InferX was deployed and validated on AWS using:

* Amazon EC2
* Amazon ECR
* Docker
* FastAPI
* ONNX Runtime
* INT8 DistilBERT

The AWS deployment was tested through the public API and load-tested with Locust.

The EC2 environment used for the experiments has since been terminated to avoid unnecessary ongoing compute costs.

The ECR image remains available for CI/CD and future deployments.

## CI/CD

GitHub Actions provides the CI/CD pipeline.

The workflow performs:

1. Repository checkout
2. Python environment setup
3. Dependency installation
4. Automated tests
5. AWS authentication using GitHub OIDC
6. Amazon ECR authentication
7. Linux AMD64 Docker build
8. Docker image publishing to ECR

AWS credentials are not stored as long-lived GitHub secrets. The workflow uses an IAM role with GitHub OIDC federation.

## Quantization

InferX evaluates dynamic INT8 quantization using ONNX Runtime.

The quantization workflow is:

```text
PyTorch / Hugging Face model
            │
            ▼
       ONNX export
            │
            ▼
      FP32 ONNX model
            │
            ▼
   Dynamic INT8 quantization
            │
            ▼
      INT8 ONNX model
            │
            ▼
       ONNX Runtime
```

This provides a practical CPU inference optimization path without requiring a GPU runtime.

## Project Structure

```text
inferx/
├── .github/
│   └── workflows/
│       └── ci.yml
├── benchmarks/
│   ├── benchmark_batching.py
│   ├── benchmark_bf16.py
│   ├── benchmark_compile.py
│   ├── benchmark_concurrency.py
│   ├── benchmark_dynamic_batching.py
│   ├── benchmark_dynamic_scheduler.py
│   ├── benchmark_fp16.py
│   ├── benchmark_int8.py
│   ├── benchmark_int8_batching.py
│   ├── benchmark_llm_baseline.py
│   ├── benchmark_llm_concurrency.py
│   ├── benchmark_llm_fp16.py
│   ├── benchmark_llm_kv_cache.py
│   ├── benchmark_llm_memory.py
│   ├── benchmark_onnx.py
│   ├── benchmark_transformer.py
│   ├── benchmark_transformer_fair.py
│   ├── benchmark_vllm.py
│   └── locustfile.py
├── monitoring/
│   └── prometheus.yml
├── reports/
│   ├── aws_load_test.md
│   └── benchmark_summary.md
├── src/
│   └── inferx/
│       ├── api.py
│       ├── batching.py
│       ├── huggingface.py
│       ├── inference.py
│       ├── model.py
│       ├── onnx.py
│       └── quantization.py
├── tests/
│   ├── test_batching.py
│   ├── test_package.py
│   └── test_performance_regression.py
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Reproducing the Project

Clone the repository:

```bash
git clone https://github.com/Ayush-Prasad4/inferx.git
cd inferx
```

Create the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project:

```bash
pip install -e .
```

Install development dependencies:

```bash
pip install pytest torch transformers onnxruntime
```

Run the tests:

```bash
pytest -q
```

Run the API locally after generating the required model artifacts:

```bash
uvicorn inferx.api:app --host 0.0.0.0 --port 8000
```

## Engineering Takeaways

InferX demonstrates practical inference engineering rather than focusing only on model training.

The experiments show several important trade-offs:

* Lower precision can improve inference latency and memory efficiency.
* ONNX Runtime can provide substantial CPU inference improvements.
* INT8 quantization can further reduce inference latency.
* Increasing batch size can improve throughput but increases individual request latency.
* Dynamic batching requires balancing batching efficiency against queueing delay.
* Concurrency has an optimal operating point rather than increasing throughput indefinitely.
* KV-cache reuse is critical for autoregressive LLM generation.
* Tail latency becomes important when moving from local benchmarks to networked deployments.
* Optimization results depend strongly on hardware, runtime, workload, and model architecture.

## vLLM

InferX includes a vLLM benchmark entry point for future GPU-oriented inference experiments.

A CPU vLLM runtime compatibility experiment was performed on an x86 AWS environment, but no representative vLLM model-performance benchmark is claimed from that experiment.

Future work includes benchmarking vLLM on an appropriate NVIDIA GPU environment.

## Future Work

Potential future improvements include:

* NVIDIA GPU benchmarking
* vLLM throughput evaluation
* Continuous batching with GPU inference
* TensorRT / TensorRT-LLM experiments
* More advanced quantization methods
* Autoscaling inference workers
* Distributed inference
* More comprehensive latency and throughput dashboards
* Model quality regression testing
* Automated benchmark reporting
* Kubernetes deployment

## Status

InferX is a completed portfolio project demonstrating an end-to-end AI inference optimization workflow across model execution, optimization, serving, benchmarking, observability, containerization, cloud deployment, and CI/CD.

The project is intentionally positioned as a production-oriented engineering platform rather than as a claim of production-scale infrastructure.

## License

This project is intended as a personal engineering and portfolio project.
