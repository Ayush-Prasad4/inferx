# InferX — AI Inference & Model Optimization Platform

> Production-oriented AI inference platform for optimizing, serving, benchmarking, and monitoring Transformer and LLM workloads.

InferX is an inference engineering project focused on understanding and improving the performance of AI models under realistic serving workloads.

Instead of building another AI chatbot, InferX explores the engineering problems behind **fast, efficient, observable, and scalable model inference**.

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
