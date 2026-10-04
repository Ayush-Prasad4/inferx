# InferX Benchmark Summary

## Transformer Inference

### DistilBERT PyTorch FP32
- Average latency: 17.739 ms
- P50 latency: 17.848 ms
- P95 latency: 19.338 ms
- P99 latency: 19.860 ms

### DistilBERT ONNX FP32
- Average latency: 4.347 ms
- P50 latency: 3.904 ms
- P95 latency: 6.704 ms
- P99 latency: 10.362 ms

### DistilBERT ONNX INT8
- Average latency: 1.531 ms
- P50 latency: 1.377 ms
- P95 latency: 1.993 ms
- P99 latency: 2.508 ms

## LLM Inference

### Qwen2.5-0.5B FP32
- Average latency: 1257.91 ms
- P50 latency: 1255.22 ms
- P95 latency: 1289.22 ms
- Generation rate: 25.44 tokens/s

### Qwen2.5-0.5B FP16
- Average latency: 843.13 ms
- P50 latency: 833.56 ms
- P95 latency: 1002.22 ms
- Generation rate: 37.95 tokens/s

### LLM Concurrency
| Concurrency | Throughput |
|---:|---:|
| 1 | 1.16 req/s |
| 2 | 1.92 req/s |
| 4 | 2.13 req/s |
| 8 | 1.68 req/s |

### KV Cache
| Configuration | Avg latency | Generation rate |
|---|---:|---:|
| Disabled | 3901.52 ms | 8.20 tokens/s |
| Enabled | 869.83 ms | 36.79 tokens/s |

### LLM Memory
| Precision | Model memory |
|---|---:|
| FP32 | 1884.59 MB |
| FP16 | 942.29 MB |

- Memory reduction: 50%
