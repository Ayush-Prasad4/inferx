import os
import torch
from transformers import AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def model_memory_mb(model):
    total_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )

    total_bytes += sum(
        buffer.numel() * buffer.element_size()
        for buffer in model.buffers()
    )

    return total_bytes / (1024 ** 2)


def load_model(dtype):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
    )
    model.eval()
    return model


def main():
    print("InferX LLM Memory Benchmark")
    print("=" * 45)
    print(f"Model: {MODEL_NAME}")

    print("\nLoading FP32 model...")
    fp32_model = load_model(torch.float32)
    fp32_memory = model_memory_mb(fp32_model)

    print(f"FP32 model memory: {fp32_memory:.2f} MB")

    del fp32_model

    print("\nLoading FP16 model...")
    fp16_model = load_model(torch.float16)
    fp16_memory = model_memory_mb(fp16_model)

    print(f"FP16 model memory: {fp16_memory:.2f} MB")

    reduction = (1 - fp16_memory / fp32_memory) * 100

    print(f"\nMemory reduction: {reduction:.2f}%")
    print(f"FP16 memory ratio: {fp16_memory / fp32_memory:.2f}x")


if __name__ == "__main__":
    main()
