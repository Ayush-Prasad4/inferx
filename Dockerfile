FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src ./src
COPY benchmarks ./benchmarks

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

RUN pip install --no-cache-dir \
    transformers \
    onnxruntime \
    fastapi \
    uvicorn \
    pydantic \
    prometheus-fastapi-instrumentator

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "inferx.api:app", "--host", "0.0.0.0", "--port", "8000"]
