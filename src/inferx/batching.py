import threading
import time
from concurrent.futures import Future

import onnxruntime as ort
from transformers import AutoTokenizer


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_PATH = "benchmarks/distilbert_int8.onnx"


class DynamicBatcher:
    def __init__(self, max_batch_size=8, max_wait_ms=10):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"],
        )

        self.queue = []
        self.condition = threading.Condition()

        self.worker = threading.Thread(
            target=self._worker,
            daemon=True,
        )
        self.worker.start()

    def submit(self, text: str) -> Future:
        future = Future()

        with self.condition:
            self.queue.append((text, future))
            self.condition.notify()

        return future

    def _worker(self):
        while True:
            with self.condition:
                while not self.queue:
                    self.condition.wait()

                deadline = (
                    time.monotonic()
                    + self.max_wait_ms / 1000
                )

                while (
                    len(self.queue) < self.max_batch_size
                ):
                    remaining = deadline - time.monotonic()

                    if remaining <= 0:
                        break

                    self.condition.wait(timeout=remaining)

                    if not self.queue:
                        break

                batch = self.queue[:self.max_batch_size]
                del self.queue[:self.max_batch_size]

            texts = [item[0] for item in batch]
            futures = [item[1] for item in batch]

            try:
                inputs = self.tokenizer(
                    texts,
                    return_tensors="np",
                    padding=True,
                    truncation=True,
                )

                outputs = self.session.run(
                    None,
                    {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                    },
                )

                logits = outputs[0]

                for i, future in enumerate(futures):
                    future.set_result(logits[i])

            except Exception as exc:
                for future in futures:
                    future.set_exception(exc)
