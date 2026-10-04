from concurrent.futures import wait

import numpy as np

import inferx.batching as batching


class FakeTokenizer:
    def __call__(self, texts, return_tensors=None, padding=True, truncation=True):
        batch_size = len(texts)
        return {
            "input_ids": np.zeros((batch_size, 4), dtype=np.int64),
            "attention_mask": np.ones((batch_size, 4), dtype=np.int64),
        }


class FakeSession:
    def run(self, output_names, inputs):
        batch_size = inputs["input_ids"].shape[0]
        return [np.arange(batch_size * 2, dtype=np.float32).reshape(batch_size, 2)]


def test_dynamic_batcher(monkeypatch):
    monkeypatch.setattr(
        batching.AutoTokenizer,
        "from_pretrained",
        lambda model_name: FakeTokenizer(),
    )

    monkeypatch.setattr(
        batching.ort,
        "InferenceSession",
        lambda *args, **kwargs: FakeSession(),
    )

    batcher = batching.DynamicBatcher(
        max_batch_size=4,
        max_wait_ms=10,
    )

    futures = [
        batcher.submit("first"),
        batcher.submit("second"),
        batcher.submit("third"),
    ]

    done, not_done = wait(futures, timeout=2)

    assert not not_done
    assert len(done) == 3

    for future in futures:
        result = future.result()
        assert result.shape == (2,)
