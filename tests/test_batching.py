from inferx.batching import DynamicBatcher


def test_dynamic_batcher():
    batcher = DynamicBatcher(
        max_batch_size=4,
        max_wait_ms=10,
    )

    futures = [
        batcher.submit("This movie was excellent.")
        for _ in range(4)
    ]

    results = [future.result(timeout=5) for future in futures]

    assert len(results) == 4

    for result in results:
        assert result.shape[0] == 2
