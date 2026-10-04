def test_inferx_package_imports():
    import inferx

    assert inferx is not None

def test_simple_model_inference():
    import torch
    from inferx.model import SimpleModel

    model = SimpleModel()

    input_data = torch.tensor([[1.0, 2.0]])

    with torch.inference_mode():
        output = model(input_data)

    assert output.shape == (1, 1)

def test_inference_function():
    import torch
    from inferx.inference import run_inference
    from inferx.model import SimpleModel

    model = SimpleModel()
    input_data = torch.tensor([[1.0, 2.0]])

    output = run_inference(model, input_data)

    assert output.shape == (1, 1)