import torch

from inferx.model import SimpleModel


def run_inference(model: SimpleModel, input_data: torch.Tensor) -> torch.Tensor:
    model.eval()

    with torch.inference_mode():
        output = model(input_data)

    return output
