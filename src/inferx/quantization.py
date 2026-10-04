from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize_model(
    input_path: str = "benchmarks/distilbert.onnx",
    output_path: str = "benchmarks/distilbert_int8.onnx",
):
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )

    return output_path
