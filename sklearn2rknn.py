import onnx
import onnx.helper as helper
from onnx import version_converter
from rknn.api import RKNN
from typing import Optional
from sklearn.neural_network import MLPClassifier
import numpy as np
import tempfile
import random
import string
import os
import argparse
import sys
import pickle
import json

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def sklearn2onnx(model, batch_size:Optional[int]=None, opset_ver=14) -> tuple[onnx.ModelProto, dict]:
    """
    Converts a scikit-learn model to ONNX format.

    Parameters
    ----------
    model : scikit-learn model
    batch_size : int, the batch size of the input to the model. This MUST be the same as the batch size used when calling the model for inference.
        Could be None.
    opset_ver : int, the ONNX operator set version to use. RKNN only works when this is >= 13 even though it claims to support 12
    """
    activation_functions_map = {
        'identity': 'Identity',
        'logistic': 'Sigmoid',
        'tanh': 'Tanh',
        'relu': 'Relu',
        'softmax': 'Softmax',
    }
    meta = {
        'input_shape': (batch_size, model.n_features_in_),
        'output_shape': (batch_size, model.n_outputs_),
    }

    is_classifier = hasattr(model, 'classes_')

    if is_classifier:
        meta['classes'] = model.classes_.tolist()
        meta['mode'] = 'classifier'
    else:
        meta['mode'] = 'regressor'

    # Build our own graph
    n_layers = model.n_layers_
    assert(len(model.coefs_) == len(model.intercepts_))
    activation_func_name = activation_functions_map[model.activation]
    nodes = []
    zero = None
    for i, (weight, bias) in enumerate(zip(model.coefs_, model.intercepts_)):
        weight_name = f"weight{i}"
        bias_name = f"bias{i}"
        nodes += [helper.make_node("Constant", inputs=[], outputs=[weight_name]
            , value=helper.make_tensor(name=weight_name, data_type=onnx.TensorProto.FLOAT, dims=weight.shape, vals=weight.flatten().tolist()))]
        nodes += [helper.make_node("Constant", inputs=[], outputs=[bias_name]
            , value=helper.make_tensor(name=bias_name, data_type=onnx.TensorProto.FLOAT, dims=bias.shape, vals=bias.flatten().tolist()))]

        act_func_name = activation_func_name if i < n_layers - 2 else activation_functions_map[model.out_activation_]
        # Can't use an Identity node as versuin converter will complain. Also we can't use GEMM as RKNN implements it in CPU,
        # However, MatMul + Add does run on the NPU
        if act_func_name == "Identity":
            nodes += [helper.make_node("MatMul", inputs=[f"output{i}", weight_name], outputs=[f"output_tmp_mul{i+1}"])]
            nodes += [helper.make_node("Add", inputs=[f"output_tmp_mul{i+1}", bias_name], outputs=[f"output{i+1}"])]
        else:
            nodes += [helper.make_node("MatMul", inputs=[f"output{i}", weight_name], outputs=[f"output_tmp_mul{i+1}"])]
            nodes += [helper.make_node("Add", inputs=[f"output_tmp_mul{i+1}", bias_name], outputs=[f"output_tmp{i+1}"])]
            nodes += [helper.make_node(act_func_name, inputs=[f"output_tmp{i+1}"], outputs=[f"output{i+1}"])]

    graph = helper.make_graph(nodes, "scikit2rknn"
        , [helper.make_tensor_value_info("output0", onnx.TensorProto.FLOAT, [batch_size, model.n_features_in_])]
        , [helper.make_tensor_value_info(f"output{n_layers-1}", onnx.TensorProto.FLOAT, [batch_size, model.n_outputs_])]
    )
    model = helper.make_model(graph)
    # Check the ONNX model is sane
    onnx.checker.check_model(model)
    # Convert to opset 14 as that's what the RKNN converter supports (not even fully support. But 12 doesn't work at all)
    model = version_converter.convert_version(model, opset_ver)
    return model, meta

def onnx2rknn(model, save_path : str, target_platform : str, quantization: bool=False, example_input = None) -> int:
    rknn = RKNN()
    if rknn.config(target_platform=target_platform) != 0: return -1
    if rknn.load_onnx(model=model) != 0: return -1
    if quantization == False:
        eprint("WARNING: Quantization is disabled. As of rknn-toolkit2 1.5.0, the NPU only supports quantized models. Inference will run on the CPU.")
    if rknn.build(do_quantization=quantization, dataset=example_input) != 0: return -1
    rknn.export_rknn(save_path)
    return 0

def sklearn2rknn(model,
        save_path: str,
        target_platform: str,
        tmp_dir: Optional[str] =None,
        remove_tmp: bool =True,
        batch_size: Optional[int] = None,
        quantization: bool = False,
        example_input: Optional[np.ndarray] = None,
    ) -> int:
    """
    Convert a scikit-learn model to RKNN format.

    Parameters
    ----------
    model : scikit-learn model
    save_path : str, the path to save the RKNN model to. Also save the meta data to save_path + ".json"
    target_platform : str, the target platform to convert to. Should be the the chip name, e.g. "rk3588", "rk3566", etc.."
    batch_size : int, the batch size of the input to the model. This MUST be the same as the batch size used when calling the model for inference.

    return : int, 0 if success, -1 if failed
    """
    tmp = tempfile.gettempdir() if tmp_dir is None else tmp_dir
    fname  = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(16))
    onnx_path = os.path.join(tmp, fname + ".onnx")
    model, meta = sklearn2onnx(model, batch_size=batch_size)
    onnx.save(model, onnx_path)
    ret = onnx2rknn(onnx_path, save_path, target_platform, quantization=quantization, example_input=example_input)
    if remove_tmp: os.remove(onnx_path)
    with open(save_path + ".json", "w") as f:
        f.write(json.dumps(meta))
    return ret

convert = sklearn2rknn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a scikit-learn model to RKNN format')
    parser.add_argument('model_path', type=str, help='The path to the scikit-learn model')
    parser.add_argument('save_path', type=str, help='The path to save the RKNN model to')
    parser.add_argument('target_platform', type=str, help='The target platform to convert to. Should be the the chip name, e.g. "rk3588", "rk3566", etc.."')
    parser.add_argument('--tmp_dir', type=str, default=None, help='The directory to save the temporary ONNX model to. If not specified, the system temporary directory will be used')
    parser.add_argument('--remove_tmp', action='store_true', help='Whether to remove the temporary ONNX model after conversion')
    parser.add_argument('--batch_size', type=int, default=None, help='The batch size of the input to the model. This MUST be the same as the batch size used when calling the model for inference.')
    parser.add_argument('--quantization', action='store_true', help='Whether to quantize the model. As of rknn-toolkit2 1.5.0, the NPU only supports quantized models. Inference will run on the CPU.')
    parser.add_argument('--example_input', type=str, default=None, help='The path to the example input to the model. Required if quantization is enabled')
    args = parser.parse_args()

    model = None
    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)
    example_input = None
    if args.example_input is not None:
        example_input = np.load(args.example_input)
    ret = sklearn2rknn(model, args.save_path, args.target_platform, args.tmp_dir, args.remove_tmp, args.batch_size, args.quantization, example_input)
    if ret != 0:
        print("Failed to convert model")
        sys.exit(-1)
    print("Successfully converted model")


