"""Model-file fixtures for ingestion tests, built without PyTorch.

ONNX files are built with onnx.helper (the reference ONNX API). The .pt files are synthesised in
PyTorch's zip checkpoint layout (archive/data.pkl + archive/data/<key>) with the same pickle globals
and persistent ids torch.save emits; they are not produced by torch itself because PyTorch is not
installable in the build sandbox. `test_real_torch_checkpoint` covers real files when torch exists.
"""
from __future__ import annotations

import io
import json
import pickle
import sys
import types
import zipfile
from collections import OrderedDict

import numpy as np


def onnx_mlp(seed: int = 0) -> bytes:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    r = np.random.default_rng(seed)
    W1, b1 = r.normal(0, .3, (16, 8)).astype(np.float32), r.normal(0, .1, 16).astype(np.float32)
    W2, b2 = r.normal(0, .3, (4, 16)).astype(np.float32), r.normal(0, .1, 4).astype(np.float32)
    nodes = [helper.make_node("Gemm", ["x", "W1", "b1"], ["h"], transB=1, name="fc1"),
             helper.make_node("Relu", ["h"], ["hr"], name="relu1"),
             helper.make_node("MatMul", ["hr", "W2t"], ["m"], name="fc2"),
             helper.make_node("Add", ["m", "b2"], ["y0"], name="fc2_bias"),
             helper.make_node("Softmax", ["y0"], ["y"], name="softmax")]
    g = helper.make_graph(nodes, "mlp", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 8])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 4])],
                          [numpy_helper.from_array(W1, "W1"), numpy_helper.from_array(b1, "b1"),
                           numpy_helper.from_array(W2.T.copy(), "W2t"), numpy_helper.from_array(b2, "b2")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m.SerializeToString()


def onnx_cnn(seed: int = 0) -> bytes:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    r = np.random.default_rng(seed)
    Wc = r.normal(0, .3, (4, 1, 3, 3)).astype(np.float32)
    bc = np.zeros(4, np.float32)
    bn = [np.ones(4, np.float32) * 1.5, np.full(4, .1, np.float32), np.full(4, .2, np.float32), np.full(4, 2.0, np.float32)]
    Wf, bf = r.normal(0, .3, (3, 4 * 4 * 4)).astype(np.float32), np.zeros(3, np.float32)
    nodes = [helper.make_node("Conv", ["x", "Wc", "bc"], ["c"], pads=[1, 1, 1, 1], strides=[1, 1], name="conv"),
             helper.make_node("BatchNormalization", ["c", "s", "b", "mu", "var"], ["cb"], name="bn"),
             helper.make_node("Relu", ["cb"], ["cr"], name="relu"),
             helper.make_node("MaxPool", ["cr"], ["p"], kernel_shape=[2, 2], strides=[2, 2], name="pool"),
             helper.make_node("Flatten", ["p"], ["f"], axis=1, name="flat"),
             helper.make_node("Gemm", ["f", "Wf", "bf"], ["y"], transB=1, name="head")]
    inits = [numpy_helper.from_array(a, n) for a, n in zip([Wc, bc, *bn, Wf, bf], ["Wc", "bc", "s", "b", "mu", "var", "Wf", "bf"])]
    g = helper.make_graph(nodes, "cnn", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 1, 8, 8])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 3])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m.SerializeToString()


def onnx_residual() -> bytes:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    W = np.eye(4, dtype=np.float32)
    nodes = [helper.make_node("Gemm", ["x", "W"], ["h"], transB=1), helper.make_node("Add", ["h", "x"], ["y"])]
    g = helper.make_graph(nodes, "res", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 4])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 4])], [numpy_helper.from_array(W, "W")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    return m.SerializeToString()


def json_graph(with_weights: bool = False) -> bytes:
    doc = {"format": "nomo.graph/1", "name": "tiny_cnn", "input_shape": [2, 16, 16], "base_accuracy": 91.0,
           "layers": [{"name": "c1", "op": "conv2d", "out_channels": 8, "kernel": 3, "stride": 2, "activation": "relu",
                       "pool": {"type": "avg", "kernel": 2, "stride": 2}},
                      {"name": "fc", "op": "dense", "out_features": 5, "activation": "linear"}]}
    if with_weights:
        doc["layers"] = [{"name": "a", "op": "dense", "out_features": 3, "activation": "relu",
                          "weights": np.ones((3, 2 * 16 * 16)).tolist(), "bias": [0, 0, 0]},
                         {"name": "b", "op": "dense", "out_features": 2, "activation": "linear"}]
    return json.dumps(doc).encode()


# ---------------------------------------------------------------------------- .pt synthesis

def _fake_torch():
    torch = types.ModuleType("torch")
    utils = types.ModuleType("torch._utils")

    class FloatStorage:            # noqa: D401 - stand-in carrying torch's global name
        pass
    FloatStorage.__module__ = "torch"
    FloatStorage.__qualname__ = "FloatStorage"

    def _rebuild_tensor_v2(*a):
        raise RuntimeError("fixture only")
    _rebuild_tensor_v2.__module__ = "torch._utils"
    _rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
    utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    torch.FloatStorage = FloatStorage
    torch._utils = utils
    return torch, utils


class _Storage:
    def __init__(self, key, arr):
        self.key, self.arr = key, arr


class _Tensor:
    def __init__(self, storage, arr, rebuild):
        self.storage, self.arr, self.rebuild = storage, arr, rebuild

    def __reduce__(self):
        stride = tuple(s // self.arr.itemsize for s in self.arr.strides)
        return (self.rebuild, (self.storage, 0, tuple(self.arr.shape), stride, False, OrderedDict()))


def torch_state_dict_bytes(tensors: "OrderedDict[str, np.ndarray]", evil: bool = False) -> bytes:
    torch, utils = _fake_torch()
    saved = {k: sys.modules.get(k) for k in ("torch", "torch._utils")}
    sys.modules["torch"], sys.modules["torch._utils"] = torch, utils
    try:
        storages = {}
        sd = OrderedDict()
        for i, (k, a) in enumerate(tensors.items()):
            a = np.ascontiguousarray(a, np.float32)
            st = _Storage(str(i), a)
            storages[str(i)] = a
            sd[k] = _Tensor(st, a, utils._rebuild_tensor_v2)

        class P(pickle.Pickler):
            def persistent_id(self, obj):
                if isinstance(obj, _Storage):
                    return ("storage", torch.FloatStorage, obj.key, "cpu", obj.arr.size)
                return None
        buf = io.BytesIO()
        obj = sd
        if evil:
            import os

            class Evil:
                def __reduce__(self):
                    return (os.system, ("echo pwned",))
            obj = OrderedDict(sd, payload=Evil())
        P(buf, protocol=2).dump(obj)
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as zf:
            zf.writestr("archive/data.pkl", buf.getvalue())
            zf.writestr("archive/version", "3\n")
            for k, a in storages.items():
                zf.writestr(f"archive/data/{k}", a.tobytes())
        return out.getvalue()
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def mlp_state_dict(seed: int = 0) -> "OrderedDict[str, np.ndarray]":
    r = np.random.default_rng(seed)
    return OrderedDict([("net.0.weight", r.normal(0, .3, (12, 6))), ("net.0.bias", r.normal(0, .1, 12)),
                        ("net.2.weight", r.normal(0, .3, (3, 12))), ("net.2.bias", r.normal(0, .1, 3))])
