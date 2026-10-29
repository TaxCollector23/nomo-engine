"""Model ingestion (SPEC §1.1a): uploaded file -> (ModelGraph, weights, report).

Supported inputs
    .onnx         sequential graphs of Gemm / MatMul(+Add) / Conv / Relu / Flatten / Reshape /
                  BatchNormalization (folded) / MaxPool / AveragePool / GlobalAveragePool /
                  Identity / Dropout / trailing Softmax|Sigmoid (dropped, reported)
    .json         nomo.graph/1 schema (below); weights optional
    .pt / .pth    PyTorch state_dict files, parsed WITHOUT importing torch through a whitelisting
                  unpickler: only tensors and ordered dicts are accepted. Whole pickled nn.Module
                  objects are rejected, because unpickling them executes arbitrary code.

Every path produces the same result: a chain of LayerSpecs with geometry attrs, float weights
{name: (W, b)}, and an IngestReport listing assumptions (e.g. activations inferred for a state_dict,
synthetic weights when a JSON graph has none) so the UI can show exactly what was guessed.

nomo.graph/1:
    {"format": "nomo.graph/1", "name": "...", "input_shape": [C, H, W] | [N],
     "base_accuracy": 92.5,
     "layers": [
        {"name": "c1", "op": "conv2d", "out_channels": 16, "kernel": 3, "stride": 1, "padding": 1,
         "activation": "relu", "pool": {"type": "max", "kernel": 2, "stride": 2},
         "weights": [...optional nested list...], "bias": [...optional...]},
        {"name": "fc", "op": "dense", "out_features": 10, "activation": "linear"}]}
"""
from __future__ import annotations

import io
import json
import pickle
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..ir import LayerSensitivity, LayerSpec, ModelGraph, conv2d, dense, operator_block

Weights = Dict[str, Tuple[np.ndarray, np.ndarray]]
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


class IngestError(ValueError):
    """Raised with a message suitable for showing to the user."""


@dataclass
class IngestReport:
    source_format: str
    layers: int = 0
    params: int = 0
    macs: int = 0
    weights_source: str = "file"                 # file | synthetic
    assumptions: List[str] = field(default_factory=list)
    dropped_ops: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# shape helpers shared by all parsers
# ---------------------------------------------------------------------------

def _pool_out(h: int, w: int, pool: Optional[dict]) -> Tuple[int, int]:
    if not pool:
        return h, w
    if pool["type"] == "global_avg":
        return 1, 1
    k, s = int(pool["kernel"]), int(pool.get("stride", pool["kernel"]))
    return (h - k) // s + 1, (w - k) // s + 1


class _ChainBuilder:
    """Accumulates layers while tracking the running activation shape."""

    def __init__(self, input_shape: Tuple[int, ...]) -> None:
        self.input_shape = tuple(int(x) for x in input_shape)
        self.shape: Tuple[int, ...] = self.input_shape
        self.layers: List[LayerSpec] = []
        self.weights: Weights = {}

    def _name(self, name: Optional[str], kind: str) -> str:
        base = (name or f"{kind}{len(self.layers)}").replace("/", "_").replace(".", "_").replace(":", "_")[:48] or kind
        n, i = base, 1
        while any(l.name == n for l in self.layers):
            n, i = f"{base}_{i}", i + 1
        return n

    def add_dense(self, name: Optional[str], W: np.ndarray, b: Optional[np.ndarray], activation: str) -> None:
        n_in = int(np.prod(self.shape))
        if W.shape[1] != n_in:
            raise IngestError(f"dense layer '{name}' expects {W.shape[1]} inputs but receives {n_in} "
                              f"(activation shape {list(self.shape)})")
        flatten = len(self.shape) > 1
        nm = self._name(name, "dense")
        self.layers.append(dense(nm, n_in, W.shape[0], activation=activation,
                                 attrs={"flatten_input": True} if flatten else {}))
        self.weights[nm] = (W.astype(np.float64), (b if b is not None else np.zeros(W.shape[0])).astype(np.float64))
        self.shape = (W.shape[0],)

    def add_conv(self, name: Optional[str], W: np.ndarray, b: Optional[np.ndarray], stride: int, padding: int,
                 activation: str, pool: Optional[dict]) -> None:
        if len(self.shape) != 3:
            raise IngestError(f"conv layer '{name}' needs a [C, H, W] input, got {list(self.shape)}")
        c, h, w = self.shape
        if W.shape[1] != c:
            raise IngestError(f"conv layer '{name}' expects {W.shape[1]} input channels, got {c}")
        if W.shape[2] != W.shape[3]:
            raise IngestError(f"conv layer '{name}': only square kernels are supported")
        nm = self._name(name, "conv")
        spec = conv2d(nm, c, W.shape[0], W.shape[2], h, w, stride, padding=padding, activation=activation)
        oc, oh, ow = spec.attrs["out_shape"]
        if pool:
            ph, pw = _pool_out(oh, ow, pool)
            attrs = dict(spec.attrs, pool=pool, pooled_shape=(oc, ph, pw))
            spec = LayerSpec(**{**spec.__dict__, "out_neurons": oc * ph * pw, "attrs": attrs})
            oh, ow = ph, pw
        self.layers.append(spec)
        self.weights[nm] = (W.astype(np.float64), (b if b is not None else np.zeros(W.shape[0])).astype(np.float64))
        self.shape = (oc, oh, ow)

    def pool_last(self, pool: dict) -> None:
        if not self.layers or self.layers[-1].op != "conv2d" or "pool" in self.layers[-1].attrs:
            raise IngestError("pooling is only supported directly after a convolution")
        last = self.layers[-1]
        oc, oh, ow = last.attrs["out_shape"]
        ph, pw = _pool_out(oh, ow, pool)
        attrs = dict(last.attrs, pool=pool, pooled_shape=(oc, ph, pw))
        self.layers[-1] = LayerSpec(**{**last.__dict__, "out_neurons": oc * ph * pw, "attrs": attrs})
        self.shape = (oc, ph, pw)

    def set_last_activation(self, act: str) -> None:
        if self.layers:
            last = self.layers[-1]
            self.layers[-1] = LayerSpec(**{**last.__dict__, "activation": act})

    def add_operator(self, name: Optional[str], family: str, W: np.ndarray, b: Optional[np.ndarray],
                     output_shape: Tuple[int, ...], activation: str = "linear", metadata: Optional[dict] = None) -> None:
        """Add a channel/token/feature-preserving operator block.

        Operator blocks keep their rank and geometry in the IR.  A 2-D weight
        matrix is applied independently at each spatial/token position by the
        runtime adapter; no implicit flattening is introduced.
        """
        input_shape = tuple(self.shape)
        feature_axis = 0 if len(input_shape) >= 3 else len(input_shape) - 1
        if len(input_shape) < 1 or W.ndim != 2 or W.shape[1] != input_shape[feature_axis]:
            raise IngestError(f"operator '{name}' needs a [out_features, in_features] channel/token matrix")
        nm = self._name(name, family)
        out_shape = tuple(int(v) for v in output_shape)
        positions = int(np.prod(input_shape[1:])) if len(input_shape) >= 3 else int(input_shape[0]) if len(input_shape) == 2 else 1
        params = int(W.size + W.shape[0])
        spec = operator_block(nm, family, input_shape, out_shape, positions * int(W.shape[0]) * int(W.shape[1]),
                              params, tuple(W.shape), activation=activation, attrs=metadata or {})
        self.layers.append(spec)
        self.weights[nm] = (W.astype(np.float64), (b if b is not None else np.zeros(W.shape[0])).astype(np.float64))
        self.shape = out_shape

    def build(self, name: str, base_accuracy: float) -> ModelGraph:
        if not self.layers:
            raise IngestError("no supported layers found in the model")
        return ModelGraph(name=name, input_shape=self.input_shape, layers=self.layers,
                          base_accuracy=float(base_accuracy))


# ---------------------------------------------------------------------------
# ONNX
# ---------------------------------------------------------------------------

def parse_onnx(data: bytes, name: str, base_accuracy: float,
               input_shape: Optional[Tuple[int, ...]] = None) -> Tuple[ModelGraph, Weights, IngestReport]:
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:  # pragma: no cover
        raise IngestError("ONNX support needs the 'onnx' package on the server") from exc
    try:
        m = onnx.load_from_string(data)
    except Exception as exc:
        raise IngestError(f"not a readable ONNX file: {exc}") from exc
    g = m.graph
    inits = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    real_inputs = [i for i in g.input if i.name not in inits]
    if len(real_inputs) != 1:
        raise IngestError(f"expected exactly one graph input, found {len(real_inputs)}")
    if input_shape is None:
        dims = [d.dim_value for d in real_inputs[0].type.tensor_type.shape.dim]
        if not dims or any(d <= 0 for d in dims[1:]):
            raise IngestError("the ONNX input has dynamic spatial dimensions; please enter the input shape")
        input_shape = tuple(dims[1:])                       # drop the batch dimension
    rep = IngestReport("onnx")
    cb = _ChainBuilder(input_shape)
    consumers: Dict[str, int] = {}
    for n in g.node:
        for i in n.input:
            consumers[i] = consumers.get(i, 0) + 1
    cur = real_inputs[0].name
    pending_matmul: Optional[Tuple[str, np.ndarray]] = None

    def attr(node, key, default=None):
        for a in node.attribute:
            if a.name == key:
                v = onnx.helper.get_attribute_value(a)
                return v.decode() if isinstance(v, bytes) else v
        return default

    for node in g.node:
        data_inputs = [i for i in node.input if i and i not in inits]
        if node.op_type == "Constant":
            inits[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)
            continue
        if cur not in data_inputs:
            raise IngestError(f"graph is not a simple chain at node '{node.name or node.op_type}' "
                              "(branches / residual connections are not supported yet)")
        if consumers.get(cur, 0) > 1:
            raise IngestError(f"tensor '{cur}' feeds several nodes (branching graphs are not supported yet)")
        op = node.op_type
        if op == "Gemm":
            W = inits[node.input[1]]
            W = W if attr(node, "transB", 0) else W.T
            if attr(node, "transA", 0):
                raise IngestError("Gemm with transA=1 is not supported")
            alpha, beta = attr(node, "alpha", 1.0), attr(node, "beta", 1.0)
            b = inits[node.input[2]] * beta if len(node.input) > 2 and node.input[2] else None
            cb.add_dense(node.name, W * alpha, b, "linear")
        elif op == "MatMul":
            pending_matmul = (node.name, inits[node.input[1]].T)
            cb.add_dense(node.name, pending_matmul[1], None, "linear")
        elif op == "Add" and pending_matmul is not None and any(i in inits for i in node.input):
            bias = inits[[i for i in node.input if i in inits][0]].reshape(-1)
            last = cb.layers[-1].name
            cb.weights[last] = (cb.weights[last][0], bias.astype(np.float64))
            pending_matmul = None
        elif op == "Conv":
            if attr(node, "group", 1) != 1:
                raise IngestError(f"grouped/depthwise convolution '{node.name}' is not supported yet")
            strides = attr(node, "strides", [1, 1])
            pads = attr(node, "pads", [0, 0, 0, 0])
            if len(set(strides)) != 1 or len(set(pads)) != 1 or any(d != 1 for d in attr(node, "dilations", [1, 1])):
                raise IngestError(f"conv '{node.name}': only equal strides/pads and dilation 1 are supported")
            W = inits[node.input[1]]
            b = inits[node.input[2]] if len(node.input) > 2 and node.input[2] else None
            cb.add_conv(node.name, W, b, int(strides[0]), int(pads[0]), "linear", None)
        elif op == "BatchNormalization":
            scale, bias, mean, var = (inits[node.input[k]] for k in range(1, 5))
            eps = attr(node, "epsilon", 1e-5)
            last = cb.layers[-1].name
            W, b = cb.weights[last]
            k = scale / np.sqrt(var + eps)
            W = W * k.reshape((-1,) + (1,) * (W.ndim - 1))
            cb.weights[last] = (W, (b - mean) * k + bias)
            rep.assumptions.append(f"BatchNormalization after '{last}' folded into its weights")
        elif op == "Relu":
            last = cb.layers[-1] if cb.layers else None
            if last is not None and last.attrs.get("pool", {}).get("type") in ("avg", "global_avg"):
                raise IngestError(f"ReLU after average pooling ('{last.name}') is not supported; use Conv-ReLU-Pool order")
            cb.set_last_activation("relu")
        elif op in ("MaxPool", "AveragePool"):
            k = attr(node, "kernel_shape")
            s = attr(node, "strides", k)
            if len(set(k)) != 1 or len(set(s)) != 1 or any(attr(node, "pads", [0, 0, 0, 0])):
                raise IngestError(f"pool '{node.name}': only square, unpadded pooling is supported")
            cb.pool_last({"type": "max" if op == "MaxPool" else "avg", "kernel": int(k[0]), "stride": int(s[0])})
        elif op == "GlobalAveragePool":
            cb.pool_last({"type": "global_avg"})
        elif op in ("Flatten", "Reshape", "Identity", "Dropout", "Squeeze"):
            pass                                            # dense layers flatten their input themselves
        elif op in ("Softmax", "LogSoftmax", "Sigmoid"):
            rep.dropped_ops.append(op)
            rep.assumptions.append(f"final {op} dropped: search and export use the logits (argmax is unchanged)")
        else:
            raise IngestError(f"unsupported ONNX operator '{op}' (node '{node.name}')")
        if op not in ("MatMul", "Add"):
            pending_matmul = None
        cur = node.output[0]
    return _finish(cb, name, base_accuracy, rep)


# ---------------------------------------------------------------------------
# JSON schema
# ---------------------------------------------------------------------------

def parse_json_graph(data: bytes, name: Optional[str] = None,
                     base_accuracy: Optional[float] = None, seed: int = 0) -> Tuple[ModelGraph, Weights, IngestReport]:
    try:
        doc = json.loads(data)
    except Exception as exc:
        raise IngestError(f"not valid JSON: {exc}") from exc
    if doc.get("format") not in ("nomo.graph/1", None):
        raise IngestError(f"unknown format '{doc.get('format')}' (expected nomo.graph/1)")
    if "input_shape" not in doc or "layers" not in doc:
        raise IngestError("JSON graph needs 'input_shape' and 'layers'")
    rep = IngestReport("json")
    rng = np.random.default_rng(seed)
    cb = _ChainBuilder(tuple(doc["input_shape"]))
    synthetic: List[str] = []
    for i, L in enumerate(doc["layers"]):
        op, act = L.get("op"), L.get("activation", "relu")
        if act not in ("relu", "linear"):
            raise IngestError(f"layer {i}: activation must be 'relu' or 'linear'")
        if op == "dense":
            n_in, n_out = int(np.prod(cb.shape)), int(L["out_features"])
            if "weights" in L:
                W = np.asarray(L["weights"], float)
            else:
                W = rng.normal(0, np.sqrt(2 / n_in), (n_out, n_in))
                synthetic.append(L.get("name") or f"layer {i}")
            b = np.asarray(L["bias"], float) if "bias" in L else np.zeros(n_out)
            cb.add_dense(L.get("name"), W, b, act)
        elif op == "conv2d":
            k, oc = int(L["kernel"]), int(L["out_channels"])
            c = cb.shape[0]
            if "weights" in L:
                W = np.asarray(L["weights"], float)
            else:
                W = rng.normal(0, np.sqrt(2 / (c * k * k)), (oc, c, k, k))
                synthetic.append(L.get("name") or f"layer {i}")
            b = np.asarray(L["bias"], float) if "bias" in L else np.zeros(oc)
            cb.add_conv(L.get("name"), W, b, int(L.get("stride", 1)), int(L.get("padding", k // 2)), act, L.get("pool"))
        elif op in ("fourier", "fno", "attention", "vit_attention", "message_passing", "gnn"):
            family = "fno" if op in ("fourier", "fno") else "vit" if op in ("attention", "vit_attention") else "gnn"
            in_features = cb.shape[0] if len(cb.shape) >= 3 else cb.shape[-1]
            out_shape = tuple(L.get("out_shape", cb.shape))
            out_features = out_shape[0] if len(out_shape) >= 3 else out_shape[-1]
            if "weights" in L:
                W = np.asarray(L["weights"], float)
            else:
                W = rng.normal(0, np.sqrt(2 / max(1, in_features)), (out_features, in_features))
                synthetic.append(L.get("name") or f"layer {i}")
            b = np.asarray(L["bias"], float) if "bias" in L else np.zeros(out_features)
            attrs = dict(L.get("metadata", {}), spectral_modes=L.get("spectral_modes"),
                         sequence_length=L.get("sequence_length"), graph_edges=L.get("graph_edges"))
            cb.add_operator(L.get("name"), family, W, b, out_shape, act, attrs)
            rep.assumptions.append(f"{family.upper()} operator block retained spatial/token/graph geometry; backend lowering is adapter-based")
        else:
            raise IngestError(f"layer {i}: unsupported op '{op}' (use 'dense' or 'conv2d')")
    if synthetic:
        rep.weights_source = "synthetic" if len(synthetic) == len(doc["layers"]) else "partial"
        rep.assumptions.append(f"no weights given for {', '.join(synthetic)}: random (He) weights were generated, so "
                               "exports are structurally correct but those layers are untrained")
    out = _finish(cb, name or doc.get("name", "uploaded_model"),
                  base_accuracy if base_accuracy is not None else doc.get("base_accuracy", 90.0), rep)
    out[0].architecture_family = doc.get("architecture_family") or ("fno" if any(l.op == "fno" for l in out[0].layers)
                                                                      else "vit" if any(l.op == "vit" for l in out[0].layers)
                                                                      else "gnn" if any(l.op == "gnn" for l in out[0].layers)
                                                                      else None)
    out[0].metadata.update({"operator_schema": "nomo.operator/1"})
    return out


# ---------------------------------------------------------------------------
# PyTorch state_dict (no torch import, no code execution)
# ---------------------------------------------------------------------------

_TORCH_DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
                 "BFloat16Storage": None, "LongStorage": np.int64, "IntStorage": np.int32,
                 "ShortStorage": np.int16, "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_}


class UnsafePickle(IngestError):
    pass


class _StorageType:
    def __init__(self, name: str) -> None:
        self.name = name


def _rebuild_tensor_v2(storage, offset, size, stride, *_args):
    arr, = storage
    itemsize = arr.itemsize
    if len(size) == 0:
        return arr[offset].copy()
    return np.lib.stride_tricks.as_strided(arr[offset:], shape=tuple(size),
                                           strides=tuple(s * itemsize for s in stride)).copy()


def _rebuild_parameter(data, *_args):
    return data


class _SafeUnpickler(pickle.Unpickler):
    ALLOWED = {
        ("collections", "OrderedDict"),
        ("torch._utils", "_rebuild_tensor_v2"),
        ("torch._utils", "_rebuild_parameter"),
        ("torch._utils", "_rebuild_parameter_with_state"),
    }

    def __init__(self, fh, zf: zipfile.ZipFile, prefix: str) -> None:
        super().__init__(fh)
        self.zf, self.prefix = zf, prefix

    def find_class(self, module: str, name: str):
        if module == "torch" and name in _TORCH_DTYPES:
            return _StorageType(name)
        if (module, name) not in self.ALLOWED:
            raise UnsafePickle(
                f"this file contains a pickled Python object ({module}.{name}), not just weights. "
                "Loading it would run code from the file, so it is refused. Save weights with "
                "torch.save(model.state_dict(), 'model.pt') or export ONNX instead.")
        if name == "OrderedDict":
            from collections import OrderedDict
            return OrderedDict
        if name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        return _rebuild_parameter

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and pid and pid[0] == "storage"):
            raise UnsafePickle("unexpected persistent object in state_dict")
        stype, key = pid[1], pid[2]
        if not isinstance(stype, _StorageType):
            raise UnsafePickle("unsupported storage type")
        dtype = _TORCH_DTYPES[stype.name]
        if dtype is None:
            raise IngestError("bfloat16 weights are not supported; convert to float32 before saving")
        raw = self.zf.read(f"{self.prefix}data/{key}")
        return (np.frombuffer(raw, dtype=dtype),)


def load_state_dict(data: bytes) -> Dict[str, np.ndarray]:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise IngestError("not a PyTorch zip checkpoint (legacy non-zip .pt files are not supported; "
                          "re-save with a recent PyTorch)") from exc
    pkl = [n for n in zf.namelist() if n.endswith("data.pkl")]
    if not pkl:
        raise IngestError("no data.pkl inside the checkpoint")
    prefix = pkl[0][: -len("data.pkl")]
    obj = _SafeUnpickler(io.BytesIO(zf.read(pkl[0])), zf, prefix).load()
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise IngestError("the checkpoint is not a state_dict")
    return {k: np.asarray(v) for k, v in obj.items() if isinstance(v, np.ndarray)}


def parse_state_dict(data: bytes, name: str, base_accuracy: float,
                     input_shape: Optional[Tuple[int, ...]] = None) -> Tuple[ModelGraph, Weights, IngestReport]:
    sd = load_state_dict(data)
    rep = IngestReport("pytorch_state_dict")
    groups: List[Tuple[str, np.ndarray, Optional[np.ndarray]]] = []
    bn: Dict[str, Dict[str, np.ndarray]] = {}
    for key, arr in sd.items():                       # state_dict order == module registration order
        mod, _, param = key.rpartition(".")
        if param == "weight" and arr.ndim in (2, 4):
            groups.append((mod, arr, sd.get(f"{mod}.bias")))
        elif param in ("running_mean", "running_var") or (param in ("weight", "bias") and arr.ndim == 1
                                                          and f"{mod}.running_mean" in sd):
            bn.setdefault(mod, {})[param] = arr
    if not groups:
        raise IngestError("no Linear or Conv2d weights found in the state_dict")
    first = groups[0][1]
    if input_shape is None:
        if first.ndim == 4:
            raise IngestError("a convolutional state_dict needs the input shape [C, H, W] (it is not stored in weights)")
        input_shape = (first.shape[1],)
    cb = _ChainBuilder(input_shape)
    for idx, (mod, W, b) in enumerate(groups):
        act = "linear" if idx == len(groups) - 1 else "relu"
        if W.ndim == 4:
            k = W.shape[2]
            cb.add_conv(mod, W, b, 1, k // 2, act, None)
        else:
            cb.add_dense(mod, W, b, act)
    rep.assumptions.append("a state_dict stores weights but not the forward pass: ReLU after every layer except "
                           "the last, stride 1 and 'same' padding for convolutions were assumed. Upload ONNX for exact structure.")
    if bn:
        rep.assumptions.append(f"{len(bn)} BatchNorm layer(s) found but not folded (their position is unknown in a state_dict)")
    return _finish(cb, name, base_accuracy, rep)


# ---------------------------------------------------------------------------

def _finish(cb: _ChainBuilder, name: str, base_accuracy: float, rep: IngestReport):
    model = cb.build(name, base_accuracy)
    rep.layers, rep.params, rep.macs = model.n, model.total_params(), model.total_macs()
    for w in cb.weights.values():
        if not np.all(np.isfinite(w[0])):
            raise IngestError("weights contain NaN or infinity")
    return model, cb.weights, rep


def ingest(filename: str, data: bytes, base_accuracy: float = 90.0,
           input_shape: Optional[Tuple[int, ...]] = None, name: Optional[str] = None):
    if len(data) > MAX_UPLOAD_BYTES:
        raise IngestError(f"file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    if not 0 < base_accuracy <= 100:
        raise IngestError("baseline accuracy must be between 0 and 100")
    stem = (name or filename.rsplit(".", 1)[0] or "uploaded_model")[:48]
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext == "onnx":
        return parse_onnx(data, stem, base_accuracy, input_shape)
    if ext == "json":
        return parse_json_graph(data, stem, base_accuracy)
    if ext in ("pt", "pth"):
        return parse_state_dict(data, stem, base_accuracy, input_shape)
    raise IngestError(f"unsupported file type '.{ext}' (use .onnx, .pt/.pth or .json)")
