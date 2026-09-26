"""QGraph: the fully-quantised, integer-only form of a mapped genome.

This is the single source of truth for deployed semantics. Three consumers must agree
bit for bit: this golden interpreter, the C++ kernel (csrc/lif_kernel.cpp), and the
emitted C11 code (export/c11.py). tests/test_c11_bitexact.py enforces it.

Value kinds on edges:
    i8   int8 vector (continuous activations, symmetric, zero-point 0)
    spk  int8 [T, N] spike raster: {0,1} from LIF, {-1,0,1} from the signed encoder
    q16  int32 Q16.16 vector (symbolic domain)

Stage semantics (per inference, all state reset to zero at frame start):

  QDense      acc = W x + b (int32/64);  y = sat8(requant(acc, m0, sh)); relu -> max(y, 0)
  QEncoder    signed sigma-delta, per step t:  a += x;  s = +1 if a >= th (a -= th),
              -1 if a <= -th (a += th), else 0
  QLIF        per step t:  v -= asr(v, k)  (k = 0: no leak);  v = sat_vbits(v + W s_t + b);
              z = v >= th;  v -= th * z
  QDecoder    c = sum_t z_t;  i8 out: sat8(requant(c, m0, sh));  q16 out: sat32(c * k_q16)
  QToQ16      y = sat32(x * k_q16)                                    (i8 -> q16)
  QFromQ16    y = sat8(requant(x, m0, sh))                            (q16 -> i8)
  QSymLinear  y = sat32(rshift_round(Phi_q16 x_q16, 16))              (q16 -> q16)
  QGuard      y = constraint.forward_q16(x, aux)                       (q16 -> q16)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np

from .qformat import INT8_MAX, INT8_MIN, INT32_MAX, INT32_MIN, check64, requant, rshift_round, sat, sat32

try:                                     # optional C++ acceleration, identical semantics
    from nomo import _core as _cxx       # type: ignore
except Exception:                        # pragma: no cover - exercised when the extension is not built
    _cxx = None


@dataclass
class QDense:
    name: str
    W: np.ndarray                        # int8 [No, Ni]
    b: np.ndarray                        # int32 [No]
    m0: int
    shift: int
    relu: bool
    w_bits: int = 8
    w_scale: float = 1.0                 # real = int * scale (for export metadata)
    in_scale: float = 1.0
    out_scale: float = 1.0
    kind: str = "dense"


@dataclass
class QEncoder:
    name: str
    theta: int
    T: int
    n: int
    in_scale: float = 1.0
    lam: float = 1.0                     # real value represented by one spike
    kind: str = "encoder"


@dataclass
class QLIF:
    name: str
    W: np.ndarray                        # int8 [No, Ni], values within w_bits
    b: np.ndarray                        # int32 [No], injected every step
    theta: int
    leak_shift: int
    v_bits: int
    T: int
    w_bits: int = 4
    v_scale: float = 1.0                 # real membrane units per integer LSB
    lam_in: float = 1.0
    lam_out: float = 1.0
    plastic: bool = False
    kind: str = "lif"


@dataclass
class QDecoder:
    name: str
    T: int
    n: int
    out: str                             # "i8" | "q16"
    m0: int = 0
    shift: int = 0
    k_q16: int = 0
    lam: float = 1.0
    out_scale: float = 1.0
    kind: str = "decoder"


@dataclass
class QToQ16:
    name: str
    k_q16: int
    n: int
    in_scale: float = 1.0
    kind: str = "to_q16"


@dataclass
class QFromQ16:
    name: str
    m0: int
    shift: int
    n: int
    out_scale: float = 1.0
    kind: str = "from_q16"


@dataclass
class QSymLinear:
    name: str
    Phi: np.ndarray                      # int64 Q16.16 [No, Ni]
    substitute_id: str = ""
    kind: str = "sym_linear"


@dataclass
class QGuard:
    name: str
    constraint: object                   # SymbolicConstraint
    n: int
    n_aux: int
    kind: str = "guard"


Stage = Union[QDense, QEncoder, QLIF, QDecoder, QToQ16, QFromQ16, QSymLinear, QGuard]


@dataclass
class QGraph:
    name: str
    stages: List[Stage]
    in_size: int
    in_scale: float
    out_size: int
    n_aux: int = 0
    meta: dict = field(default_factory=dict)

    def out_kind(self) -> str:
        return "q16"                     # the compiled graph always returns Q16.16 (see finalize in quantize.py)


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------

def lif_numpy(W: np.ndarray, b: np.ndarray, s_in: np.ndarray, theta: int, leak_shift: int, v_bits: int) -> np.ndarray:
    T = s_in.shape[0]
    Wl, bl = W.astype(np.int64), b.astype(np.int64)
    v = np.zeros(W.shape[0], dtype=np.int64)
    out = np.zeros((T, W.shape[0]), dtype=np.int8)
    lo, hi = -(1 << (v_bits - 1)), (1 << (v_bits - 1)) - 1
    for t in range(T):
        if leak_shift > 0:
            v = v - (v >> leak_shift)                     # numpy >> on int64 is arithmetic
        inc = Wl @ s_in[t].astype(np.int64) + bl
        if np.abs(inc).max(initial=0) > (1 << 31) - 1:        # C kernel accumulates in int32
            raise OverflowError("LIF synaptic accumulator overflow")
        v = np.clip(v + inc, lo, hi)
        z = v >= theta
        v = v - theta * z
        out[t] = z
    return out


def lif(W, b, s_in, theta, leak_shift, v_bits) -> np.ndarray:
    if _cxx is not None:
        return np.asarray(_cxx.lif_layer(np.ascontiguousarray(W, np.int8), np.ascontiguousarray(b, np.int32),
                                         np.ascontiguousarray(s_in, np.int8), int(theta), int(leak_shift), int(v_bits)))
    return lif_numpy(W, b, s_in, theta, leak_shift, v_bits)


def encode(x: np.ndarray, theta: int, T: int) -> np.ndarray:
    a = np.zeros_like(x, dtype=np.int64)
    out = np.zeros((T, len(x)), dtype=np.int8)
    xl = x.astype(np.int64)
    for t in range(T):
        a = a + xl
        pos = a >= theta
        neg = a <= -theta
        a = a - theta * pos + theta * neg
        out[t] = pos.astype(np.int8) - neg.astype(np.int8)
    return out


def _requant_vec(v: Sequence[int], m0: int, shift: int) -> np.ndarray:
    return np.array([sat(requant(int(x), m0, shift), 8) for x in v], dtype=np.int64)


# ---------------------------------------------------------------------------
# golden interpreter
# ---------------------------------------------------------------------------

def run(qg: QGraph, x_i8: np.ndarray, aux_q16: Optional[Sequence[int]] = None, trace: Optional[list] = None) -> np.ndarray:
    kind, val = "i8", np.asarray(x_i8, dtype=np.int64)
    if val.min() < INT8_MIN or val.max() > INT8_MAX:
        raise ValueError("input must be int8")
    for st in qg.stages:
        if isinstance(st, QDense):
            assert kind == "i8", st.name
            acc = st.W.astype(np.int64) @ val + st.b.astype(np.int64)
            if acc.min() < INT32_MIN or acc.max() > INT32_MAX:      # C kernel accumulates in int32
                raise OverflowError(f"{st.name}: int32 accumulator overflow")
            y = _requant_vec(acc, st.m0, st.shift)
            val = np.maximum(y, 0) if st.relu else y
        elif isinstance(st, QEncoder):
            assert kind == "i8", st.name
            val, kind = encode(val, st.theta, st.T).astype(np.int64), "spk"
        elif isinstance(st, QLIF):
            assert kind == "spk", st.name
            val = lif(st.W, st.b, val.astype(np.int8), st.theta, st.leak_shift, st.v_bits).astype(np.int64)
        elif isinstance(st, QDecoder):
            assert kind == "spk", st.name
            c = val.sum(axis=0)
            if st.out == "i8":
                val, kind = _requant_vec(c, st.m0, st.shift), "i8"
            else:
                val, kind = np.array([sat32(int(ci) * st.k_q16) for ci in c], dtype=np.int64), "q16"
        elif isinstance(st, QToQ16):
            assert kind == "i8", st.name
            val, kind = np.array([sat32(int(x) * st.k_q16) for x in val], dtype=np.int64), "q16"
        elif isinstance(st, QFromQ16):
            assert kind == "q16", st.name
            val, kind = _requant_vec(val, st.m0, st.shift), "i8"
        elif isinstance(st, QSymLinear):
            assert kind == "q16", st.name
            out = []
            for r in range(st.Phi.shape[0]):
                acc = 0
                for c_ in range(st.Phi.shape[1]):
                    acc = check64(acc + int(st.Phi[r, c_]) * int(val[c_]))
                out.append(sat32(rshift_round(acc, 16)))
            val = np.array(out, dtype=np.int64)
        elif isinstance(st, QGuard):
            assert kind == "q16", st.name
            val = np.array(st.constraint.forward_q16(val.tolist(), aux_q16), dtype=np.int64)
        else:  # pragma: no cover
            raise TypeError(type(st))
        if trace is not None:
            trace.append((st.name, kind, val.copy()))
    assert kind == "q16", "graph must terminate in Q16.16 (quantize.finalize appends the conversion)"
    return val
