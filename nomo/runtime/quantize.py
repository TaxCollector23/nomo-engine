"""Quantisation planner: (model, float weights, genome, calibration data) -> QGraph.

Scale alignment across crossings (SPEC §6):

  int8 edge e              s_e = A_e / 127,   A_e = percentile_{99.99}(|a_e|) on calibration data
  ANN layer                W_q = round(W / s_w), s_w = max|W| / (2^(b-1) - 1)
                           b_q = round(b / (s_in s_w)),  M = s_in s_w / s_out  -> (m0, shift)
  encoder (int8 -> spikes) theta_enc = round(A_e / s_e) = 127, one spike carries lambda_in = theta_enc s_e
  rate-coded LIF layer     data-based normalisation (Rueckauer et al. 2017): lambda_out = A_{e+1}
                           s_v = max(max|W| lambda_in / (2^(b-1) - 1),  lambda_out / 2^(v_bits - 2))
                           W_q = round(W lambda_in / s_v), b_q = round(b / s_v), theta = round(lambda_out / s_v)
                           so that E[rate_out] ~= ReLU(W x + b) / lambda_out
  decoder (spikes -> int8) M = lambda / (T s_out);   decoder (spikes -> Q16.16) k = q16(lambda / T)
  int8 -> Q16.16           k = q16(s_in);   Q16.16 -> int8   M = 1 / (2^16 s_out)

The integer-only C11 backend supports: ANN a_bits = 8, w_bits <= 8; SNN rate coding,
w_bits <= 8, v_bits <= 24. TTFS and 16-bit activations are exported to NIR (metadata)
but rejected here with an explicit error rather than silently approximated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..ir import ModelGraph
from ..search.genome import Coding, Domain, Genome
from .qformat import q16, quantize_multiplier, sat
from .qgraph import (QDecoder, QDense, QEncoder, QFromQ16, QGraph, QGuard, QLIF, QSymLinear, QToQ16)

Weights = Dict[str, Tuple[np.ndarray, np.ndarray]]


class UnsupportedLowering(NotImplementedError):
    pass


def float_forward(model: ModelGraph, weights: Weights, genome: Genome, X: np.ndarray,
                  aux: Optional[np.ndarray] = None, apply_guards: bool = True) -> List[np.ndarray]:
    """Reference float64 forward. Returns activations per edge: acts[0] = input, acts[i+1] = out of layer i
    (after the guard if one sits on that edge). X: [B, in]."""
    acts = [np.asarray(X, np.float64)]
    x = acts[0]
    for i, (spec, gene) in enumerate(zip(model.layers, genome.layers)):
        if gene.domain == Domain.SYM:
            x = x @ model.substitutes[spec.symbolic_substitute].phi().T
        else:
            W, b = weights[spec.name]
            x = x @ W.T + b
            if spec.activation == "relu":
                x = np.maximum(x, 0.0)
        site = model.guard_at(i)
        if site is not None and apply_guards:
            con = model.constraints[site.constraint_id]
            x = np.stack([con.forward_float(row, None if aux is None else aux[k]) for k, row in enumerate(x)])
        acts.append(x)
    return acts


@dataclass
class CompileOptions:
    percentile: float = 99.99
    leak_shift: int = 0                  # 0: integrate-and-fire (exact rate conversion); k>0: LIF with tau = 2^k dt


def _amax(a: np.ndarray, pct: float) -> float:
    return float(max(np.percentile(np.abs(a), pct), 1e-8))


def _wq(W: np.ndarray, scale: float, bits: int) -> np.ndarray:
    q = (1 << (bits - 1)) - 1
    return np.clip(np.round(W / scale), -q, q).astype(np.int8)


def compile_qgraph(model: ModelGraph, weights: Weights, genome: Genome, calib_X: np.ndarray,
                   calib_aux: Optional[np.ndarray] = None, opts: Optional[CompileOptions] = None) -> QGraph:
    opts = opts or CompileOptions()
    acts = float_forward(model, weights, genome, calib_X, calib_aux, apply_guards=True)
    pre = float_forward(model, weights, genome, calib_X, calib_aux, apply_guards=False)
    A = [_amax(a, opts.percentile) for a in acts]
    s = [a / 127.0 for a in A]

    stages: list = []
    kind, lam, T_cur = "i8", 0.0, 0          # current edge representation

    def to_i8(edge: int, tag: str) -> None:
        nonlocal kind
        if kind == "spk":
            m0, sh = quantize_multiplier(lam / (T_cur * s[edge]))
            stages.append(QDecoder(f"{tag}.decode", T_cur, model.edge_size(edge), "i8", m0, sh, lam=lam, out_scale=s[edge]))
        elif kind == "q16":
            m0, sh = quantize_multiplier(1.0 / (65536.0 * s[edge]))
            stages.append(QFromQ16(f"{tag}.from_q16", m0, sh, model.edge_size(edge), out_scale=s[edge]))
        kind = "i8"

    def to_q16(edge: int, tag: str) -> None:
        nonlocal kind
        if kind == "i8":
            stages.append(QToQ16(f"{tag}.to_q16", q16(s[edge]), model.edge_size(edge), in_scale=s[edge]))
        elif kind == "spk":
            stages.append(QDecoder(f"{tag}.decode", T_cur, model.edge_size(edge), "q16", k_q16=q16(lam / T_cur), lam=lam))
        kind = "q16"

    for i, (spec, gene) in enumerate(zip(model.layers, genome.layers)):
        if gene.domain == Domain.ANN:
            if gene.a_bits != 8 or gene.w_bits > 8:
                raise UnsupportedLowering(f"{spec.name}: C11 backend lowers ANN a_bits=8, w_bits<=8 (got {gene.w_bits}/{gene.a_bits})")
            to_i8(i, spec.name)
            W, b = weights[spec.name]
            s_w = max(float(np.abs(W).max()), 1e-12) / ((1 << (gene.w_bits - 1)) - 1)
            Wq = _wq(W, s_w, gene.w_bits)
            bq = np.round(b / (s[i] * s_w)).astype(np.int64)
            bq = np.clip(bq, -(1 << 31), (1 << 31) - 1).astype(np.int32)
            m0, sh = quantize_multiplier(s[i] * s_w / s[i + 1])
            stages.append(QDense(spec.name, Wq, bq, m0, sh, spec.activation == "relu", gene.w_bits, s_w, s[i], s[i + 1]))

        elif gene.domain == Domain.SNN:
            if gene.coding != Coding.RATE:
                raise UnsupportedLowering(f"{spec.name}: C11 backend lowers rate-coded LIF only (TTFS is NIR-export only)")
            if gene.w_bits > 8 or gene.a_bits > 24:
                raise UnsupportedLowering(f"{spec.name}: w_bits<=8 and membrane bits<=24 required")
            if kind != "spk":
                to_i8(i, spec.name)
                theta_enc = max(1, int(round(A[i] / s[i])))
                lam, T_cur = theta_enc * s[i], gene.timesteps
                stages.append(QEncoder(f"{spec.name}.encode", theta_enc, T_cur, model.edge_size(i), s[i], lam))
                kind = "spk"
            elif gene.timesteps != T_cur:
                raise UnsupportedLowering("SNN segment with mixed T (repair should have prevented this)")
            W, b = weights[spec.name]
            lam_out = _amax(pre[i + 1], opts.percentile)
            if gene.w_bits == 1:                  # binary weights {-1, +1}, magnitude = mean |W|
                s_v = max(float(np.abs(W).mean()) * lam, lam_out / (1 << (gene.a_bits - 2)), 1e-12)
                Wq = np.where(W >= 0, 1, -1).astype(np.int8)
            else:
                qmax = (1 << (gene.w_bits - 1)) - 1
                s_v = max(float(np.abs(W).max()) * lam / qmax, lam_out / (1 << (gene.a_bits - 2)), 1e-12)
                Wq = np.clip(np.round(W * lam / s_v), -qmax, qmax).astype(np.int8)
            bq = np.round(b / s_v).astype(np.int32)
            theta = max(1, int(round(lam_out / s_v)))
            stages.append(QLIF(spec.name, Wq, bq, theta, opts.leak_shift, gene.a_bits, T_cur, gene.w_bits,
                               s_v, lam, lam_out, gene.plastic))
            lam = lam_out

        else:  # SYM substitute
            to_q16(i, spec.name)
            sub = model.substitutes[spec.symbolic_substitute]
            stages.append(QSymLinear(spec.name, sub.phi_q16(), sub.id))

        site = model.guard_at(i)
        if site is not None:
            to_q16(i + 1, f"guard{i}")
            con = model.constraints[site.constraint_id]
            stages.append(QGuard(f"guard:{con.id}", con, model.layers[i].out_neurons, con.n_aux))

    to_q16(model.n, "output")
    n_aux = max([st.n_aux for st in stages if isinstance(st, QGuard)] or [0])
    return QGraph(model.name, stages, model.input_size(), s[0], model.layers[-1].out_neurons, n_aux,
                  meta={"edge_scales": s, "edge_amax": A, "genome": genome.key, "leak_shift": opts.leak_shift})


def quantize_input(qg: QGraph, x: np.ndarray) -> np.ndarray:
    return np.array([sat(int(np.floor(v / qg.in_scale + 0.5)), 8) for v in np.asarray(x, float)], dtype=np.int64)


def quantize_aux(aux: Sequence[float]) -> List[int]:
    return [q16(float(a)) for a in aux]
