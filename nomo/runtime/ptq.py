"""Calibration-driven post-training quantisation utilities.

This module deliberately uses small, dependency-light numpy routines.  It is
used by the service endpoint as well as offline CLI runs, and its output is a
JSON-compatible report that travels with an export bundle.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CalibrationData:
    inputs: np.ndarray
    aux: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    source: str = "upload"

    @property
    def sample_count(self) -> int:
        return int(self.inputs.shape[0])

    def to_dict(self) -> Dict[str, Any]:
        return {"sample_count": self.sample_count, "input_shape": list(self.inputs.shape[1:]),
                "aux_shape": list(self.aux.shape[1:]) if self.aux is not None else None,
                "has_labels": self.labels is not None, "source": self.source}


def _as_float_array(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"calibration field '{name}' must be a finite tensor batch")
    return arr


def parse_calibration_bytes(filename: str, data: bytes, input_shape: Sequence[int],
                            n_aux: int = 0, min_samples: int = 100, max_samples: int = 500) -> CalibrationData:
    """Parse `.npz`, `.npy`, or JSON calibration tensors without pickle loading."""
    if len(data) > 200 * 1024 * 1024:
        raise ValueError("calibration upload is larger than 200 MB")
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    labels = None
    aux = None
    try:
        if ext == "npz":
            with np.load(io.BytesIO(data), allow_pickle=False) as z:
                key = next((k for k in ("inputs", "X", "x", "data") if k in z.files), None)
                if key is None:
                    raise ValueError(".npz calibration needs an 'inputs' (or X/x/data) array")
                inputs = _as_float_array(z[key], "inputs")
                aux_key = next((k for k in ("aux", "A", "metadata") if k in z.files), None)
                aux = _as_float_array(z[aux_key], "aux") if aux_key else None
                if "labels" in z.files:
                    labels = np.asarray(z["labels"])
        elif ext == "npy":
            inputs = _as_float_array(np.load(io.BytesIO(data), allow_pickle=False), "inputs")
        elif ext in ("json", "jsn"):
            doc = json.loads(data.decode("utf-8"))
            if isinstance(doc, list):
                inputs = _as_float_array(doc, "inputs")
            else:
                inputs = _as_float_array(doc.get("inputs", doc.get("X", doc.get("x"))), "inputs")
                if doc.get("aux") is not None:
                    aux = _as_float_array(doc["aux"], "aux")
                if doc.get("labels") is not None:
                    labels = np.asarray(doc["labels"])
        else:
            raise ValueError("unsupported calibration format; use .npz, .npy, or .json")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(".npz"):
            raise
        raise ValueError(f"could not read calibration data: {exc}") from exc

    expected = tuple(int(v) for v in input_shape)
    if inputs.ndim < 2 or tuple(inputs.shape[1:]) != expected:
        # A flat [N, prod(input_shape)] upload is safe to reshape and useful for
        # tabular/MLP models; spatial tensors must retain their explicit shape.
        if inputs.ndim == 2 and int(np.prod(expected)) == inputs.shape[1]:
            inputs = inputs.reshape((inputs.shape[0],) + expected)
        else:
            raise ValueError(f"calibration inputs have shape {list(inputs.shape[1:])}; expected {list(expected)}")
    n = int(inputs.shape[0])
    if n < min_samples or n > max_samples:
        raise ValueError(f"calibration sample count must be between {min_samples} and {max_samples} (got {n})")
    if n_aux:
        if aux is None or aux.ndim != 2 or aux.shape != (n, n_aux):
            raise ValueError(f"calibration aux must have shape [{n}, {n_aux}]")
    elif aux is not None and (aux.ndim != 2 or aux.shape[0] != n):
        raise ValueError(f"calibration aux must have {n} rows")
    if labels is not None and len(labels) != n:
        raise ValueError("calibration labels must have one value per input")
    return CalibrationData(inputs, aux, labels, source=f"upload:{filename}")


def _symmetric_quant_error(values: np.ndarray, threshold: float, bits: int) -> float:
    qmax = max(1, (1 << (max(2, int(bits)) - 1)) - 1)
    scale = max(float(threshold), 1e-12) / qmax
    clipped = np.clip(values, -threshold, threshold)
    rounded = np.floor(clipped / scale + 0.5) * scale
    return float(np.mean((values - rounded) ** 2))


def _candidate_thresholds(values: np.ndarray, bins: int = 128) -> np.ndarray:
    abs_v = np.abs(values)
    hi = max(float(abs_v.max(initial=0.0)), 1e-8)
    percentiles = np.array([90, 95, 97, 98, 99, 99.5, 99.9, 99.99, 100], dtype=float)
    pct = np.percentile(abs_v, percentiles)
    return np.unique(np.maximum(pct, hi / bins))


def _mse_threshold(values: np.ndarray, bits: int) -> float:
    candidates = _candidate_thresholds(values)
    errors = [_symmetric_quant_error(values, float(t), bits) for t in candidates]
    return float(candidates[int(np.argmin(errors))])


def _kl_threshold(values: np.ndarray, bits: int) -> float:
    """Choose a clipping range by KL divergence over an absolute histogram."""
    flat = np.abs(values).ravel()
    hi = max(float(flat.max(initial=0.0)), 1e-8)
    hist, edges = np.histogram(flat, bins=256, range=(0.0, hi))
    hist = hist.astype(np.float64)
    if hist.sum() == 0 or np.count_nonzero(hist) < 2:
        return hi
    # Quantised bins are deliberately larger than the 256-bin source histogram.
    qbins = max(2, min(128, 1 << max(2, int(bits))))
    best_t, best_kl = hi, float("inf")
    for end in range(max(qbins, 16), len(hist) + 1, 8):
        p = hist[:end].copy()
        tail = hist[end:].sum()
        p[-1] += tail
        if p.sum() == 0:
            continue
        # Merge source bins into qbins, then expand back to the source shape.
        q = np.zeros_like(p)
        for i, count in enumerate(p):
            j = min(qbins - 1, int(i * qbins / len(p)))
            q[j * len(p) // qbins:(j + 1) * len(p) // qbins] += count
        q = q / max(q.sum(), 1.0)
        p = p / max(p.sum(), 1.0)
        mask = (p > 0) & (q > 0)
        kl = float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
        if kl < best_kl:
            best_kl, best_t = kl, float(edges[min(end, len(edges) - 1)])
    return max(best_t, 1e-8)


def calibrate(model: Any, weights: Dict[str, Tuple[np.ndarray, np.ndarray]], genome: Any,
              inputs: np.ndarray, aux: Optional[np.ndarray] = None, *, method: str = "mse") -> Dict[str, Any]:
    """Calculate activation ranges from real calibration tensors.

    The returned edge ranges can be passed to ``CompileOptions.activation_amax``
    and are intentionally independent of the accuracy proxy used by search.
    """
    from .quantize import float_forward

    X = np.asarray(inputs, dtype=np.float64)
    acts = float_forward(model, weights, genome, X, aux, apply_guards=True)
    selected = method.lower()
    if selected not in ("mse", "kl", "both"):
        raise ValueError("PTQ method must be 'mse', 'kl', or 'both'")
    edges = []
    ranges = []
    for i, values in enumerate(acts):
        v = np.asarray(values, dtype=np.float64)
        bits = 8
        if i > 0 and i - 1 < len(genome.layers):
            bits = int(getattr(genome.layers[i - 1], "a_bits", 8) or 8)
        raw = max(float(np.max(np.abs(v))), 1e-8)
        mse = _mse_threshold(v, bits)
        kl = _kl_threshold(v, bits)
        if selected == "mse":
            chosen, chosen_by = mse, "mse"
        elif selected == "kl":
            chosen, chosen_by = kl, "kl"
        else:
            # Select whichever objective actually reconstructs this tensor best.
            chosen = mse if _symmetric_quant_error(v, mse, bits) <= _symmetric_quant_error(v, kl, bits) else kl
            chosen_by = "mse" if chosen == mse else "kl"
        ranges.append(float(chosen))
        edges.append({"edge": i, "bits": bits, "raw_amax": raw,
                      "percentile_amax": float(np.percentile(np.abs(v), 99.99)),
                      "mse_amax": mse, "kl_amax": kl, "selected_amax": float(chosen),
                      "selected_by": chosen_by, "values": int(v.size)})
    return {"format": "nomo.ptq/1", "method": selected, "sample_count": int(X.shape[0]),
            "input_shape": list(X.shape[1:]), "has_aux": aux is not None,
            "edge_amax": ranges, "edges": edges, "source": "calibration_data"}
