"""Measured cost look-up table.

Records come from on-silicon microbenchmarks (one layer or one crossing kernel,
swept over size). For each key k = (platform, kernel, domain, w_bits, a_bits, T)
the LUT stores samples (s_i, E_i, L_i) where s is the work size (MACs, synaptic
ops, or values converted). Queries use:

  * piecewise-linear interpolation in (log s, log E) and (log s, log L) inside the
    sampled range, and
  * a least-squares power law  log E = log a + b log s  outside it,

so the returned cost is monotone in s whenever the samples are. When a key has no
samples the caller falls back to the analytic profile; `source` reports which path
was taken so that telemetry can show how much of a candidate's cost is measured.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

LUTKey = Tuple[str, str, str, int, int, int]   # platform, kernel, domain, w_bits, a_bits, T


@dataclass
class _Series:
    log_s: np.ndarray
    log_e: np.ndarray
    log_l: np.ndarray
    fit_e: Tuple[float, float]     # (log a, b)
    fit_l: Tuple[float, float]


@dataclass(frozen=True)
class LUTHit:
    energy_j: float
    latency_s: float
    source: str                    # "lut" | "lut-extrapolated"


def _fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) == 1:
        return float(y[0] - x[0]), 1.0          # assume linear scaling through the single sample
    A = np.stack([np.ones_like(x), x], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


class CostLUT:
    def __init__(self) -> None:
        self._raw: Dict[LUTKey, List[Tuple[float, float, float]]] = {}
        self._series: Dict[LUTKey, _Series] = {}

    # ---------------- ingestion
    def add(self, key: LUTKey, size: float, energy_j: float, latency_s: float) -> None:
        if size <= 0 or energy_j <= 0 or latency_s <= 0:
            raise ValueError(f"LUT sample must be positive, got size={size} E={energy_j} L={latency_s}")
        self._raw.setdefault(key, []).append((float(size), float(energy_j), float(latency_s)))
        self._series.pop(key, None)

    def extend(self, rows: Iterable[dict]) -> "CostLUT":
        for r in rows:
            key = (r["platform"], r["kernel"], r["domain"], int(r["w_bits"]), int(r["a_bits"]), int(r.get("T", 0)))
            self.add(key, float(r["size"]), float(r["energy_j"]), float(r["latency_s"]))
        return self

    @classmethod
    def from_csv(cls, path: str | Path) -> "CostLUT":
        with open(path, newline="") as fh:
            return cls().extend(csv.DictReader(fh))

    @classmethod
    def from_json(cls, path: str | Path) -> "CostLUT":
        return cls().extend(json.loads(Path(path).read_text()))

    def __len__(self) -> int:
        return sum(len(v) for v in self._raw.values())

    # ---------------- query
    def _build(self, key: LUTKey) -> Optional[_Series]:
        if key in self._series:
            return self._series[key]
        raw = self._raw.get(key)
        if not raw:
            return None
        # aggregate repeated sizes by median (robust to measurement noise)
        by_size: Dict[float, List[Tuple[float, float]]] = {}
        for s, e, l in raw:
            by_size.setdefault(s, []).append((e, l))
        sizes = np.array(sorted(by_size))
        E = np.array([np.median([x[0] for x in by_size[s]]) for s in sizes])
        L = np.array([np.median([x[1] for x in by_size[s]]) for s in sizes])
        ls, le, ll = np.log(sizes), np.log(E), np.log(L)
        ser = _Series(ls, le, ll, _fit(ls, le), _fit(ls, ll))
        self._series[key] = ser
        return ser

    def query(self, key: LUTKey, size: float) -> Optional[LUTHit]:
        ser = self._build(key)
        if ser is None or size <= 0:
            return None
        x = np.log(size)
        if ser.log_s[0] <= x <= ser.log_s[-1]:
            e = np.interp(x, ser.log_s, ser.log_e)
            l = np.interp(x, ser.log_s, ser.log_l)
            return LUTHit(float(np.exp(e)), float(np.exp(l)), "lut")
        e = ser.fit_e[0] + ser.fit_e[1] * x
        l = ser.fit_l[0] + ser.fit_l[1] * x
        return LUTHit(float(np.exp(e)), float(np.exp(l)), "lut-extrapolated")
