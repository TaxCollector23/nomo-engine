"""Integer arithmetic primitives. Each function has a byte-for-byte C11 twin in
export/c11.py (NOMO_* helpers). Python ints are unbounded, so every helper that
models an int32/int64 register checks its range and raises instead of silently
diverging from the C semantics.
"""
from __future__ import annotations

import math
from typing import Tuple

INT8_MIN, INT8_MAX = -128, 127
INT16_MIN, INT16_MAX = -(1 << 15), (1 << 15) - 1
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1
INT64_MIN, INT64_MAX = -(1 << 63), (1 << 63) - 1
Q16_ONE = 1 << 16


def check64(v: int) -> int:
    if not INT64_MIN <= v <= INT64_MAX:
        raise OverflowError(f"int64 overflow in fixed-point kernel: {v}")
    return v


def sat(v: int, bits: int) -> int:
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return lo if v < lo else hi if v > hi else v


def sat8(v: int) -> int:
    return sat(v, 8)


def sat32(v: int) -> int:
    return sat(v, 32)


def asr(v: int, n: int) -> int:
    """Arithmetic shift right (floor division by 2^n). Python >> already floors."""
    return v >> n


def rshift_round(v: int, n: int) -> int:
    """Round-half-up right shift: floor((v + 2^(n-1)) / 2^n)."""
    if n <= 0:
        return check64(v << (-n))
    return check64(v + (1 << (n - 1))) >> n


def q16(x: float) -> int:
    return sat32(int(math.floor(x * Q16_ONE + 0.5)))


def q16mul(a: int, b: int) -> int:
    return sat32(rshift_round(check64(a * b), 16))


def quantize_multiplier(m: float) -> Tuple[int, int]:
    """Represent a positive real multiplier as m ~= m0 * 2^-(31 + shift), m0 in [2^30, 2^31).

    `shift` may be negative (multiplier >= 1) down to -30.
    """
    if m <= 0:
        raise ValueError(f"multiplier must be positive, got {m}")
    frac, exp = math.frexp(m)                 # m = frac * 2^exp, frac in [0.5, 1)
    m0 = int(round(frac * (1 << 31)))
    if m0 == (1 << 31):
        m0 //= 2
        exp += 1
    shift = -exp
    if shift < -30:
        raise ValueError(f"multiplier {m} too large for Q31 representation")
    if shift > 31:
        return 0, 0                           # underflows to zero
    return m0, shift


def requant(acc: int, m0: int, shift: int) -> int:
    """round(acc * m0 * 2^-(31+shift)) with int64 intermediate, as in the C kernel."""
    return rshift_round(check64(acc * m0), 31 + shift)
