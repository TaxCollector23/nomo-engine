"""C11 bare-metal backend.

Emits `nomo_model.h` / `nomo_model.c`: integer-only, no heap, no floating point, no libc
beyond <stdint.h> and <string.h> (memset). Arithmetic helpers are the exact twins of
runtime/qformat.py; `asr` is written portably because `>>` on negative signed values is
implementation-defined in C11 (6.5.7p5).

Memory model: every stage owns a static output buffer (SSA-style). Peak RAM is therefore
the sum of buffer sizes, reported in the manifest. A liveness-based buffer allocator is on
the roadmap (SPEC §8.4); for the chain-structured graphs lowered today two ping-pong buffers
per value kind would suffice.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..runtime.qgraph import (QDecoder, QDense, QEncoder, QFromQ16, QGraph, QGuard, QLIF, QSymLinear,
                              QToQ16, run)
from ..symbolic.constraints import BoxBound, RotationalRateBound, ThrustLimit, rot_params_q16
from ..runtime.qformat import q16

HELPERS = r"""
static inline int64_t nomo_asr(int64_t v, int n) { return v < 0 ? ~((~v) >> n) : (v >> n); }
static inline int64_t nomo_rsr(int64_t v, int n) {
  if (n <= 0) return v * ((int64_t)1 << (-n));
  return nomo_asr(v + ((int64_t)1 << (n - 1)), n);
}
static inline int64_t nomo_clamp(int64_t v, int64_t lo, int64_t hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline int8_t nomo_sat8(int64_t v) { return (int8_t)nomo_clamp(v, -128, 127); }
static inline int32_t nomo_sat32(int64_t v) { return (int32_t)nomo_clamp(v, INT32_MIN, INT32_MAX); }
static inline int32_t nomo_q16mul(int64_t a, int64_t b) { return nomo_sat32(nomo_rsr(a * b, 16)); }
"""


def _lit(v: int) -> str:
    v = int(v)
    if v == -(1 << 31):
        return "INT32_MIN"
    if v == -(1 << 63):
        return "INT64_MIN"
    return f"{v}LL" if abs(v) > (1 << 31) - 1 else str(v)


def _array(ctype: str, name: str, data: Sequence[int], per_line: int = 24) -> str:
    vals = [_lit(x) for x in np.asarray(data).ravel().tolist()]
    lines = [", ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line)]
    return f"static const {ctype} {name}[{len(vals)}] = {{\n  " + ",\n  ".join(lines) + "\n};\n"


@dataclass
class CEmission:
    header: str
    source: str
    manifest: Dict[str, object]

    def write(self, out_dir: str | Path) -> List[Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = [out / "nomo_model.h", out / "nomo_model.c", out / "nomo_model.manifest.json"]
        paths[0].write_text(self.header)
        paths[1].write_text(self.source)
        paths[2].write_text(json.dumps(self.manifest, indent=2))
        return paths


def emit_c11(qg: QGraph, prefix: str = "nomo") -> CEmission:
    P = prefix.upper()
    consts: List[str] = []
    bufs: List[str] = []
    body: List[str] = []
    ram = 0
    cur = "in"                     # name of current value buffer
    cur_kind = "i8"
    cur_n = qg.in_size

    for k, st in enumerate(qg.stages):
        tag = f"s{k}"
        body.append(f"  /* stage {k}: {st.kind} {st.name} */")
        if isinstance(st, QDense):
            no, ni = st.W.shape
            consts += [_array("int8_t", f"{tag}_W", st.W), _array("int32_t", f"{tag}_B", st.b)]
            bufs.append(f"static int8_t {tag}_out[{no}];")
            ram += no
            relu = f"\n      if (y < 0) y = 0;" if st.relu else ""
            body.append(f"""  for (int j = 0; j < {no}; ++j) {{
    int64_t acc = {tag}_B[j];
    const int8_t *w = &{tag}_W[j * {ni}];
    for (int i = 0; i < {ni}; ++i) acc += (int64_t)w[i] * (int64_t){cur}[i];
    {{
      int8_t y = nomo_sat8(nomo_rsr(acc * (int64_t){st.m0}, {31 + st.shift}));{relu}
      {tag}_out[j] = y;
    }}
  }}""")
            cur, cur_kind, cur_n = f"{tag}_out", "i8", no

        elif isinstance(st, QEncoder):
            n, T = st.n, st.T
            bufs += [f"static int8_t {tag}_out[{T * n}];", f"static int32_t {tag}_acc[{n}];"]
            ram += T * n + 4 * n
            body.append(f"""  memset({tag}_acc, 0, sizeof {tag}_acc);
  for (int t = 0; t < {T}; ++t) {{
    for (int i = 0; i < {n}; ++i) {{
      {tag}_acc[i] += {cur}[i];
      int8_t s = 0;
      if ({tag}_acc[i] >= {st.theta}) {{ s = 1; {tag}_acc[i] -= {st.theta}; }}
      else if ({tag}_acc[i] <= -{st.theta}) {{ s = -1; {tag}_acc[i] += {st.theta}; }}
      {tag}_out[t * {n} + i] = s;
    }}
  }}""")
            cur, cur_kind, cur_n = f"{tag}_out", "spk", n

        elif isinstance(st, QLIF):
            no, ni = st.W.shape
            T = st.T
            lo, hi = -(1 << (st.v_bits - 1)), (1 << (st.v_bits - 1)) - 1
            consts += [_array("int8_t", f"{tag}_W", st.W), _array("int32_t", f"{tag}_B", st.b)]
            bufs += [f"static int8_t {tag}_out[{T * no}];", f"static int64_t {tag}_v[{no}];"]
            ram += T * no + 8 * no
            leak = f"\n      vj -= nomo_asr(vj, {st.leak_shift});" if st.leak_shift > 0 else ""
            body.append(f"""  memset({tag}_v, 0, sizeof {tag}_v);
  for (int t = 0; t < {T}; ++t) {{
    const int8_t *si = &{cur}[t * {ni}];
    for (int j = 0; j < {no}; ++j) {{
      int64_t inc = {tag}_B[j];
      const int8_t *w = &{tag}_W[j * {ni}];
      for (int i = 0; i < {ni}; ++i) if (si[i]) inc += (int64_t)w[i] * (int64_t)si[i];
      int64_t vj = {tag}_v[j];{leak}
      vj = nomo_clamp(vj + inc, {_lit(lo)}, {_lit(hi)});
      {{
        const int8_t z = vj >= {st.theta} ? 1 : 0;
        vj -= (int64_t){st.theta} * z;
        {tag}_v[j] = vj;
        {tag}_out[t * {no} + j] = z;
      }}
    }}
  }}""")
            cur, cur_kind, cur_n = f"{tag}_out", "spk", no

        elif isinstance(st, QDecoder):
            n, T = st.n, st.T
            ctype = "int8_t" if st.out == "i8" else "int32_t"
            bufs.append(f"static {ctype} {tag}_out[{n}];")
            ram += n * (1 if st.out == "i8" else 4)
            conv = (f"nomo_sat8(nomo_rsr((int64_t)c * (int64_t){st.m0}, {31 + st.shift}))" if st.out == "i8"
                    else f"nomo_sat32((int64_t)c * (int64_t){st.k_q16})")
            body.append(f"""  for (int j = 0; j < {n}; ++j) {{
    int32_t c = 0;
    for (int t = 0; t < {T}; ++t) c += {cur}[t * {n} + j];
    {tag}_out[j] = {conv};
  }}""")
            cur, cur_kind, cur_n = f"{tag}_out", st.out, n

        elif isinstance(st, QToQ16):
            bufs.append(f"static int32_t {tag}_out[{st.n}];")
            ram += 4 * st.n
            body.append(f"  for (int i = 0; i < {st.n}; ++i) {tag}_out[i] = nomo_sat32((int64_t){cur}[i] * (int64_t){st.k_q16});")
            cur, cur_kind, cur_n = f"{tag}_out", "q16", st.n

        elif isinstance(st, QFromQ16):
            bufs.append(f"static int8_t {tag}_out[{st.n}];")
            ram += st.n
            body.append(f"  for (int i = 0; i < {st.n}; ++i) {tag}_out[i] = nomo_sat8(nomo_rsr((int64_t){cur}[i] * (int64_t){st.m0}, {31 + st.shift}));")
            cur, cur_kind, cur_n = f"{tag}_out", "i8", st.n

        elif isinstance(st, QSymLinear):
            no, ni = st.Phi.shape
            consts.append(_array("int32_t", f"{tag}_PHI", st.Phi))
            bufs.append(f"static int32_t {tag}_out[{no}];")
            ram += 4 * no
            body.append(f"""  for (int j = 0; j < {no}; ++j) {{
    int64_t acc = 0;
    for (int i = 0; i < {ni}; ++i) acc += (int64_t){tag}_PHI[j * {ni} + i] * (int64_t){cur}[i];
    {tag}_out[j] = nomo_sat32(nomo_rsr(acc, 16));
  }}""")
            cur, cur_kind, cur_n = f"{tag}_out", "q16", no

        elif isinstance(st, QGuard):
            n = st.n
            bufs.append(f"static int32_t {tag}_out[{n}];")
            ram += 4 * n
            body.append(f"  for (int i = 0; i < {n}; ++i) {tag}_out[i] = {cur}[i];")
            y = f"{tag}_out"
            for term in st.constraint.terms:
                if isinstance(term, BoxBound):
                    for c, lo, hi in zip(term.channels, term.lo, term.hi):
                        body.append(f"  {y}[{c}] = (int32_t)nomo_clamp({y}[{c}], {_lit(q16(lo))}, {_lit(q16(hi))});")
                elif isinstance(term, ThrustLimit):
                    body.append(f"""  {{ /* thrust: 0 <= T <= T_max * rho/rho0 */
    const int64_t cap = nomo_q16mul({q16(term.t_max_n)}, aux_q16[{term.aux_density_ratio}]);
    int64_t v = {y}[{term.channel}];
    if (v < 0) v = 0;
    if (v > cap) v = cap;
    {y}[{term.channel}] = (int32_t)v;
  }}""")
                elif isinstance(term, RotationalRateBound):
                    p = rot_params_q16(term)
                    I = p["I"]
                    w = [f"(int64_t)aux_q16[{i}]" for i in term.aux_omega]
                    body.append(f"""  {{ /* rotational-rate admissible torque: tau in [I(-wmax-w)/dt + g, I(wmax-w)/dt + g] cap [-tmax, tmax] */
    const int64_t w0 = {w[0]}, w1 = {w[1]}, w2 = {w[2]};
    const int64_t g[3] = {{ nomo_q16mul(nomo_q16mul({I[2] - I[1]}, w1), w2),
                            nomo_q16mul(nomo_q16mul({I[0] - I[2]}, w2), w0),
                            nomo_q16mul(nomo_q16mul({I[1] - I[0]}, w0), w1) }};
    const int64_t Iax[3] = {{ {I[0]}, {I[1]}, {I[2]} }};
    const int64_t wax[3] = {{ w0, w1, w2 }};
    const int ch[3] = {{ {term.channels[0]}, {term.channels[1]}, {term.channels[2]} }};
    for (int a = 0; a < 3; ++a) {{
      const int64_t r_lo = nomo_sat32((int64_t)nomo_q16mul(nomo_q16mul(Iax[a], -{p['w_max']} - wax[a]), {p['inv_dt']}) + g[a]);
      const int64_t r_hi = nomo_sat32((int64_t)nomo_q16mul(nomo_q16mul(Iax[a], {p['w_max']} - wax[a]), {p['inv_dt']}) + g[a]);
      int64_t v = {y}[ch[a]];
      if (r_lo > {p['tau_max']}) v = {p['tau_max']};            /* disjoint: saturate toward rate interval */
      else if (r_hi < -{p['tau_max']}) v = -{p['tau_max']};
      else v = nomo_clamp(v, r_lo > -{p['tau_max']} ? r_lo : -{p['tau_max']}, r_hi < {p['tau_max']} ? r_hi : {p['tau_max']});
      {y}[ch[a]] = (int32_t)v;
    }}
  }}""")
                else:  # pragma: no cover
                    raise TypeError(term)
            cur, cur_kind, cur_n = y, "q16", n
        else:  # pragma: no cover
            raise TypeError(st)

    assert cur_kind == "q16" and cur_n == qg.out_size
    body.append(f"  for (int i = 0; i < {qg.out_size}; ++i) out_q16[i] = {cur}[i];")

    uses_aux = qg.n_aux > 0
    header = f"""/* Generated by Nomo {qg.meta.get('genome', '')}. Do not edit. */
#ifndef {P}_MODEL_H
#define {P}_MODEL_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {{
#endif
#define {P}_IN_SIZE {qg.in_size}
#define {P}_OUT_SIZE {qg.out_size}
#define {P}_AUX_SIZE {max(1, qg.n_aux)}
/* input:  int8, real = in * {qg.in_scale!r}
   aux:    Q16.16 (body rates rad/s, density ratio, ...) as declared by the symbolic constraints
   output: Q16.16 physical units */
void {prefix}_infer(const int8_t in[{P}_IN_SIZE], const int32_t aux_q16[{P}_AUX_SIZE], int32_t out_q16[{P}_OUT_SIZE]);
#ifdef __cplusplus
}}
#endif
#endif
"""
    source = (f"/* Generated by Nomo. Integer-only C11; see SPEC §8. */\n#include \"{prefix}_model.h\"\n#include <string.h>\n"
              + HELPERS + "\n" + "".join(consts) + "\n" + "\n".join(bufs) + "\n\n"
              + f"void {prefix}_infer(const int8_t in[{P}_IN_SIZE], const int32_t aux_q16[{P}_AUX_SIZE], int32_t out_q16[{P}_OUT_SIZE]) {{\n"
              + ("" if uses_aux else "  (void)aux_q16;\n")
              + "\n".join(body) + "\n}\n")
    rom = sum(int(np.asarray(getattr(st, a)).size) * sz for st in qg.stages
              for a, sz in (("W", 1), ("b", 4), ("Phi", 4)) if hasattr(st, a))
    manifest = {"graph": qg.name, "genome": qg.meta.get("genome"), "stages": [f"{s.kind}:{s.name}" for s in qg.stages],
                "in_size": qg.in_size, "out_size": qg.out_size, "aux_size": qg.n_aux,
                "in_scale": qg.in_scale, "static_ram_bytes": ram, "const_rom_bytes": rom}
    return CEmission(header, source, manifest)


def emit_test_harness(qg: QGraph, X_i8: np.ndarray, AUX_q16: np.ndarray, prefix: str = "nomo") -> str:
    """A self-checking main(): runs every vector through the generated model and compares with
    the golden interpreter's outputs. Exit status is the number of mismatching vectors."""
    expected = np.stack([run(qg, x, a) for x, a in zip(X_i8, AUX_q16)])
    P = prefix.upper()
    n = len(X_i8)
    aux = AUX_q16 if AUX_q16.size else np.zeros((n, 1), np.int64)
    return (f'#include "{prefix}_model.h"\n#include <stdio.h>\n'
            + _array("int8_t", "X", X_i8) + _array("int32_t", "AUX", aux) + _array("int32_t", "Y", expected)
            + f"""int main(void) {{
  int bad = 0;
  int32_t out[{P}_OUT_SIZE];
  for (int k = 0; k < {n}; ++k) {{
    {prefix}_infer(&X[k * {P}_IN_SIZE], &AUX[k * {aux.shape[1]}], out);
    for (int i = 0; i < {P}_OUT_SIZE; ++i) {{
      if (out[i] != Y[k * {P}_OUT_SIZE + i]) {{
        printf("mismatch vector %d output %d: got %ld want %ld\\n", k, i, (long)out[i], (long)Y[k * {P}_OUT_SIZE + i]);
        ++bad;
        break;
      }}
    }}
  }}
  printf("%d/%d vectors bit-exact\\n", {n} - bad, {n});
  return bad;
}}
""")


def emit_c11_single_header(qg: QGraph, prefix: str = "nomo") -> str:
    """Header-only (stb-style) form of `emit_c11`: include `nomo_model.h` anywhere; in exactly one
    .c file write `#define NOMO_MODEL_IMPLEMENTATION` before the include to compile the kernels."""
    em = emit_c11(qg, prefix)
    P = prefix.upper()
    decl = em.header
    end = f"#endif\n"
    assert decl.rstrip().endswith("#endif")
    decl_body = decl[: decl.rstrip().rfind("#endif")]
    impl = em.source.replace(f'#include "{prefix}_model.h"\n', "")
    return (decl_body
            + f"\n#ifdef {P}_MODEL_IMPLEMENTATION\n/* ---- implementation (compile in exactly one translation unit) ---- */\n"
            + impl + f"#endif /* {P}_MODEL_IMPLEMENTATION */\n" + end)
