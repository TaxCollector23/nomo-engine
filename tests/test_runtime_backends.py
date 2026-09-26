import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

from nomo.export.c11 import emit_c11, emit_test_harness
from nomo.hardware.profiles import AKD1500, LOIHI2
from nomo.models.zoo import attitude_calibration, attitude_policy, synthetic_weights
from nomo.runtime import qgraph
from nomo.runtime.quantize import (CompileOptions, UnsupportedLowering, compile_qgraph, float_forward,
                                   quantize_aux, quantize_input)
from nomo.search.genome import Coding, Domain, Genome, GuardGene, GuardImpl, LayerGene, repair

M = attitude_policy()
W = synthetic_weights(M)
X, AUX = attitude_calibration(360)


def genome(spec: str, hw=LOIHI2, T=16, w=4, v=16):
    L = []
    for c in spec:
        if c == "A":
            L.append(LayerGene(Domain.ANN, 8, 8))
        elif c == "Y":
            L.append(LayerGene(Domain.SYM, 32, 32))
        else:
            L.append(LayerGene(Domain.SNN, w, v, Coding.RATE, T))
    return repair(Genome(tuple(L), (GuardGene(6, GuardImpl.HOST),)), M, hw)


GENOMES = [("AAAAAAA", 0), ("AAAAAYA", 0), ("ASSSAYA", 0), ("SSSSAYA", 5), ("ASSSAAA", 3)]
# minimum float-vs-integer correlation at w8/v24/T32. Leak bleeds charge that rate coding
# needs (tau = 2^k steps against T = 32), so leaky neurons convert less faithfully.
MIN_CORR = {0: 0.93, 5: 0.9, 3: 0.8}


@pytest.mark.skipif(qgraph._cxx is None, reason="C++ extension not built")
def test_cpp_lif_kernel_matches_numpy():
    rng = np.random.default_rng(0)
    for _ in range(30):
        no, ni, T = rng.integers(1, 80), rng.integers(1, 80), rng.integers(1, 24)
        Wq = rng.integers(-8, 8, (no, ni)).astype(np.int8)
        b = rng.integers(-20, 20, no).astype(np.int32)
        s = rng.integers(-1, 2, (T, ni)).astype(np.int8)
        th, k, vb = int(rng.integers(1, 200)), int(rng.integers(0, 6)), int(rng.choice([16, 24]))
        a = qgraph.lif_numpy(Wq, b, s, th, k, vb)
        c = np.asarray(qgraph._cxx.lif_layer(Wq, b, s, th, k, vb))
        assert np.array_equal(a, c)


def _fidelity(g, leak=0):
    qg = compile_qgraph(M, W, g, X[:256], AUX[:256], CompileOptions(leak_shift=leak))
    ref = float_forward(M, W, g, X[256:], AUX[256:])[-1]
    out = np.array([qgraph.run(qg, quantize_input(qg, x), quantize_aux(a)) for x, a in zip(X[256:], AUX[256:])]) / 65536.0
    return np.corrcoef(out.ravel(), ref.ravel())[0, 1]


@pytest.mark.parametrize("spec,leak", GENOMES)
def test_quantised_graph_tracks_float(spec, leak):
    g = genome(spec, w=8, v=24, T=32)
    assert _fidelity(g, leak) > (0.99 if "S" not in spec else MIN_CORR[leak])


def test_conversion_fidelity_improves_with_precision_and_timesteps():
    """Post-training conversion error must fall monotonically along the axes the accuracy proxy
    penalises (SPEC 5.1): weight bits and rate-coding window T."""
    f_lo = _fidelity(genome("ASSSAYA", w=4, v=16, T=16))
    f_w = _fidelity(genome("ASSSAYA", w=8, v=16, T=16))
    f_wt = _fidelity(genome("ASSSAYA", w=8, v=24, T=32))
    assert f_lo < f_w < f_wt
    g = genome("ASSSAAA", w=8, v=24, T=32)
    assert _fidelity(g, 3) < _fidelity(g, 0)                   # leak degrades rate conversion


def test_unsupported_lowerings_are_explicit():
    L = list(genome("ASSSAAA").layers)
    L[1] = LayerGene(Domain.SNN, 4, 16, Coding.TTFS, 16)
    with pytest.raises(UnsupportedLowering):
        compile_qgraph(M, W, Genome(tuple(L), (GuardGene(6),)), X[:64], AUX[:64])


def test_binary_weights_lower():
    g = genome("ASSSAAA", hw=AKD1500, w=1, T=8)
    assert g.layers[1].w_bits == 1
    qg = compile_qgraph(M, W, g, X[:256], AUX[:256])
    lif = [s for s in qg.stages if isinstance(s, qgraph.QLIF)][0]
    assert set(np.unique(lif.W)) <= {-1, 1}


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not available")
@pytest.mark.parametrize("spec,leak", GENOMES)
@pytest.mark.parametrize("sanitize", [False, True])
def test_c11_backend_bit_exact(spec, leak, sanitize):
    g = genome(spec)
    qg = compile_qgraph(M, W, g, X[:256], AUX[:256], CompileOptions(leak_shift=leak))
    Xi = np.stack([quantize_input(qg, x) for x in X[256:]])
    Ai = np.array([quantize_aux(a) for a in AUX[256:]])
    d = tempfile.mkdtemp()
    emit_c11(qg).write(d)
    with open(os.path.join(d, "test_main.c"), "w") as fh:
        fh.write(emit_test_harness(qg, Xi, Ai))
    flags = ["-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic", "-O2"]
    if sanitize:
        flags += ["-fsanitize=undefined", "-fno-sanitize-recover=undefined"]
    r = subprocess.run(["gcc", *flags, "-o", f"{d}/t", f"{d}/nomo_model.c", f"{d}/test_main.c"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([f"{d}/t"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "runtime error" not in r.stderr
