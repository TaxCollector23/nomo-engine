import json
import tempfile

import nir
import numpy as np
import pytest

from nomo.export.nir_export import build_graph, export_all, read_extended
from nomo.hardware.profiles import LOIHI2
from nomo.models.zoo import attitude_calibration, attitude_policy, synthetic_weights
from nomo.runtime.quantize import CompileOptions, compile_qgraph
from nomo.search.genome import Coding, Domain, Genome, GuardGene, LayerGene, repair

M = attitude_policy()
W = synthetic_weights(M)
X, AUX = attitude_calibration(200)


def _qg(leak):
    L = [LayerGene(Domain.ANN, 8, 8)] + [LayerGene(Domain.SNN, 4, 16, Coding.RATE, 16)] * 3 + \
        [LayerGene(Domain.ANN, 8, 8), LayerGene(Domain.SYM, 32, 32), LayerGene(Domain.ANN, 8, 8)]
    g = repair(Genome(tuple(L), (GuardGene(6),)), M, LOIHI2)
    return g, compile_qgraph(M, W, g, X, AUX, CompileOptions(leak_shift=leak))


@pytest.mark.parametrize("leak", [0, 4])
def test_strict_loads_in_stock_nir_with_type_check(leak):
    g, qg = _qg(leak)
    p = export_all(qg, tempfile.mkdtemp(), genome_wire=g.to_wire())
    G = nir.read(p["strict"], type_check=True)
    types = {type(n).__name__ for n in G.nodes.values()}
    assert ("LIF" in types) == (leak > 0) and ("IF" in types)
    cons = [n for n in G.nodes.values() if n.metadata.get("nomo.op") == "SymbolicConstraint"]
    assert len(cons) == 2                                         # Output(pre) + Input(post) cut pair
    spec = json.loads(cons[0].metadata["nomo.constraint_json"])
    assert spec["id"] == "actuator_limits" and len(spec["terms"]) == 2
    dense = [n for n in G.nodes.values() if n.metadata.get("nomo.q.w_int") is not None]
    for n in dense:                                               # fixed-point metadata reconstructs weights
        w_int = np.asarray(n.metadata["nomo.q.w_int"])
        scale = n.metadata.get("nomo.q.w_scale", n.metadata.get("nomo.q.v_scale"))
        assert np.allclose(w_int * scale, n.weight, rtol=1e-6, atol=1e-7)


def test_lif_parameters_encode_integer_leak():
    g, qg = _qg(3)
    G = nir.read(export_all(qg, tempfile.mkdtemp())["strict"])
    lif = [n for n in G.nodes.values() if type(n).__name__ == "LIF"][0]
    dt = lif.metadata["nomo.dt"]
    assert np.allclose(dt / lif.tau, 2.0 ** -3) and np.allclose(lif.r * dt / lif.tau, 1.0)


def test_extended_roundtrip_and_stock_rejection():
    g, qg = _qg(0)
    p = export_all(qg, tempfile.mkdtemp(), genome_wire=g.to_wire())
    with pytest.raises(Exception):
        nir.read(p["extended"])
    E, B = read_extended(p["extended"]), build_graph(qg, genome_wire=g.to_wire())
    assert E.edges == B.edges and set(E.nodes) == set(B.nodes)
    assert {"nomo.SpikeEncoder", "nomo.SpikeDecoder", "nomo.SymbolicConstraint"} <= {n.type for n in E.nodes.values()}
    for k, n in B.nodes.items():
        assert E.nodes[k].type == n.type
        for pk, pv in n.params.items():
            ev = E.nodes[k].params[pk]
            assert ev == pv if isinstance(pv, str) else np.array_equal(np.asarray(ev), np.asarray(pv))
    assert json.loads(E.metadata["nomo.genome_json"]) == json.loads(json.dumps(g.to_wire()))


def test_manifest_is_complete_integer_program():
    g, qg = _qg(0)
    man = json.loads(open(export_all(qg, tempfile.mkdtemp())["manifest"]).read())
    assert man["format"] == "nomo.qgraph/1" and len(man["stages"]) == len(qg.stages)
    lif = [s for s in man["stages"] if s["kind"] == "lif"][0]
    assert lif["W"]["dtype"] == "int8" and isinstance(lif["theta"], int)
