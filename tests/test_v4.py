"""v4 tests: policy/pins, crossing penalty, hardware overrides, presets, ingestion, exports, copilot, API."""
from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fixtures_v4 import (json_graph, mlp_state_dict, onnx_cnn, onnx_mlp, onnx_residual,  # noqa: E402
                         torch_state_dict_bytes)

from nomo.hardware.custom import HardwareOverrides, apply_overrides
from nomo.hardware.profiles import AKD1500, LOIHI2
from nomo.ingest import IngestError, ingest, load_state_dict
from nomo.models.zoo import attitude_policy, perception_cnn, synthetic_weights
from nomo.search.evaluator import Budgets, NeurosymbolicEvaluator
from nomo.search.genome import Coding, Domain, Genome, GuardGene, LayerGene, repair
from nomo.search.nsga2 import NSGA2Config, TriDomainNSGA2Optimizer
from nomo.search.policy import Pin, SearchPolicy, parse_policy, validate_policy
from nomo.search.presets import PRESETS


# ----------------------------------------------------------------------------- policy

def random_genome_for_tests(m, hw, rng):
    """Arbitrary genes ignoring every rule, so repair has to do all the work."""
    L = []
    for _ in m.layers:
        d = [Domain.ANN, Domain.SNN, Domain.SYM][int(rng.integers(3))]
        L.append(LayerGene(d, int(rng.choice([1, 2, 4, 8, 16])), int(rng.choice([4, 8, 16, 24])),
                           [Coding.RATE, Coding.TTFS][int(rng.integers(2))] if d == Domain.SNN else Coding.NONE,
                           int(rng.choice([1, 4, 8, 16, 32])) if d == Domain.SNN else 0, bool(rng.integers(2)) and d == Domain.SNN))
    return Genome(tuple(L), tuple(GuardGene(s.after_layer) for s in m.guard_sites))


def _search(model, hw=AKD1500, pop=24, gens=6, **budget):
    ev = NeurosymbolicEvaluator(model, hw, Budgets(acc_min=model.base_accuracy - 5, **budget))
    opt = TriDomainNSGA2Optimizer(model, hw, ev, NSGA2Config(pop_size=pop, generations=gens, seed=3))
    return opt.run(), ev, opt


def test_toggle_off_spiking_means_no_spiking_layers():
    m = attitude_policy()
    m.policy = parse_policy(m, {"spiking": False}, ["rate", "ttfs"], {})
    res, ev, _ = _search(m)
    assert all(l.domain != Domain.SNN for e in ev.cache.values() for l in e.genome.layers)


def test_pins_are_hard_invariants():
    m = attitude_policy()
    m.policy = parse_policy(m, {}, ["rate", "ttfs"], {"fuse1": {"domain": "SNN", "w_bits": 4},
                                                     "fuse0": {"domain": "ANN", "w_bits": "INT8", "a_bits": 8}})
    assert validate_policy(m, AKD1500, m.policy).ok
    res, ev, _ = _search(m)
    for e in ev.cache.values():
        assert e.genome.layers[1].domain == Domain.SNN and e.genome.layers[1].w_bits == 4
        assert e.genome.layers[0] .domain == Domain.ANN and e.genome.layers[0].w_bits == 8 and e.genome.layers[0].a_bits == 8


def test_repair_idempotent_under_policy():
    m = attitude_policy()
    m.policy = SearchPolicy(frozenset({Domain.ANN, Domain.SNN}), frozenset({Coding.TTFS}), ((2, Pin(Domain.SNN, 4)),))
    rng = np.random.default_rng(0)
    for _ in range(200):
        g = repair(random_genome_for_tests(m, LOIHI2, rng), m, LOIHI2)
        assert repair(g, m, LOIHI2) == g
        assert all(l.coding == Coding.TTFS for l in g.layers if l.domain == Domain.SNN)
        assert all(l.domain != Domain.SYM for l in g.layers) and g.layers[2].domain == Domain.SNN


def test_policy_rejects_impossible_pins_with_reasons():
    m = attitude_policy()
    rep = validate_policy(m, AKD1500, parse_policy(m, {}, ["rate"], {"torque_head": {"domain": "SNN"}}))
    assert not rep.ok and "negative" in rep.errors[0] or "signed" in rep.errors[0]
    rep = validate_policy(m, AKD1500, parse_policy(m, {}, ["rate"], {"fuse0": {"domain": "SYM"}}))
    assert not rep.ok and "formula" in rep.errors[0]
    rep = validate_policy(m, AKD1500, parse_policy(m, {}, ["rate"], {"fuse0": {"w_bits": "INT16"}}))
    assert not rep.ok and "not supported" in rep.errors[0]
    with pytest.raises(ValueError):
        parse_policy(m, {}, ["phase"], {})
    with pytest.raises(ValueError):
        parse_policy(m, {}, ["rate"], {"nope": {"domain": "ANN"}})


def test_toggle_warning_when_layer_has_no_other_option():
    m = attitude_policy()
    rep = validate_policy(m, AKD1500, parse_policy(m, {"continuous": False}, ["rate"], {}))
    assert rep.ok and any("torque_head" in w for w in rep.warnings)


# ----------------------------------------------------------------------------- crossing penalty

def test_crossing_penalty_scales_objectives_not_reported_values():
    m = attitude_policy()
    g = repair(Genome(tuple([LayerGene(Domain.ANN, 8, 8), LayerGene(Domain.SNN, 4, 16, Coding.RATE, 8),
                             LayerGene(Domain.ANN, 8, 8)] + [LayerGene(Domain.ANN, 8, 8)] * 4), (GuardGene(6),)), m, AKD1500)
    e0 = NeurosymbolicEvaluator(m, AKD1500, Budgets()).evaluate(g)
    e1 = NeurosymbolicEvaluator(m, AKD1500, Budgets(crossing_penalty=0.5)).evaluate(g)
    k = e1.metrics["domain_switches"]
    assert k == 2
    assert e1.F[0] == pytest.approx(e0.F[0] * (1 + 0.5 * k)) and e1.to_wire()["f"][0] == pytest.approx(e0.cost.energy_j)


def test_crossing_min_saving_makes_weak_hybrids_infeasible():
    m = attitude_policy()
    g = repair(Genome(tuple([LayerGene(Domain.ANN, 8, 8), LayerGene(Domain.SNN, 4, 16, Coding.RATE, 8)]
                            + [LayerGene(Domain.ANN, 8, 8)] * 5), (GuardGene(6),)), m, AKD1500)
    e = NeurosymbolicEvaluator(m, AKD1500, Budgets(crossing_min_saving_pct=90)).evaluate(g)
    assert "crossing_saving" in e.g and e.g["crossing_saving"] > 0 and not e.feasible


def test_archive_capacity_and_asf_weights():
    m = attitude_policy()
    ev = NeurosymbolicEvaluator(m, AKD1500, Budgets(acc_min=m.base_accuracy - 6))
    opt = TriDomainNSGA2Optimizer(m, AKD1500, ev, NSGA2Config(pop_size=32, generations=10, seed=1, archive_capacity=3,
                                                              asf_weights=(10.0, 1.0, 1.0), p_mutation=0.5))
    opt.run()
    assert len(opt.archive_front()) <= 3
    rec_e = opt.recommend()
    opt.cfg.asf_weights = (1.0, 1.0, 10.0)
    assert opt.recommend().accuracy >= rec_e.accuracy


# ----------------------------------------------------------------------------- hardware + presets

def test_hardware_overrides_change_cost_and_provenance():
    hw = apply_overrides(AKD1500, HardwareOverrides(mac_energy_pj=10.0, static_power_mw=1.0, clock_mhz=150))
    assert hw.ann_cost(8, 8)[0] == pytest.approx(10e-12) and hw.provenance["p_static_w"] == "user"
    m = attitude_policy()
    g = repair(Genome(tuple(LayerGene(Domain.ANN, 8, 8) for _ in m.layers), (GuardGene(6),)), m, AKD1500)
    a = NeurosymbolicEvaluator(m, AKD1500, Budgets()).evaluate(g)
    b = NeurosymbolicEvaluator(m, hw, Budgets()).evaluate(g)
    assert b.cost.latency_s > a.cost.latency_s                  # half the clock
    with pytest.raises(ValueError):
        apply_overrides(AKD1500, HardwareOverrides(n_cores=0))


def test_presets_are_valid_run_settings():
    from nomo.telemetry.server import RunIn
    for name, p in PRESETS.items():
        s = p["settings"]
        cfg = RunIn(model="attitude_policy", budgets=s["budgets"], search=s["search"], lock_symbolic=s.get("lock_symbolic", False))
        assert cfg.search.asf_weights and len(cfg.search.asf_weights) == 3, name


# ----------------------------------------------------------------------------- ingestion

def test_onnx_mlp_and_cnn_match_onnxruntime():
    import onnxruntime as ort
    from nomo.export.plan import build_plan, run_plan
    for data, shape in ((onnx_mlp(), (4, 8)), (onnx_cnn(), (4, 1, 8, 8))):
        m, W, rep = ingest("m.onnx", data, 85.0)
        x = np.random.default_rng(0).normal(size=shape).astype(np.float32)
        ref = ort.InferenceSession(data).run(None, {"x": x})[0]
        plan = build_plan(m, W, Genome(tuple(LayerGene(Domain.ANN, 16, 16) for _ in m.layers), ()), x.astype(float))
        for L, spec in zip(plan["layers"], m.layers):
            L["W"], L["act_bits"] = W[spec.name][0], 0
        y = run_plan(plan, x.astype(float))
        if "Softmax" in rep.dropped_ops:
            y = np.exp(y) / np.exp(y).sum(1, keepdims=True)
        assert np.abs(y - ref).max() < 1e-5


def test_onnx_rejects_branching_graphs():
    with pytest.raises(IngestError, match="branch"):
        ingest("r.onnx", onnx_residual())


def test_json_graph_with_and_without_weights():
    m, W, rep = ingest("g.json", json_graph())
    assert rep.weights_source == "synthetic" and m.layers[0].attrs["pool"]["type"] == "avg" and m.layers[1].fan_in == 128
    m, W, rep = ingest("g.json", json_graph(with_weights=True))
    assert np.all(W["a"][0] == 1.0) and rep.weights_source == "partial" and "b" in rep.assumptions[0]


def test_state_dict_loader_is_exact_and_safe():
    sd = mlp_state_dict()
    back = load_state_dict(torch_state_dict_bytes(sd))
    assert all(np.array_equal(back[k], sd[k].astype(np.float32)) for k in sd)
    m, W, rep = ingest("m.pth", torch_state_dict_bytes(sd), 80)
    assert [l.activation for l in m.layers] == ["relu", "linear"] and any("assumed" in a for a in rep.assumptions)
    with pytest.raises(IngestError, match="refused"):
        ingest("evil.pt", torch_state_dict_bytes(sd, evil=True))
    with pytest.raises(IngestError):
        ingest("x.pt", b"not a zip")


def test_real_torch_checkpoint():
    torch = pytest.importorskip("torch")
    net = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.ReLU(), torch.nn.Linear(7, 2))
    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    back = load_state_dict(buf.getvalue())
    assert np.allclose(back["0.weight"], net[0].weight.detach().numpy())


def test_uploaded_model_runs_through_search():
    m, W, _ = ingest("m.onnx", onnx_cnn(), 85.0)
    res, ev, _ = _search(m, pop=16, gens=4)
    assert res.front


# ----------------------------------------------------------------------------- exports

@pytest.fixture(scope="module")
def attitude_ctx():
    from nomo.export.bundle import RunContext
    m = attitude_policy()
    ev = NeurosymbolicEvaluator(m, AKD1500, Budgets(acc_min=m.base_accuracy - 5))
    opt = TriDomainNSGA2Optimizer(m, AKD1500, ev, NSGA2Config(pop_size=32, generations=8, seed=2))
    opt.run()
    return RunContext(m, synthetic_weights(m), "synthetic (test)", AKD1500, ev, {"budgets": {"accuracy_drop_max": 5}},
                      [], opt.archive_front, opt.recommend)


def _hybrid(ctx):
    g = repair(Genome(tuple([LayerGene(Domain.ANN, 8, 8)] + [LayerGene(Domain.SNN, 4, 16, Coding.RATE, 8)] * 3
                            + [LayerGene(Domain.ANN, 8, 8), LayerGene(Domain.SYM, 32, 32), LayerGene(Domain.ANN, 8, 8)]),
                      (GuardGene(6),)), ctx.model, ctx.hw)
    return ctx.evaluator.evaluate(g)


def test_capabilities_explain_unavailable_formats(attitude_ctx):
    from nomo.export.bundle import capabilities
    caps = capabilities(attitude_ctx, _hybrid(attitude_ctx))
    assert caps["c11"]["available"] and caps["onnx"]["available"] and "continuous section" in caps["onnx"]["note"]
    from nomo.export.bundle import RunContext
    m = perception_cnn()
    ctx = RunContext(m, synthetic_weights(m), "synthetic (test)", AKD1500, NeurosymbolicEvaluator(m, AKD1500, Budgets()))
    g = repair(Genome(tuple(LayerGene(Domain.ANN, 8, 8) for _ in m.layers), ()), m, AKD1500)
    caps = capabilities(ctx, ctx.evaluator.evaluate(g))
    assert not caps["c11"]["available"] and "convolution" in caps["c11"]["reason"]


def test_full_bundle_every_artifact_verifies(attitude_ctx, tmp_path):
    import nir
    from nomo.export.bundle import FORMATS, build_bundle
    ev = _hybrid(attitude_ctx)
    data, man = build_bundle(attitude_ctx, ev, list(FORMATS))
    assert not man["errors"] and not man["skipped"]
    zipfile.ZipFile(io.BytesIO(data)).extractall(tmp_path)
    root = tmp_path / man["root"]
    assert (root / "report.pdf").read_bytes()[:5] == b"%PDF-"
    d = json.loads((root / "design.json").read_text())
    assert d["design_key"] == ev.key and "baseline_all_continuous" in d
    r = subprocess.run([sys.executable, "deploy_model.py", "--check", "--backend", "numpy"], cwd=root / "pytorch",
                       capture_output=True, text=True)
    assert "PASS" in r.stdout, r.stdout + r.stderr
    nir.read(str(root / "nir" / "attitude_policy_float.nir"))
    assert list((root / "onnx").glob("*.onnx")) and list((root / "coreml").glob("*.mlpackage.zip"))
    r = subprocess.run("cc -std=c11 -Wall -Werror -O2 example_main.c -o demo && ./demo", shell=True, cwd=root / "c11",
                       capture_output=True, text=True)
    assert r.returncode == 0 and "out[0]" in r.stdout, r.stderr


def test_onnx_sections_match_reference_including_guard(attitude_ctx):
    from nomo.export import hybrid_runtime as hr
    from nomo.export.onnx_coreml import continuous_sections, run_onnx, to_onnx
    ev = _hybrid(attitude_ctx)
    plan = attitude_ctx.plan(ev)
    X, A = attitude_ctx.calib()
    for s, e in continuous_sections(plan):
        pre = {**plan, "layers": plan["layers"][:s], "guards": [G for G in plan["guards"] if G["after"] < s]}
        xin = X if s == 0 else np.asarray(hr.forward(hr.NumpyOps(), pre, X, A))
        sub = {**plan, "layers": plan["layers"][s:e + 1],
               "guards": [{**G, "after": G["after"] - s} for G in plan["guards"] if s <= G["after"] <= e]}
        ref = np.asarray(hr.forward(hr.NumpyOps(), sub, xin, A))
        y = run_onnx(to_onnx(plan, s, e, "t"), xin, A)
        assert np.mean(~np.isclose(y, ref, rtol=1e-3, atol=1e-4 * np.abs(ref).max())) < 0.01


# ----------------------------------------------------------------------------- copilot

def test_copilot_grounded_answers(attitude_ctx):
    from nomo.copilot import answer
    ev = _hybrid(attitude_ctx)
    front, rec = attitude_ctx.front_fn(), attitude_ctx.recommend_fn()
    a = answer(attitude_ctx, ev, front, rec, "Why is fuse1 spiking?")
    assert a.facts["layer"] == "fuse1" and a.facts["alternatives"] and a.source == "rules"
    alt = a.facts["alternatives"][0]
    assert alt["key"] in attitude_ctx.evaluator.cache                  # counterfactual really evaluated
    a = answer(attitude_ctx, ev, front, rec, "How do I cut energy by another 20%?")
    assert a.actions and a.actions[0]["type"] in ("select", "rerun")
    a = answer(attitude_ctx, rec, front, rec, "Explain this Pareto front in plain English")
    assert "trade-offs" in a.text and any(x["type"] == "select" for x in a.actions)
    assert "spike" in answer(attitude_ctx, ev, front, rec, "what is TTFS?").text.lower()
    a = answer(attitude_ctx, ev, front, rec, "why is torque_head continuous?")
    assert "negative" in a.text


def test_copilot_llm_failure_falls_back(attitude_ctx, monkeypatch):
    from nomo import copilot
    monkeypatch.setenv("NOMO_ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(copilot, "_llm", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    ev = _hybrid(attitude_ctx)
    a = copilot.answer(attitude_ctx, ev, [], None, "what is rate coding")
    assert a.source == "rules" and "rate" in a.text.lower()
    monkeypatch.setattr(copilot, "_llm", lambda *a, **k: "phrased by model")
    assert copilot.answer(attitude_ctx, ev, [], None, "what is rate coding").source == "llm"


# ----------------------------------------------------------------------------- API

@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("NOMO_LOG_DIR", str(tmp_path))
    from fastapi.testclient import TestClient
    from nomo.telemetry.server import create_app
    with TestClient(create_app()) as c:
        yield c


def _wait(c, rid, limit=120):
    t = time.time()
    while time.time() - t < limit:
        s = c.get(f"/runs/{rid}").json()
        if s["status"] not in ("pending", "running"):
            return s
        time.sleep(0.2)
    raise AssertionError("run did not finish")


def test_api_upload_run_design_export_copilot(client):
    r = client.post("/models/upload", params={"filename": "net.onnx", "base_accuracy": 87.5}, content=onnx_cnn())
    assert r.status_code == 200, r.text
    up = r.json()
    assert up["model_id"].startswith("upload:") and up["layer_table"][0]["op"] == "conv2d"
    assert "SNN" in up["layer_table"][0]["can_be"] and up["layer_table"][-1]["can_be"] == ["ANN"]
    assert client.get(f"/models/{up['model_id']}").json()["name"] == "net"
    bad = client.post("/models/upload", params={"filename": "r.onnx"}, content=onnx_residual())
    assert bad.status_code == 422 and "branch" in bad.json()["detail"]

    cfg = {"model": up["model_id"], "hardware": "akd1500", "pop_size": 16, "generations": 4,
           "budgets": {"accuracy_drop_max": 6},
           "search": {"codings": ["ttfs"], "crossing_penalty": 0.1, "asf_weights": [3, 1, 1]},
           "pins": {"conv": {"domain": "ANN", "w_bits": "INT8"}},
           "hardware_overrides": {"mac_energy_pj": 2.0, "sram_kb_per_core": 64}}
    rid = client.post("/runs", json=cfg).json()["run_id"]
    assert _wait(client, rid)["status"] == "completed"
    d = client.get(f"/runs/{rid}/designs/recommended").json()
    assert d["design"]["layers"][0]["domain"] == "ANN" and d["design"]["layers"][0]["w_bits"] == 8
    assert d["design"]["hardware"]["provenance"]["ann_e_mac"] == "user"
    assert d["capabilities"]["pytorch"]["available"] and d["summary"]["text"]
    key = d["design"]["design_key"]
    r = client.post(f"/runs/{rid}/export", json={"key": key, "formats": ["design", "pytorch", "onnx"]})
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert any(n.endswith("deploy_model.py") for n in names) and any(n.endswith(".onnx") for n in names)
    a = client.post(f"/runs/{rid}/copilot", json={"question": "Explain the pareto front", "key": key}).json()
    assert a["text"] and "actions" in a
    assert client.post(f"/runs/{rid}/export", json={"key": "nope", "formats": ["pdf"]}).status_code == 404
    assert client.post(f"/runs/{rid}/export", json={"key": key, "formats": ["docx"]}).status_code == 422


def test_api_rejects_impossible_policy_with_reason(client):
    r = client.post("/runs", json={"model": "attitude_policy", "pins": {"fuse0": {"domain": "SYM"}}})
    assert r.status_code == 422 and "formula" in r.json()["detail"]
    r = client.post("/runs", json={"model": "attitude_policy", "search": {"codings": ["phase"]}})
    assert r.status_code == 422
    r = client.post("/runs", json={"model": "attitude_policy", "hardware_overrides": {"n_cores": -1}})
    assert r.status_code == 422


def test_api_presets_catalog_and_strict_safety(client):
    p = client.get("/presets").json()
    assert set(p) == {"battery_saver", "ultra_low_latency", "balanced_edge", "strict_safety"}
    cat = client.get("/catalog").json()
    assert "defaults" in cat["hardware"]["akd1500"] and cat["models"]["attitude_policy"]["layer_table"]
    s = p["strict_safety"]["settings"]
    rid = client.post("/runs", json={"model": "attitude_policy", "pop_size": 16, "generations": 3, **s}).json()["run_id"]
    assert _wait(client, rid)["status"] == "completed"
    d = client.get(f"/runs/{rid}/designs/recommended").json()["design"]
    assert d["layers"][5]["domain"] == "SYM"                            # physics layer locked in


def test_design_endpoints_wait_for_completion(client):
    rid = client.post("/runs", json={"model": "perception_cnn", "pop_size": 32, "generations": 40}).json()["run_id"]
    r = client.post(f"/runs/{rid}/copilot", json={"question": "hi"})
    assert r.status_code in (409, 200)
    client.post(f"/runs/{rid}/stop")
    _wait(client, rid)
