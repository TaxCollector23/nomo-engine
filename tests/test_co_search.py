import math

import numpy as np
import pytest

from nomo.hardware.profiles import AKD1500
from nomo.models.zoo import attitude_policy
from nomo.search.co_design import (
    AcceleratorArchitecture,
    CoDesignCandidate,
    CoDesignConstraints,
    CoDesignEvaluator,
    CoDesignSearch,
    CoDesignSpace,
    CoDesignWorkload,
    CO_DESIGN_CAPABILITY,
    generate_candidates,
    generate_co_design_candidates,
    pareto_rank,
)
from nomo.search.evaluator import Budgets, NeurosymbolicEvaluator
from nomo.search.genome import Domain, uniform_genome


def _space():
    return CoDesignSpace(
        pe_rows=(2, 1, 2),
        pe_cols=(1, 2),
        sram_bytes=(4096, 1024),
        memory_bandwidth_bytes_s=(4e9, 1e9),
        precision_bits=(8, 4, 8),
    )


def test_candidate_generation_is_deterministic_and_canonical():
    first = generate_candidates(_space(), "deployment-a")
    second = generate_candidates(_space(), "deployment-a")

    assert first == second
    assert len(first) == 2 * 2 * 2 * 2 * 2
    assert [candidate.key for candidate in first] == sorted(candidate.key for candidate in first)
    assert all(candidate.deployment_key == "deployment-a" for candidate in first)

    pairs = generate_co_design_candidates(["z", "a", "a"], _space())
    assert len(pairs) == 2 * len(first)
    assert pairs[0].deployment_key == "a"
    assert len({candidate.key for candidate in pairs}) == len(pairs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pe_rows", 0),
        ("pe_cols", -1),
        ("sram_bytes", 128),
        ("memory_bandwidth_bytes_s", math.nan),
        ("memory_bandwidth_bytes_s", 0),
        ("precision_bits", 3),
    ],
)
def test_custom_hardware_values_are_rejected(field, value):
    values = {
        "pe_rows": 1,
        "pe_cols": 1,
        "sram_bytes": 1024,
        "memory_bandwidth_bytes_s": 1e9,
        "precision_bits": 8,
    }
    values[field] = value
    with pytest.raises(ValueError):
        AcceleratorArchitecture(**values)


def test_constraints_filter_candidates_and_reject_direct_unsafe_use():
    limits = CoDesignConstraints(
        max_pe_count=4,
        min_sram_bytes=1024,
        max_sram_bytes=4096,
        min_memory_bandwidth_bytes_s=1e9,
        max_memory_bandwidth_bytes_s=4e9,
        allowed_precision_bits=(4, 8),
        max_array_aspect_ratio=2,
    )
    candidates = generate_candidates(_space(), "g", constraints=limits)
    assert candidates
    assert all(candidate.architecture.pe_count <= 4 for candidate in candidates)
    assert all(candidate.architecture.precision_bits in (4, 8) for candidate in candidates)
    assert all(candidate.architecture.pe_rows / candidate.architecture.pe_cols <= 2 for candidate in candidates)

    unsafe = CoDesignCandidate("g", AcceleratorArchitecture(4, 4, 1024, 1e9, 8))
    workload = CoDesignWorkload(1000, 1000, 100, 1000)
    with pytest.raises(ValueError, match="PE count"):
        CoDesignEvaluator(workload, limits).evaluate(unsafe)


def test_evaluation_exposes_constraints_reasons_and_evidence():
    limits = CoDesignConstraints(allowed_precision_bits=(4, 8), max_latency_s=1e-3)
    workload = CoDesignWorkload(
        macs=100_000,
        parameters=10_000,
        activation_values=2_000,
        working_set_values=20_000,
        required_precision_bits=8,
        base_accuracy_loss_pp=1.0,
    )
    evaluator = CoDesignEvaluator(workload, limits)
    candidate = CoDesignCandidate("g", AcceleratorArchitecture(1, 1, 1024, 1e9, 4))
    result = evaluator.evaluate(candidate)

    assert result.cv > 0
    assert result.g["sram_capacity"] > 0
    assert result.g["precision_support"] > 0
    assert result.evidence_sources["energy_j"] == "analytic_prior"
    assert result.evidence_sources["accuracy_loss_pp"] == "deployment_proxy"
    assert any("infeasible" in reason for reason in result.reasons)
    wire = result.to_wire()
    assert wire["capability"] == CO_DESIGN_CAPABILITY
    assert wire["objective_names"] == ["energy_j", "latency_s", "accuracy_loss_pp", "area_proxy"]
    assert wire["architecture"]["pe_count"] == 1
    assert wire["evidence_sources"] == result.evidence_sources


def test_pareto_ranking_keeps_architecture_tradeoffs_visible():
    workload = CoDesignWorkload(
        macs=100_000_000,
        parameters=1_000_000,
        activation_values=100_000,
        working_set_values=1_000_000,
        required_precision_bits=8,
    )
    evaluator = CoDesignEvaluator(workload)
    architectures = (
        AcceleratorArchitecture(1, 1, 2_000_000, 1e9, 8),
        AcceleratorArchitecture(16, 16, 2_000_000, 64e9, 8),
        AcceleratorArchitecture(1, 1, 2_000_000, 64e9, 8),
    )
    results = [evaluator.evaluate(CoDesignCandidate("g", architecture)) for architecture in architectures]
    ranked = pareto_rank(results)
    front = [result for result in ranked if result.rank == 0 and result.feasible]

    assert len(front) >= 2
    assert any(result.metrics["compute_latency_s"] > result.metrics["memory_latency_s"] for result in front)
    assert any("Pareto rank 0" in reason for result in front for reason in result.reasons)
    assert np.all(np.isfinite(np.stack([result.F for result in ranked])))


def test_existing_evaluator_api_remains_compatible_and_co_search_is_additive():
    model = attitude_policy()
    evaluator = NeurosymbolicEvaluator(model, AKD1500, Budgets())
    genome = uniform_genome(model, AKD1500, Domain.ANN)
    legacy = evaluator.evaluate(genome)
    legacy_wire = legacy.to_wire()

    result = evaluator.co_design(
        [genome],
        space=CoDesignSpace(
            pe_rows=(1, 4), pe_cols=(1, 4), sram_bytes=(1 << 20,),
            memory_bandwidth_bytes_s=(1e9, 16e9), precision_bits=(8,),
        ),
    )

    assert legacy.key == genome.key
    assert legacy_wire["key"] == genome.key
    assert len(result.evaluations) == 8
    assert result.front
    assert result.recommended in result.front
    assert result.recommended.candidate.deployment_key == genome.key
    assert result.recommended.evidence_sources["energy_j"] in {"analytic_prior", "calibrated_hardware_prior"}


def test_search_accepts_cached_deployment_evaluations():
    model = attitude_policy()
    evaluator = NeurosymbolicEvaluator(model, AKD1500, Budgets())
    genome = uniform_genome(model, AKD1500, Domain.ANN)
    base = evaluator.evaluate(genome)
    search = CoDesignSearch(
        evaluator,
        CoDesignSpace(pe_rows=(1,), pe_cols=(1,), sram_bytes=(1 << 20,),
                      memory_bandwidth_bytes_s=(1e9,), precision_bits=(8,)),
    )
    result = search.run([base])
    assert len(result.evaluations) == 1
    assert result.evaluations[0].deployment_evaluation is base
