"""AMD Instinct catalog pins + end-to-end feasibility on AMD clusters.

The datasheet values below were transcribed from the AMD product pages
(MI300X / MI325X / MI355X, dense non-sparse rates). If a test fails here,
someone changed the catalog away from the published datasheet on purpose or
by accident - either way it deserves a conscious decision, not a silent drift.
"""
import pytest

from nomo_planner.domains.hardware import (
    CLUSTERS,
    flops,
    link_for_group,
    roofline_time,
)


def test_amd_instinct_devices_match_datasheet_dense_rates():
    mi300x = CLUSTERS["mi300x_if_roce"].device
    mi325x = CLUSTERS["mi325x_if_roce"].device
    mi355x = CLUSTERS["mi355x_if_roce"].device

    assert mi300x.name == "AMD Instinct MI300X"
    assert mi300x.peak_flops == {"bf16": 1307.4e12, "fp8": 2614.9e12, "fp32": 163.4e12}
    assert mi300x.mem_bytes == 192e9 and mi300x.mem_bw == 5.3e12 and mi300x.power_w == 750

    assert mi325x.peak_flops == mi300x.peak_flops
    assert mi325x.mem_bytes == 256e9 and mi325x.mem_bw == 6.0e12 and mi325x.power_w == 1000

    assert mi355x.peak_flops == {"bf16": 2500e12, "fp8": 5000e12, "fp32": 157.3e12}
    assert mi355x.mem_bytes == 288e9 and mi355x.mem_bw == 8.0e12 and mi355x.power_w == 1400

    for key in ("mi300x_if_roce", "mi325x_if_roce", "mi355x_if_roce"):
        cluster = CLUSTERS[key]
        assert cluster.devices_per_node == 8
        # 8 OAMs per node stay inside the Infinity Fabric full mesh: intra link for TP<=8.
        assert link_for_group(cluster, 8) is cluster.intra
        assert link_for_group(cluster, 16) is cluster.inter
        # Honest provenance: fabric/latency/price are disclosed, never presented as measured.
        assert "placeholder" in cluster.provenance["latency"]
        assert cluster.provenance["peak"].startswith("spec")


def test_amd_fp8_dtype_is_usable_by_the_planner():
    # The planner raises if a dtype is missing, so pin fp8 (the search space uses it).
    for key in ("mi300x_if_roce", "mi325x_if_roce", "mi355x_if_roce"):
        d = CLUSTERS[key].device
        assert flops(d, "fp8") > flops(d, "bf16") > flops(d, "fp32") > 0


def test_amd_domain_search_is_feasible_and_roofline_consistent():
    from nomo_planner.domains import build
    from nomo_planner.domains.search import enumerate_plans, Planner

    for key in ("mi300x_if_roce", "mi325x_if_roce", "mi355x_if_roce"):
        pack = build("llm_training", {"model": "llama3-8b", "cluster": key})
        spack = build("llm_inference", {"model": "llama3-8b", "cluster": key})
        tresult = Planner(pack).run()
        assert tresult["front"], key
        sresult = Planner(spack).run()
        assert sresult["front"], f"serving {key}"

    # Roofline sanity on fp8 compute: MI355X fp8 is ~1.9x MI300X fp8, so the same
    # compute-bound workload must be strictly faster on MI355X (ratio in (1.5, 2.0)).
    t300 = roofline_time(1e12, 0, CLUSTERS["mi300x_if_roce"].device, "fp8")
    t355 = roofline_time(1e12, 0, CLUSTERS["mi355x_if_roce"].device, "fp8")
    assert 1.5 < t300 / t355 < 2.0


def test_amd_plans_are_not_silently_calibrated():
    # Uncertainty intervals exist ONLY for the calibrated A100 cluster; AMD plans must
    # not inherit them (honesty boundary).
    from nomo_planner.domains.calibration import CALIBRATED

    for key in ("mi300x_if_roce", "mi325x_if_roce", "mi355x_if_roce"):
        assert key not in CALIBRATED
    assert "a100_nvlink_ib" in CALIBRATED
