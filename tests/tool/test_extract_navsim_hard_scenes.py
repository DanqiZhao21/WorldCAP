import pytest

from tool.smalltool.extract_navsim_hard_scenes import build_subsets


def test_build_subsets_uses_quantiles_and_composes_union():
    rows = [
        {
            "token": "straight_easy",
            "abs_yaw_deg": 1.0,
            "latacc_proxy": 0.01,
            "speed_delta_mps": 0.1,
            "max_acc_mps2": 0.2,
            "max_yaw_rate_deg_s": 0.5,
            "near_agents_30m": 1,
            "has_traffic_light": False,
        },
        {
            "token": "curved",
            "abs_yaw_deg": 30.0,
            "latacc_proxy": 0.3,
            "speed_delta_mps": 0.2,
            "max_acc_mps2": 0.4,
            "max_yaw_rate_deg_s": 4.0,
            "near_agents_30m": 4,
            "has_traffic_light": False,
        },
        {
            "token": "dynamic",
            "abs_yaw_deg": 5.0,
            "latacc_proxy": 0.05,
            "speed_delta_mps": 5.0,
            "max_acc_mps2": 3.0,
            "max_yaw_rate_deg_s": 8.0,
            "near_agents_30m": 2,
            "has_traffic_light": False,
        },
        {
            "token": "interaction",
            "abs_yaw_deg": 10.0,
            "latacc_proxy": 0.02,
            "speed_delta_mps": 0.3,
            "max_acc_mps2": 0.5,
            "max_yaw_rate_deg_s": 1.0,
            "near_agents_30m": 10,
            "has_traffic_light": True,
        },
        {
            "token": "fast_curve",
            "abs_yaw_deg": 20.0,
            "latacc_proxy": 2.0,
            "speed_delta_mps": 0.4,
            "max_acc_mps2": 0.6,
            "max_yaw_rate_deg_s": 3.0,
            "near_agents_30m": 3,
            "has_traffic_light": False,
        },
    ]

    subsets, thresholds = build_subsets(rows, hard_percentile=80.0, curved_strict_percentile=90.0)

    assert thresholds["abs_yaw_deg_p80"] == pytest.approx(22.0)
    assert subsets["navtest_hard_curved_p80"] == ["curved"]
    assert subsets["navtest_hard_curved_p90"] == ["curved"]
    assert subsets["navtest_hard_dynamic_p80"] == ["dynamic"]
    assert subsets["navtest_hard_fast_curve_p80"] == ["fast_curve"]
    assert subsets["navtest_hard_interaction_p80"] == ["interaction"]
    assert subsets["navtest_hard_composite_p80"] == [
        "curved",
        "dynamic",
        "fast_curve",
        "interaction",
    ]
