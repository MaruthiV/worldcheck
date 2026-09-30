import json
from pathlib import Path

import pytest

from worldcheck import calibrate as cal
from worldcheck.policies import POLICIES

COMMITTED = Path(__file__).resolve().parent.parent / "results" / "calibration.json"


@pytest.fixture(scope="module")
def res():
    return cal.calibrate()


def test_committed_results_match_a_fresh_run(res):
    assert json.loads(COMMITTED.read_text()) == json.loads(json.dumps(res)), \
        "results/calibration.json is stale, rerun python -m worldcheck.calibrate"


def test_clean_simulator_shadow_is_exactly_truth(res):
    gt = res["configs"]["ground_truth"]
    assert gt["shadow"] == res["truth"]
    assert gt["shadow_vs_truth"]["counts"]["agree"] == 36


def test_clean_simulator_training_signal_is_the_reward_only_view(res):
    assert res["configs"]["ground_truth"]["training_signal"] == res["reward_only"]


def test_verifier_is_silent_on_real_policy_traffic_against_the_reference(res):
    assert res["configs"]["ground_truth"]["verifier"]["episodes_flagged"] == 0


def test_paging_matters_in_the_reference(res):
    t = res["truth"]
    assert all(t[f"one_page+{w}"] < t[f"until_empty+{w}"] for w in ("blind", "check_first", "write_then_check"))


def test_policy_set_is_the_full_factorial():
    assert len(POLICIES) == len(set(POLICIES)) == 9


def test_proxy_counts_an_empty_page_with_a_total_as_success():
    trace = [{"tool": "list_orders", "args": {"page_index": 1}, "observation": {"total": 7, "orders": []}}]
    assert cal.proxy(trace, None, None) == pytest.approx(0.6)


def test_proxy_uses_their_substring_match_for_answers():
    assert cal.proxy([], "16", 6) == 1.0
    assert cal.proxy([], "5", 6) < 1.0


def test_same_run_twice_is_identical(res):
    assert cal.calibrate() == res
