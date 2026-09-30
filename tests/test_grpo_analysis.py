import json

import pytest

import gpu.analyze as an


def episode(task, gt, answer, pages, reward, sources=("guard",)):
    return {"task_id": task, "sample": 0, "ground_truth": gt, "answer": answer, "reward": reward,
            "turns": 4, "unparsed_turns": 0, "list_pages": list(pages), "sources": list(sources), "tools": []}


def fake_run(correct_share, pages_deep):
    eps = []
    for i in range(10):
        ok = i < 10 * correct_share
        eps.append(episode(f"q{i}", "24", "24" if ok else "7", [0, 1] if pages_deep else [0], 0.5))
        eps.append(episode(f"a{i}", "", None, [0, 1, 2] if pages_deep else [0], 0.4))
    return {"episodes": eps}


@pytest.fixture
def raw(tmp_path, monkeypatch):
    monkeypatch.setattr(an, "RAW", tmp_path)
    monkeypatch.setattr(an, "OUT", tmp_path / "summary.json")
    monkeypatch.setattr(an, "RESAMPLES", 500)
    for s in an.SEEDS:
        for ev in ("patched", "shipped"):
            (tmp_path / f"grpo-patched-s{s}-{ev}.json").write_text(json.dumps(fake_run(0.8, True)))
            (tmp_path / f"grpo-shipped-s{s}-{ev}.json").write_text(json.dumps(fake_run(0.2, False)))
    (tmp_path / "sft-s0-patched.json").write_text(json.dumps(fake_run(0.5, True)))
    return tmp_path


def test_a_planted_difference_is_supported(raw):
    an.main()
    eff = json.loads((raw / "summary.json").read_text())["effects"]["under the patched plugin"]
    assert eff["question_accuracy"]["patched_minus_shipped"] == pytest.approx(0.6)
    assert eff["question_accuracy"]["supported"]
    assert eff["pages_past_first"]["supported"]


def test_no_difference_is_not_supported(raw):
    for s in an.SEEDS:
        (raw / f"grpo-patched-s{s}-patched.json").write_text(json.dumps(fake_run(0.2, False)))
    an.main()
    eff = json.loads((raw / "summary.json").read_text())["effects"]["under the patched plugin"]
    assert eff["question_accuracy"]["patched_minus_shipped"] == 0
    assert not eff["question_accuracy"]["supported"]


def test_strict_answer_match_rejects_the_dump():
    assert an.correct("24", "24") and an.correct(" 24 ", "24") and an.correct("24.0", "24")
    assert not an.correct("0 1 2 24 100", "24") and not an.correct("124", "24")
    assert an.correct("Azure Skies", "azure skies") and not an.correct("Azure Skies, Rain", "Azure Skies")


def test_multi_number_answers_are_flagged():
    assert an.METRICS["multi_number_answer"](episode("q", "24", "0 1 2 24", [], 1.0))
    assert not an.METRICS["multi_number_answer"](episode("q", "24", "24", [], 1.0))
    assert an.METRICS["multi_number_answer"](episode("q", "Azure", "1 2", [], 1.0)) is None
