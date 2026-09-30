import json

import pytest

from worldcheck import answers, upstream
from worldcheck.driver import SPLIT


@pytest.fixture(scope="module")
def plugin(patronus):
    return upstream.load_adapter(patronus)


@pytest.fixture(scope="module")
def counting_rows(patronus):
    rows = [json.loads(l) for l in open(patronus / SPLIT)]
    return [r for r in rows if r["ground_truth"].isdigit()]


def test_every_counting_question_is_covered(counting_rows):
    assert len(counting_rows) == 6


def test_listing_every_number_scores_like_the_exact_answer(plugin, counting_rows):
    for row in counting_rows:
        assert answers.score(plugin, row, row["ground_truth"]) == 1.0
        assert answers.score(plugin, row, answers.NUMBER_DUMP) == 1.0, row["task_id"]


def test_a_wrong_number_that_contains_the_answer_gets_full_credit(plugin, counting_rows):
    for row in counting_rows:
        assert answers.score(plugin, row, "1" + row["ground_truth"]) == 1.0


def test_off_by_one_is_scored_far_below_either(plugin, counting_rows):
    for row in counting_rows:
        assert answers.score(plugin, row, str(int(row["ground_truth"]) + 1)) < 0.3
