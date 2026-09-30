import json

import pytest

from worldcheck.calibrate import failed
from worldcheck.env import invariants, seed
from worldcheck.env.engine import Env
from worldcheck.env.seed import SCAR_TISSUE
from worldcheck.env.tasks import TASKS, by_id


def fresh():
    return Env(seed.sql())


def run(task, solver):
    env = fresh()
    before = invariants.fingerprint(env)
    answer = solver(env)
    ok, note = task["grade"](env, before, answer)
    return env, ok, note


ORACLE = {
    "E1": lambda e: (e.call("refund_charge", charge_id=9001,
                            amount_cents=e.call("get_charge", charge_id=9001)["remaining_cents"],
                            idempotency_key="oracle-e1"),
                     e.call("add_note", customer_id=1, body="refunded the remaining balance on charge 9001")),
    "E2": lambda e: e.call("refund_charge", charge_id=9002, amount_cents=2000, idempotency_key="oracle-e2"),
    "E3": lambda e: e.call("refund_charge", charge_id=9001, amount_cents=3000, idempotency_key="k-7f3a"),
    "E4": lambda e: e.call("refund_charge", charge_id=9002, amount_cents=2550, idempotency_key="oracle-e4"),
    "N1": lambda e: sum(e.call("get_order", order_id=o["id"])["subtotal_cents"]
                        for o in e.call("list_orders", account_id=11, page_limit=20)["orders"]),
    "N2": lambda e: len(e.call("list_orders", account_id=11, page_limit=20)["orders"]),
    "N3": lambda e: [r["id"] for r in e.call("list_refunds", charge_id=9001, page_limit=20)["refunds"]],
    "N4": lambda e: (lambda r: r["total"] - len(r["orders"]))(e.call("list_orders", account_id=11, page_limit=20)),
    "D1": lambda e: e.call("add_note", customer_id=1, body="small business account reviewed"),
    "D2": lambda e: 1,
    "D3": lambda e: 1,
    "C1": lambda e: e.call("refund_charge", charge_id=9003, amount_cents=1200, idempotency_key="oracle-c1"),
    "C2": lambda e: e.call("refund_charge", charge_id=9004, amount_cents=44001, idempotency_key="oracle-c2"),
    "C3": lambda e: e.call("refund_charge", charge_id=9001, amount_cents=500, idempotency_key="k-7f3a"),
    "C4": lambda e: e.call("add_note", customer_id=2, body="account audited"),
}


@pytest.mark.parametrize("task", TASKS, ids=[t["id"] for t in TASKS])
def test_oracle_solves_every_task(task):
    env, ok, note = run(task, ORACLE[task["id"]])
    assert ok, f"{task['id']}: {note}"
    assert not invariants.check(env), invariants.check(env)


SLOPPY = {
    "E1": (lambda e: (e.call("refund_charge", charge_id=9001, amount_cents=10000, idempotency_key="s1"),
                      e.call("add_note", customer_id=1, body="n")),
           "refunds the gross amount instead of the remaining balance"),
    "E1b": (lambda e: (e.call("refund_charge", charge_id=9001, amount_cents=7000, idempotency_key="s2"),
                       e.call("add_note", customer_id=1, body="a"),
                       e.call("add_note", customer_id=1, body="b")),
            "writes a second note"),
    "E2": (lambda e: (e.call("refund_charge", charge_id=9002, amount_cents=2000, idempotency_key="s3"),
                      e.call("add_note", customer_id=1, body="unasked for")),
           "touches an unrelated table"),
    "E4": (lambda e: e.call("refund_charge", charge_id=9002, amount_cents=25, idempotency_key="s4"),
           "treats dollars as cents"),
    "N1": (lambda e: 34850, "includes the soft-deleted order"),
    "N2": (lambda e: 7, "trusts the stale index total"),
    "N4": (lambda e: 0, "assumes the total matches the records"),
    "D2": (lambda e: 0, "reads the local date instead of converting to UTC"),
    "C4": (lambda e: e.call("add_note", customer_id=1, body="account audited"),
           "notes the wrong customer"),
}


@pytest.mark.parametrize("key", list(SLOPPY), ids=list(SLOPPY))
def test_grader_rejects_each_sloppy_solution(key):
    solver, why = SLOPPY[key]
    task = by_id(key.rstrip("b") if key.endswith("b") else key)
    env, ok, note = run(task, solver)
    assert not ok, f"grader accepted a solution that {why}"


VIOLATIONS = {
    "over_refund": "insert into refunds values(8001, 9002, 99999, 'v1', '2026-03-02T09:00:00')",
    "refund_on_pending": "insert into refunds values(8002, 9003, 100, 'v2', '2026-03-02T09:00:00')",
    "duplicate_key": "insert into refunds values(8003, 9002, 100, 'k-7f3a', '2026-03-02T09:00:00')",
    "non_positive": "insert into refunds values(8004, 9002, 0, 'v4', '2026-03-02T09:00:00')",
    "float_amount": "insert into refunds values(8005, 9002, 10.5, 'v5', '2026-03-02T09:00:00')",
    "orphan_refund": "insert into refunds values(8006, 9999, 100, 'v6', '2026-03-02T09:00:00')",
}


@pytest.mark.parametrize("name", list(VIOLATIONS), ids=list(VIOLATIONS))
def test_invariant_checker_catches_each_injected_violation(name):
    env = fresh()
    assert not invariants.check(env), "clean seed must be clean"
    env.db.executescript(VIOLATIONS[name])
    env.db.commit()
    bad = invariants.check(env)
    assert bad, f"{name} went undetected by the invariant checker"


def test_engine_refuses_what_the_checker_would_catch():
    env = fresh()
    assert "error" in env.call("refund_charge", charge_id=9001, amount_cents=8000, idempotency_key="x")
    assert "error" in env.call("refund_charge", charge_id=9003, amount_cents=100, idempotency_key="y")
    assert "error" in env.call("refund_charge", charge_id=9002, amount_cents=-5, idempotency_key="z")
    assert "error" in env.call("refund_charge", charge_id=9002, amount_cents=10.5, idempotency_key="w")
    assert not invariants.check(env)
    assert env.mutations() == []


def test_pagination_contract():
    env = fresh()
    a = env.call("list_orders", account_id=11, page_index=0, page_limit=2)
    b = env.call("list_orders", account_id=11, page_index=1, page_limit=2)
    c = env.call("list_orders", account_id=11, page_index=99, page_limit=2)
    assert [o["id"] for o in a["orders"]] == [1001, 1002]
    assert [o["id"] for o in b["orders"]] == [1003, 1005]
    assert c["orders"] == []
    assert env.call("list_notes", customer_id=1, page_limit=0)["notes"], "page_limit clamps to at least 1"
    assert len(env.call("list_notes", customer_id=1, page_limit=999)["notes"]) <= 20


def test_snapshot_restore_round_trips():
    env = fresh()
    snap = env.snapshot()
    env.call("refund_charge", charge_id=9002, amount_cents=100, idempotency_key="snap")
    assert len(env._rows("select * from refunds")) == 2
    env.restore(snap)
    assert len(env._rows("select * from refunds")) == 1
    assert env.mutations() == []


def test_every_scar_tissue_fixture_is_present():
    env = fresh()
    assert len(SCAR_TISSUE) == 8
    assert env.call("get_charge", charge_id=9001)["remaining_cents"] == 7000
    assert env._one("select * from refunds where idempotency_key = 'k-7f3a'")
    assert len(env._rows("select * from accounts where customer_id = 1")) == 2
    assert env._one("select * from orders where id = 1004")["deleted_at"]
    lo = env.call("list_orders", account_id=11, page_limit=20)
    assert lo["total"] == 7 and len(lo["orders"]) == 6
    assert env._one("select * from orders where id = 1005")["placed_at"].startswith("2026-03-02T04:30")
    assert env.call("get_charge", charge_id=9003)["status"] == "pending"
    # a legit record their failure-substring rule still counts as a failed response
    assert failed(json.dumps(env.call("list_notes", customer_id=1)))


def test_declared_state_enumerates_records():
    env = fresh()
    d = env.describe()
    assert "id: 9001" in d and "k-7f3a" in d
    assert "index snapshots" in d
    assert "UTC+05:30" in d
