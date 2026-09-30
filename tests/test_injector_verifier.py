import pytest

from worldcheck import verify
from worldcheck.env import invariants, seed
from worldcheck.env.engine import Env
from worldcheck.sim.injector import FAULTS, TABLE7, GroundTruth, Injector


def fresh():
    return Env(seed.sql())


def declared(env):
    return invariants.fingerprint(env)


def sweep(backend, account=11, limit=2, pages=4):
    backend.call("find_customer", email="dana@example.com")
    for p in range(pages):
        backend.call("list_orders", account_id=account, page_index=p, page_limit=limit)


def write_then_read(backend):
    backend.call("refund_charge", charge_id=9002, amount_cents=1000, idempotency_key="w1")
    backend.call("get_charge", charge_id=9002)
    backend.call("list_refunds", charge_id=9002, page_limit=20)


def findings_for(backend, decl):
    out, hist = [], []
    for step in backend.trace:
        out += verify.verify(decl, step["tool"], step["args"], step["observation"], hist)
        hist.append(step)
    return out


def checks(findings):
    return {f["check"] for f in findings}


def test_ground_truth_run_is_clean():
    env = fresh()
    decl = declared(env)
    b = GroundTruth(env)
    sweep(b)
    write_then_read(b)
    f = findings_for(b, decl)
    assert f == [], f


def test_clean_injector_is_also_clean():
    env = fresh()
    decl = declared(env)
    b = Injector(env, faults=())
    sweep(b)
    write_then_read(b)
    assert findings_for(b, decl) == []


EXPECTED = {
    "pagination_repeat": ("pagination_conformance", sweep),
    "pagination_empty": ("pagination_conformance", sweep),
    "entity_hallucination": ("entity_fabrication", sweep),
    "json_wrapped": ("envelope_contamination", sweep),
    "ack_without_persist": ("read_after_write", write_then_read),
    "stale_read": ("read_after_write", write_then_read),
}


@pytest.mark.parametrize("fault", FAULTS)
def test_every_fault_is_caught_by_its_intended_check(fault):
    want, drive = EXPECTED[fault]
    env = fresh()
    decl = declared(env)
    b = Injector(env, faults=(fault,))
    drive(b)
    got = checks(findings_for(b, decl))
    assert want in got, f"{fault} produced {got or 'nothing'}, expected {want}"


@pytest.mark.parametrize("fault", FAULTS)
def test_no_fault_goes_entirely_undetected(fault):
    _, drive = EXPECTED[fault]
    env = fresh()
    decl = declared(env)
    b = Injector(env, faults=(fault,))
    drive(b)
    assert findings_for(b, decl), f"{fault} went undetected"


def test_table7_modes_are_all_covered():
    assert set(TABLE7) <= set(FAULTS)
    for fault in TABLE7:
        env = fresh()
        decl = declared(env)
        b = Injector(env, faults=(fault,))
        sweep(b)
        assert findings_for(b, decl), f"their Table 7 mode {TABLE7[fault]} went undetected"


def test_faults_corrupt_the_view_not_the_state():
    env = fresh()
    b = Injector(env, faults=("pagination_empty", "entity_hallucination", "json_wrapped"))
    sweep(b)
    b.call("refund_charge", charge_id=9002, amount_cents=1000, idempotency_key="v1")
    assert not invariants.check(env)
    refunds = env._rows("select * from refunds where charge_id = 9002")
    assert len(refunds) == 1 and refunds[0]["amount_cents"] == 1000


def test_ack_without_persist_leaves_state_untouched():
    env = fresh()
    b = Injector(env, faults=("ack_without_persist",))
    obs = b.call("refund_charge", charge_id=9002, amount_cents=1000, idempotency_key="v2")
    assert obs["status"] == "success"
    assert env._rows("select * from refunds where charge_id = 9002") == []
    assert env.mutations() == []


def test_stale_read_reproduces_their_appendix_g_behaviour():
    env = fresh()
    b = Injector(env, faults=("stale_read",))
    b.call("refund_charge", charge_id=9002, amount_cents=1000, idempotency_key="v3")
    charge = b.call("get_charge", charge_id=9002)
    assert charge["refunded_cents"] == 0, "a stale read should not see the write"
    assert env.call("get_charge", charge_id=9002)["refunded_cents"] == 1000, "real state did advance"


def test_unknown_fault_is_rejected():
    with pytest.raises(ValueError):
        Injector(fresh(), faults=("not_a_fault",))


def test_documented_stale_total_is_not_flagged_as_fabrication():
    env = fresh()
    decl = declared(env)
    b = GroundTruth(env)
    b.call("list_orders", account_id=11, page_limit=20)
    f = findings_for(b, decl)
    assert "total_consistency" not in checks(f), "total_as_of marks a documented snapshot, not a lie"


def test_findings_name_a_layer():
    env = fresh()
    decl = declared(env)
    b = Injector(env, faults=("pagination_empty",))
    sweep(b)
    f = findings_for(b, decl)
    assert f and all(x["layer"] in ("reference", "adapter", "simulator", "reward") for x in f)


def acked(calls):
    env = fresh()
    decl = declared(env)
    b = Injector(env, faults=("ack_without_persist",))
    for tool, args in calls:
        b.call(tool, **args)
    return findings_for(b, decl)


def test_ack_on_a_forbidden_refund_is_a_phantom_mutation():
    f = acked([("refund_charge", {"charge_id": 9003, "amount_cents": 1200, "idempotency_key": "p1"})])
    assert checks(f) == {"phantom_mutation"}, f


def test_ack_for_a_missing_customer_is_a_phantom_mutation():
    f = acked([("add_note", {"customer_id": 77, "body": "x"})])
    assert checks(f) == {"phantom_mutation"}, f


def test_ack_on_a_reused_key_is_a_legit_retry():
    assert acked([("refund_charge", {"charge_id": 9001, "amount_cents": 3000, "idempotency_key": "k-7f3a"})]) == []


def test_unpersisted_note_is_only_missed_on_the_page_it_belongs_on():
    f = acked([("add_note", {"customer_id": 1, "body": "hello"}),
               ("list_notes", {"customer_id": 1, "page_limit": 2}),
               ("list_notes", {"customer_id": 1, "page_index": 1, "page_limit": 2})])
    assert len(f) == 1 and f[0]["check"] == "read_after_write" and "hello" in f[0]["detail"], f


def test_stale_read_counts_once_not_as_paging_or_total():
    env = fresh()
    decl = declared(env)
    b = Injector(env, faults=("stale_read",))
    write_then_read(b)
    assert checks(findings_for(b, decl)) == {"read_after_write"}
