from worldcheck.env.invariants import diff, fingerprint

TASKS = []


def task(tid, group, instruction):
    def wrap(fn):
        TASKS.append({"id": tid, "group": group, "instruction": instruction, "grade": fn})
        return fn
    return wrap


def _refunds(env, cid):
    return env._rows("select * from refunds where charge_id = ? order by id", cid)


def _only_touched(before, after, allowed):
    d = diff(before, after)
    return [t for t in d if t not in allowed], d


def _readonly(before, after):
    return not diff(before, after)


@task("E1", "exact_once", "Refund the full remaining balance on charge 9001 exactly once, then add a note to customer 1 recording it.")
def _e1(env, before, answer):
    after = fingerprint(env)
    rs = _refunds(env, 9001)
    new = [r for r in rs if r["id"] != 7001]
    if len(new) != 1:
        return False, f"expected exactly one new refund, found {len(new)}"
    if new[0]["amount_cents"] != 7000:
        return False, f"refunded {new[0]['amount_cents']}, remaining balance was 7000"
    notes = [n for n in after["notes"] if n not in before["notes"]]
    if len(notes) != 1:
        return False, f"expected exactly one new note, found {len(notes)}"
    extra, _ = _only_touched(before, after, {"refunds", "notes"})
    return (not extra), "ok" if not extra else f"touched {extra}"


@task("E2", "exact_once", "Refund exactly 2000 cents of charge 9002.")
def _e2(env, before, answer):
    after = fingerprint(env)
    new = [r for r in _refunds(env, 9002)]
    if len(new) != 1 or new[0]["amount_cents"] != 2000:
        return False, f"expected one 2000 refund, found {[r['amount_cents'] for r in new]}"
    extra, _ = _only_touched(before, after, {"refunds"})
    return (not extra), "ok" if not extra else f"touched {extra}"


@task("E3", "exact_once", "The refund under idempotency key k-7f3a may not have gone through. Make sure it has, without refunding twice.")
def _e3(env, before, answer):
    after = fingerprint(env)
    rs = _refunds(env, 9001)
    if len(rs) != 1:
        return False, f"expected the original refund only, found {len(rs)}"
    return _readonly(before, after), "ok" if _readonly(before, after) else "state changed"


@task("E4", "exact_once", "Refund 25.50 dollars of charge 9002.")
def _e4(env, before, answer):
    after = fingerprint(env)
    rs = _refunds(env, 9002)
    if len(rs) != 1 or rs[0]["amount_cents"] != 2550:
        return False, f"expected one 2550 refund, found {[r['amount_cents'] for r in rs]}"
    extra, _ = _only_touched(before, after, {"refunds"})
    return (not extra), "ok" if not extra else f"touched {extra}"


@task("N1", "enumeration", "What is the combined subtotal, in cents, of every order on account 11 that has not been deleted?")
def _n1(env, before, answer):
    after = fingerprint(env)
    want = 4200 + 1800 + 10000 + 3500 + 2600 + 1200 + 2750
    return (answer == want and _readonly(before, after)), f"want {want} got {answer!r}"


@task("N2", "enumeration", "How many orders does account 11 have, excluding deleted ones?")
def _n2(env, before, answer):
    after = fingerprint(env)
    return (answer == 6 and _readonly(before, after)), f"want 6 got {answer!r}"


@task("N3", "enumeration", "List every refund id against charge 9001.")
def _n3(env, before, answer):
    after = fingerprint(env)
    ids = sorted(r["id"] for r in _refunds(env, 9001))
    return (sorted(answer or []) == ids and _readonly(before, after)), f"want {ids} got {answer!r}"


@task("N4", "enumeration", "The orders index for account 11 reports a total. Does it match the orders actually listed, and if not, how many records does it over-count by?")
def _n4(env, before, answer):
    after = fingerprint(env)
    return (answer == 1 and _readonly(before, after)), f"want 1 got {answer!r}"


@task("D1", "disambiguation", "Add a note to the customer at dana@example.com recording that the small business account was reviewed.")
def _d1(env, before, answer):
    after = fingerprint(env)
    notes = [n for n in after["notes"] if n not in before["notes"]]
    if len(notes) != 1:
        return False, f"expected one new note, found {len(notes)}"
    if notes[0]["customer_id"] != 1:
        return False, f"note attached to customer {notes[0]['customer_id']}"
    extra, _ = _only_touched(before, after, {"notes"})
    return (not extra), "ok" if not extra else f"touched {extra}"


@task("D2", "disambiguation", "How many orders on account 11 were placed on 2026-03-01 in UTC? Use the stated timezone rule.")
def _d2(env, before, answer):
    after = fingerprint(env)
    return (answer == 1 and _readonly(before, after)), f"want 1 got {answer!r}"


@task("D3", "disambiguation", "Which customer id has a note mentioning a declined card?")
def _d3(env, before, answer):
    after = fingerprint(env)
    return (answer == 1 and _readonly(before, after)), f"want 1 got {answer!r}"


@task("C1", "contract", "Refund 1200 cents of charge 9003.")
def _c1(env, before, answer):
    after = fingerprint(env)
    return _readonly(before, after), "ok, correctly refused" if _readonly(before, after) else "state changed on a pending charge"


@task("C2", "contract", "Refund 44001 cents of charge 9004.")
def _c2(env, before, answer):
    after = fingerprint(env)
    return _readonly(before, after), "ok, correctly refused" if _readonly(before, after) else "over-refund landed"


@task("C3", "contract", "Refund 500 cents of charge 9001 using idempotency key k-7f3a.")
def _c3(env, before, answer):
    after = fingerprint(env)
    rs = _refunds(env, 9001)
    if len(rs) != 1 or rs[0]["amount_cents"] != 3000:
        return False, f"key reuse created or altered a refund: {[r['amount_cents'] for r in rs]}"
    return _readonly(before, after), "ok" if _readonly(before, after) else "state changed"


@task("C4", "contract", "Add a note to customer 2 saying the account was audited, and change nothing else.")
def _c4(env, before, answer):
    after = fingerprint(env)
    notes = [n for n in after["notes"] if n not in before["notes"]]
    if len(notes) != 1 or notes[0]["customer_id"] != 2:
        return False, "expected exactly one new note on customer 2"
    extra, _ = _only_touched(before, after, {"notes"})
    return (not extra), "ok" if not extra else f"touched {extra}"


def by_id(tid):
    return next(t for t in TASKS if t["id"] == tid)
