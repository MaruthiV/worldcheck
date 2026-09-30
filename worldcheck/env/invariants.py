TABLES = ("customers", "accounts", "orders", "order_items", "charges", "refunds", "notes")


def fingerprint(env):
    return {t: env._rows(f"select * from {t} order by id") for t in TABLES}


def diff(before, after):
    out = {}
    for t in TABLES:
        b = {r["id"]: r for r in before[t]}
        a = {r["id"]: r for r in after[t]}
        added = [a[i] for i in a if i not in b]
        removed = [b[i] for i in b if i not in a]
        changed = [(b[i], a[i]) for i in a if i in b and a[i] != b[i]]
        if added or removed or changed:
            out[t] = {"added": added, "removed": removed, "changed": changed}
    return out


# re-checked straight from the db so an engine bug cannot hide behind a tool
def check(env):
    bad = []
    for c in env._rows("select id, amount_cents, status from charges"):
        s = env._one("select coalesce(sum(amount_cents),0) as s from refunds where charge_id = ?", c["id"])["s"]
        if s > c["amount_cents"]:
            bad.append(f"charge {c['id']} over-refunded: {s} of {c['amount_cents']}")
        if s > 0 and c["status"] != "captured":
            bad.append(f"charge {c['id']} is {c['status']} but has {s} cents refunded")
    dupes = env._rows("select idempotency_key, count(*) n from refunds group by idempotency_key having n > 1")
    for d in dupes:
        bad.append(f"idempotency_key {d['idempotency_key']} used by {d['n']} refunds")
    for r in env._rows("select id, amount_cents from refunds"):
        if not isinstance(r["amount_cents"], int) or r["amount_cents"] <= 0:
            bad.append(f"refund {r['id']} has a non-positive or non-integer amount: {r['amount_cents']!r}")
    for c in env._rows("select id, amount_cents from charges"):
        if not isinstance(c["amount_cents"], int):
            bad.append(f"charge {c['id']} amount is not an integer: {c['amount_cents']!r}")
    orphan = env._rows("select id, charge_id from refunds where charge_id not in (select id from charges)")
    for o in orphan:
        bad.append(f"refund {o['id']} points at missing charge {o['charge_id']}")
    return bad
