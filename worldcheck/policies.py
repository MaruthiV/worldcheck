import json
import math
from datetime import datetime, timedelta

KEYS = {"find_customer": "customers", "list_orders": "orders",
        "list_refunds": "refunds", "list_notes": "notes"}
PAGED = {"list_orders", "list_refunds", "list_notes"}
PAGE_LIMIT = 5
MAX_PAGES = 10
PAGING = ("one_page", "until_empty", "by_total")
WRITING = ("blind", "check_first", "write_then_check")
POLICIES = [f"{p}+{w}" for p in PAGING for w in WRITING]


# agents read through a role/content envelope, same as their plugin's json unwrapping
def unwrap(obs):
    if isinstance(obs, dict) and {"role", "content"} <= set(obs):
        try:
            obs = json.loads(obs["content"])
        except (TypeError, ValueError):
            return {}
    return obs if isinstance(obs, dict) else {}


class Agent:
    def __init__(self, backend, policy):
        self.b = backend
        self.paging, self.writing = policy.split("+")
        self.n_keys = 0

    def call(self, tool, **args):
        return unwrap(self.b.call(tool, **args))

    def fresh_key(self):
        self.n_keys += 1
        return f"agent-{self.n_keys}"

    # no dedupe on purpose, their table 7 (i) agent double counts
    def listing(self, tool, **args):
        key = KEYS[tool]
        if tool not in PAGED:
            got = self.call(tool, **args)
            return list(got.get(key, [])), got.get("total")
        first = self.call(tool, page_index=0, page_limit=PAGE_LIMIT, **args)
        recs, total = list(first.get(key, [])), first.get("total")
        if self.paging == "one_page":
            return recs, total
        if self.paging == "by_total":
            pages = range(1, min(MAX_PAGES, math.ceil((total or 0) / PAGE_LIMIT)))
        else:
            pages = range(1, MAX_PAGES)
        for p in pages:
            got = self.call(tool, page_index=p, page_limit=PAGE_LIMIT, **args).get(key, [])
            if self.paging == "until_empty" and not got:
                break
            recs += got
        return recs, total

    def _refund_visible(self, charge_id, key):
        done, _ = self.listing("list_refunds", charge_id=charge_id)
        return any(r.get("idempotency_key") == key for r in done)

    def refund(self, charge_id, amount, key=None):
        key = key or self.fresh_key()
        if self.writing == "check_first":
            if self._refund_visible(charge_id, key):
                return
            c = self.call("get_charge", charge_id=charge_id)
            if c.get("status") != "captured" or amount > c.get("remaining_cents", 0):
                return
        self.call("refund_charge", charge_id=charge_id, amount_cents=amount, idempotency_key=key)
        if self.writing == "write_then_check":
            for _ in range(2):
                if self._refund_visible(charge_id, key):
                    return
                key = self.fresh_key()
                self.call("refund_charge", charge_id=charge_id, amount_cents=amount, idempotency_key=key)

    def _note_visible(self, customer_id, body):
        notes, _ = self.listing("list_notes", customer_id=customer_id)
        return any(n.get("body") == body for n in notes)

    def note(self, customer_id, body):
        if self.writing == "check_first" and self._note_visible(customer_id, body):
            return
        self.call("add_note", customer_id=customer_id, body=body)
        if self.writing == "write_then_check":
            for _ in range(2):
                if self._note_visible(customer_id, body):
                    return
                self.call("add_note", customer_id=customer_id, body=body)


def _utc_date(local):
    try:
        return (datetime.fromisoformat(local) - timedelta(hours=5, minutes=30)).date().isoformat()
    except (TypeError, ValueError):
        return None


def _n1(a):
    orders, _ = a.listing("list_orders", account_id=11)
    return sum(a.call("get_order", order_id=o["id"]).get("subtotal_cents", 0) for o in orders)


def _n4(a):
    orders, total = a.listing("list_orders", account_id=11)
    return (total or 0) - len(orders)


def _d1(a):
    custs, _ = a.listing("find_customer", email="dana@example.com")
    if custs:
        a.note(custs[0]["id"], "small business account reviewed")


def _d2(a):
    orders, _ = a.listing("list_orders", account_id=11)
    return sum(1 for o in orders if _utc_date(o.get("placed_at", "")) == "2026-03-01")


def _d3(a):
    for cid in (1, 2):
        notes, _ = a.listing("list_notes", customer_id=cid)
        if any("declined" in (n.get("body") or "") for n in notes):
            return cid
    return None


def _e1(a):
    c = a.call("get_charge", charge_id=9001)
    a.refund(9001, c.get("remaining_cents", 0))
    a.note(1, "refunded the remaining balance on charge 9001")


PLANS = {
    "E1": _e1,
    "E2": lambda a: a.refund(9002, 2000),
    "E3": lambda a: a.refund(9001, 3000, key="k-7f3a"),
    "E4": lambda a: a.refund(9002, 2550),
    "N1": _n1,
    "N2": lambda a: len(a.listing("list_orders", account_id=11)[0]),
    "N3": lambda a: [r.get("id") for r in a.listing("list_refunds", charge_id=9001)[0]],
    "N4": _n4,
    "D1": _d1,
    "D2": _d2,
    "D3": _d3,
    "C1": lambda a: a.refund(9003, 1200),
    "C2": lambda a: a.refund(9004, 44001),
    "C3": lambda a: a.refund(9001, 500, key="k-7f3a"),
    "C4": lambda a: a.note(2, "account audited"),
}


def run(backend, policy, task_id):
    return PLANS[task_id](Agent(backend, policy))
