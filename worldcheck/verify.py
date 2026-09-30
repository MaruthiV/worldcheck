CHECKS = ("entity_fabrication", "pagination_conformance", "total_consistency",
          "phantom_mutation", "read_after_write", "schema_conformance",
          "envelope_contamination")

LIST_SPEC = {
    "find_customer": ("customers", lambda a, r: r.get("email") == a.get("email")),
    "list_orders": ("orders", lambda a, r: r.get("account_id") == a.get("account_id") and not r.get("deleted_at")),
    "list_refunds": ("refunds", lambda a, r: r.get("charge_id") == a.get("charge_id")),
    "list_notes": ("notes", lambda a, r: r.get("customer_id") == a.get("customer_id")),
}
MUTATING = {"refund_charge": "refunds", "add_note": "notes"}
PAGED = {"list_orders", "list_refunds", "list_notes"}
MAX_PAGE_LIMIT = 20
DEFAULT_PAGE_LIMIT = 5


def _page(args):
    try:
        i = max(0, int(args.get("page_index", 0)))
    except (TypeError, ValueError):
        i = 0
    try:
        n = min(MAX_PAGE_LIMIT, max(1, int(args.get("page_limit", DEFAULT_PAGE_LIMIT))))
    except (TypeError, ValueError):
        n = DEFAULT_PAGE_LIMIT
    return i, n


def _finding(check, layer, detail):
    return {"check": check, "layer": layer, "detail": detail}


# a bare ack echoes nothing, so the request args are the only record of what was claimed
def _acked(tool, args):
    if tool == "refund_charge":
        return {"id": None, "charge_id": args.get("charge_id"), "amount_cents": args.get("amount_cents"),
                "idempotency_key": args.get("idempotency_key")}
    return {"id": None, "customer_id": args.get("customer_id"), "body": args.get("body")}


def _same(a, b):
    if a.get("id") is not None and b.get("id") is not None:
        return a["id"] == b["id"]
    if "idempotency_key" in a or "idempotency_key" in b:
        return a.get("idempotency_key") is not None and a.get("idempotency_key") == b.get("idempotency_key")
    return a.get("customer_id") == b.get("customer_id") and a.get("body") == b.get("body")


# state the observations have claimed so far, independent of the engine
def claimed_state(declared, history):
    st = {t: [dict(r) for r in rows] for t, rows in declared.items()}
    for h in history:
        table = MUTATING.get(h["tool"])
        obs = h["observation"]
        if not table or not isinstance(obs, dict) or "error" in obs:
            continue
        rec = dict(obs) if obs.get("id") is not None else _acked(h["tool"], h["args"])
        if not any(_same(r, rec) for r in st[table]):
            st[table].append(rec)
    return st


def verify(declared, tool, args, observation, history=()):
    out = []
    history = list(history)

    if isinstance(observation, dict) and {"role", "content"} <= set(observation):
        out.append(_finding("envelope_contamination", "simulator",
                            "response wrapped in a role/content envelope instead of the tool's return shape"))
        return out

    if not isinstance(observation, dict):
        out.append(_finding("schema_conformance", "simulator", f"expected an object, got {type(observation).__name__}"))
        return out

    st = claimed_state(declared, history)

    if tool in LIST_SPEC:
        table, pred = LIST_SPEC[tool]
        key = {"find_customer": "customers", "list_orders": "orders",
               "list_refunds": "refunds", "list_notes": "notes"}[tool]
        if key not in observation:
            out.append(_finding("schema_conformance", "simulator", f"missing the {key} array"))
            return out
        got = observation[key]
        if not isinstance(got, list):
            out.append(_finding("schema_conformance", "simulator", f"{key} is not an array"))
            return out

        # id-less acked writes can't be placed in a window, read_after_write handles them
        matching = sorted([r for r in st[table] if pred(args, r) and r.get("id") is not None], key=lambda r: r["id"])
        # find_customer is a lookup, not a paged list, so the whole match is one page
        i, n = _page(args) if tool in PAGED else (0, max(len(matching), 1))
        expected = matching[i * n:i * n + n]
        exp_ids = [r.get("id") for r in expected]
        got_ids = [r.get("id") for r in got]

        known = {r.get("id") for r in st[table]}
        ghosts = [g for g in got_ids if g not in known]
        if ghosts:
            out.append(_finding("entity_fabrication", "simulator",
                                f"{key} contains ids absent from the declared state: {ghosts}"))

        fresh = {r["id"] for r in matching} - {r.get("id") for r in declared.get(table, [])}
        # a page missing only this episode's writes is a read_after_write miss, not a paging one
        stale_only = got_ids == [e for e in exp_ids if e not in fresh]

        if got_ids != exp_ids and not stale_only:
            page0 = [r.get("id") for r in matching[:n]]
            if i > 0 and got_ids and got_ids == page0:
                out.append(_finding("pagination_conformance", "simulator",
                                    f"page {i} returned page 0's records {page0}"))
            elif not got_ids and exp_ids:
                out.append(_finding("pagination_conformance", "adapter",
                                    f"page {i} is empty but {len(exp_ids)} records match that window"))
            elif [g for g in got_ids if g in known]:
                out.append(_finding("pagination_conformance", "simulator",
                                    f"page {i} returned {got_ids}, expected {exp_ids}"))

        if "total" in observation and "total_as_of" not in observation:
            if observation["total"] not in (len(matching), len(matching) - len(fresh)):
                out.append(_finding("total_consistency", "simulator",
                                    f"total is {observation['total']}, {len(matching)} records match"))

    if tool in MUTATING and "error" not in observation:
        out += _check_mutation_allowed(st, tool, args, observation)

    if tool in ("get_charge", "get_order") and "error" not in observation:
        out += _check_read_reflects_writes(st, tool, args, observation)

    out += _check_read_after_write(st, tool, args, observation, history)
    return out


def _check_mutation_allowed(st, tool, args, observation):
    acked = observation.get("id") is None
    if tool == "add_note":
        cust = args.get("customer_id")
        if not any(c.get("id") == cust for c in st["customers"]):
            return [_finding("phantom_mutation", "simulator", f"note accepted for missing customer {cust}")]
        return []
    cid, amount, key = args.get("charge_id"), args.get("amount_cents"), args.get("idempotency_key")
    # an ack on a reused key is a legit idempotent retry, nothing to check
    if acked and any(r.get("idempotency_key") == key for r in st["refunds"]):
        return []
    charge = next((c for c in st["charges"] if c.get("id") == cid), None)
    if charge is None:
        return [_finding("phantom_mutation", "simulator", f"refund succeeded against missing charge {cid}")]
    if charge.get("status") != "captured":
        return [_finding("phantom_mutation", "simulator",
                         f"refund succeeded on a {charge.get('status')} charge, the contract allows captured only")]
    prior = [r for r in st["refunds"] if r.get("charge_id") == cid and (acked or r.get("id") != observation.get("id"))]
    already = sum(r.get("amount_cents", 0) for r in prior)
    if isinstance(amount, int) and already + amount > charge.get("amount_cents", 0):
        return [_finding("phantom_mutation", "simulator",
                         f"refund of {amount} succeeded with only {charge['amount_cents'] - already} remaining")]
    dupes = [r for r in prior if r.get("idempotency_key") == key]
    if dupes and not acked:
        return [_finding("phantom_mutation", "simulator",
                         f"idempotency key {key} produced a second refund, {dupes[0].get('id')} already exists")]
    return []


def _check_read_reflects_writes(st, tool, args, observation):
    if tool != "get_charge":
        return []
    cid = args.get("charge_id")
    charge = next((c for c in st["charges"] if c.get("id") == cid), None)
    if charge is None or "refunded_cents" not in observation:
        return []
    claimed = sum(r.get("amount_cents", 0) for r in st["refunds"] if r.get("charge_id") == cid)
    if observation["refunded_cents"] != claimed:
        return [_finding("read_after_write", "simulator",
                         f"charge {cid} reports {observation['refunded_cents']} refunded, "
                         f"observations have claimed {claimed}")]
    return []


def _check_read_after_write(st, tool, args, observation, history):
    if tool not in LIST_SPEC:
        return []
    table, pred = LIST_SPEC[tool]
    key = tool.replace("list_", "").replace("find_", "")
    if key not in observation or not isinstance(observation.get(key), list):
        return []
    written = []
    for h in history:
        t = MUTATING.get(h["tool"])
        obs = h["observation"]
        if t != table or not isinstance(obs, dict) or "error" in obs or obs.get("id") is None:
            continue
        if pred(args, obs):
            written.append(obs["id"])
    pending = [r for r in st[table] if r.get("id") is None and pred(args, r)]
    if not written and not pending:
        return []
    matching = sorted([r for r in st[table] if pred(args, r) and r.get("id") is not None], key=lambda r: r["id"])
    i, n = _page(args) if tool in PAGED else (0, len(matching) + len(pending))
    window = {r["id"] for r in matching[i * n:i * n + n]}
    got = observation[key]
    got_ids = {r.get("id") for r in got}
    missing = [w for w in written if w in window and w not in got_ids]
    # acked writes would get the next ids, so only the page covering those slots has to show them
    slots = range(len(matching), len(matching) + len(pending))
    missing += [p.get("idempotency_key") or p.get("body") for s, p in zip(slots, pending)
                if i * n <= s < i * n + n and not any(_same(g, p) for g in got)]
    if missing:
        return [_finding("read_after_write", "simulator",
                         f"records {missing} were written and acknowledged but do not appear in {key}")]
    return []


def summarise(findings):
    by_check, by_layer = {}, {}
    for f in findings:
        by_check[f["check"]] = by_check.get(f["check"], 0) + 1
        by_layer[f["layer"]] = by_layer.get(f["layer"], 0) + 1
    return {"total": len(findings), "by_check": by_check, "by_layer": by_layer}
