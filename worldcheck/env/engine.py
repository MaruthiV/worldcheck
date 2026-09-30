import json
import sqlite3

SCHEMA = """
create table customers(id integer primary key, email text, full_name text, created_at text);
create table accounts(id integer primary key, customer_id integer, label text, closed_at text);
create table orders(id integer primary key, account_id integer, placed_at text, status text, deleted_at text);
create table order_items(id integer primary key, order_id integer, sku text, qty integer, unit_cents integer);
create table charges(id integer primary key, order_id integer, amount_cents integer, status text, captured_at text);
create table refunds(id integer primary key, charge_id integer, amount_cents integer, idempotency_key text, created_at text);
create table notes(id integer primary key, customer_id integer, body text, created_at text);
create table audit(seq integer primary key autoincrement, at text, tool text, args_json text, effect_json text);
create table list_snapshots(key text primary key, total integer, taken_at text);
"""

MAX_PAGE_LIMIT = 20
DEFAULT_PAGE_LIMIT = 5
TIMEZONE_RULE = "every timestamp below is local time at UTC+05:30"


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


class Env:
    def __init__(self, seed_sql, clock="2026-03-02T09:00:00"):
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.executescript(seed_sql)
        self.db.commit()
        self.now = clock

    def tick(self, seconds=60):
        h, m, s = (int(x) for x in self.now[11:].split(":"))
        total = h * 3600 + m * 60 + s + seconds
        self.now = f"{self.now[:11]}{total // 3600 % 24:02d}:{total // 60 % 60:02d}:{total % 60:02d}"

    def snapshot(self):
        return self.db.serialize(), self.now

    def restore(self, snap):
        blob, now = snap
        self.db.deserialize(blob)
        self.now = now

    def _rows(self, sql, *a):
        return [dict(r) for r in self.db.execute(sql, a).fetchall()]

    def _one(self, sql, *a):
        r = self.db.execute(sql, a).fetchone()
        return dict(r) if r else None

    def _audit(self, tool, args, effect):
        self.db.execute("insert into audit(at, tool, args_json, effect_json) values(?,?,?,?)",
                        (self.now, tool, json.dumps(args, sort_keys=True), json.dumps(effect, sort_keys=True)))
        self.db.commit()

    def mutations(self):
        return self._rows("select seq, at, tool, args_json, effect_json from audit order by seq")

    def call(self, tool, **args):
        fn = getattr(self, f"t_{tool}", None)
        if fn is None:
            return {"error": f"unknown tool {tool}"}
        self.tick()
        return fn(args)

    def t_find_customer(self, args):
        email = args.get("email")
        rows = self._rows("select * from customers where email = ? order by id", email)
        return {"total": len(rows), "customers": rows}

    def t_list_orders(self, args):
        acct = args.get("account_id")
        i, n = _page(args)
        rows = self._rows(
            "select id, account_id, placed_at, status from orders "
            "where account_id = ? and deleted_at is null order by id", acct)
        snap = self._one("select total, taken_at from list_snapshots where key = ?", f"orders:{acct}")
        out = {"orders": rows[i * n:i * n + n]}
        if snap:
            out["total"] = snap["total"]
            out["total_as_of"] = snap["taken_at"]
        else:
            out["total"] = len(rows)
        return out

    def t_get_order(self, args):
        o = self._one("select * from orders where id = ?", args.get("order_id"))
        if not o:
            return {"error": "order not found"}
        o["items"] = self._rows("select id, sku, qty, unit_cents from order_items where order_id = ? order by id", o["id"])
        o["subtotal_cents"] = sum(it["qty"] * it["unit_cents"] for it in o["items"])
        return o

    def t_get_charge(self, args):
        c = self._one("select * from charges where id = ?", args.get("charge_id"))
        if not c:
            return {"error": "charge not found"}
        refunded = self._one("select coalesce(sum(amount_cents), 0) as s from refunds where charge_id = ?", c["id"])["s"]
        c["refunded_cents"] = refunded
        c["remaining_cents"] = c["amount_cents"] - refunded
        return c

    def t_refund_charge(self, args):
        cid, amount, key = args.get("charge_id"), args.get("amount_cents"), args.get("idempotency_key")
        if not isinstance(amount, int) or isinstance(amount, bool):
            return {"error": "amount_cents must be an integer number of cents"}
        if not key:
            return {"error": "idempotency_key is required"}
        prior = self._one("select * from refunds where idempotency_key = ?", key)
        if prior:
            return prior
        c = self._one("select * from charges where id = ?", cid)
        if not c:
            return {"error": "charge not found"}
        if c["status"] != "captured":
            return {"error": f"charge is {c['status']}, only a captured charge can be refunded"}
        if amount <= 0:
            return {"error": "amount_cents must be positive"}
        refunded = self._one("select coalesce(sum(amount_cents), 0) as s from refunds where charge_id = ?", cid)["s"]
        if refunded + amount > c["amount_cents"]:
            return {"error": f"refund of {amount} exceeds remaining {c['amount_cents'] - refunded}"}
        cur = self.db.execute(
            "insert into refunds(charge_id, amount_cents, idempotency_key, created_at) values(?,?,?,?)",
            (cid, amount, key, self.now))
        self.db.commit()
        out = self._one("select * from refunds where id = ?", cur.lastrowid)
        self._audit("refund_charge", args, out)
        return out

    def t_list_refunds(self, args):
        cid = args.get("charge_id")
        i, n = _page(args)
        rows = self._rows("select * from refunds where charge_id = ? order by id", cid)
        return {"total": len(rows), "refunds": rows[i * n:i * n + n]}

    def t_add_note(self, args):
        cust, body = args.get("customer_id"), args.get("body")
        if not self._one("select id from customers where id = ?", cust):
            return {"error": "customer not found"}
        if not body:
            return {"error": "body is required"}
        cur = self.db.execute("insert into notes(customer_id, body, created_at) values(?,?,?)",
                              (cust, body, self.now))
        self.db.commit()
        out = self._one("select * from notes where id = ?", cur.lastrowid)
        self._audit("add_note", args, out)
        return out

    def t_list_notes(self, args):
        cust = args.get("customer_id")
        i, n = _page(args)
        rows = self._rows("select * from notes where customer_id = ? order by id", cust)
        return {"total": len(rows), "notes": rows[i * n:i * n + n]}

    def describe(self):
        lines = [f"Current time: {self.now}", f"Timezone rule: {TIMEZONE_RULE}", "", "Database records:"]
        for table, cols in (
            ("customers", "id, email, full_name"),
            ("accounts", "id, customer_id, label, closed_at"),
            ("orders", "id, account_id, placed_at, status, deleted_at"),
            ("order_items", "id, order_id, sku, qty, unit_cents"),
            ("charges", "id, order_id, amount_cents, status"),
            ("refunds", "id, charge_id, amount_cents, idempotency_key"),
            ("notes", "id, customer_id, body"),
        ):
            rows = self._rows(f"select {cols} from {table} order by id")
            lines.append(f"  {table} ({len(rows)} total, {len(rows)} shown):")
            for r in rows:
                lines.append("    - " + ", ".join(f"{k}: {v}" for k, v in r.items() if v is not None))
        snaps = self._rows("select key, total, taken_at from list_snapshots order by key")
        if snaps:
            lines.append("  index snapshots (totals served by list_orders, may lag the records above):")
            for s in snaps:
                lines.append(f"    - {s['key']}: total {s['total']} as of {s['taken_at']}")
        return "\n".join(lines)
