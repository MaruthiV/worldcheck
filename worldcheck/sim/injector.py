import json

from worldcheck.env import seed
from worldcheck.env.engine import Env, _page

LIST_TOOLS = {"find_customer": "customers", "list_orders": "orders",
              "list_refunds": "refunds", "list_notes": "notes"}
MUTATING = {"refund_charge", "add_note"}

FAULTS = ("pagination_repeat", "pagination_empty", "entity_hallucination",
          "json_wrapped", "ack_without_persist", "stale_read")

TABLE7 = {"pagination_repeat": "i, incorrect pagination",
          "entity_hallucination": "ii, object hallucination",
          "json_wrapped": "iii, json-wrapped responses"}


class GroundTruth:
    name = "ground_truth"

    def __init__(self, env):
        self.env = env
        self.trace = []

    def call(self, tool, **args):
        obs = self.env.call(tool, **args)
        self.trace.append({"tool": tool, "args": args, "observation": obs, "faults": []})
        return obs


class Injector:
    def __init__(self, env, faults=(), tools=None):
        bad = set(faults) - set(FAULTS)
        if bad:
            raise ValueError(f"unknown faults {sorted(bad)}")
        self.env = env
        self.faults = tuple(faults)
        self.tools = tools
        self.trace = []
        self.name = "injector:" + (",".join(faults) or "clean")
        # a shadow of the initial state, never mutated, so stale_read has somewhere to read from
        self._shadow = Env(seed.sql(), clock=env.now)
        self._mutated = False

    def _on(self, fault, tool):
        if fault not in self.faults:
            return False
        return self.tools is None or tool in self.tools

    def call(self, tool, **args):
        applied = []

        if self._on("ack_without_persist", tool) and tool in MUTATING:
            applied.append("ack_without_persist")
            obs = {"status": "success", "message": "Action completed"}
            self.trace.append({"tool": tool, "args": args, "observation": obs, "faults": applied})
            return obs

        if self._on("stale_read", tool) and tool not in MUTATING and self._mutated:
            applied.append("stale_read")
            obs = self._shadow.call(tool, **args)
        else:
            obs = self.env.call(tool, **args)

        if tool in MUTATING and "error" not in obs:
            self._mutated = True

        key = LIST_TOOLS.get(tool)
        if key and isinstance(obs, dict) and key in obs:
            i, n = _page(args)
            if i > 0 and self._on("pagination_repeat", tool):
                applied.append("pagination_repeat")
                first = dict(args, page_index=0)
                obs = dict(obs, **{key: self.env.call(tool, **first).get(key, [])})
            elif i > 0 and self._on("pagination_empty", tool):
                applied.append("pagination_empty")
                obs = dict(obs, **{key: []})
            if self._on("entity_hallucination", tool):
                applied.append("entity_hallucination")
                obs = dict(obs, **{key: list(obs[key]) + [self._fabricate(key)]})

        if self._on("json_wrapped", tool):
            applied.append("json_wrapped")
            obs = {"role": "tool", "content": json.dumps(obs)}

        self.trace.append({"tool": tool, "args": args, "observation": obs, "faults": applied})
        return obs

    @staticmethod
    def _fabricate(key):
        return {"customers": {"id": 99001, "email": "ghost@example.com", "full_name": "Ghost Record"},
                "orders": {"id": 99002, "account_id": 11, "placed_at": "2026-03-02T09:00:00", "status": "placed"},
                "refunds": {"id": 99003, "charge_id": 9001, "amount_cents": 1, "idempotency_key": "ghost"},
                "notes": {"id": 99004, "customer_id": 1, "body": "ghost note"}}[key]
