import json
import re

from worldcheck import upstream

SPLIT = "appworld/data/appworld_rl_split_clean.jsonl"


def load_row(patronus, index=0):
    with open(patronus / SPLIT) as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(index)


def find_row(patronus, needle):
    with open(patronus / SPLIT) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            if needle in r["wm_system_prompt"]:
                return i, r
    raise LookupError(needle)


def credentials_in(prompt):
    return {m.group(1): {"username": m.group(2), "password": m.group(3)}
            for m in re.finditer(r"- (\w+): username=([^,]+), password=(\S+)", prompt)}


def collections_in(prompt):
    out = {}
    for m in re.finditer(r"^\s+(\w+) \((\d+) total(?:, (\d+) shown)?\)", prompt, re.M):
        total = int(m.group(2))
        out[m.group(1)] = {"total": total, "shown": int(m.group(3)) if m.group(3) else total}
    return out


def ids(payload, key):
    return [r.get("id") for r in payload.get(key, [])] if isinstance(payload, dict) else None


class Episode:
    def __init__(self, plugin, row, uuid):
        self.plugin = plugin
        self.scheduler = None
        self.row = row
        self.uuid = uuid
        self.req = upstream.RolloutInferRequest(
            messages=[],
            data_dict={"instruction": row["instruction"], "wm_system_prompt": row["wm_system_prompt"]},
            uuid=uuid,
        )
        self.turn = 0

    def bind(self, scheduler):
        self.scheduler = scheduler
        return self

    def login(self, app, creds):
        got = self.call(f"{app}__login", **creds[app])
        if "access_token" not in got:
            raise AssertionError(f"{app} login did not succeed: {got}")
        return got

    def call(self, tool, **args):
        raw = json.dumps([{"name": tool, "parameters": args}])
        self.req.messages.append({"role": "assistant", "content": raw})
        choice = upstream.ChatCompletionResponseChoice(
            index=0, message=upstream.ChatMessage(role="assistant", content=raw), finish_reason="stop"
        )
        ret = self.scheduler.step(self.req, choice, self.turn)
        self.req = ret["infer_request"]
        self.turn += 1
        return json.loads(self.body(self.req.messages[-1]["content"]))

    @staticmethod
    def body(content):
        return content.split("<tool_response>\n", 1)[1].rsplit("\n</tool_response>", 1)[0]

    def guard_only(self, tool, logged_in, **args):
        out = self.plugin.expected_appworld_response(
            self.row["wm_system_prompt"], tool, args, logged_in_apps=logged_in
        )
        return json.loads(out) if out is not None else None


def new_scheduler(plugin):
    s = plugin.AppWorldScheduler()
    s.use_wm = True
    return s


def setup():
    patronus = upstream.ensure_checkout(upstream.PATRONUS_REPO, upstream.PATRONUS_COMMIT, "mdlm_world_modeling")
    msswift = upstream.ensure_checkout(upstream.MSSWIFT_REPO, upstream.MSSWIFT_COMMIT, "ms-swift")
    fidelity = upstream.check_shim_fidelity(msswift)
    if not fidelity["ok"]:
        raise AssertionError("shim fidelity failed: " + "; ".join(fidelity["problems"]))
    plugin = upstream.load_adapter(patronus)

    def no_model(*a, **k):
        raise AssertionError("world model was called; guard path should need no model")

    plugin.call_world_model = no_model
    return patronus, plugin, fidelity
