import json
import re
import statistics
from itertools import combinations
from pathlib import Path

from worldcheck import verify
from worldcheck.env import invariants, seed
from worldcheck.env.engine import Env
from worldcheck.env.tasks import TASKS
from worldcheck.policies import POLICIES, run
from worldcheck.sim.injector import FAULTS, GroundTruth, Injector

GT = {"N1": 26050, "N2": 6, "N3": [7001], "N4": 1, "D2": 1, "D3": 1}
MARGIN = 0.5 / len(TASKS)
CONFIGS = {"ground_truth": ()} | {f: (f,) for f in FAULTS} | {
    "appworld_like": ("pagination_empty", "ack_without_persist")}
OUT = Path("results/calibration.json")


# appworld_plugin.py:581
def failed(text):
    t = text.lower()
    return '"error"' in t or "execution failed" in t or "traceback" in t


# shape of AppWorldReward minus the credential multipliers, which have nothing to act on here
def proxy(trace, answer, gt):
    texts = [json.dumps(s["observation"]) for s in trace]
    ok = [t for t in texts if t.strip() and not failed(t)]
    data = any(len(t) > 50 or '"id"' in t or '"title"' in t for t in ok)
    if gt is not None:
        ans = str(answer)
        # substring match, their rule at :666
        if str(gt).lower() in ans.lower():
            return 1.0
        if isinstance(gt, int):
            nums = [int(x) for x in re.findall(r"\d+", ans)]
            q = 0.0
            if nums:
                near = min(nums, key=lambda x: abs(x - gt))
                q = (1 - min(abs(near - gt) / max(gt, 1), 1)) * 0.25
            return max(q, 0.15) if data else q
        return 0.15 if data else 0.08 if ok else 0.03
    if not texts:
        return 0.0
    r = 0.6 * len(ok) / len(texts) + (0.1 if data else 0.0)
    sigs = [s["tool"] + json.dumps(s["args"], sort_keys=True) for s in trace]
    if len(sigs) > 2 and (len(sigs) - len(set(sigs))) / len(sigs) > 0.5:
        r *= 0.5
    return r


def grade(task, env, before, answer):
    return 1.0 if task["grade"](env, before, answer)[0] else 0.0


def episode(faults, policy, task):
    env = Env(seed.sql())
    before = invariants.fingerprint(env)
    b = Injector(env, faults=faults) if faults else GroundTruth(env)
    answer = run(b, policy, task["id"])
    return b.trace, answer, env, before


# the simulated run's actions, open loop, against a fresh reference
def shadow(task, trace, answer):
    env = Env(seed.sql())
    before = invariants.fingerprint(env)
    for s in trace:
        env.call(s["tool"], **s["args"])
    return grade(task, env, before, answer)


def findings(trace, declared):
    out, hist = [], []
    for s in trace:
        out += verify.verify(declared, s["tool"], s["args"], s["observation"], hist)
        hist.append(s)
    return out


def _order(a, b):
    return 0 if abs(a - b) < MARGIN else (1 if a > b else -1)


def compare(truth, view):
    classes = {"agree": [], "collapse": [], "spurious": [], "reversal": []}
    for x, y in combinations(POLICIES, 2):
        t, v = _order(truth[x], truth[y]), _order(view[x], view[y])
        c = "agree" if t == v else "collapse" if v == 0 else "spurious" if t == 0 else "reversal"
        classes[c].append([x, y])
    top = max(view.values())
    picks = [p for p in POLICIES if top - view[p] < MARGIN]
    return {"counts": {k: len(v) for k, v in classes.items()},
            "pairs": {k: v for k, v in classes.items() if k != "agree"},
            "top_by_view": picks,
            "regret": round(max(truth.values()) - min(truth[p] for p in picks), 4)}


def _r(xs, ys):
    try:
        return round(statistics.correlation(xs, ys), 4)
    except statistics.StatisticsError:
        return None


def calibrate():
    ref = {p: {t["id"]: episode((), p, t) for t in TASKS} for p in POLICIES}
    cell_truth = {}
    for p in POLICIES:
        for t in TASKS:
            _, answer, env, before = ref[p][t["id"]]
            cell_truth[(p, t["id"])] = grade(t, env, before, answer)
    truth = {p: statistics.mean(cell_truth[(p, t["id"])] for t in TASKS) for p in POLICIES}
    reward_only = {p: statistics.mean(proxy(ref[p][t["id"]][0], ref[p][t["id"]][1], GT.get(t["id"]))
                                      for t in TASKS) for p in POLICIES}
    declared = invariants.fingerprint(Env(seed.sql()))

    configs = {}
    for name, faults in CONFIGS.items():
        train, shad, cells, flagged, by_check = {}, {}, [], 0, {}
        for p in POLICIES:
            tr, sh = [], []
            for t in TASKS:
                trace, answer, _, _ = episode(faults, p, t)
                tr.append(proxy(trace, answer, GT.get(t["id"])))
                sh.append(shadow(t, trace, answer))
                cells.append((tr[-1], cell_truth[(p, t["id"])]))
                f = findings(trace, declared)
                flagged += bool(f)
                for x in f:
                    by_check[x["check"]] = by_check.get(x["check"], 0) + 1
            train[p], shad[p] = statistics.mean(tr), statistics.mean(sh)
        configs[name] = {
            "faults": list(faults),
            "training_signal": {p: round(v, 4) for p, v in train.items()},
            "shadow": {p: round(v, 4) for p, v in shad.items()},
            "training_signal_vs_truth": compare(truth, train),
            "shadow_vs_truth": compare(truth, shad),
            "r_proxy_vs_truth_per_episode": _r([c[0] for c in cells], [c[1] for c in cells]),
            "verifier": {"episodes_flagged": flagged, "episodes": len(cells), "by_check": by_check},
        }

    return {
        "meta": {"policies": POLICIES, "tasks": [t["id"] for t in TASKS], "tie_margin": round(MARGIN, 4),
                 "proxy_omits": "credential, placeholder and schema-argument multipliers"},
        "truth": {p: round(v, 4) for p, v in truth.items()},
        "reward_only": {p: round(v, 4) for p, v in reward_only.items()},
        "reward_only_vs_truth": compare(truth, reward_only),
        "configs": configs,
    }


def main():
    res = calibrate()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"{'policy':28} truth  reward_only")
    for p in POLICIES:
        print(f"{p:28} {res['truth'][p]:.3f}  {res['reward_only'][p]:.3f}")
    c = res["reward_only_vs_truth"]
    print(f"\nreward only vs truth: {c['counts']}, regret {c['regret']}\n")
    print(f"{'config':22} {'training signal vs truth':44} {'shadow vs truth':44} {'r':>7} flagged")
    for name, cfg in res["configs"].items():
        t, s = cfg["training_signal_vs_truth"], cfg["shadow_vs_truth"]
        fmt = lambda c: f"{c['counts']['agree']:2}a {c['counts']['collapse']:2}c {c['counts']['spurious']:2}s {c['counts']['reversal']:2}r  regret {c['regret']:.3f}"
        v = cfg["verifier"]
        print(f"{name:22} {fmt(t):44} {fmt(s):44} {str(cfg['r_proxy_vs_truth_per_episode']):>7} "
              f"{v['episodes_flagged']}/{v['episodes']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
