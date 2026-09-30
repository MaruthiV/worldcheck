import json
import random
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "grpo"
OUT = ROOT / "results" / "grpo_summary.json"
SEEDS = (0, 1, 2, 3, 4)
RESAMPLES = 10000
LIST_TOOLS = {"spotify__show_song_library", "spotify__show_liked_songs", "spotify__search_songs",
              "spotify__show_album_library", "spotify__show_playlist_library", "spotify__show_recommendations",
              "spotify__show_following_artists", "venmo__show_transactions", "venmo__show_social_feed",
              "venmo__show_received_payment_requests", "venmo__show_sent_payment_requests",
              "venmo__search_friends", "file_system__show_directory", "simple_note__search_notes"}


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def correct(answer, gt):
    if answer is None:
        return False
    a, g = str(answer).strip().lower(), str(gt).strip().lower()
    if g.isdigit():
        return _num(a) == int(g)
    return a == g


METRICS = {
    "question_accuracy": lambda e: correct(e["answer"], e["ground_truth"]) if e["ground_truth"] else None,
    "question_accuracy_substring_rule": lambda e: (e["answer"] is not None and str(e["ground_truth"]).lower()
                                                   in str(e["answer"]).lower()) if e["ground_truth"] else None,
    "pages_past_first": lambda e: any(p >= 1 for p in e["list_pages"]) if e["list_pages"] else None,
    "highest_page": lambda e: max(e["list_pages"]) if e["list_pages"] else None,
    "reward": lambda e: e["reward"],
    "unparsed_turn_rate": lambda e: e["unparsed_turns"] / max(e["turns"], 1),
    "model_observation_rate": lambda e: (e["sources"].count("model") / len(e["sources"])) if e["sources"] else None,
    "multi_number_answer": lambda e: (len(set(re.findall(r"\d+", str(e["answer"] or "")))) >= 2)
                                     if str(e["ground_truth"]).isdigit() else None,
    "answer_chars": lambda e: len(str(e["answer"] or "")) if e["ground_truth"] else None,
}


def load(tag, arm):
    p = RAW / f"{tag}-{arm}.json"
    return json.loads(p.read_text())["episodes"] if p.exists() else None


# per-row means first, so a row with many samples can't outweigh one with few
def by_row(eps, fn):
    rows = {}
    for e in eps:
        v = fn(e)
        if v is not None:
            rows.setdefault(e["task_id"], []).append(float(v))
    return {k: statistics.mean(v) for k, v in rows.items()}


def mean_of(rows):
    return statistics.mean(rows.values()) if rows else None


def boot_diff(a, b, rng):
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    diffs = []
    for _ in range(RESAMPLES):
        pick = [rng.choice(keys) for _ in keys]
        diffs.append(statistics.mean(a[k] for k in pick) - statistics.mean(b[k] for k in pick))
    diffs.sort()
    return [round(diffs[int(0.025 * RESAMPLES)], 4), round(diffs[int(0.975 * RESAMPLES) - 1], 4)]


def training_curve(arm, seed, bins=10):
    p = RAW / "train" / f"grpo-{arm}-s{seed}" / "trajectories.jsonl"
    if not p.exists():
        return None
    calls = []
    for line in p.read_text().splitlines():
        tc = json.loads(line).get("tool_call") or {}
        if tc.get("name") in LIST_TOOLS:
            calls.append(_num(tc.get("args", {}).get("page_index", "0")) or 0)
    if not calls:
        return None
    size = max(1, len(calls) // bins)
    return [round(sum(c >= 1 for c in calls[i:i + size]) / len(calls[i:i + size]), 3)
            for i in range(0, size * bins, size) if calls[i:i + size]]


# appworld_wm_prompt.py:384-390, the guard's own list of tools that change state
MUTATION_PREFIXES = (
    "spotify__like_", "spotify__unlike_", "spotify__review_", "spotify__create_", "spotify__add_", "spotify__remove_",
    "spotify__follow_", "spotify__download_", "venmo__create_", "venmo__approve_", "venmo__like_", "venmo__remind_",
    "venmo__deny_", "phone__send_", "phone__delete_", "file_system__create_", "file_system__move_",
    "file_system__delete_", "file_system__compress_", "simple_note__update_", "simple_note__create_",
    "simple_note__delete_")


def finished_without_acting(eps):
    if not eps:
        return None
    acts = [e for e in eps if not e["ground_truth"]]
    hit = [e for e in acts if "supervisor__complete_task" in e["tools"] and e["reward"] >= 0.9
           and not any(t and t.startswith(MUTATION_PREFIXES) for t in e["tools"])]
    return round(len(hit) / len(acts), 4) if acts else None


def _list_len(text):
    try:
        d = json.loads(text)
    except (TypeError, ValueError):
        return None
    return next((len(v) for v in d.values() if isinstance(v, list)), None) if isinstance(d, dict) else None


# replay every later-page request from a training log through the guard, which pages correctly
def live_page_losses(arm, seed, plugin, rows_by_instruction):
    p = RAW / "train" / f"grpo-{arm}-s{seed}" / "trajectories.jsonl"
    if not p.exists():
        return None
    out = {"later_page_requests": 0, "held_records_but_came_back_empty": 0, "correctly_empty": 0, "other": 0}
    for line in p.read_text().splitlines():
        step = json.loads(line)
        tc = step.get("tool_call") or {}
        if (_num(tc.get("args", {}).get("page_index", "0")) or 0) < 1:
            continue
        out["later_page_requests"] += 1
        args = {k: (int(v) if str(v).lstrip("-").isdigit() else v) for k, v in tc.get("args", {}).items()
                if k != "access_token"}
        should = [_list_len(plugin.expected_appworld_response(r["wm_system_prompt"], tc["name"], args,
                                                              logged_in_apps={tc["name"].split("__")[0]}))
                  for r in rows_by_instruction.get(step.get("instruction"), [])]
        should = max((n for n in should if n is not None), default=None)
        got = _list_len(step.get("wm_response"))
        if got == 0 and should:
            out["held_records_but_came_back_empty"] += 1
        elif got == 0 and should == 0:
            out["correctly_empty"] += 1
        else:
            out["other"] += 1
    return out


def main():
    rng = random.Random(0)
    out = {"decision_rule": "supported if the row-bootstrap 95% interval excludes zero and at least 4 of 5 "
                            "per-seed differences share its sign", "levels": {}, "effects": {}, "training_curves": {}}
    pooled = {}
    for eval_arm in ("patched", "shipped"):
        for train in ("sft", "shipped", "patched"):
            tags = ["sft-s0"] if train == "sft" else [f"grpo-{train}-s{s}" for s in SEEDS]
            eps = [e for t in tags for e in (load(t, eval_arm) or [])]
            if not eps:
                continue
            pooled[(train, eval_arm)] = eps
            out["levels"][f"{train}-trained, {eval_arm} plugin"] = {
                m: (round(v, 4) if (v := mean_of(by_row(eps, fn))) is not None else None) for m, fn in METRICS.items()}

    for eval_arm in ("patched", "shipped"):
        if ("patched", eval_arm) not in pooled or ("shipped", eval_arm) not in pooled:
            continue
        eff = {}
        for m, fn in METRICS.items():
            a, b = by_row(pooled[("patched", eval_arm)], fn), by_row(pooled[("shipped", eval_arm)], fn)
            if not a or not b:
                continue
            diff = mean_of(a) - mean_of(b)
            ci = boot_diff(a, b, rng)
            per_seed = []
            for s in SEEDS:
                pa, sh = load(f"grpo-patched-s{s}", eval_arm), load(f"grpo-shipped-s{s}", eval_arm)
                if pa and sh:
                    ra, rb = by_row(pa, fn), by_row(sh, fn)
                    if ra and rb:
                        per_seed.append(round(mean_of(ra) - mean_of(rb), 4))
            agree = sum((d > 0) == (diff > 0) and d != 0 for d in per_seed)
            eff[m] = {"patched_minus_shipped": round(diff, 4), "ci95": ci, "per_seed": per_seed,
                      "supported": bool(ci and (ci[0] > 0 or ci[1] < 0) and agree >= max(2, len(per_seed) - 1))}
        out["effects"][f"under the {eval_arm} plugin"] = eff

    for arm in ("shipped", "patched"):
        for s in SEEDS:
            c = training_curve(arm, s)
            if c:
                out["training_curves"][f"grpo-{arm}-s{s}"] = c

    if list((RAW / "train").glob("grpo-*/trajectories.jsonl")):
        import os
        import tempfile
        os.environ.setdefault("TRAJECTORY_LOG", str(Path(tempfile.mkdtemp()) / "traj.jsonl"))
        from worldcheck.driver import setup
        patronus, plugin, _ = setup()
        rows = {}
        for line in open(patronus / "appworld" / "data" / "appworld_rl_split_clean.jsonl"):
            r = json.loads(line)
            rows.setdefault(r["instruction"][:60], []).append(r)
        out["live_page_losses"] = {f"grpo-{arm}-s{s}": v for arm in ("shipped", "patched") for s in SEEDS
                                   if (v := live_page_losses(arm, s, plugin, rows))}

    out["exploratory_finishing_without_acting"] = {
        "note": "not predeclared; found by reading episodes. share of action-task episodes that call "
                "complete_task, make no call from the guard's own mutation_prefixes, and still score >= 0.9",
        **{ev: {t: v for t in ["sft-s0"] + [f"grpo-{a}-s{s}" for a in ("shipped", "patched") for s in SEEDS]
                if (v := finished_without_acting(load(t, ev))) is not None}
           for ev in ("patched", "shipped")}}

    OUT.write_text(json.dumps(out, indent=2) + "\n")
    for k, v in out["levels"].items():
        print(f"{k:34} " + "  ".join(f"{m}={x}" for m, x in v.items()))
    for k, v in out["effects"].items():
        print(f"\npatched-trained minus shipped-trained, {k}:")
        for m, e in v.items():
            print(f"  {m:34} {e['patched_minus_shipped']:+.4f}  ci {e['ci95']}  seeds {e['per_seed']}  "
                  f"{'SUPPORTED' if e['supported'] else 'not supported'}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
