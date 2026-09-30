import json
import random
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "grpo"
OUT = ROOT / "results" / "grpo_summary.json"
SEEDS = (0, 1, 2)
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


def main():
    rng = random.Random(0)
    out = {"decision_rule": "supported if the row-bootstrap 95% interval excludes zero and at least 2 of 3 "
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
                      "supported": bool(ci and (ci[0] > 0 or ci[1] < 0) and agree >= 2)}
        out["effects"][f"under the {eval_arm} plugin"] = eff

    for arm in ("shipped", "patched"):
        for s in SEEDS:
            c = training_curve(arm, s)
            if c:
                out["training_curves"][f"grpo-{arm}-s{s}"] = c

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
