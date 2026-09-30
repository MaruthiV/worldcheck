import json
import os
import sys
import tempfile
from pathlib import Path

from worldcheck.driver import credentials_in, setup

RESULTS = Path(__file__).resolve().parent.parent / "results" / "t1_answer_match.json"
SPLIT = "appworld/data/appworld_rl_split_clean.jsonl"
PAGE_LIMIT = 20
NUMBER_DUMP = " ".join(str(i) for i in range(101))
FIRST_PAGE_TOOLS = ["spotify__show_song_library", "spotify__show_album_library", "spotify__show_playlist_library",
                    "spotify__show_following_artists", "spotify__show_recommendations"]


def turn(tool, args, payload):
    return [{"role": "assistant", "content": json.dumps([{"name": tool, "parameters": args}])},
            {"role": "user", "content": f"<tool_response>\n{json.dumps(payload)}\n</tool_response>"}]


# the credential steps the reward expects, then a single complete_task
def trajectory(row, answer):
    creds = credentials_in(row["wm_system_prompt"])
    app = "spotify" if "spotify" in creds else next(iter(creds))
    pw = [{"app": a, "username": c["username"], "password": c["password"]} for a, c in creds.items()]
    msgs = turn("supervisor__show_account_passwords", {}, pw)
    msgs += turn(f"{app}__login", creds[app], {"access_token": "tok_valid", "token_type": "Bearer"})
    return msgs, json.dumps([{"name": "supervisor__complete_task", "parameters": {"answer": answer}}])


def score(plugin, row, answer):
    msgs, final = trajectory(row, answer)
    return plugin.AppWorldReward()([final], ground_truth=[row["ground_truth"]], messages=[msgs],
                                   instruction=[row["instruction"]])[0]


def _strings(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("title", "name") and isinstance(v, str):
                yield v
            else:
                yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from ([v] if isinstance(v, str) else _strings(v))


# everything an agent sees on the first page of each spotify list, i.e. what it could paste back
def first_page_names(plugin, row):
    names = []
    for tool in FIRST_PAGE_TOOLS:
        out = plugin.expected_appworld_response(row["wm_system_prompt"], tool,
                                                {"page_index": 0, "page_limit": PAGE_LIMIT},
                                                logged_in_apps={"spotify"})
        if out:
            names += [s for s in _strings(json.loads(out)) if s not in names]
    return names


def main():
    os.environ.setdefault("TRAJECTORY_LOG", str(Path(tempfile.mkdtemp()) / "traj.jsonl"))
    patronus, plugin, _ = setup()
    rows = [json.loads(l) for l in open(patronus / SPLIT)]
    per_row = []
    for row in rows:
        gt = row["ground_truth"]
        if not gt:
            continue
        rec = {"task_id": row["task_id"], "ground_truth_kind": "number" if gt.isdigit() else "text",
               "exact": score(plugin, row, gt)}
        if gt.isdigit():
            n = int(gt)
            rec["off_by_one"] = score(plugin, row, str(n + 1))
            rec["digit_added_in_front"] = score(plugin, row, f"1{gt}")
            rec["dump"] = {"answer": "every integer 0 to 100", "chars": len(NUMBER_DUMP),
                           "reward": score(plugin, row, NUMBER_DUMP)}
        else:
            names = first_page_names(plugin, row)
            dump = " | ".join(names)
            rec["dump"] = {"answer": "every title or name on the first page of each spotify list",
                           "names": len(names), "chars": len(dump), "contains_ground_truth": gt in names,
                           "reward": score(plugin, row, dump)}
        per_row.append(rec)

    numbers = [r for r in per_row if r["ground_truth_kind"] == "number"]
    texts = [r for r in per_row if r["ground_truth_kind"] == "text"]
    out = {
        "note": "rewards come from the unmodified AppWorldReward at the pinned commit",
        "mechanism": "appworld_plugin.py:666 gives full credit when the ground truth is a substring of the "
                     "answer, with no penalty for length or for extra candidates",
        "question_rows": len(per_row),
        "number_rows_where_dump_scores_full": sum(r["dump"]["reward"] == 1.0 for r in numbers),
        "number_rows": len(numbers),
        "text_rows_where_dump_scores_full": sum(r["dump"]["reward"] == 1.0 for r in texts),
        "text_rows": len(texts),
        "limits": ["trajectories are hand-built, not sampled from a trained agent",
                   "whether GRPO actually discovers these answers is a separate question"],
        "per_row": per_row,
    }
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2) + "\n")
    for r in per_row:
        extra = (f"off by one {r['off_by_one']:.3f}  '1'+answer {r['digit_added_in_front']:.3f}"
                 if "off_by_one" in r else f"ground truth on a first page: {r['dump']['contains_ground_truth']}")
        print(f"{r['task_id']:10} exact {r['exact']:.3f}  dump ({r['dump']['chars']} chars) "
              f"{r['dump']['reward']:.3f}  {extra}")
    print(f"\nnumber questions where '0 1 2 ... 100' scores 1.0: "
          f"{out['number_rows_where_dump_scores_full']} of {out['number_rows']}")
    print(f"text questions where pasting every first-page name scores 1.0: "
          f"{out['text_rows_where_dump_scores_full']} of {out['text_rows']}")
    print(f"results: {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
