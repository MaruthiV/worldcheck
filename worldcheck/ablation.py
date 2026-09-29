import json
import os
import sys
import tempfile
from pathlib import Path

from worldcheck import fix, upstream
from worldcheck.driver import Episode, collections_in, credentials_in, new_scheduler, setup

RESULTS = Path(__file__).resolve().parent.parent / "results" / "t3_guard_ablation.json"
SPLIT = "appworld/data/appworld_rl_split_clean.jsonl"
PAGE_LIMIT = 20
MAX_PAGES = 8

SWEEP_TOOLS = {
    "songs": ["spotify__show_song_library", "spotify__show_liked_songs"],
    "albums": ["spotify__show_album_library"],
    "playlists": ["spotify__show_playlist_library"],
    "following_artists": ["spotify__show_following_artists"],
    "transactions": ["venmo__show_transactions", "venmo__show_social_feed"],
    "payment_requests": ["venmo__show_sent_payment_requests",
                         "venmo__show_received_payment_requests"],
    "venmo_friends": ["venmo__search_friends"],
    "notes": ["simple_note__search_notes"],
}


def target(plugin, prompt):
    # pick the tool that can actually enumerate something, and let the guard define the reference
    best = None
    for section, tools in SWEEP_TOOLS.items():
        for tool in tools:
            out = plugin.expected_appworld_response(
                prompt, tool, {"page_index": 0, "page_limit": PAGE_LIMIT},
                logged_in_apps={tool.split("__", 1)[0]})
            if not out:
                continue
            try:
                d = json.loads(out)
            except Exception:
                continue
            if not isinstance(d, dict) or "total" not in d:
                continue
            size = int(d["total"])
            if size > PAGE_LIMIT and (best is None or size > best[2]):
                best = (section, tool, size)
    return best


def sweep(plugin, row, tag, obedient_stub=False):
    prompt = row["wm_system_prompt"]
    t = target(plugin, prompt)
    if not t:
        return None
    section, tool, size = t
    header = collections_in(prompt).get(section, {})
    advertised = header.get("total", size)
    app = tool.split("__", 1)[0]
    creds = credentials_in(prompt)
    if app not in creds:
        return None

    if obedient_stub:
        def stub(wm_prompt, messages, extra):
            # a world model that returns exactly the records the prompt declares
            call = None
            for m in reversed(messages):
                if m.get("role") == "assistant" and m.get("content"):
                    try:
                        call = json.loads(m["content"])[0]
                        break
                    except Exception:
                        continue
            if not call:
                return json.dumps({"error": "could not parse the action"})
            out = plugin.expected_appworld_response(prompt, call["name"],
                                                    call.get("parameters", {}),
                                                    logged_in_apps={app})
            return out if out is not None else json.dumps(
                {"status": "success", "message": "Action completed"})
        plugin.call_world_model = stub
    else:
        def refuse(*a, **k):
            raise AssertionError("world model called with the guard on")
        plugin.call_world_model = refuse

    ep = Episode(plugin, row, tag).bind(new_scheduler(plugin))
    ep.login(app, creds)

    seen, pages, stop = [], 0, None
    for page in range(MAX_PAGES):
        got = ep.call(tool, page_index=page, page_limit=PAGE_LIMIT)
        pages += 1
        recs = next((v for v in got.values() if isinstance(v, list)), []) if isinstance(got, dict) else []
        seen += [json.dumps(r, sort_keys=True) for r in recs]
        if not recs:
            stop = page
            break
    return {
        "section": section, "tool": tool,
        "records_declared": size, "records_advertised": advertised,
        "prompt_withholds": advertised - size,
        "pages_requested": pages, "first_empty_page": stop,
        "records_seen": len(set(seen)),
        "fraction_seen": round(len(set(seen)) / size, 4) if size else None,
        "complete": len(set(seen)) >= size,
        "messages": ep.req.messages,
    }


def main():
    os.environ.setdefault("TRAJECTORY_LOG", str(Path(tempfile.mkdtemp()) / "traj.jsonl"))
    patronus, _, fidelity = setup()
    patched = fix.patched_tree(patronus / "appworld")

    arms = {
        "guard_on_shipped": {"tree": patronus, "guard": "1", "stub": False},
        "guard_on_patched": {"tree": patched, "guard": "1", "stub": False},
        "guard_off_obedient_stub": {"tree": patronus, "guard": "0", "stub": True},
    }

    rows = [json.loads(l) for l in (patronus / SPLIT).open()]
    per_arm = {}

    for name, cfg in arms.items():
        os.environ["APPWORLD_WM_GUARD"] = cfg["guard"]
        plugin = upstream.load_adapter(cfg["tree"])
        orm = plugin.AppWorldReward()
        out = []
        for i, row in enumerate(rows):
            r = sweep(plugin, row, f"{name}-{i}", obedient_stub=cfg["stub"])
            if r is None:
                continue
            msgs = r.pop("messages")
            r["reward"] = orm([""], ground_truth=[row.get("ground_truth", "")],
                              messages=[msgs], instruction=[row["instruction"]])[0]
            r["task_id"] = row.get("task_id")
            out.append(r)
        per_arm[name] = out

    def agg(rs):
        n = len(rs)
        return {
            "rows_swept": n,
            "rows_fully_enumerated": sum(1 for r in rs if r["complete"]),
            "mean_fraction_of_collection_seen": round(sum(r["fraction_seen"] for r in rs) / n, 4) if n else None,
            "mean_records_seen": round(sum(r["records_seen"] for r in rs) / n, 2) if n else None,
            "mean_records_declared": round(sum(r["records_declared"] for r in rs) / n, 2) if n else None,
            "rows_where_prompt_withholds_records": sum(1 for r in rs if r["prompt_withholds"] > 0),
            "mean_reward": round(sum(r["reward"] for r in rs) / n, 5) if n else None,
            "rows_stopping_on_page_1_or_earlier": sum(1 for r in rs if r["first_empty_page"] is not None and r["first_empty_page"] <= 1),
        }

    result = {
        "note": "the ablation the upstream README flags as 'optional - worth ablation' and never runs",
        "patronus_commit": upstream.PATRONUS_COMMIT,
        "page_limit": PAGE_LIMIT,
        "reference": "correct pagination over the records DECLARED in each row prompt. note the header reads (N total, M shown) and only M records are listed, so M is the ceiling any simulator can serve and N-M is withheld by the prompt itself",
        "arms_explained": {
            "guard_on_shipped": "APPWORLD_WM_GUARD=1, unmodified adapter. the shipped default",
            "guard_on_patched": "APPWORLD_WM_GUARD=1 with patches/appworld_plugin_pagination.patch applied",
            "guard_off_obedient_stub": "APPWORLD_WM_GUARD=0 with a stub world model that returns exactly the records the prompt declares. isolates the wrapper, and is NOT a neural model",
        },
        "limits": [
            "no neural world model was run, so the guard-off arm bounds the wrapper's contribution only",
            "fixed scripted sweeps, not a trained policy",
            "the reference is pagination over declared records, not a full executable environment",
        ],
        "shim_fidelity_ok": fidelity["ok"],
        "summary": {k: agg(v) for k, v in per_arm.items()},
        "per_row": per_arm,
    }
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(result, indent=2) + "\n")

    w = max(len(k) for k in per_arm)
    print(f"{'arm':<{w}}  rows  complete  mean_seen/size   frac   mean_reward  stop<=p1")
    for k, v in result["summary"].items():
        print(f"{k:<{w}}  {v['rows_swept']:>4}  {v['rows_fully_enumerated']:>8}  "
              f"{v['mean_records_seen']:>6}/{v['mean_records_declared']:<6}  "
              f"{v['mean_fraction_of_collection_seen']:>5}  {v['mean_reward']:>11}  {v['rows_stopping_on_page_1_or_earlier']:>7}")
    print(f"\nresults: {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
