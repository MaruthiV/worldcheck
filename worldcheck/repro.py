import json
import os
import re
import sys
import tempfile
from pathlib import Path

from worldcheck import upstream
from worldcheck.driver import (Episode, collections_in, credentials_in, find_row,
                               ids, load_row, new_scheduler, setup)

RESULTS = Path(__file__).resolve().parent.parent / "results" / "t1_pagination.json"

def main():
    os.environ.setdefault("TRAJECTORY_LOG", str(Path(tempfile.mkdtemp()) / "traj.jsonl"))
    patronus, plugin, fidelity = setup()

    import json_repair
    assert json_repair.loads('{"a": 1}') == {"a": 1}

    row = load_row(patronus, 0)
    prompt = row["wm_system_prompt"]
    cols = collections_in(prompt)
    songs_total = cols.get("songs")

    # one scheduler for the whole run, matching rollout_mixin instantiating it once per trainer
    scheduler = new_scheduler(plugin)

    cases = {}
    creds = credentials_in(prompt)

    def episode(tag):
        return Episode(plugin, row, f"worldcheck-{tag}").bind(scheduler)

    # ascending sweep at the page_limit the agent prompt recommends
    ep = episode("sweep")
    ep.login("spotify", creds)
    sweep = []
    for page in range(4):
        got = ep.call("spotify__show_song_library", page_index=page, page_limit=20)
        want = ep.guard_only("spotify__show_song_library", {"spotify"}, page_index=page, page_limit=20)
        sweep.append({
            "page_index": page,
            "scheduler_ids": ids(got, "songs"),
            "guard_ids": ids(want, "songs"),
            "scheduler_total": got.get("total"),
            "n_returned": len(got.get("songs", [])),
            "payload_keys": sorted(got) if isinstance(got, dict) else None,
        })
    seen = sum(p["n_returned"] for p in sweep)
    stop_at = next((p["page_index"] for p in sweep if p["n_returned"] == 0), None)
    before_stop = sum(p["n_returned"] for p in sweep if stop_at is None or p["page_index"] < stop_at)
    cases["ascending_sweep"] = {
        "library_size": songs_total,
        "pages": sweep,
        "records_seen_all_pages": seen,
        "first_empty_page": stop_at,
        "records_seen_before_agent_stops": before_stop,
        "agent_stop_rule": "appworld_prompt.py:36 increment page_index until the response is empty",
    }

    # first ever call is page 1, nothing cached, proves this is not cache staleness
    ep = episode("first-page-1")
    ep.login("spotify", creds)
    got = ep.call("spotify__show_song_library", page_index=1, page_limit=20)
    want = ep.guard_only("spotify__show_song_library", {"spotify"}, page_index=1, page_limit=20)
    cases["first_call_is_page_1"] = {
        "scheduler_ids": ids(got, "songs"),
        "guard_ids": ids(want, "songs"),
        "empty": got.get("songs") == [],
        "total_still_reported": got.get("total"),
    }

    # out of order, page 1 then page 0
    ep = episode("out-of-order")
    ep.login("spotify", creds)
    first = ep.call("spotify__show_song_library", page_index=1, page_limit=20)
    second = ep.call("spotify__show_song_library", page_index=0, page_limit=20)
    cases["out_of_order"] = {
        "page1_then_page0_ids": ids(second, "songs"),
        "guard_page0_ids": ids(ep.guard_only("spotify__show_song_library", {"spotify"}, page_index=0, page_limit=20), "songs"),
        "page1_ids": ids(first, "songs"),
    }

    # clamp divergence, the plugin lacks the guard's max(0,..) and max(1,..)
    clamps = {}
    for label, args in [("page_limit_0", {"page_index": 0, "page_limit": 0}),
                        ("page_index_negative", {"page_index": -1, "page_limit": 20})]:
        ep = episode(f"clamp-{label}")
        ep.login("spotify", creds)
        got = ep.call("spotify__show_song_library", **args)
        want = ep.guard_only("spotify__show_song_library", {"spotify"}, **args)
        clamps[label] = {"scheduler_n": len(got.get("songs", [])), "guard_n": len(want.get("songs", []))}
    cases["clamp_divergence"] = clamps

    # single record response gets an unrelated list field sliced and a bogus total injected
    playlist_ids = re.findall(r"^\s+- id: (\d+),", prompt.split("playlists (")[1], re.M)[:1] if "playlists (" in prompt else []
    if playlist_ids:
        ep = episode("playlist")
        ep.login("spotify", creds)
        got = ep.call("spotify__show_playlist", playlist_id=int(playlist_ids[0]))
        want = ep.guard_only("spotify__show_playlist", {"spotify"}, playlist_id=int(playlist_ids[0]))
        listish = [k for k, v in (want or {}).items() if isinstance(v, list) and v]
        cases["single_record_truncation"] = {
            "playlist_id": int(playlist_ids[0]),
            "guard_list_fields": {k: len(want[k]) for k in listish},
            "scheduler_list_fields": {k: len(got.get(k, [])) for k in listish},
            "total_injected": "total" in got and "total" not in (want or {}),
        }

    # documented design, not a bug: mutation acked, later read unchanged
    _, note_row = find_row(patronus, "notes (")
    note_prompt = note_row["wm_system_prompt"]
    note_creds = credentials_in(note_prompt)
    first_note = re.search(r"^\s+- (?=.*\bid: )(.*)$", note_prompt.split("notes (", 1)[1], re.M)
    note_fields = dict(re.findall(r"(\w+): ([^,]+?)(?:,|$)", first_note.group(1))) if first_note else {}
    nid = int(note_fields["id"]) if "id" in note_fields else None
    if nid is not None:
        ep = Episode(plugin, note_row, "worldcheck-raw").bind(scheduler)
        ep.login("simple_note", note_creds)
        before = ep.call("simple_note__show_note", note_id=nid)
        upd = ep.call("simple_note__update_note", note_id=nid, title="WORLDCHECK_AFTER")
        after = ep.call("simple_note__show_note", note_id=nid)
        bogus = ep.call("simple_note__update_note", note_id=999999, title="nope")
        deleted = ep.call("simple_note__delete_note", note_id=nid)
        after_delete = ep.call("simple_note__show_note", note_id=nid)
        cases["read_after_write"] = {
            "row_task_id": note_row.get("task_id"),
            "note_id": nid,
            "title_before": before.get("title"),
            "update_status": upd.get("status"),
            "title_after_update": after.get("title"),
            "stale_read": before.get("title") == after.get("title"),
            "update_of_missing_record": bogus.get("status"),
            "delete_status": deleted.get("status"),
            "readable_after_delete": "title" in after_delete,
            "classification": "documented design, paper Appendix G, do not report as a defect",
        }

    out = {
        "apps_logged_in": sorted(creds),
        "patronus_commit": upstream.PATRONUS_COMMIT,
        "msswift_commit": upstream.MSSWIFT_COMMIT,
        "task_id": row.get("task_id"),
        "instruction": row["instruction"][:80],
        "instruction_note": "truncated; full row text stays upstream and is not republished here",
        "guard_env": {"APPWORLD_WM_GUARD": os.environ.get("APPWORLD_WM_GUARD", "unset, defaults to 1")},
        "real_dependencies": ["json_repair", "requests", "unmodified upstream appworld adapter"],
        "shimmed": upstream.SHIMMED,
        "shim_fidelity": fidelity,
        "world_model_calls": 0,
        "cases": cases,
    }
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2) + "\n")

    s = cases["ascending_sweep"]
    print(f"library: {s['library_size']} songs, page_limit 20")
    for p in s["pages"]:
        print(f"  page {p['page_index']}: scheduler {p['n_returned']:>2} ids={p['scheduler_ids']} | guard ids={p['guard_ids']}")
    print(f"agent stops at page {s['first_empty_page']}, having seen {s['records_seen_before_agent_stops']} of {s['library_size']}")
    print(f"first-ever page 1 empty: {cases['first_call_is_page_1']['empty']} (nothing cached)")
    print(f"results: {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
