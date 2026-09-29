import json
import os
import sys
import tempfile
from pathlib import Path

from worldcheck.driver import Episode, credentials_in, ids, load_row, new_scheduler, setup

RESULTS = Path(__file__).resolve().parent.parent / "results" / "t1_reward.json"
PAGE_LIMIT = 20


def turn_pair(tool, args, payload):
    return [
        {"role": "assistant", "content": json.dumps([{"name": tool, "parameters": args}])},
        {"role": "user", "content": f"<tool_response>\n{json.dumps(payload)}\n</tool_response>"},
    ]


def score(plugin, row, messages):
    orm = plugin.AppWorldReward()
    return orm([""], ground_truth=[row.get("ground_truth", "")],
               messages=[messages], instruction=[row["instruction"]])[0]


def main():
    os.environ.setdefault("TRAJECTORY_LOG", str(Path(tempfile.mkdtemp()) / "traj.jsonl"))
    patronus, plugin, _ = setup()
    row = load_row(patronus, 0)
    creds = credentials_in(row["wm_system_prompt"])
    scheduler = new_scheduler(plugin)

    # what the agent actually gets: page 0 fine, page 1 empty, stop rule fires
    ep = Episode(plugin, row, "reward-defective").bind(scheduler)
    ep.login("spotify", creds)
    seen_defective = []
    for page in range(4):
        got = ep.call("spotify__show_song_library", page_index=page, page_limit=PAGE_LIMIT)
        n = len(got.get("songs", []))
        seen_defective.append(n)
        if n == 0:
            break
    defective_msgs = list(ep.req.messages)

    # counterfactual with correct pagination, built from the guard's own output
    probe = Episode(plugin, row, "reward-guard-source").bind(scheduler)
    repaired_msgs = list(turn_pair("spotify__login", creds["spotify"],
                                   {"access_token": "tok_valid", "token_type": "Bearer"}))
    seen_repaired = []
    for page in range(5):
        want = probe.guard_only("spotify__show_song_library", {"spotify"},
                                page_index=page, page_limit=PAGE_LIMIT)
        n = len(want.get("songs", []))
        repaired_msgs += turn_pair("spotify__show_song_library",
                                   {"page_index": page, "page_limit": PAGE_LIMIT}, want)
        seen_repaired.append(n)
        if n == 0:
            break

    empty_page = {"total": 80, "songs": []}
    probes = {
        "empty_page_with_nonzero_total": turn_pair(
            "spotify__show_song_library", {"page_index": 1, "page_limit": PAGE_LIMIT}, empty_page),
        "explicit_error": turn_pair(
            "spotify__show_song_library", {"page_index": 1, "page_limit": PAGE_LIMIT},
            {"error": "500 Internal Server Error"}),
        "all_four_pages_empty": (
            turn_pair("spotify__login", creds["spotify"], {"access_token": "tok_valid", "token_type": "Bearer"})
            + [m for page in range(1, 5)
               for m in turn_pair("spotify__show_song_library",
                                  {"page_index": page, "page_limit": PAGE_LIMIT}, empty_page)]),
        "full_page": turn_pair(
            "spotify__show_song_library", {"page_index": 0, "page_limit": PAGE_LIMIT},
            probe.guard_only("spotify__show_song_library", {"spotify"}, page_index=0, page_limit=PAGE_LIMIT)),
    }

    out = {
        "note": "rewards come from the unmodified AppWorldReward in appworld_plugin.py at the pinned commit",
        "task_id": row.get("task_id"),
        "instruction": row["instruction"][:80],
        "instruction_note": "truncated; full row text stays upstream and is not republished here",
        "ground_truth_present": bool(row.get("ground_truth")),
        "page_limit": PAGE_LIMIT,
        "arms": {
            "defective": {
                "records_per_page": seen_defective,
                "records_seen": sum(seen_defective),
                "turns": len(defective_msgs) // 2,
                "reward": score(plugin, row, defective_msgs),
            },
            "repaired": {
                "records_per_page": seen_repaired,
                "records_seen": sum(seen_repaired),
                "turns": len(repaired_msgs) // 2,
                "reward": score(plugin, row, repaired_msgs),
            },
        },
        "single_observation_probes": {
            k: score(plugin, row, v) for k, v in probes.items()
        },
    }
    a = out["arms"]
    out["reward_gap"] = round(a["repaired"]["reward"] - a["defective"]["reward"], 6)
    out["reward_distinguishes_them"] = out["reward_gap"] != 0
    out["mechanism"] = (
        "appworld_plugin.py:583-584 classifies a tool_response as success when its text is non-empty "
        "and contains no 'error'. {\"total\": 80, \"songs\": []} satisfies both, so num_success counts it "
        "and success_rate stays 1.0 whether the pages carry records or not. reward = success_rate * 0.6 "
        "is therefore blind to how much of the collection the agent actually saw."
    )
    out["limits"] = [
        "trajectories are hand-built, not sampled from a trained policy",
        "absolute rewards are small because these trajectories skip the credential workflow the reward expects",
        "the supported claim is about the reward's discrimination between arms, not about absolute magnitudes",
        "no training run was performed, so downstream effect on a trained agent remains unmeasured",
    ]

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2) + "\n")

    print(f"defective: saw {a['defective']['records_seen']}/80 songs in {a['defective']['turns']} turns"
          f" -> reward {a['defective']['reward']:.4f}")
    print(f"repaired : saw {a['repaired']['records_seen']}/80 songs in {a['repaired']['turns']} turns"
          f" -> reward {a['repaired']['reward']:.4f}")
    print(f"gap: {out['reward_gap']:+.4f}")
    print()
    for k, v in out["single_observation_probes"].items():
        print(f"  probe {k:32s} reward {v:.4f}")
    print(f"\nresults: {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
