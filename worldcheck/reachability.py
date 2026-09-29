import json
import re
import sys
from pathlib import Path

from worldcheck import upstream

RESULTS = Path(__file__).resolve().parent.parent / "results" / "t1_reachability.json"
RECOMMENDED_PAGE_LIMIT = 20


def sections(prompt):
    out = {}
    for m in re.finditer(r"^\s+(\w+) \((\d+) total(?:, (\d+) shown)?\)", prompt, re.M):
        total = int(m.group(2))
        out[m.group(1)] = {"total": total, "shown": int(m.group(3)) if m.group(3) else total}
    return out


def main():
    patronus = upstream.ensure_checkout(upstream.PATRONUS_REPO, upstream.PATRONUS_COMMIT, "mdlm_world_modeling")
    path = patronus / "appworld/data/appworld_rl_split_clean.jsonl"

    rows, counts = [], {}
    for line in path.open():
        r = json.loads(line)
        s = sections(r["wm_system_prompt"])
        over = {k: v for k, v in s.items() if v["shown"] > RECOMMENDED_PAGE_LIMIT}
        withheld = {k: v["total"] - v["shown"] for k, v in s.items() if v["total"] > v["shown"]}
        rows.append({
            "task_id": r.get("task_id"),
            "max_turns": r.get("max_turns"),
            "collections": s,
            "sections_needing_a_second_page": over,
            "affected": bool(over),
            "sections_where_prompt_withholds": withheld,
            "records_withheld": sum(withheld.values()),
            "has_notes": "notes" in s,
        })
        for k in s:
            counts[k] = counts.get(k, 0) + 1

    affected = [r for r in rows if r["affected"]]
    out = {
        "patronus_commit": upstream.PATRONUS_COMMIT,
        "split": "appworld/data/appworld_rl_split_clean.jsonl",
        "recommended_page_limit": RECOMMENDED_PAGE_LIMIT,
        "page_limit_source": "appworld_prompt.py:35",
        "stop_rule_source": "appworld_prompt.py:36",
        "rows": len(rows),
        "rows_where_listed_records_exceed_page_limit": len(affected),
        "rows_where_prompt_advertises_more_than_it_lists": sum(1 for r in rows if r["records_withheld"] > 0),
        "total_records_withheld_by_prompts": sum(r["records_withheld"] for r in rows),
        "largest_listed_collection_per_row": sorted(
            (max((v["shown"] for v in r["collections"].values()), default=0)) for r in rows
        ),
        "collection_frequency": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "rows_with_notes": [r["task_id"] for r in rows if r["has_notes"]],
        "per_row": rows,
    }
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2) + "\n")

    print(f"rows: {out['rows']}")
    print(f"rows whose LISTED records exceed page_limit {RECOMMENDED_PAGE_LIMIT}: "
          f"{out['rows_where_listed_records_exceed_page_limit']}  (the patchable harm)")
    print(f"rows advertising a total larger than the records they list: "
          f"{out['rows_where_prompt_advertises_more_than_it_lists']}, "
          f"{out['total_records_withheld_by_prompts']} records withheld  (not a plugin bug)")
    print(f"collections seen: {out['collection_frequency']}")
    print(f"rows containing notes: {len(out['rows_with_notes'])}")
    print(f"results: {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
