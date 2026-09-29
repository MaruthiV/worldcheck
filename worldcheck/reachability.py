import json
import re
import sys
from pathlib import Path

from worldcheck import upstream

RESULTS = Path(__file__).resolve().parent.parent / "results" / "t1_reachability.json"
RECOMMENDED_PAGE_LIMIT = 20


def sections(prompt):
    return {m.group(1): int(m.group(2))
            for m in re.finditer(r"^\s+(\w+) \((\d+) total", prompt, re.M)}


def main():
    patronus = upstream.ensure_checkout(upstream.PATRONUS_REPO, upstream.PATRONUS_COMMIT, "mdlm_world_modeling")
    path = patronus / "appworld/data/appworld_rl_split_clean.jsonl"

    rows, counts = [], {}
    for line in path.open():
        r = json.loads(line)
        s = sections(r["wm_system_prompt"])
        over = {k: v for k, v in s.items() if v > RECOMMENDED_PAGE_LIMIT}
        rows.append({
            "task_id": r.get("task_id"),
            "max_turns": r.get("max_turns"),
            "collections": s,
            "over_page_limit": over,
            "affected": bool(over),
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
        "rows_with_a_collection_over_page_limit": len(affected),
        "largest_collection_per_row": sorted(
            (max(r["collections"].values()) if r["collections"] else 0) for r in rows
        ),
        "collection_frequency": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "rows_with_notes": [r["task_id"] for r in rows if r["has_notes"]],
        "per_row": rows,
    }
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2) + "\n")

    print(f"rows: {out['rows']}")
    print(f"rows with a collection larger than page_limit {RECOMMENDED_PAGE_LIMIT}: {out['rows_with_a_collection_over_page_limit']}")
    print(f"collections seen: {out['collection_frequency']}")
    print(f"rows containing notes: {len(out['rows_with_notes'])}")
    print(f"results: {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
