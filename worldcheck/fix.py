import difflib
import shutil
import tempfile
from pathlib import Path

TARGET = "appworld_plugin.py"

OLD_IMPORT = "from appworld_wm_prompt import build_action_local_wm_prompt, expected_appworld_response"
NEW_IMPORT = ("from appworld_wm_prompt import (RETURN_KEY_BY_SECTION, SECTION_BY_TOOL, _page_args,\n"
              "                               build_action_local_wm_prompt, expected_appworld_response)")

OLD_CACHE_INIT = "        self._list_caches = {}  # per-request cache of full list responses for pagination\n"
NEW_CACHE_INIT = ""

OLD_BLOCK = '''        # Enforce pagination on list responses
        page_index = tool_args.get("page_index", 0)
        page_limit = tool_args.get("page_limit", 5)
        try:
            page_index = int(page_index)
            page_limit = min(int(page_limit), 20)
        except (ValueError, TypeError):
            page_index, page_limit = 0, 5

        req_id = getattr(infer_request, 'uuid', None) or id(infer_request)
        cache_args = {
            k: v for k, v in wm_args.items()
            if k not in {"page_index", "page_limit", "access_token"}
        }
        cache_key = f"{req_id}:{tool_name}:{json.dumps(cache_args, sort_keys=True, default=str)}"

        if wm_response.strip().startswith("{"):
            try:
                import json_repair
                parsed = json_repair.loads(wm_response)
                if isinstance(parsed, dict):
                    # Find the array field
                    for key in parsed:
                        if isinstance(parsed[key], list) and len(parsed[key]) > 0:
                            # Cache the full list on first call for this API
                            if cache_key not in self._list_caches:
                                self._list_caches[cache_key] = {
                                    "key": key,
                                    "data": parsed[key],
                                    "total": parsed.get("total", len(parsed[key])),
                                }
                            cached = self._list_caches[cache_key]
                            total = cached["total"]
                            full_data = cached["data"]
                            start = page_index * page_limit
                            parsed[key] = full_data[start:start + page_limit]
                            parsed["total"] = total
                            wm_response = json.dumps(parsed)
                            break
            except Exception:
                pass
'''

NEW_BLOCK = '''        # Enforce pagination on list responses
        page_index, page_limit = _page_args(tool_args)

        section = SECTION_BY_TOOL.get(tool_name)
        list_key = RETURN_KEY_BY_SECTION.get(section) if section else None

        if list_key and wm_response.strip().startswith("{"):
            try:
                import json_repair
                parsed = json_repair.loads(wm_response)
                data = parsed.get(list_key) if isinstance(parsed, dict) else None
                # guard and wm are each asked for one page, so only re-page an over-long reply
                if isinstance(data, list) and len(data) > page_limit:
                    start = page_index * page_limit
                    parsed[list_key] = data[start:start + page_limit]
                    wm_response = json.dumps(parsed)
            except Exception:
                pass
'''

REPLACEMENTS = [
    ("import", OLD_IMPORT, NEW_IMPORT),
    ("list cache init", OLD_CACHE_INIT, NEW_CACHE_INIT),
    ("pagination block", OLD_BLOCK, NEW_BLOCK),
]


def apply_to_text(src):
    out = src
    for label, old, new in REPLACEMENTS:
        n = out.count(old)
        if n != 1:
            raise AssertionError(f"{label}: expected exactly 1 occurrence upstream, found {n}")
        out = out.replace(old, new, 1)
    return out


def patched_tree(appworld_dir, dest=None):
    appworld_dir = Path(appworld_dir)
    dest = Path(dest or tempfile.mkdtemp(prefix="worldcheck-patched-"))
    dest.mkdir(parents=True, exist_ok=True)
    for py in appworld_dir.glob("*.py"):
        shutil.copy2(py, dest / py.name)
    target = dest / TARGET
    target.write_text(apply_to_text(target.read_text()))
    return dest


def unified_diff(appworld_dir):
    src = (Path(appworld_dir) / TARGET).read_text()
    return "".join(difflib.unified_diff(
        src.splitlines(keepends=True),
        apply_to_text(src).splitlines(keepends=True),
        fromfile=f"a/appworld/{TARGET}",
        tofile=f"b/appworld/{TARGET}",
    ))
