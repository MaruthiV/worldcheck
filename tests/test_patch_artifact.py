import subprocess
from pathlib import Path

from worldcheck import fix

PATCH = Path(__file__).resolve().parent.parent / "patches" / "appworld_plugin_pagination.patch"


def test_patch_file_matches_the_replacements(patronus):
    body = fix.unified_diff(patronus / "appworld")
    on_disk = PATCH.read_text()
    assert body in on_disk, "patches/ is stale; regenerate it from worldcheck.fix"


def test_patch_applies_cleanly_to_the_pinned_checkout(patronus):
    r = subprocess.run(["git", "apply", "--check", "-p1", str(PATCH)],
                       cwd=patronus, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_replacements_are_anchored_to_upstream_text(patronus):
    src = (patronus / "appworld" / fix.TARGET).read_text()
    for label, old, _ in fix.REPLACEMENTS:
        assert src.count(old) == 1, f"{label}: upstream text drifted"
