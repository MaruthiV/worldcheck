import pytest

from tests.conftest import PAGE_LIMIT, logged_in_episode
from worldcheck.driver import ids

UPSTREAM = "upstream"
PATCHED = "patched"


def trees(request):
    return request.getfixturevalue("patronus"), request.getfixturevalue("patched_dir")


@pytest.fixture
def both(request, patronus, patched_dir):
    return {UPSTREAM: patronus, PATCHED: patched_dir}


def load(plugin_dir):
    from worldcheck import upstream as u
    plugin = u.load_adapter(plugin_dir)

    def no_model(*a, **k):
        raise AssertionError("world model called")

    plugin.call_world_model = no_model
    return plugin


def sweep(plugin, row, pages=4, limit=PAGE_LIMIT):
    ep = logged_in_episode(plugin, row, "spotify", "sweep")
    out = []
    for page in range(pages):
        got = ep.call("spotify__show_song_library", page_index=page, page_limit=limit)
        out.append(ids(got, "songs"))
    return ep, out


def test_upstream_loses_every_page_after_the_first(both, song_row):
    plugin = load(both[UPSTREAM])
    ep, pages = sweep(plugin, song_row)
    assert len(pages[0]) == PAGE_LIMIT
    assert pages[1] == [] and pages[2] == [] and pages[3] == []


def test_patched_returns_every_page(both, song_row):
    plugin = load(both[PATCHED])
    ep, pages = sweep(plugin, song_row)
    for page, got in enumerate(pages):
        want = ids(ep.guard_only("spotify__show_song_library", {"spotify"},
                                 page_index=page, page_limit=PAGE_LIMIT), "songs")
        assert got == want, f"page {page}"
    flat = [i for p in pages for i in p]
    assert len(flat) == 80 and len(set(flat)) == 80


def test_patched_first_call_to_a_later_page_works(both, song_row):
    plugin = load(both[PATCHED])
    ep = logged_in_episode(plugin, song_row, "spotify", "first-later")
    got = ep.call("spotify__show_song_library", page_index=1, page_limit=PAGE_LIMIT)
    want = ep.guard_only("spotify__show_song_library", {"spotify"}, page_index=1, page_limit=PAGE_LIMIT)
    assert ids(got, "songs") == ids(want, "songs") != []


def test_patched_out_of_order_pages_are_not_mislabelled(both, song_row):
    plugin = load(both[PATCHED])
    ep = logged_in_episode(plugin, song_row, "spotify", "ooo")
    p1 = ep.call("spotify__show_song_library", page_index=1, page_limit=PAGE_LIMIT)
    p0 = ep.call("spotify__show_song_library", page_index=0, page_limit=PAGE_LIMIT)
    want0 = ep.guard_only("spotify__show_song_library", {"spotify"}, page_index=0, page_limit=PAGE_LIMIT)
    want1 = ep.guard_only("spotify__show_song_library", {"spotify"}, page_index=1, page_limit=PAGE_LIMIT)
    assert ids(p0, "songs") == ids(want0, "songs")
    assert ids(p1, "songs") == ids(want1, "songs")
    assert ids(p0, "songs") != ids(p1, "songs")


@pytest.mark.parametrize("args", [
    {"page_index": 0, "page_limit": 0},
    {"page_index": -1, "page_limit": PAGE_LIMIT},
])
def test_patched_clamps_agree_with_the_guard(both, song_row, args):
    plugin = load(both[PATCHED])
    ep = logged_in_episode(plugin, song_row, "spotify", "clamp")
    got = ep.call("spotify__show_song_library", **args)
    want = ep.guard_only("spotify__show_song_library", {"spotify"}, **args)
    assert ids(got, "songs") == ids(want, "songs")


def test_patched_leaves_single_record_responses_alone(both, song_row):
    plugin = load(both[PATCHED])
    ep = logged_in_episode(plugin, song_row, "spotify", "playlist")
    pid = ep.guard_only("spotify__show_playlist", {"spotify"}, playlist_id=300)
    got = ep.call("spotify__show_playlist", playlist_id=300)
    assert got == pid
    assert "total" not in got


def test_upstream_truncates_a_single_record_list_field(both, song_row):
    plugin = load(both[UPSTREAM])
    ep = logged_in_episode(plugin, song_row, "spotify", "playlist-upstream")
    want = ep.guard_only("spotify__show_playlist", {"spotify"}, playlist_id=300)
    got = ep.call("spotify__show_playlist", playlist_id=300)
    assert len(got["songs"]) < len(want["songs"])
    assert "total" in got and "total" not in want


def test_patched_still_pages_an_overlong_reply(both, song_row):
    # the guard never over-returns, but a neural wm can; this is the Table 7 (i) defence
    plugin = load(both[PATCHED])
    ep = logged_in_episode(plugin, song_row, "spotify", "overlong")
    full = ep.guard_only("spotify__show_song_library", {"spotify"}, page_index=0, page_limit=20)
    oversized = dict(full)
    page1 = ep.guard_only("spotify__show_song_library", {"spotify"}, page_index=1, page_limit=20)
    oversized["songs"] = full["songs"] + page1["songs"]

    import json
    captured = {}
    orig = plugin.expected_appworld_response

    def over(*a, **k):
        if a[1] == "spotify__show_song_library":
            captured["used"] = True
            return json.dumps(oversized)
        return orig(*a, **k)

    plugin.expected_appworld_response = over
    try:
        got = ep.call("spotify__show_song_library", page_index=1, page_limit=20)
    finally:
        plugin.expected_appworld_response = orig
    assert captured.get("used")
    assert ids(got, "songs") == ids(page1, "songs")


def test_read_after_write_is_unchanged_by_the_patch(both, note_row):
    # documented design in the paper appendix, the patch must not pretend to fix it
    out = {}
    for name in (UPSTREAM, PATCHED):
        plugin = load(both[name])
        ep = logged_in_episode(plugin, note_row, "simple_note", f"raw-{name}")
        nid = 2704
        before = ep.call("simple_note__show_note", note_id=nid)
        upd = ep.call("simple_note__update_note", note_id=nid, title="WORLDCHECK_AFTER")
        after = ep.call("simple_note__show_note", note_id=nid)
        out[name] = (before.get("title"), upd.get("status"), after.get("title"))
    assert out[UPSTREAM] == out[PATCHED]
    assert out[PATCHED][1] == "success"
    assert out[PATCHED][0] == out[PATCHED][2]
