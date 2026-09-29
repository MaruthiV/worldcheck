import os
import tempfile
from pathlib import Path

import pytest

from worldcheck import fix, upstream
from worldcheck.driver import credentials_in, find_row, load_row

os.environ.setdefault("TRAJECTORY_LOG", str(Path(tempfile.mkdtemp()) / "traj.jsonl"))

PAGE_LIMIT = 20


@pytest.fixture(scope="session")
def patronus():
    return upstream.ensure_checkout(upstream.PATRONUS_REPO, upstream.PATRONUS_COMMIT, "mdlm_world_modeling")


@pytest.fixture(scope="session")
def msswift():
    return upstream.ensure_checkout(upstream.MSSWIFT_REPO, upstream.MSSWIFT_COMMIT, "ms-swift")


@pytest.fixture(scope="session")
def patched_dir(patronus):
    return fix.patched_tree(patronus / "appworld")


@pytest.fixture(scope="session")
def song_row(patronus):
    return load_row(patronus, 0)


@pytest.fixture(scope="session")
def note_row(patronus):
    return find_row(patronus, "notes (")[1]


@pytest.fixture
def adapter(request):
    tree = request.param
    plugin = upstream.load_adapter(tree)

    def no_model(*a, **k):
        raise AssertionError("world model called; the guard path needs no model")

    plugin.call_world_model = no_model
    return plugin


def logged_in_episode(plugin, row, app, tag):
    from worldcheck.driver import Episode, new_scheduler
    ep = Episode(plugin, row, f"test-{tag}").bind(new_scheduler(plugin))
    ep.login(app, credentials_in(row["wm_system_prompt"]))
    return ep
